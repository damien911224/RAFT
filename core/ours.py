import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from update import Decoder, PositionEmbedding

from deformable import DeformableTransformer
from utils.misc import inverse_sigmoid
import copy


try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):

    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if "dropout" not in self.args:
            self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=128, norm_fn="batch", dropout=args.dropout)

        d_model = 64
        num_feature_levels = 3
        self.num_feature_levels = num_feature_levels
        h, w = args.image_size[0], args.image_size[1]
        self.row_pos_embed = nn.ModuleList([nn.Embedding(w // (2 ** i), d_model // 2)
                                            for i in range(3 - self.num_feature_levels + 1, 4)])
        self.col_pos_embed = nn.ModuleList([nn.Embedding(h // (2 ** i), d_model // 2)
                                            for i in range(3 - self.num_feature_levels + 1, 4)])

        # self.row_query_embed = nn.ModuleList([nn.Embedding(w // (2 ** i), d_model // 2) for i in range(1, 4)])
        # self.col_query_embed = nn.ModuleList([nn.Embedding(h // (2 ** i), d_model // 2) for i in range(1, 4)])

        # self.row_tgt_embed = nn.Embedding(w // 8, d_model // 2)
        # self.col_tgt_embed = nn.Embedding(h // 8, d_model // 2)

        self.reset_parameters()

        self.transformer = DeformableTransformer(d_model=d_model, nhead=8,
                                                 num_encoder_layers=3, num_decoder_layers=3,
                                                 dim_feedforward=d_model * 4, dropout=0.1,
                                                 activation="relu", return_intermediate_dec=True,
                                                 num_feature_levels=num_feature_levels, dec_n_points=4, enc_n_points=4)

        self.flow_embed = MLP(d_model, d_model, 2, 3)
        input_proj_list = []
        for l_i in range(3 - num_feature_levels, 3):
            in_channels = (128, 192, 256)[l_i]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, d_model, kernel_size=1),
                nn.GroupNorm(d_model // 2, d_model)))
        self.input_proj = nn.ModuleList(input_proj_list)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        num_pred = self.transformer.decoder.num_layers
        split = 0
        # self.flow_embed = self._get_clones(self.flow_embed, num_pred)
        # self.transformer.decoder.flow_embed = self.flow_embed
        split = 0
        self.flow_embed = nn.ModuleList([self.flow_embed for _ in range(num_pred)])
        self.transformer.decoder.flow_embed = None

    def reset_parameters(self):
        for embed in self.row_pos_embed:
            nn.init.uniform_(embed.weight)
        for embed in self.col_pos_embed:
            nn.init.uniform_(embed.weight)
        # nn.init.xavier_uniform_(self.row_query_embed.weight)
        # nn.init.xavier_uniform_(self.col_query_embed.weight)
        # nn.init.xavier_uniform_(self.row_tgt_embed.weight)
        # nn.init.xavier_uniform_(self.col_tgt_embed.weight)

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8, device=img.device)
        coords1 = coords_grid(N, H//8, W//8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def get_embedding(self, target_feat, col_embed, row_embed):
        f_n, _, f_h, f_w = target_feat.size()
        p_h, _ = col_embed.weight.size()
        p_w, _ = row_embed.weight.size()

        this_embed = torch.cat((col_embed.weight.unsqueeze(1).repeat(1, p_w, 1),
                                row_embed.weight.unsqueeze(0).repeat(p_h, 1, 1)), dim=-1)
        this_embed = this_embed.permute(2, 0, 1).unsqueeze(0).repeat(f_n, 1, 1, 1)

        if f_h != p_h:
            this_embed = F.interpolate(this_embed, size=(f_h, f_w), mode='bilinear', align_corners=True)

        return this_embed

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        features_01 = self.fnet(image1)[3 - self.num_feature_levels:]
        features_02 = self.fnet(image2)[3 - self.num_feature_levels:]

        features_01 = [self.input_proj[l](feat) for l, feat in enumerate(features_01)]
        features_02 = [self.input_proj[l](feat) for l, feat in enumerate(features_02)]

        pos_embeds = [self.get_embedding(feat, col_embed, row_embed)
                      for feat, col_embed, row_embed in zip(features_01, self.col_pos_embed, self.row_pos_embed)]

        hs, init_reference, inter_references = \
            self.transformer(features_01, features_02, pos_embeds)

        i_h, i_w = self.args.image_size[0], self.args.image_size[1]
        flow_raws = list()
        flow_predictions = list()
        for lid in range(len(hs)):
            this_flow = list()
            this_pred = list()
            tmp = self.flow_embed[lid](hs[lid])
            if lid == 0:
                reference = init_reference
            else:
                reference = inter_references[lid - 1]
            prev_idx = 0
            for lvl in range(len(features_01)):
                bs, c, h, w = features_01[lvl].shape
                this_len = h * w
                split = 0
                # reference = inverse_sigmoid(reference[prev_idx:prev_idx + this_len])
                # flow = tmp[prev_idx:prev_idx + this_len] + reference
                # flow = init_reference[prev_idx:prev_idx + this_len] - flow.sigmoid()
                # flow = flow.view(bs, h, w, 2).permute(0, 3, 1, 2)
                # this_pred.append(flow)
                # flow *= torch.tensor((i_h, i_w), dtype=torch.float32).view(1, 2, 1, 1).to(flow.device)
                # flow = F.interpolate(flow, size=(i_h, i_w), mode="bilinear", align_corners=True)
                # this_flow.append(flow)
                split = 0
                flow = tmp[:, prev_idx:prev_idx + this_len]
                flow = flow.view(bs, h, w, 2).permute(0, 3, 1, 2)
                this_pred.append(flow)
                flow = F.interpolate(flow, size=(i_h, i_w), mode="bilinear", align_corners=True)
                flow *= torch.tensor((i_h / h, i_w / w), dtype=torch.float32).view(1, 2, 1, 1).to(flow.device)
                this_flow.append(flow)
                split = 0
                prev_idx += this_len
            this_pred = torch.stack(this_pred, dim=0).mean(dim=0)
            this_flow = torch.stack(this_flow, dim=0).mean(dim=0)
            flow_raws.append(this_pred)
            flow_predictions.append(this_flow)

        if test_mode:
            return flow_raws[-1], flow_predictions[-1]
        else:
            return flow_predictions

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--image_size', type=int, nargs='+', default=[368, 496])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')

    args = parser.parse_args()

    dummy_image_01 = torch.zeros(size=(2, 3, 368, 496), dtype=torch.uint8)
    dummy_image_02 = torch.zeros(size=(2, 3, 368, 496), dtype=torch.uint8)

    model = RAFT(args)

    # model.cuda()
    model.train()

    flow_predictions = model(dummy_image_01, dummy_image_02)
