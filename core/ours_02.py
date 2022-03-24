import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from update import Decoder, PositionEmbedding

from deformable_02 import DeformableTransformer
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
        num_queries = 100

        self.num_feature_levels = num_feature_levels
        h, w = args.image_size[0], args.image_size[1]
        self.row_pos_embed = nn.ModuleList([nn.Embedding(w // (2 ** i), d_model // 2)
                                            for i in range(3, self.num_feature_levels + 3)])
        self.col_pos_embed = nn.ModuleList([nn.Embedding(h // (2 ** i), d_model // 2)
                                            for i in range(3, self.num_feature_levels + 3)])

        # self.row_query_embed = nn.ModuleList([nn.Embedding(w // (2 ** i), d_model // 2) for i in range(1, 4)])
        # self.col_query_embed = nn.ModuleList([nn.Embedding(h // (2 ** i), d_model // 2) for i in range(1, 4)])

        # self.row_tgt_embed = nn.Embedding(w // 8, d_model // 2)
        # self.col_tgt_embed = nn.Embedding(h // 8, d_model // 2)

        self.query_embed = nn.Embedding(num_queries, d_model)

        input_proj_list = []
        for l_i in range(num_feature_levels):
            in_channels = (128, 192, 256)[l_i]
            input_proj_list.append(nn.Sequential(
                nn.Conv2d(in_channels, d_model, kernel_size=1),
                nn.GroupNorm(d_model // 2, d_model)))
        self.input_proj = nn.ModuleList(input_proj_list)

        self.transformer = DeformableTransformer(d_model=d_model, nhead=8,
                                                 num_encoder_layers=3, num_decoder_layers=6,
                                                 dim_feedforward=d_model * 4, dropout=0.1,
                                                 activation="relu", return_intermediate_dec=True,
                                                 num_feature_levels=num_feature_levels,
                                                 dec_n_points=4, enc_n_points=4)

        self.flow_embed = MLP(d_model, d_model, 2, 3)
        self.corr_embed = MLP(d_model, d_model, d_model, 3)
        # self.context_embed = nn.Sequential(
        #         nn.Conv2d(d_model, d_model, kernel_size=1),
        #         nn.GroupNorm(d_model // 2, d_model))
        self.context_embed = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=1),
                nn.GroupNorm(d_model // 2, d_model))

        num_pred = self.transformer.decoder.num_layers
        split = 0
        # self.flow_embed = self._get_clones(self.flow_embed, num_pred)
        # self.transformer.decoder.flow_embed = self.flow_embed
        split = 0
        self.flow_embed = nn.ModuleList([self.flow_embed for _ in range(num_pred)])
        self.corr_embed = nn.ModuleList([self.corr_embed for _ in range(num_pred)])
        self.transformer.decoder.flow_embed = None
        split = 0

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for embed in self.row_pos_embed:
            nn.init.uniform_(embed.weight)
        for embed in self.col_pos_embed:
            nn.init.uniform_(embed.weight)
        nn.init.xavier_uniform_(self.query_embed.weight)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        nn.init.xavier_uniform_(self.context_embed[0].weight, gain=1)
        nn.init.constant_(self.context_embed[0].bias, 0)

        # for p in self.flow_embed:
        #     nn.init.xavier_uniform_(p.weight, gain=1)
        #     nn.init.constant_(p.bias, 0)
        #
        # for p in self.corr_embed.layers:
        #     nn.init.xavier_uniform_(p.weight, gain=1)
        #     nn.init.constant_(p.bias, 0)
        #
        # for p in self.context_embed.layers:
        #     nn.init.xavier_uniform_(p.weight, gain=1)
        #     nn.init.constant_(p.bias, 0)

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

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

    def forward(self, image1, image2, iters=6, test_mode=False):
        """ Estimate optical flow between pair of frames """
        with autocast(enabled=self.args.mixed_precision):
            image1 = 2 * (image1 / 255.0) - 1.0
            image2 = 2 * (image2 / 255.0) - 1.0

            image1 = image1.contiguous()
            image2 = image2.contiguous()

            features_01 = self.fnet(image1)[:self.num_feature_levels]
            features_02 = self.fnet(image2)[:self.num_feature_levels]

            features_01 = [self.input_proj[l](feat) for l, feat in enumerate(features_01)]
            features_02 = [self.input_proj[l](feat) for l, feat in enumerate(features_02)]

            pos_embeds = [self.get_embedding(feat, col_embed, row_embed)
                          for feat, col_embed, row_embed in zip(features_01, self.col_pos_embed, self.row_pos_embed)]
            bs, c, h, w = features_01[0].shape
            query_embeds = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

            hs, init_reference, inter_references, memory_01 = \
                self.transformer(features_01, features_02, pos_embeds, query_embeds)
            memory_01 = memory_01[:, :h * w].view(bs, h, w, c).permute(0, 3, 1, 2)

            i_h, i_w = h * 8, w * 8
            flow_predictions = list()
            for lid in range(len(hs)):
                # bs, n, c
                corr_embed = self.corr_embed[lid](hs[lid].permute(0, 2, 1)).permute(0, 2, 1)
                _, n, c = corr_embed.shape
                # bs, c, h * w
                context_embed = self.context_embed(memory_01).view(bs, c, h * w)
                # bs, n, h * w
                corr = torch.bmm(corr_embed, context_embed)
                # bs, 2, n
                reg = self.flow_embed[lid](hs[lid].permute(0, 2, 1))
                # bs, 2, h, w
                flow = torch.bmm(reg, corr).view(bs, 2, h, w)
                # flow *= torch.tensor((i_h, i_w), dtype=torch.float32).view(1, 2, 1, 1).to(flow.device)
                flow = F.interpolate(flow, size=(i_h, i_w), mode="bilinear", align_corners=True)
                flow_predictions.append(flow)

            if test_mode:
                return flow_predictions[-1], flow_predictions[-1]
            else:
                return flow_predictions


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers = nn.ModuleList(nn.Conv1d(n, k, kernel_size=1) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.norms = nn.ModuleList([nn.GroupNorm(c // 2, c) if c is not None else None
                                    for c in [input_dim] + h + [None]])

    def forward(self, x):
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = F.relu(norm(layer(x))) if i < self.num_layers - 1 else layer(x)
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
