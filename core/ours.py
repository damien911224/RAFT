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

        h, w = args.image_size[0], args.image_size[1]
        self.row_pos_embed = nn.ModuleList([nn.Embedding(w // (2 ** i), 128 // 2) for i in range(1, 4)])
        self.col_pos_embed = nn.ModuleList([nn.Embedding(h // (2 ** i), 128 // 2) for i in range(1, 4)])

        self.row_query_embed = nn.Embedding(w // 8, 128 // 2)
        self.col_query_embed = nn.Embedding(h // 8, 128 // 2)

        self.row_tgt_embed = nn.Embedding(w // 8, 128 // 2)
        self.col_tgt_embed = nn.Embedding(h // 8, 128 // 2)

        self.reset_parameters()

        self.transformer = DeformableTransformer(d_model=128, nhead=8,
                                                 num_encoder_layers=6, num_decoder_layers=6,
                                                 dim_feedforward=128 * 4, dropout=0.1,
                                                 activation="relu", return_intermediate_dec=True,
                                                 num_feature_levels=3, dec_n_points=4, enc_n_points=4)

        self.flow_embed = MLP(128, 128, 2, 3)

        num_pred = self.transformer.decoder.num_layers
        self.flow_embed = self._get_clones(self.flow_embed, num_pred)
        # nn.init.constant_(self.flow_embed[0].layers[-1].bias.data[2:], -2.0)
        # hack implementation for iterative bounding box refinement
        self.transformer.decoder.bbox_embed = self.bbox_embed

    def reset_parameters(self):
        for embed in self.row_pos_embed:
            nn.init.uniform_(embed.weight)
        for embed in self.col_pos_embed:
            nn.init.uniform_(embed.weight)
        nn.init.xavier_uniform_(self.row_query_embed.weight)
        nn.init.xavier_uniform_(self.col_query_embed.weight)
        nn.init.xavier_uniform_(self.row_tgt_embed.weight)
        nn.init.xavier_uniform_(self.col_tgt_embed.weight)

    def _get_clones(module, N):
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
        f_n, f_h, f_w, _ = target_feat.size()
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

        features_01 = self.fnet(image1)
        features_02 = self.fnet(image2)

        pos_embeds = [self.get_embedding(feat, col_embed, row_embed)
                      for feat, col_embed, row_embed in zip(features_01, self.col_pos_embed, self.row_pos_embed)]
        query_embed = self.get_embedding(features_01[-1], self.col_query_embed, self.row_query_embed)
        tgt_embed = self.get_embedding(features_01[-1], self.col_tgt_embed, self.row_tgt_embed)

        hs, init_reference, inter_references = \
            self.decoder(features_01, features_02, pos_embeds, query_embed, tgt_embed)

        flow_predictions = [upflow8(flow) for flow in inter_references]

        if test_mode:
            return inter_references[-1], flow_predictions[-1]
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
