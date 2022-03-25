import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from update import Decoder, PositionEmbedding

from deformable import DeformableTransformerDecoderLayer
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

        d_model = 64
        self.extractor = BasicEncoder(base_channel=d_model, norm_fn="batch")
        self.extractor_projection = \
            nn.Sequential(nn.Conv2d(self.extractor.down_dim, d_model, kernel_size=1),
            nn.GroupNorm(d_model // 8, d_model))

        # self.encoder = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=d_model * 4, nhead=8)

        # self.context_decoder = \
        #     nn.ModuleList((nn.TransformerDecoderLayer(d_model=d_model, dim_feedforward=d_model * 4, nhead=8)
        #                    for _ in range(6)))
        self.context_decoder = nn.TransformerDecoderLayer(d_model=d_model, dim_feedforward=d_model * 4, nhead=8)
        self.correlation_decoder = \
            nn.ModuleList((nn.TransformerDecoderLayer(d_model=d_model, dim_feedforward=d_model * 4, nhead=8)
                           for _ in range(6)))

        h, w = args.image_size[0], args.image_size[1]
        self.row_pos_embed = nn.Embedding(w // (2 ** 3), d_model // 2)
        self.col_pos_embed = nn.Embedding(h // (2 ** 3), d_model // 2)
        # self.context_query_embed = nn.Embedding(50, d_model)
        self.context_query_embed = nn.Linear(d_model, d_model)
        self.correlation_query_embed = nn.Linear(d_model, d_model)

        self.context_correlation_embed = MLP(d_model, d_model, d_model, 3)
        self.context_extractor_embed = MLP(d_model, d_model, self.extractor.up_dim, 3)
        # self.correlation_context_embed = MLP(d_model, d_model, d_model, 3)
        self.correlation_flow_embed = MLP(d_model, d_model, 2, 3)

        iterations = 6
        # self.context_correlation_embed = nn.ModuleList([self.context_correlation_embed for _ in range(iterations)])
        # self.context_extractor_embed = nn.ModuleList([self.context_extractor_embed for _ in range(iterations)])
        # self.correlation_context_embed = nn.ModuleList([self.correlation_context_embed for _ in range(iterations)])
        self.correlation_flow_embed = nn.ModuleList([self.correlation_flow_embed for _ in range(iterations)])

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        nn.init.xavier_uniform_(self.extractor_projection[0].weight, gain=1)
        nn.init.constant_(self.extractor_projection[0].bias, 0)

        nn.init.uniform_(self.row_pos_embed.weight)
        nn.init.uniform_(self.col_pos_embed.weight)

        # nn.init.uniform_(self.context_query_embed.weight)
        nn.init.xavier_uniform_(self.context_query_embed.weight.data, gain=1.0)
        nn.init.constant_(self.context_query_embed.bias.data, 0.)
        nn.init.xavier_uniform_(self.correlation_query_embed.weight.data, gain=1.0)
        nn.init.constant_(self.correlation_query_embed.bias.data, 0.)

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H, W, device=img.device)
        # coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0

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
        this_embed = this_embed.permute(2, 0, 1).unsqueeze(0)

        if f_h != p_h:
            this_embed = F.interpolate(this_embed, size=(f_h, f_w), mode="bilinear", align_corners=True)

        return this_embed

    def forward(self, image1, image2, iters=6, test_mode=False):
        """ Estimate optical flow between pair of frames """
        with autocast(enabled=self.args.mixed_precision):
            image1 = 2 * (image1 / 255.0) - 1.0
            image2 = 2 * (image2 / 255.0) - 1.0

            image1 = image1.contiguous()
            image2 = image2.contiguous()
            _, _, I_H, I_W = image1.shape

            D1, D2, U1 = self.extractor(torch.cat((image1, image2), dim=0))
            bs, c, h, w = D1.shape
            _, C, H, W = U1.shape
            pos_embeds = self.get_embedding(D1, self.col_pos_embed, self.row_pos_embed)
            # hw, bs, c
            D1, D2 = torch.split(
                torch.flatten(self.extractor_projection(torch.cat((D1, D2), dim=0)) + pos_embeds, 2).permute(2, 0, 1),
                bs, dim=1)

            # hw, bs, c
            # D1, D2 = self.encoder(torch.cat((D1, D2), dim=1)).split(bs, dim=1)

            # bs, HW, C
            U1 = torch.flatten(U1, 2).permute(0, 2, 1)

            # n, bs, c
            # context = self.context_query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            context = self.context_query_embed(D1.permute(1, 0, 2))
            correlation = self.correlation_query_embed(D1.permute(1, 0, 2))

            coord_0 = self.initialize_flow(image1)

            flow_predictions = list()
            corr_predictions = list()
            # bs, n, c
            context = self.context_decoder(context.permute(1, 0, 2), D1).permute(1, 0, 2)
            # bs, n, c
            context_correlation = self.context_correlation_embed(context)
            # bs, n, C
            context_extractor = self.context_extractor_embed(context)
            for i in range(len(self.correlation_decoder)):
                # bs, n, c
                # context = context.permute(1, 0, 2)
                # context = self.context_decoder[i](context, D1).permute(1, 0, 2)
                # bs, hw, c
                correlation = correlation.permute(1, 0, 2)
                correlation = self.correlation_decoder[i](correlation, D2).permute(1, 0, 2)

                # bs, n, c
                # context_correlation = self.context_correlation_embed[i](context)
                # bs, n, C
                # context_extractor = self.context_extractor_embed[i](context)
                # bs, hw, c
                # correlation_context = self.correlation_context_embed[i](correlation)
                # correlation_context = correlation
                correlation_context = correlation.detach()
                # bs, hw, 2
                correlation_flow = self.correlation_flow_embed[i](correlation)

                # bs, n, hw
                context_flow = torch.bmm(context_correlation, correlation_context.permute(0, 2, 1))
                # bs, n, 2
                # context_flow = torch.bmm(context_flow, correlation_flow)
                context_flow = torch.bmm(context_flow, correlation_flow.detach())

                # bs, HW, n
                extractor_flow = torch.bmm(U1, context_extractor.permute(0, 2, 1))
                # bs, HW, 2
                extractor_flow = torch.bmm(extractor_flow, context_flow)
                # bs, 2, H, W
                flow = torch.sigmoid(extractor_flow.permute(0, 2, 1).view(bs, 2, H, W))

                flow = flow * torch.tensor((I_W - 1, I_H - 1), dtype=torch.float32).view(1, 2, 1, 1).to(flow.device)
                if I_H != H or I_W != W:
                    flow = F.interpolate(flow, size=(I_H, I_W), mode="bilinear", align_corners=True)
                flow = coord_0 - flow

                # bs, 2, H, W
                corr_flow = torch.sigmoid(correlation_flow.permute(0, 2, 1).view(bs, 2, h, w))
                corr_flow = \
                    corr_flow * torch.tensor((I_W - 1, I_H - 1),
                                             dtype=torch.float32).view(1, 2, 1, 1).to(extractor_flow.device)
                if I_H != H or I_W != W:
                    corr_flow = F.interpolate(corr_flow, size=(I_H, I_W), mode="bilinear", align_corners=True)
                corr_flow = coord_0 - corr_flow

                flow_predictions.append(flow)
                corr_predictions.append(corr_flow)

            if test_mode:
                return flow_predictions[-1], flow_predictions[-1]
            else:
                return flow_predictions, corr_predictions


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
        x = x.permute(0, 2, 1)
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = F.relu(norm(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.permute(0, 2, 1)
        return x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="raft", help="name your experiment")
    parser.add_argument("--stage", help="determines which dataset to use for training")
    parser.add_argument("--restore_ckpt", help="restore checkpoint")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--validation", type=str, nargs="+")

    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, nargs="+", default=[368, 496])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")

    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--wdecay", type=float, default=.00005)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.8, help="exponential weighting")
    parser.add_argument("--add_noise", action="store_true")

    args = parser.parse_args()

    dummy_image_01 = torch.zeros(size=(2, 3, 368, 496), dtype=torch.uint8)
    dummy_image_02 = torch.zeros(size=(2, 3, 368, 496), dtype=torch.uint8)

    model = RAFT(args)

    # model.cuda()
    model.train()

    flow_predictions = model(dummy_image_01, dummy_image_02)
