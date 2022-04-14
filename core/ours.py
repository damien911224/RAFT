import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from backbone import Backbone
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from update import Decoder, PositionEmbedding

from deformable import DeformableTransformerEncoderLayer, DeformableTransformerDecoderLayer
from utils.misc import inverse_sigmoid
import copy
import math

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

        # self.extractor = BasicEncoder(base_channel=64, norm_fn="bach")
        self.extractor = Backbone("resnet50", train_backbone=False, return_interm_layers=True, dilation=False)
        # self.context_extractor = Backbone("resnet50", train_backbone=True, return_interm_layers=True, dilation=False)
        d_model = 256
        self.num_feature_levels = 3
        # self.extractor_projection = \
        #     nn.Sequential(nn.Conv2d(self.extractor.down_dim, d_model, kernel_size=1),
        #     nn.GroupNorm(d_model // 8, d_model))

        input_proj_list = []
        channels = (512, 1024, 2048)
        for l_i in range(self.num_feature_levels):
            in_channels = channels[l_i]
            input_proj_list.append(nn.Sequential(
                nn.Conv1d(in_channels, d_model, kernel_size=1),
                nn.GroupNorm(d_model // 2, d_model)))
        self.input_proj = nn.ModuleList(input_proj_list)

        self.encoder = \
            nn.ModuleList((DeformableTransformerEncoderLayer(d_model=d_model, d_ffn=d_model * 4,
                                                             dropout=0.1, activation="gelu",
                                                             n_levels=self.num_feature_levels * 2,
                                                             n_heads=8, n_points=4)
                           for _ in range(6)))

        self.keypoint_decoder = \
            nn.ModuleList((DeformableTransformerDecoderLayer(d_model=d_model, d_ffn=d_model * 4,
                                                             dropout=0.1, activation="gelu",
                                                             n_levels=self.num_feature_levels * 2,
                                                             n_heads=8, n_points=4, self_deformable=False)
                           for _ in range(1)))

        # self.keypoint_decoder = \
        #     nn.ModuleList((nn.TransformerDecoderLayer(d_model=d_model, dim_feedforward=d_model * 4,
        #                                               nhead=8, dropout=0.1, activation="gelu")
        #                    for _ in range(6)))

        # self.correlation_decoder = \
        #     nn.ModuleList((nn.TransformerDecoderLayer(d_model=d_model, dim_feedforward=d_model * 4,
        #                                               nhead=8, dropout=0.1, activation="gelu")
        #                    for _ in range(6)))

        # self.correlation_decoder = \
        #     nn.ModuleList((DeformableTransformerDecoderLayer(d_model=d_model, d_ffn=d_model * 4,
        #                                                      dropout=0.1, activation="gelu",
        #                                                      n_levels=self.num_feature_levels, n_heads=8, n_points=4,
        #                                                      self_deformable=False)
        #                    for _ in range(6)))

        # self.context_decoder = \
        #     nn.ModuleList((DeformableTransformerDecoderLayer(d_model=d_model, d_ffn=d_model * 4,
        #                                                      dropout=0.1, activation="gelu",
        #                                                      n_levels=self.num_feature_levels, n_heads=8, n_points=4,
        #                                                      self_deformable=False)
        #                    for _ in range(6)))

        h, w = args.image_size[0], args.image_size[1]
        # self.pos_embed = NerfPositionalEncoding(depth=d_model // 4)
        self.row_pos_embed = nn.ModuleList([nn.Embedding(w // (2 ** i), d_model // 2)
                                            for i in range(3, self.num_feature_levels + 3)])
        self.col_pos_embed = nn.ModuleList([nn.Embedding(h // (2 ** i), d_model // 2)
                                            for i in range(3, self.num_feature_levels + 3)])
        self.lvl_pos_embed = nn.Embedding(self.num_feature_levels, d_model)
        self.img_pos_embed = nn.Embedding(2, d_model)

        self.query_embed = nn.Embedding(25, d_model)
        self.query_pos_embed = nn.Embedding(25, d_model)
        self.flow_embed = MLP(d_model, d_model, 2, 3)
        # self.flow_embed = nn.Linear(d_model, 2)
        self.context_embed = MLP(d_model, d_model, d_model, 3)
        self.reference_embed = MLP(d_model, d_model, 2, 3)
        # self.reference_embed = nn.Linear(d_model, 2)
        self.extractor_embed = MLP(512, d_model, d_model, 3)

        iterations = 6
        self.flow_embed = nn.ModuleList([self.flow_embed for _ in range(iterations)])
        self.context_embed = nn.ModuleList([self.context_embed for _ in range(iterations)])
        self.reference_embed = nn.ModuleList([self.reference_embed for _ in range(iterations)])
        # self.flow_embed = nn.ModuleList([copy.deepcopy(self.flow_embed) for _ in range(iterations)])
        # self.context_embed = nn.ModuleList([copy.deepcopy(self.context_embed) for _ in range(iterations)])
        # self.reference_embed = nn.ModuleList([copy.deepcopy(self.reference_embed) for _ in range(iterations)])
        # self.confidence_embed = nn.ModuleList([copy.deepcopy(self.confidence_embed) for _ in range(iterations)])

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # nn.init.xavier_uniform_(self.extractor_projection[0].weight)
        # nn.init.constant_(self.extractor_projection[0].bias, 0)

        # nn.init.xavier_uniform_(self.flow_embed.weight)
        # nn.init.constant_(self.flow_embed.bias, 0)
        # nn.init.xavier_uniform_(self.reference_embed.weight)
        # nn.init.constant_(self.reference_embed.bias, 0)
        # nn.init.xavier_uniform_(self.confidence_embed.weight)
        # nn.init.constant_(self.confidence_embed.bias, 0)

        for embed in self.row_pos_embed:
            nn.init.normal_(embed.weight)
        for embed in self.col_pos_embed:
            nn.init.normal_(embed.weight)
        nn.init.xavier_uniform_(self.query_embed.weight)
        nn.init.normal_(self.query_pos_embed.weight)
        nn.init.normal_(self.lvl_pos_embed.weight)
        nn.init.normal_(self.img_pos_embed.weight)

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

        this_embed = this_embed.flatten(2).permute(0, 2, 1)

        return this_embed

    def get_sine_embedding(self, target_feat):
        f_n, _, f_h, f_w = target_feat.size()

        col_embed = (torch.arange(f_h) + 0.5) / f_h
        col_embed = self.pos_embed(col_embed[:, None])
        row_embed = (torch.arange(f_w) + 0.5) / f_w
        row_embed = self.pos_embed(row_embed[:, None])

        this_embed = torch.cat((col_embed.unsqueeze(1).repeat(1, f_w, 1),
                                row_embed.unsqueeze(0).repeat(f_h, 1, 1)), dim=-1)
        this_embed = this_embed.permute(2, 0, 1).unsqueeze(0).to(target_feat.device)

        this_embed = this_embed.flatten(2).permute(0, 2, 1)

        return this_embed

    def get_reference_points(self, spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device) / H_,
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device) / W_)
            ref_y = ref_y.reshape(-1)[None]
            ref_x = ref_x.reshape(-1)[None]
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, image1, image2, iters=6, test_mode=False):
        """ Estimate optical flow between pair of frames """
        with autocast(enabled=self.args.mixed_precision):
            image1 = 2 * (image1 / 255.0) - 1.0
            image2 = 2 * (image2 / 255.0) - 1.0

            image1 = image1.contiguous()
            image2 = image2.contiguous()
            bs, _, I_H, I_W = image1.shape

            # D1, D2, U1 = self.extractor(torch.cat((image1, image2), dim=0))
            features = self.extractor(torch.cat((image1, image2), dim=0))
            D1 = list()
            D2 = list()
            for f_i in range(len(features)):
                x1, x2 = features["{}".format(f_i)].split(bs, dim=0)
                D1.append(x1)
                D2.append(x2)
            # features_01 = self.extractor(image1)
            # features_02 = self.extractor(image2)
            # D1 = [features_01["{}".format(i)] for i in range(len(features_01))]
            # D2 = [features_02["{}".format(i)] for i in range(len(features_02))]
            _, c, h, w = D1[-1].shape
            # bs, hw, c
            # src_pos = self.get_embedding(D1, self.col_pos_embed, self.row_pos_embed).flatten(2).permute(0, 2, 1)
            src_pos = [self.get_embedding(feat, col_embed, row_embed) + self.lvl_pos_embed.weight[i]
                       for i, (feat, col_embed, row_embed)
                       in enumerate(zip(D1, self.col_pos_embed, self.row_pos_embed))]
            # src_pos = [self.get_sine_embedding(feat) + self.lvl_pos_embed.weight[i]
            #            for i, (feat, col_embed, row_embed)
            #            in enumerate(zip(D1, self.col_pos_embed, self.row_pos_embed))]
            # src_pos = torch.cat(src_pos, dim=1)
            src_pos = torch.flatten(torch.cat(src_pos, dim=1).unsqueeze(1) + self.img_pos_embed.weight[None, :, None],
                                    start_dim=1, end_dim=2)
            # src_pos = torch.cat(src_pos, dim=1)
            src = [self.input_proj[i](torch.cat((feat1.flatten(2), feat2.flatten(2)), dim=0)).permute(0, 2, 1)
                   for i, (feat1, feat2) in enumerate(zip(D1, D2))]
            src = torch.cat(torch.cat(src, dim=1).split(bs, dim=0), dim=1)
            # src = torch.cat(src, dim=1)

            # bs, HW, CU1
            U1 = D1[0]
            # U1 = self.context_extractor(image1)["0"]
            _, C, H, W = U1.shape
            U1 = torch.flatten(U1, 2).permute(0, 2, 1)
            U1 = self.extractor_embed(U1)

            # bs, n, c
            query = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
            query_pos = self.query_pos_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

            init_reference_points = self.get_reference_points([(5, 5), ], device=src.device).squeeze(2)
            init_reference_points = init_reference_points.repeat(bs, 1, 1)

            spatial_shapes = torch.as_tensor([feat.shape[2:] for feat in D1] * 2, dtype=torch.long, device=src.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

            src_ref = self.get_reference_points(spatial_shapes, device=src.device)
            for i in range(len(self.encoder)):
                src = self.encoder[i](src, src_pos, src_ref, spatial_shapes, level_start_index)

            # D1, D2 = src.split(bs, dim=0)

            flow_predictions = list()
            sparse_predictions = list()
            for i in range(len(self.keypoint_decoder)):
                # bs, n, 2
                reference_points = self.reference_embed[i](query + query_pos).sigmoid()
                # reference_points = init_reference_points

                # bs, n, c
                query = self.keypoint_decoder[i](query, query_pos, reference_points.unsqueeze(2),
                                                 src, src_pos, spatial_shapes, level_start_index)
                # keypoint = self.keypoint_decoder[i]((query + query_pos).permute(1, 0, 2),
                #                                     (D1 + src_pos).permute(1, 0, 2)).permute(1, 0, 2)
                # reference_points = self.reference_embed[i](keypoint).sigmoid()

                # bs, n, 2
                # reference_points = (inverse_sigmoid(reference_points.detach()) +
                #                     self.reference_embed[i](keypoint)).sigmoid()

                # bs, n, c
                # correlation = self.correlation_decoder[i](query, query_pos, reference_points.unsqueeze(2),
                #                                           D2, src_pos, spatial_shapes, level_start_index)
                # correlation = self.correlation_decoder[i](keypoint.permute(1, 0, 2),
                #                                           (D2 + src_pos).permute(1, 0, 2)).permute(1, 0, 2)

                # bs, n, c
                # context = self.context_decoder[i](keypoint, query_pos, reference_points.unsqueeze(2),
                #                                   D1, src_pos, spatial_shapes, level_start_index)

                # bs, n, 2
                flow_embed = self.flow_embed[i](query)
                flow = inverse_sigmoid(reference_points) + flow_embed
                flow = reference_points - flow.sigmoid()
                # flow = flow_embed.tanh()
                # confidence = flow_embed[..., 2:].sigmoid()
                # flow = inverse_sigmoid(reference_points) + self.flow_embed[i](query)
                # flow = reference_points - flow.sigmoid()
                # bs, n, c
                context = self.context_embed[i](query)
                # bs, n, c
                # reference_points = inverse_sigmoid(reference_points.detach()) + self.reference_embed[i](query)
                # reference_points = reference_points.unsqueeze(2).sigmoid()

                # bs, HW, n
                context_flow = F.softmax(torch.bmm(U1, context.permute(0, 2, 1)), dim=-1)
                scores = torch.max(context_flow, dim=1)[0]
                # context_flow = torch.sigmoid(torch.bmm(U1, context.permute(0, 2, 1)))
                # bs, HW, 2
                context_flow = torch.bmm(context_flow, flow)
                # bs, 2, H, W
                context_flow = context_flow.permute(0, 2, 1).view(bs, 2, H, W)

                context_flow = context_flow * \
                               torch.tensor((I_W, I_H), dtype=torch.float32).view(1, 2, 1, 1).to(context_flow.device)
                if I_H != H or I_W != W:
                    context_flow = F.interpolate(context_flow, size=(I_H, I_W), mode="bilinear", align_corners=False)

                flow_predictions.append(context_flow)
                sparse_predictions.append((reference_points, flow, scores))

            if test_mode:
                return flow_predictions[-1], flow_predictions[-1]
            else:
                return flow_predictions, sparse_predictions


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, last_activate=False):
        super().__init__()
        self.num_layers = num_layers
        self.last_activate = last_activate
        h = [hidden_dim] * (num_layers - 1)
        # self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers = nn.ModuleList(nn.Conv1d(n, k, kernel_size=1) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.norms = nn.ModuleList([nn.BatchNorm1d(c) if c is not None else None
                                    for c in [hidden_dim] + h + [output_dim]])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
        # for i, layer in enumerate(self.layers):
        #     x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            # x = F.relu(norm(layer(x))) if i < self.num_layers - 1 else layer(x)
            x = F.gelu(norm(layer(x))) if (i < self.num_layers - 1) or self.last_activate else layer(x)
        x = x.permute(0, 2, 1)
        return x


class NerfPositionalEncoding(nn.Module):
    def __init__(self, depth=10, sine_type='lin_sine'):
        '''
        out_dim = in_dim * depth * 2
        '''
        super().__init__()
        if sine_type == 'lin_sine':
            self.bases = [i + 1 for i in range(depth)]
        elif sine_type == 'exp_sine':
            self.bases = [2 ** i for i in range(depth)]
        print(f'using {sine_type} as positional encoding')

    @torch.no_grad()
    def forward(self, inputs):
        out = torch.cat([torch.sin(i * math.pi * inputs) for i in self.bases] +
                        [torch.cos(i * math.pi * inputs) for i in self.bases], dim=-1)
        assert not torch.isnan(out).any()
        return out


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
