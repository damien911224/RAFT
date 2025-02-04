import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, CNNEncoder, CNNDecoder
from backbone import Backbone
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
# from update import Decoder, PositionEmbedding

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

        # self.extractor = BasicEncoder(base_channel=64, norm_fn="instance")
        # self.up_dim = self.extractor.up_dim
        # self.extractor = Backbone("resnet50", train_backbone=True, return_interm_layers=True, dilation=False)
        # self.feature_extractor = Backbone("resnet50", train_backbone=False, return_interm_layers=True, dilation=False)
        # self.context_extractor = BasicEncoder(base_channel=64, norm_fn="batch")
        # self.up_dim = self.context_extractor.up_dim

        self.cnn_encoder = CNNEncoder(base_channel=64, norm_fn="instance")
        self.cnn_decoder = CNNDecoder(base_channel=64, norm_fn="batch")
        self.up_dim = self.cnn_decoder.up_dim
        self.num_feature_levels = 3

        # channels = (512, 1024, 2048)
        # channels = (128, 192, 256)
        channels = [96, 128, 192, 256][4 - self.num_feature_levels:]
        self.d_model = 128
        # self.d_model = channels[0] // 2
        # self.up_dim = self.d_model
        # self.extractor_embed = nn.Sequential(
        #         nn.Conv2d(channels[0], self.d_model, kernel_size=1, padding=0),
        #         nn.GroupNorm(16, self.d_model))
        split = 0
        # input_proj_list = list()
        # for l_i in range(self.num_feature_levels):
        #     in_channels = channels[l_i]
        #     input_proj_list.append(nn.Sequential(
        #         nn.Conv1d(in_channels, self.d_model, kernel_size=1, padding=0),
        #         nn.GroupNorm(16, self.d_model)))
        # self.input_proj = nn.ModuleList(input_proj_list)
        split = 0
        h, w = args.image_size[0], args.image_size[1]
        input_proj_list = list()
        for l_i in range(self.num_feature_levels):
            in_channels = channels[l_i]
            input_proj_list.append(nn.Sequential(
                nn.Conv1d(in_channels, self.d_model // 2 * 2 // 2, kernel_size=1, padding=0),
                nn.GroupNorm(16, self.d_model // 2 * 2 // 2)))
        self.input_proj = nn.ModuleList(input_proj_list)
        corr_proj_list = list()
        for l_i in range(self.num_feature_levels):
            # in_channels = (w // (2 ** (3 + l_i))) * (h // (2 ** (3 + l_i)))
            in_channels = 2 * (2 * 4 + 1) ** 2
            # corr_proj_list.append(nn.Sequential(
            #     nn.Conv1d(in_channels, self.d_model, kernel_size=1, padding=0),
            #     nn.GroupNorm(16, self.d_model)))
            corr_proj_list.append(MLP(in_channels, self.d_model // 2 * 2 // 2, self.d_model // 2 * 2 // 2, 3))
        self.corr_proj = nn.ModuleList(corr_proj_list)

        self.encoder_iterations = 1
        self.outer_iterations = 6
        self.inner_iterations = 1
        # self.inner_iterations = self.num_feature_levels
        self.num_keypoints = 100

        # self.encoder = \
        #     nn.ModuleList((DeformableTransformerEncoderLayer(d_model=self.d_model, d_ffn=self.d_model * 4,
        #                                                      dropout=0.1, activation="gelu",
        #                                                      n_levels=self.num_feature_levels * 2,
        #                                                      n_heads=8, n_points=4)
        #                    for _ in range(self.encoder_iterations)))
        #
        # self.context_encoder = \
        #     nn.ModuleList((DeformableTransformerEncoderLayer(d_model=self.d_model, d_ffn=self.d_model * 4,
        #                                                      dropout=0.1, activation="gelu",
        #                                                      n_levels=self.num_feature_levels * 2,
        #                                                      n_heads=8, n_points=4)
        #                    for _ in range(self.encoder_iterations)))

        self.decoder = \
            nn.ModuleList((DeformableTransformerDecoderLayer(d_model=self.d_model, d_ffn=self.d_model * 4,
                                                             dropout=0.1, activation="gelu",
                                                             n_levels=2 * self.num_feature_levels
                                                             if self.inner_iterations <= 1 else 2,
                                                             n_heads=8, n_points=4, self_deformable=False)
                           for _ in range(self.outer_iterations * self.inner_iterations)))

        self.lvl_pos_embed = nn.Embedding(self.num_feature_levels, self.d_model)
        self.img_pos_embed = nn.Embedding(2 + 0 + 1, self.d_model)
        self.row_pos_embed = nn.Embedding(1000, self.d_model // 2)
        self.col_pos_embed = nn.Embedding(1000, self.d_model // 2)
        self.query_embed = nn.Embedding(self.num_keypoints, self.d_model)
        self.query_pos_embed = nn.Embedding(self.num_keypoints, self.d_model)
        # self.reference_embed = nn.Embedding(self.num_keypoints, 4)
        self.flow_embed = MLP(self.d_model, self.d_model, 2, 3)
        # self.flow_embed = MLP(self.d_model, self.d_model, 4, 3)
        self.context_embed = MLP(self.d_model, self.up_dim, self.up_dim, 3)
        # self.reference_embed = MLP(self.d_model, self.d_model, 2, 3)
        # self.confidence_embed = MLP(self.d_model, self.d_model, 2, 3)
        self.context_pos_embed = nn.Linear(self.d_model, self.up_dim)
        self.use_dab = True
        if self.use_dab:
            # self.context_flow_head = MLP(2, self.up_dim, self.up_dim, 3)
            # self.context_scale = MLP(self.up_dim, self.up_dim, self.up_dim, 2)
            self.attention_pos_head = MLP(self.up_dim, self.d_model, self.d_model, 3)
            # self.src_pos_head = MLP(2, self.d_model, self.d_model, 3)
            # self.src_scale = MLP(self.d_model, self.d_model, self.d_model, 3)

            self.no_sine_embed = True
            self.query_scale = MLP(self.d_model, self.d_model, self.d_model, 2)
            # self.motion_query_scale = MLP(self.d_model, self.d_model, self.d_model, 2)
            # self.context_query_scale = MLP(self.d_model, self.d_model, self.d_model, 2)
            if self.no_sine_embed:
                self.ref_point_head = MLP(4, self.d_model, self.d_model, 3)
            else:
                self.ref_point_head = MLP(2 * self.d_model, self.d_model, self.d_model, 2)
            self.high_dim_query_update = True
            if self.high_dim_query_update:
                # self.high_dim_query_proj = MLP(self.d_model, self.d_model, self.d_model, 2)
                self.motion_high_dim_query_proj = MLP(self.d_model, self.d_model, self.d_model, 2)
                self.context_high_dim_query_proj = MLP(self.d_model, self.d_model, self.d_model, 2)
                # self.motion2context_high_dim_query_proj = MLP(self.d_model, self.d_model, self.d_model, 2)
                # self.context2motion_high_dim_query_proj = MLP(self.d_model, self.d_model, self.d_model, 2)
                # self.context_high_dim_query_proj = MLP(self.up_dim, self.up_dim, self.up_dim, 2)
                # self.src_high_dim_query_proj = MLP(self.d_model, self.d_model, self.d_model, 2)

        self.flow_embed = nn.ModuleList([copy.deepcopy(self.flow_embed)
                                         for _ in range(self.outer_iterations)])
        self.context_embed = nn.ModuleList([copy.deepcopy(self.context_embed)
                                            for _ in range(self.outer_iterations)])

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

        # for embed in self.row_pos_embed:
        #     nn.init.normal_(embed.weight)
        # for embed in self.col_pos_embed:
        #     nn.init.normal_(embed.weight)
        # nn.init.normal_(self.context_row_pos_embed.weight)
        # nn.init.normal_(self.context_col_pos_embed.weight)
        # nn.init.xavier_uniform_(self.query_embed.weight)
        nn.init.xavier_uniform_(self.query_embed.weight)
        nn.init.normal_(self.query_pos_embed.weight)
        # nn.init.uniform_(self.reference_embed.weight)
        # nn.init.xavier_uniform_(self.reference_embed.weight)
        nn.init.normal_(self.lvl_pos_embed.weight)
        nn.init.normal_(self.img_pos_embed.weight)
        nn.init.normal_(self.row_pos_embed.weight)
        nn.init.normal_(self.col_pos_embed.weight)
        # nn.init.normal_(self.iter_pos_embed.weight)

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

        if f_h != p_h or f_w != p_w:
            this_embed = F.interpolate(this_embed, size=(f_h, f_w), mode="bilinear", align_corners=False)

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

    def get_reference_points(self, spatial_shapes, device, normalize=True):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            if normalize:
                ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device) / H_,
                                              torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device) / W_)
            else:
                ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                              torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None]
            ref_x = ref_x.reshape(-1)[None]
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def gen_sineembed_for_position(self, pos_tensor):
        # n_query, bs, _ = pos_tensor.size()
        # sineembed_tensor = torch.zeros(n_query, bs, 256)
        scale = 2 * math.pi
        dim_t = torch.arange(self.d_model // 2, dtype=torch.float32, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / (self.d_model // 2))
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
        return pos

    def forward(self, image1, image2, iters=12, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # with autocast(enabled=self.args.mixed_precision):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        bs, _, I_H, I_W = image1.shape

        # D1, D2, U1 = self.extractor(torch.cat((image1, image2), dim=0))
        E1, E2 = self.cnn_encoder(torch.cat((image1, image2), dim=0))
        D1, D2, U1 = self.cnn_decoder(torch.cat((image1, image2), dim=0))
        # D1, U1 = self.cnn_decoder(image1)
        # features = self.extractor(torch.cat((image1, image2), dim=0))
        # _, _, U1 = self.context_extractor(torch.cat((image1, image2), dim=0))
        # D1 = list()
        # D2 = list()
        # for f_i in range(len(features)):
        #     x1, x2 = features["{}".format(f_i)].split(bs, dim=0)
        #     D1.append(x1)
        #     D2.append(x2)
        # U1 = self.extractor_embed(D1[0])

        E1 = E1[4 - self.num_feature_levels:]
        E2 = E2[4 - self.num_feature_levels:]
        D1 = D1[4 - self.num_feature_levels:]
        D2 = D2[4 - self.num_feature_levels:]

        _, c, h, w = D1[-1].shape
        # bs, hw, c
        raw_src_pos = [self.get_embedding(feat, self.col_pos_embed, self.row_pos_embed) + self.lvl_pos_embed.weight[i]
                       for i, feat in enumerate(D1)]
        raw_src_pos = torch.flatten(
            torch.cat(raw_src_pos, dim=1).unsqueeze(1) + self.img_pos_embed.weight[None, :-1, None],
            start_dim=1, end_dim=2)
        # raw_src_pos = torch.cat(raw_src_pos, dim=1)
        raw_context_pos = self.get_embedding(U1, self.col_pos_embed, self.row_pos_embed)
        # raw_context_pos = self.context_pos_embed(raw_context_pos)
        raw_context_pos = self.context_pos_embed(raw_context_pos + self.img_pos_embed.weight[None, -1][:, None])
        raw_context_pos = raw_context_pos.repeat(bs, 1, 1)
        # src = [self.input_proj[i](torch.cat((feat1.flatten(2), feat2.flatten(2)), dim=0)).permute(0, 2, 1)
        #        for i, (feat1, feat2) in enumerate(zip(D1, D2))]
        # src = [torch.cat((self.input_proj[i](torch.cat((feat1.flatten(2), feat2.flatten(2)), dim=0)).permute(0, 2, 1),
        #                   self.corr_proj[i](torch.cat((
        #                       F.interpolate(torch.bmm(feat1.flatten(2).permute(0, 2, 1),
        #                                               feat2.flatten(2)).view(bs * feat1.shape[2] * feat1.shape[3], 1,
        #                                                                      feat1.shape[2], feat1.shape[3]),
        #                                     ((self.args.image_size[0] // (2 ** (3 + i))),
        #                                      (self.args.image_size[1] // (2 ** (3 + i)))),
        #                                     mode="bilinear", align_corners=False).view(
        #                           bs, feat1.shape[2] * feat1.shape[3], -1)
        #                       if feat1.shape[2] != self.args.image_size[0] // (2 ** (3 + i)) or
        #                          feat1.shape[3] != self.args.image_size[1] // (2 ** (3 + i))
        #                       else torch.bmm(feat1.flatten(2).permute(0, 2, 1), feat2.flatten(2)),
        #                       F.interpolate(torch.bmm(feat2.flatten(2).permute(0, 2, 1),
        #                                               feat1.flatten(2)).view(bs * feat1.shape[2] * feat1.shape[3], 1,
        #                                                                      feat1.shape[2], feat1.shape[3]),
        #                                     ((self.args.image_size[0] // (2 ** (3 + i))),
        #                                      (self.args.image_size[1] // (2 ** (3 + i)))),
        #                                     mode="bilinear", align_corners=False).view(
        #                           bs, feat1.shape[2] * feat1.shape[3], -1)
        #                       if feat1.shape[2] != self.args.image_size[0] // (2 ** (3 + i)) or
        #                          feat1.shape[3] != self.args.image_size[1] // (2 ** (3 + i))
        #                       else torch.bmm(feat2.flatten(2).permute(0, 2, 1), feat1.flatten(2))
        #                   ), dim=0))), dim=-1)
        #        for i, (feat1, feat2) in enumerate(zip(D1, D2))]
        corr_01 = [CorrBlock(feat1, feat2, radius=4)(
            self.get_reference_points([feat1.shape[2:], ],
                                      device=feat1.device, normalize=False).squeeze(2).repeat(bs, 1, 1))
                      for i, (feat1, feat2) in enumerate(zip(E1, E2))]
        corr_02 = [CorrBlock(feat1, feat2, radius=4)(
            self.get_reference_points([feat1.shape[2:], ],
                                      device=feat1.device, normalize=False).squeeze(2).repeat(bs, 1, 1))
                      for i, (feat1, feat2) in enumerate(zip(E2, E1))]
        # src = [torch.cat((self.input_proj[i](torch.cat((feat1.flatten(2), feat2.flatten(2)), dim=0)).permute(0, 2, 1),
        #                   self.corr_proj[i](torch.cat((corr_01[i], corr_02[i]), dim=0))), dim=-1)
        #        for i, (feat1, feat2) in enumerate(zip(D1, D2))]
        # src = [torch.cat((self.input_proj[i](torch.cat((feat1.flatten(2), feat2.flatten(2)), dim=0)).permute(0, 2, 1),
        #                   self.corr_proj[i](corr_01[i])), dim=0)
        #        for i, (feat1, feat2) in enumerate(zip(D1, D2))]
        # src = [torch.cat((self.input_proj[i](feat1.flatten(2)).permute(0, 2, 1), self.corr_proj[i](corr_01[i])), dim=0)
        #        for i, (feat1, feat2) in enumerate(zip(D1, D2))]
        # src = [torch.cat((self.input_proj[i](feat1.flatten(2)).permute(0, 2, 1), self.corr_proj[i](corr_01[i])), dim=0)
        #        for i, feat1 in enumerate(D1)]
        # src = torch.cat(torch.cat(src, dim=1).split(bs, dim=0), dim=1)
        # src = torch.cat(src, dim=1)

        # bs, HW, CU1
        split = 0
        motion_src = [self.corr_proj[i](torch.cat((corr_01[i], corr_02[i]), dim=0))
                      for i, (feat1, feat2) in enumerate(zip(D1, D2))]
        motion_src = torch.cat(torch.cat(motion_src, dim=1).split(bs, dim=0), dim=1)
        context_src = [self.input_proj[i](torch.cat((feat1.flatten(2), feat2.flatten(2)), dim=0)).permute(0, 2, 1)
               for i, (feat1, feat2) in enumerate(zip(D1, D2))]
        context_src = torch.cat(torch.cat(context_src, dim=1).split(bs, dim=0), dim=1)
        split = 0
        _, C, H, W = U1.shape
        U1 = torch.flatten(U1, 2).permute(0, 2, 1)
        # EU1 = torch.flatten(EU1, 2).permute(0, 2, 1)

        # bs, n, c
        # query = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        query = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        if not self.use_dab:
            query_pos = self.query_pos_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        spatial_shapes = torch.as_tensor([feat.shape[2:] for feat in D1] * 2, dtype=torch.long, device=D1[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        src_ref = self.get_reference_points(spatial_shapes, device=D1[0].device)
        # for i in range(len(self.encoder)):
        #     src = self.encoder[i](src, raw_src_pos, src_ref, spatial_shapes, level_start_index)
        split = 0
        # for i in range(len(self.encoder)):
        #     motion_src = self.encoder[i](motion_src, raw_src_pos, src_ref, spatial_shapes, level_start_index)
        #     context_src = self.context_encoder[i](context_src, raw_src_pos, src_ref, spatial_shapes, level_start_index)
        # src = torch.cat((motion_src, context_src), dim=-1)
        split = 0
        if self.inner_iterations > 1:
            new_src_pos = list()
            new_motion_src = list()
            new_context_src = list()
            new_spatial_shapes = list()
            new_level_start_index = list()
            for l_i in range(self.num_feature_levels):
                this_H, this_W = spatial_shapes[l_i]
                this_start_index = level_start_index[l_i]
                this_length = this_H * this_W
                this_motion_src_01 = motion_src[:, this_start_index:this_start_index + this_length]
                this_context_src_01 = context_src[:, this_start_index:this_start_index + this_length]
                this_src_pos_01 = raw_src_pos[:, this_start_index:this_start_index + this_length]
                this_H, this_W = spatial_shapes[l_i + self.num_feature_levels]
                this_start_index = level_start_index[l_i + self.num_feature_levels]
                this_length = this_H * this_W
                this_motion_src_02 = motion_src[:, this_start_index:this_start_index + this_length]
                this_context_src_02 = context_src[:, this_start_index:this_start_index + this_length]
                this_src_pos_02 = raw_src_pos[:, this_start_index:this_start_index + this_length]
                this_motion_src = torch.cat((this_motion_src_01, this_motion_src_02), dim=1)
                this_context_src = torch.cat((this_context_src_01, this_context_src_02), dim=1)
                this_src_pos = torch.cat((this_src_pos_01, this_src_pos_02), dim=1)
                new_motion_src.append(this_motion_src)
                new_context_src.append(this_context_src)
                new_src_pos.append(this_src_pos)

                this_spatial_shapes = torch.as_tensor([(this_H, this_W), ] * 2, dtype=torch.long, device=D1[0].device)
                this_level_start_index = torch.cat((this_spatial_shapes.new_zeros((1,)), this_spatial_shapes.prod(1).cumsum(0)[:-1]))
                new_spatial_shapes.append(this_spatial_shapes)
                new_level_start_index.append(this_level_start_index)
            motion_src = new_motion_src[::-1]
            context_src = new_context_src[::-1]
            raw_src_pos = new_src_pos[::-1]
            spatial_shapes = new_spatial_shapes[::-1]
            level_start_index = new_level_start_index[::-1]
        split = 0
        flow_predictions = list()
        sparse_predictions = list()
        root = round(math.sqrt(self.num_keypoints))
        base_reference_points = self.get_reference_points([(root, root), ], device=D1[0].device).squeeze(2)
        base_reference_points = base_reference_points.repeat(bs, 1, 1)
        if self.inner_iterations <= 1:
            reference_points = base_reference_points.detach().unsqueeze(2).repeat(1, 1, self.num_feature_levels * 2, 1)
        else:
            reference_points = base_reference_points.detach().unsqueeze(2).repeat(1, 1, 2, 1)
        reference_flows = torch.zeros(dtype=torch.float32, size=(bs, self.num_keypoints, 2), device=D1[0].device) + 0.5
        # reference_flows = torch.zeros(dtype=torch.float32, size=(bs, H * W, 2), device=D1[0].device)
        # reference_context = \
        #     torch.zeros(dtype=torch.float32, size=(bs, self.num_keypoints, self.up_dim), device=D1[0].device)
        for o_i in range(self.outer_iterations):
            if self.use_dab:
                # raw_query_pos = torch.cat((reference_points[:, :, 0], reference_flows), dim=-1)
                # raw_query_pos = torch.cat((reference_points[:, :, 0],
                #                            reference_points[:, :, self.num_feature_levels]), dim=-1)
                raw_query_pos = torch.cat((reference_points[:, :, 0],
                                           reference_points[:, :, 1]), dim=-1)
                split = 0
                # raw_query_pos = reference_points.detach()
                # if self.no_sine_embed:
                #     raw_query_pos = self.ref_point_head(raw_query_pos)
                # else:
                #     query_sine_embed = self.gen_sineembed_for_position(raw_query_pos)  # bs, nq, 256*2
                #     raw_query_pos = self.ref_point_head(query_sine_embed)  # bs, nq, 256
                # pos_scale = self.query_scale(query) if not (o_i == 0 and i_i == 0) else 1
                # query_pos = pos_scale * raw_query_pos
                #
                # if not (o_i == 0 and i_i == 0) or self.first_query:
                #     masks = masks.flatten(2)
                #     attention_pos = torch.bmm(masks, context_pos.detach())
                #     query_pos = query_pos + self.attention_pos_head(attention_pos)
                #
                # if self.inner_iterations > 1:
                #     query_pos = query_pos + self.iter_pos_embed.weight[i_i].unsqueeze(0)
                split = 0
                # if self.high_dim_query_update and (not (o_i == 0 and i_i == 0) or self.first_query):
                #     query_pos = query_pos + self.high_dim_query_proj(query)
                split = 0
                if self.no_sine_embed:
                    query_pos = self.ref_point_head(raw_query_pos)
                else:
                    query_sine_embed = self.gen_sineembed_for_position(raw_query_pos)  # bs, nq, 256*2
                    query_pos = self.ref_point_head(query_sine_embed)  # bs, nq, 256
                # if not (o_i == 0 and i_i == 0):
                if not (o_i == 0):
                    pos_scale = self.query_scale(query)
                    query_pos = pos_scale * query_pos

                # if not (o_i == 0 and i_i == 0) or self.first_query:
                #     masks = masks.flatten(2)
                #     attention_pos = torch.bmm(masks, context_pos.detach())
                #     query_pos = query_pos + self.attention_pos_head(attention_pos)

                # if self.inner_iterations > 1:
                #     query_pos = query_pos + self.iter_pos_embed.weight[i_i].unsqueeze(0)

                # if self.high_dim_query_update and not (o_i == 0 and i_i == 0):
                if self.high_dim_query_update and not (o_i == 0):
                    query_pos = query_pos + self.motion_high_dim_query_proj(query)
                split = 0
                # context_pos = raw_context_pos + self.context_flow_head(context_flow.detach())
                # context_pos_scale = self.context_scale(U1) if not (o_i == 0 and i_i == 0) else 1
                # context_pos = context_pos_scale * context_pos
                #
                # if self.high_dim_query_update and not (o_i == 0 and i_i == 0):
                #     context_pos = context_pos + self.context_high_dim_query_proj(U1)

                context_pos = raw_context_pos
                split = 0
                # if not (o_i == 0 and i_i == 0):
                #     # bs, HW, N
                #     flow_pos = list()
                #     context_flow = context_flow.detach().permute(0, 2, 1).view(bs, 2, H, W)
                #     for H_, W_ in spatial_shapes:
                #         this_flow = F.interpolate(context_flow, size=(H_, W_), mode="bilinear", align_corners=False)
                #         this_flow = this_flow.flatten(2).permute(0, 2, 1)
                #         flow_pos.append(this_flow)
                #     flow_pos = torch.cat(flow_pos, dim=1)
                #     # bs, HW, d_model
                #     src_pos = raw_src_pos + self.src_pos_head(flow_pos)
                #     src_pos_scale = self.src_scale(src) if not (o_i == 0 and i_i == 0) else 1
                #     src_pos = src_pos_scale * src_pos
                #
                #     if self.high_dim_query_update and not (o_i == 0 and i_i == 0):
                #         src_pos = src_pos + self.src_high_dim_query_proj(src)
                # else:
                #     src_pos = raw_src_pos
                split = 0
                src_pos = raw_src_pos
            else:
                context_pos = raw_context_pos
                src_pos = raw_src_pos

            for i_i in range(self.inner_iterations):
                d_i = i_i + self.num_feature_levels * o_i if self.inner_iterations > 1 else o_i
                this_motion_src = motion_src[i_i] if self.inner_iterations > 1 else motion_src
                this_context_src = context_src[i_i] if self.inner_iterations > 1 else context_src
                this_src = torch.cat((this_motion_src, this_context_src), dim=-1)
                this_src_pos = src_pos[i_i] if self.inner_iterations > 1 else src_pos
                this_spatial_shapes = spatial_shapes[i_i] if self.inner_iterations > 1 else spatial_shapes
                this_level_start_index = level_start_index[i_i] if self.inner_iterations > 1 else level_start_index
                query = self.decoder[d_i](query, query_pos, reference_points,
                                          this_src, this_src_pos, this_spatial_shapes,
                                          this_level_start_index)

            # query = self.decoder[o_i](query, query_pos, reference_points,
            #                           src, src_pos, spatial_shapes, level_start_index)

            # bs, n, 2
            flow_embed = self.flow_embed[o_i](query)
            flow_embed = flow_embed + inverse_sigmoid(reference_flows)
            reference_flows = flow_embed.detach().sigmoid()

            src_points = reference_points[:, :, 0].detach()
            dst_points = (inverse_sigmoid(src_points) + flow_embed).sigmoid()
            key_flow = src_points - dst_points
            # reference_points[:, :, self.num_feature_levels:] = dst_points.detach().unsqueeze(2)
            reference_points[:, :, 1:] = dst_points.detach().unsqueeze(2)
            split = 0
            # bs, HW, n
            context_embed = self.context_embed[o_i](query)
            # context_embed = reference_context + self.context_embed[o_i](context_query)
            # reference_context = context_embed.detach()
            context_flow = F.softmax(torch.bmm(U1 + context_pos, context_embed.permute(0, 2, 1)), dim=-1)
            # bs, n, HW
            masks = context_flow.permute(0, 2, 1).detach()
            # bs, n
            scores = torch.max(context_flow, dim=1)[0].detach()
            # bs, HW, 2
            context_flow = torch.bmm(context_flow, key_flow)
            # context_flow = reference_flows + context_flow
            # reference_flows = context_flow.detach()
            # bs, 2, H, W
            flow = context_flow.permute(0, 2, 1).view(bs, 2, H, W)
            flow = flow * torch.as_tensor((I_W, I_H), dtype=torch.float32, device=D1[0].device).view(1, 2, 1, 1)

            if I_H != H or I_W != W:
                flow = F.interpolate(flow, size=(I_H, I_W), mode="bilinear", align_corners=False)
                masks = masks.reshape(bs, -1, 1, H, W)

            flow_predictions.append(flow)
            sparse_predictions.append((reference_points[:, :, 0].clone(), key_flow, masks, scores))
            split = 0
            # src_points = (inverse_sigmoid(src_points) + flow_embed[..., :2]).sigmoid()
            # reference_points[:, :, :self.num_feature_levels] = src_points.unsqueeze(2)
            split = 0
            # # bs, n
            # areas = torch.sum(masks, dim=(-1, -2)).squeeze(-1)
            # # bs, topk
            # topk_indices = torch.topk(scores, 25, dim=-1)[1]
            # # bs, topk, 2
            # topk_areas = torch.gather(areas, dim=1, index=topk_indices)
            # topk_dst_points = torch.gather(dst_points, dim=1, index=topk_indices.unsqueeze(-1).repeat(1, 1, 2))
            # topk_motion_query = torch.gather(motion_query, dim=1,
            #                                  index=topk_indices.unsqueeze(-1).repeat(1, 1, self.d_model))
            # topk_context_query = torch.gather(context_query, dim=1,
            #                                   index=topk_indices.unsqueeze(-1).repeat(1, 1, self.d_model))
            # new_src_points = torch.gather(src_points, dim=1, index=topk_indices.unsqueeze(-1).repeat(1, 1, 2))
            # new_src_points = new_src_points.repeat(1, 4, 1)
            # new_src_points = torch.normal(mean=new_src_points,
            #                               std=torch.sqrt(topk_areas).unsqueeze(-1).repeat(1, 4, 1))
            # new_src_points = torch.clip(new_src_points, 0.0, 1.0)
            # reference_points[:, :, :self.num_feature_levels] = new_src_points.detach().unsqueeze(2)
            # dst_points = topk_dst_points.repeat(1, 4, 1)
            # reference_points[:, :, self.num_feature_levels:] = dst_points.detach().unsqueeze(2)
            # motion_query = topk_motion_query.repeat(1, 4, 1)
            # context_query = topk_context_query.repeat(1, 4, 1)
            split = 0

        if test_mode:
            return flow_predictions, sparse_predictions
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
        self.layers = nn.ModuleList(nn.Conv1d(n, k, kernel_size=1, padding=0)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))
        self.norms = nn.ModuleList(nn.GroupNorm(32, k)
                                   for n, k in zip([input_dim] + h, h + [output_dim]))

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
