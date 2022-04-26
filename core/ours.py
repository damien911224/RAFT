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

        self.extractor = BasicEncoder(base_channel=64, norm_fn="batch")
        self.up_dim = self.extractor.up_dim
        # self.feature_extractor = Backbone("resnet50", train_backbone=False, return_interm_layers=True, dilation=False)
        # self.context_extractor = BasicEncoder(base_channel=64, norm_fn="batch")
        # self.up_dim = self.context_extractor.up_dim
        self.num_feature_levels = 3

        input_proj_list = []
        # channels = (512, 1024, 2048)
        channels = (128, 192, 256)
        self.d_model = channels[0]
        for l_i in range(self.num_feature_levels):
            in_channels = channels[l_i]
            input_proj_list.append(nn.Sequential(
                nn.Conv1d(in_channels, self.d_model, kernel_size=1, padding=0),
                nn.GroupNorm(32, self.d_model)))
        self.input_proj = nn.ModuleList(input_proj_list)

        self.encoder_iterations = 1
        self.outer_iterations = 6
        self.inner_iterations = 1
        # self.inner_iterations = self.num_feature_levels
        self.num_keypoints = 100
        # self.num_keypoints = 25

        self.encoder = \
            nn.ModuleList((DeformableTransformerEncoderLayer(d_model=self.d_model, d_ffn=self.d_model * 4,
                                                             dropout=0.1, activation="gelu",
                                                             n_levels=self.num_feature_levels * 2,
                                                             n_heads=8, n_points=4)
                           for _ in range(self.encoder_iterations)))

        self.decoder = \
            nn.ModuleList((DeformableTransformerDecoderLayer(d_model=self.d_model, d_ffn=self.d_model * 4,
                                                             dropout=0.1, activation="gelu",
                                                             n_levels=self.num_feature_levels * 2,
                                                             n_heads=8, n_points=4, self_deformable=False)
                           for _ in range(self.outer_iterations)))

        # self.encoder = \
        #     nn.ModuleList((DeformableTransformerEncoderLayer(d_model=self.d_model, d_ffn=self.d_model * 4,
        #                                                      dropout=0.1, activation="gelu",
        #                                                      n_levels=self.num_feature_levels,
        #                                                      n_heads=8, n_points=4)
        #                    for _ in range(self.encoder_iterations)))
        #
        # self.decoder = \
        #     nn.ModuleList((DeformableTransformerDecoderLayer(d_model=self.d_model, d_ffn=self.d_model * 4,
        #                                                      dropout=0.1, activation="gelu",
        #                                                      n_levels=2,
        #                                                      n_heads=8, n_points=4, self_deformable=False)
        #                    for _ in range(self.outer_iterations * self.inner_iterations)))

        # self.query_selector = nn.TransformerDecoderLayer(d_model=d_model, dim_feedforward=d_model * 4,
        #                                                  nhead=8, dropout=0.1, activation="gelu")

        # self.keypoint_decoder = \
        #     nn.ModuleList((DeformableTransformerDecoderLayer(d_model=d_model, d_ffn=d_model * 4,
        #                                                      dropout=0.1, activation="gelu",
        #                                                      n_levels=self.num_feature_levels * 2,
        #                                                      n_heads=8, n_points=4, self_deformable=False)
        #                    for _ in range(self.outer_iterations)))

        # self.keypoint_decoder = nn.TransformerDecoderLayer(d_model=d_model, dim_feedforward=d_model * 4,
        #                                                    nhead=8, dropout=0.1, activation="gelu")

        # self.keypoint_decoder = \
        #     nn.ModuleList((nn.TransformerDecoderLayer(d_model=d_model, dim_feedforward=d_model * 4,
        #                                               nhead=8, dropout=0.1, activation="gelu")
        #                    for _ in range(6)))

        # self.correlation_decoder = \
        #     nn.ModuleList((DeformableTransformerDecoderLayer(d_model=d_model, d_ffn=d_model * 4,
        #                                                      dropout=0.1, activation="gelu",
        #                                                      n_levels=self.num_feature_levels * 2,
        #                                                      n_heads=8, n_points=4, self_deformable=False)
        #                    for _ in range(self.outer_iterations)))

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
        self.lvl_pos_embed = nn.Embedding(self.num_feature_levels, self.d_model)
        self.img_pos_embed = nn.Embedding(3, self.d_model)
        self.row_pos_embed = nn.Embedding(w // (2 ** 2), self.d_model // 2)
        self.col_pos_embed = nn.Embedding(h // (2 ** 2), self.d_model // 2)

        self.iter_pos_embed = nn.Embedding(self.inner_iterations, self.d_model)

        self.query_embed = nn.Embedding(self.num_keypoints, self.d_model)
        self.query_pos_embed = nn.Embedding(self.num_keypoints, self.d_model)
        self.flow_embed = MLP(self.d_model, self.d_model, 2, 3)
        self.context_embed = MLP(self.d_model, self.up_dim, self.up_dim, 3)
        # self.reference_embed = MLP(self.d_model, self.d_model, 2, 3)
        # self.confidence_embed = MLP(self.d_model, self.d_model, 1, 3)
        self.context_pos_embed = nn.Linear(self.d_model, self.up_dim)
        self.use_dab = True
        if self.use_dab:
            self.context_flow_head = MLP(2, self.up_dim, self.up_dim, 3)
            self.context_scale = MLP(self.up_dim, self.up_dim, self.up_dim, 2)
            self.src_pos_head = MLP(self.d_model, self.d_model, self.d_model, 3)
            self.src_scale = MLP(self.d_model, self.d_model, self.d_model, 2)

            self.no_sine_embed = True
            self.query_scale = MLP(self.d_model, self.d_model, self.d_model, 2)
            if self.no_sine_embed:
                self.ref_point_head = MLP(4, self.d_model, self.d_model, 3)
            else:
                self.ref_point_head = MLP(2 * self.d_model, self.d_model, self.d_model, 2)
            self.high_dim_query_update = True
            if self.high_dim_query_update:
                self.high_dim_query_proj = MLP(self.d_model, self.d_model, self.d_model, 2)
                self.context_high_dim_query_proj = MLP(self.up_dim, self.up_dim, self.up_dim, 2)
                self.src_high_dim_query_proj = MLP(self.d_model, self.d_model, self.d_model, 2)

        # self.flow_embed = nn.ModuleList([copy.deepcopy(self.flow_embed) for _ in range(self.outer_iterations)])
        # self.context_embed = nn.ModuleList([copy.deepcopy(self.context_embed) for _ in range(self.outer_iterations)])
        # self.reference_embed = nn.ModuleList([copy.deepcopy(self.reference_embed) for _ in range(self.outer_iterations)])
        self.flow_embed = nn.ModuleList([copy.deepcopy(self.flow_embed) for _ in range(self.outer_iterations * self.inner_iterations)])
        self.context_embed = nn.ModuleList([copy.deepcopy(self.context_embed) for _ in range(self.outer_iterations * self.inner_iterations)])

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
        nn.init.xavier_uniform_(self.query_embed.weight)
        nn.init.normal_(self.query_pos_embed.weight)
        # nn.init.uniform_(self.reference_embed.weight)
        # nn.init.xavier_uniform_(self.reference_embed.weight)
        nn.init.normal_(self.lvl_pos_embed.weight)
        nn.init.normal_(self.img_pos_embed.weight)
        nn.init.normal_(self.row_pos_embed.weight)
        nn.init.normal_(self.col_pos_embed.weight)
        nn.init.normal_(self.iter_pos_embed.weight)

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
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.0, H_ - 1.0, H_, dtype=torch.float32, device=device) / (H_ - 1),
                                          torch.linspace(0.0, W_ - 1.0, W_, dtype=torch.float32, device=device) / (W_ - 1))
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

    def forward(self, image1, image2, iters=6, test_mode=False):
        """ Estimate optical flow between pair of frames """
        # with autocast(enabled=self.args.mixed_precision):
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()
        bs, _, I_H, I_W = image1.shape

        D1, D2, U1 = self.extractor(torch.cat((image1, image2), dim=0))
        # features = self.feature_extractor(torch.cat((image1, image2), dim=0))
        # _, _, U1 = self.context_extractor(torch.cat((image1, image2), dim=0))
        # D1 = list()
        # D2 = list()
        # for f_i in range(len(features)):
        #     x1, x2 = features["{}".format(f_i)].split(bs, dim=0)
        #     D1.append(x1)
        #     D2.append(x2)
        _, c, h, w = D1[-1].shape
        # bs, hw, c
        raw_src_pos = [self.get_embedding(feat, self.col_pos_embed, self.row_pos_embed) + self.lvl_pos_embed.weight[i]
                       for i, feat in enumerate(D1)]
        raw_src_pos = torch.flatten(
            torch.cat(raw_src_pos, dim=1).unsqueeze(1) + self.img_pos_embed.weight[None, :2, None],
            start_dim=1, end_dim=2)
        # raw_src_pos = torch.cat(raw_src_pos, dim=1)
        raw_context_pos = self.get_embedding(U1, self.col_pos_embed, self.row_pos_embed)
        # raw_context_pos = self.context_pos_embed(raw_context_pos)
        raw_context_pos = self.context_pos_embed(raw_context_pos + self.img_pos_embed.weight[None, -1][:, None])
        src = [self.input_proj[i](torch.cat((feat1.flatten(2), feat2.flatten(2)), dim=0)).permute(0, 2, 1)
               for i, (feat1, feat2) in enumerate(zip(D1, D2))]
        src = torch.cat(torch.cat(src, dim=1).split(bs, dim=0), dim=1)
        # src = torch.cat(src, dim=1)

        # bs, HW, CU1
        _, C, H, W = U1.shape
        U1 = torch.flatten(U1, 2).permute(0, 2, 1)

        # bs, n, c
        query = self.query_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        if not self.use_dab:
            query_pos = self.query_pos_embed.weight.unsqueeze(0).repeat(bs, 1, 1)

        spatial_shapes = torch.as_tensor([feat.shape[2:] for feat in D1] * 2, dtype=torch.long, device=src.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        src_ref = self.get_reference_points(spatial_shapes, device=src.device)
        for i in range(len(self.encoder)):
            src = self.encoder[i](src, raw_src_pos, src_ref, spatial_shapes, level_start_index)

        # new_src = list()
        # new_src_pos = list()
        # for l_i in range(self.num_feature_levels):
        #     this_H, this_W = spatial_shapes[l_i]
        #     this_start_index = level_start_index[l_i]
        #     this_length = this_H * this_W
        #     this_src = torch.cat(src[:, this_start_index:this_start_index + this_length].split(bs, dim=0), dim=1)
        #     this_src_pos = (src_pos[:, this_start_index:this_start_index + this_length].unsqueeze(1) +
        #                     self.img_pos_embed.weight[None, :2, None]).flatten(start_dim=1, end_dim=2)
        #     new_src.append(this_src)
        #     new_src_pos.append(this_src_pos)
        # src = new_src
        # src_pos = new_src_pos

        # src = torch.cat(src.split(bs, dim=0), dim=1)
        # raw_src_pos = (raw_src_pos.unsqueeze(1) +
        #                self.img_pos_embed.weight[None, :2, None]).flatten(start_dim=1, end_dim=2)
        # spatial_shapes = torch.as_tensor([feat.shape[2:] for feat in D1] * 2, dtype=torch.long, device=src.device)
        # level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        flow_predictions = list()
        sparse_predictions = list()
        root = round(math.sqrt(self.num_keypoints))
        base_reference_points = self.get_reference_points([(root, root), ], device=src.device).squeeze(2)
        base_reference_points = base_reference_points.repeat(bs, 1, 1)
        reference_points = base_reference_points.detach().unsqueeze(2).repeat(1, 1, self.num_feature_levels * 2, 1)
        reference_flows = torch.zeros(dtype=torch.float32, size=(bs, self.num_keypoints, 2), device=src.device) + 0.5
        context_flow = torch.zeros(dtype=torch.float32, size=(bs, H * W, 2), device=src.device)
        for o_i in range(self.outer_iterations):
            for i_i in range(self.inner_iterations):
                # if o_i >= 1:
                #     step = 1
                #     N = round(math.sqrt(self.num_keypoints)) + ((o_i - 1) * step)
                #     reference_points = reference_points[:, :, 0].permute(0, 2, 1)
                #     reference_points = reference_points.reshape(bs, 2, N, N)
                #     reference_points = F.interpolate(reference_points, (N + step, N + step),
                #                                      mode="bilinear", align_corners=False)
                #     reference_points = reference_points.flatten(2).permute(0, 2, 1)
                #     reference_points = reference_points.unsqueeze(2).repeat(1, 1, self.num_feature_levels * 2, 1)
                #
                #     query = query.permute(0, 2, 1)
                #     query = query.reshape(bs, self.d_model, N, N)
                #     query = F.interpolate(query, (N + step, N + step), mode="bilinear", align_corners=False)
                #     query = query.flatten(2).permute(0, 2, 1)
                #
                #     reference_flows = reference_flows.permute(0, 2, 1)
                #     reference_flows = reference_flows.reshape(bs, 2, N, N)
                #     reference_flows = F.interpolate(reference_flows, (N + step, N + step),
                #                                     mode="bilinear", align_corners=False)
                #     reference_flows = reference_flows.flatten(2).permute(0, 2, 1)

                if self.use_dab:
                    # if o_i >= 1:
                    #     # bs, HW, N
                    #     attention_pos = list()
                    #     masks = masks.flatten(start_dim=0, end_dim=1)
                    #     for H_, W_ in spatial_shapes:
                    #         this_mask = F.interpolate(masks, size=(H_, W_), mode="bilinear", align_corners=False)
                    #         this_mask = torch.stack(this_mask.split(bs, dim=0), dim=1)
                    #         this_mask = this_mask.squeeze(2).flatten(2).permute(0, 2, 1)
                    #         attention_pos.append(torch.bmm(this_mask, query_pos.detach()))
                    #     attention_pos = torch.cat(attention_pos, dim=1)
                    #     # bs, HW, d_model
                    #     src_pos = raw_src_pos + self.src_pos_head(attention_pos)
                    #     src_pos_scale = self.src_scale(src) if not (o_i == 0 and i_i == 0) else 1
                    #     src_pos = src_pos_scale * src_pos
                    #
                    #     if self.high_dim_query_update and not (o_i == 0 and i_i == 0):
                    #         src_pos = src_pos + self.src_high_dim_query_proj(src)
                    # else:
                    #     src_pos = raw_src_pos

                    src_pos = raw_src_pos

                    raw_query_pos = torch.cat((reference_points[:, :, 0], reference_flows), dim=-1)
                    if self.no_sine_embed:
                        raw_query_pos = self.ref_point_head(raw_query_pos)
                    else:
                        query_sine_embed = self.gen_sineembed_for_position(raw_query_pos)  # bs, nq, 256*2
                        raw_query_pos = self.ref_point_head(query_sine_embed)  # bs, nq, 256
                    pos_scale = self.query_scale(query) if not (o_i == 0 and i_i == 0) else 1
                    query_pos = pos_scale * raw_query_pos

                    if self.inner_iterations > 1:
                        query_pos = query_pos + self.iter_pos_embed.weight[i_i].unsqueeze(0)

                    if self.high_dim_query_update and not (o_i == 0 and i_i == 0):
                        query_pos = query_pos + self.high_dim_query_proj(query)

                    context_pos = raw_context_pos + self.context_flow_head(context_flow.detach())
                    context_pos_scale = self.context_scale(U1) if not (o_i == 0 and i_i == 0) else 1
                    context_pos = context_pos_scale * context_pos

                    if self.high_dim_query_update and not (o_i == 0 and i_i == 0):
                        context_pos = context_pos + self.context_high_dim_query_proj(U1)

                query = self.decoder[o_i](query, query_pos, reference_points,
                                          src, src_pos, spatial_shapes, level_start_index)

                # bs, n, 2
                flow_embed = self.flow_embed[o_i](query)
                flow_embed = flow_embed + inverse_sigmoid(reference_flows)

                src_points = reference_points[:, :, 0].detach()
                dst_points = (inverse_sigmoid(src_points) + flow_embed).sigmoid()
                key_flow = src_points - dst_points
                reference_points[:, :, self.num_feature_levels:] = dst_points.detach().unsqueeze(2)
                reference_flows = flow_embed.detach().sigmoid()

                # bs, HW, n
                context = self.context_embed[o_i](query)
                context_flow = F.softmax(torch.bmm(U1 + context_pos, context.permute(0, 2, 1)), dim=-1)
                masks = context_flow.permute(0, 2, 1).detach()
                scores = torch.max(context_flow, dim=1)[0].detach()
                # bs, HW, 2
                context_flow = torch.bmm(context_flow, key_flow)
                # bs, 2, H, W
                flow = context_flow.permute(0, 2, 1).view(bs, 2, H, W)
                flow = flow * torch.as_tensor((I_W, I_H), dtype=torch.float32, device=src.device).view(1, 2, 1, 1)

                if I_H != H or I_W != W:
                    flow = F.interpolate(flow, size=(I_H, I_W), mode="bilinear", align_corners=False)
                    masks = masks.reshape(bs, -1, 1, H, W)
                    # masks = F.interpolate(masks, size=(I_H, I_W), mode="bilinear", align_corners=False)
                    # masks = masks.view(bs, self.num_keypoints, I_H, I_W)

                flow_predictions.append(flow)
                sparse_predictions.append((reference_points[:, :, 0], key_flow, masks, scores))

        # flow_predictions = list()
        # sparse_predictions = list()
        # for o_i in range(self.outer_iterations):
        #     root = round(math.sqrt(self.num_keypoints))
        #     base_reference_points = self.get_reference_points([(root, root), ], device=image1.device).squeeze(2)
        #     base_reference_points = base_reference_points.repeat(bs, 1, 1)
        #     reference_points = base_reference_points.detach().unsqueeze(2).repeat(1, 1, 2, 1)
        #     reference_flows = torch.zeros(dtype=torch.float32, size=(bs, self.num_keypoints, 2),
        #                                   device=image1.device) + 0.5
        #     context_flow = torch.zeros(dtype=torch.float32, size=(bs, H * W, 2), device=image1.device)
        #     for i_i in range(self.inner_iterations):
        #         this_spatial_shapes = spatial_shapes[i_i].unsqueeze(0).repeat(2, 1)
        #         level_start_index = \
        #             torch.cat((this_spatial_shapes.new_zeros((1,)), this_spatial_shapes.prod(1).cumsum(0)[:-1]))
        #
        #         if i_i >= 1:
        #             N = round(math.sqrt(self.num_keypoints)) * (2 ** (i_i - 1))
        #             reference_points = reference_points[:, :, 0].permute(0, 2, 1)
        #             reference_points = reference_points.reshape(bs, 2, N, N)
        #             reference_points = F.interpolate(reference_points, (N * 2, N * 2),
        #                                              mode="bilinear", align_corners=False)
        #             reference_points = reference_points.flatten(2).permute(0, 2, 1)
        #             reference_points = reference_points.unsqueeze(2).repeat(1, 1, 2, 1)
        #
        #             query = query.permute(0, 2, 1)
        #             query = query.reshape(bs, self.d_model, N, N)
        #             query = F.interpolate(query, (N * 2, N * 2), mode="bilinear", align_corners=False)
        #             query = query.flatten(2).permute(0, 2, 1)
        #
        #             reference_flows = reference_flows.permute(0, 2, 1)
        #             reference_flows = reference_flows.reshape(bs, 2, N, N)
        #             reference_flows = F.interpolate(reference_flows, (N * 2, N * 2),
        #                                             mode="bilinear", align_corners=False)
        #             reference_flows = reference_flows.flatten(2).permute(0, 2, 1)
        #         elif o_i >= 1:
        #             N = round(math.sqrt(self.num_keypoints)) * (2 ** (self.num_feature_levels - 1))
        #             n = round(math.sqrt(self.num_keypoints))
        #             query = query.permute(0, 2, 1)
        #             query = query.reshape(bs, self.d_model, N, N)
        #             query = F.interpolate(query, (n, n), mode="bilinear", align_corners=False)
        #             query = query.flatten(2).permute(0, 2, 1)
        #
        #         if self.use_dab:
        #             raw_query_pos = torch.cat((reference_points[:, :, 0], reference_flows), dim=-1)
        #             if self.no_sine_embed:
        #                 raw_query_pos = self.ref_point_head(raw_query_pos)
        #             else:
        #                 query_sine_embed = self.gen_sineembed_for_position(raw_query_pos)  # bs, nq, 256*2
        #                 raw_query_pos = self.ref_point_head(query_sine_embed)  # bs, nq, 256
        #             pos_scale = self.query_scale(query) if not (o_i == 0 and i_i == 0) else 1
        #             query_pos = pos_scale * raw_query_pos
        #
        #             if self.inner_iterations > 1:
        #                 query_pos = query_pos + self.iter_pos_embed.weight[i_i].unsqueeze(0)
        #
        #             if self.high_dim_query_update and not (o_i == 0 and i_i == 0):
        #                 query_pos = query_pos + self.high_dim_query_proj(query)
        #
        #             context_pos = raw_context_pos + self.context_flow_head(context_flow.detach())
        #             context_pos_scale = self.context_scale(U1) if not (o_i == 0 and i_i == 0) else 1
        #             context_pos = context_pos_scale * context_pos
        #
        #             if self.high_dim_query_update and not (o_i == 0 and i_i == 0):
        #                 context_pos = context_pos + self.context_high_dim_query_proj(U1)
        #
        #         query = self.decoder[o_i * i_i](query, query_pos, reference_points,
        #                                         src[i_i], src_pos[i_i], this_spatial_shapes, level_start_index)
        #
        #         # bs, n, 2
        #         flow_embed = self.flow_embed[o_i * i_i](query)
        #         flow_embed = flow_embed + inverse_sigmoid(reference_flows)
        #
        #         src_points = reference_points[:, :, 0].detach()
        #         dst_points = (inverse_sigmoid(src_points) + flow_embed).sigmoid()
        #         key_flow = src_points - dst_points
        #         reference_points[:, :, 1] = dst_points.detach()
        #         reference_flows = flow_embed.detach().sigmoid()
        #
        #         # bs, HW, n
        #         context = self.context_embed[o_i * i_i](query)
        #         context_flow = F.softmax(torch.bmm(U1 + context_pos, context.permute(0, 2, 1)), dim=-1)
        #         if i_i >= self.num_feature_levels - 1:
        #             masks = context_flow.permute(0, 2, 1).detach()
        #             scores = torch.max(context_flow, dim=1)[0].detach()
        #         # bs, HW, 2
        #         context_flow = torch.bmm(context_flow, key_flow)
        #         if i_i >= self.num_feature_levels - 1:
        #             # bs, 2, H, W
        #             flow = context_flow.permute(0, 2, 1).view(bs, 2, H, W)
        #             flow = flow * torch.as_tensor((I_W, I_H), dtype=torch.float32, device=image1.device).view(1, 2, 1, 1)
        #
        #             if I_H != H or I_W != W:
        #                 flow = F.interpolate(flow, size=(I_H, I_W), mode="bilinear", align_corners=False)
        #                 masks = masks.reshape(bs, -1, 1, H, W)
        #                 # masks = F.interpolate(masks, size=(I_H, I_W), mode="bilinear", align_corners=False)
        #                 # masks = masks.view(bs, self.num_keypoints, I_H, I_W)
        #
        #             flow_predictions.append(flow)
        #             sparse_predictions.append((reference_points[:, :, 0], key_flow, masks, scores))

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
