import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock, AlternateCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from update import Decoder, PositionEmbedding

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

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)

        self.embedding = PositionEmbedding(self.args, hidden_dim=256)
        self.memory_embedding = nn.Parameter(torch.empty((1, 256, 1, 1, 1)))

        w, h = args.image_size[1] // 8, args.image_size[0] // 8
        self.query = nn.Parameter(torch.empty((1, 256, h, w)))

        self.decoders = \
            nn.ModuleList([Decoder(self.args, hidden_dim=256, num_heads=8, ff_dim=256 * 4, dropout=0.1)
                           for _ in range(args.iters)])

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

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

        fmap1 = self.embedding(fmap1)
        fmap2 = self.embedding(fmap2)

        n, c, h, w = fmap1.size()

        # net = fmap1
        net = self.query.repeat((n, 1, 1, 1))
        _, _, q_h, q_w = net.size()
        if h != q_h or w != q_w:
            net = F.interpolate(net, size=(h, w), mode='bilinear', align_corners=True)

        memory = torch.stack((fmap1, fmap2), axis=2)
        memory = memory + self.memory_embedding
        flow_predictions = []
        for itr in range(iters):
            net, preds = self.decoders[itr](query=net, key=memory)

            # upsample predictions
            flow_up = upflow8(preds)

            flow_predictions.append(flow_up)

        if test_mode:
            return preds, flow_up
            
        return flow_predictions
