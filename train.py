from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
# from raft import RAFT
from core.ours import RAFT
import evaluate
import datasets
import flow_vis

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds[0])
    flow_loss = 0.0
    sparse_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    bs, _, I_H, I_W = flow_gt.shape

    for i in range(n_predictions):
        # i_weight = gamma ** (n_predictions - i - 1)
        i_weight = 1.0
        i_loss = (flow_preds[0][i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

        ref, sparse_flow = flow_preds[1][i]
        scale = torch.tensor((I_W, I_H), dtype=torch.float32).view(1, 1, 2).to(sparse_flow.device)
        flatten_gt = flow_gt.flatten(2).permute(0, 2, 1)
        ceil_coords = torch.ceil(ref * scale).long()
        floor_coords = torch.floor(ref * scale).long()
        sparse_gt = flatten_gt[torch.arange(bs),
                    floor_coords[torch.arange(bs), :, 1] * floor_coords[torch.arange(bs), :, 0]] * ref.frac() + \
                    flatten_gt[torch.arange(bs),
                    ceil_coords[torch.arange(bs), :, 1] * ceil_coords[torch.arange(bs), :, 0]] * (1 - ref.frac())
        print(sparse_gt.shape)
        exit()



    loss = flow_loss

    epe = torch.sum((flow_preds[0][-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
        'loss': loss
    }

    return loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wdecay)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, round(args.num_steps * 0.8))
    # scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
    #     pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def write_images(self, image1, image2, targets, preds):
        if self.writer is None:
            self.writer = SummaryWriter()

        image1 = image1.detach().cpu().numpy()
        image1 = np.transpose(image1, (0, 2, 3, 1))
        image2 = image2.detach().cpu().numpy()
        image2 = np.transpose(image2, (0, 2, 3, 1))
        targets = targets.detach().cpu().numpy()
        targets = np.transpose(targets, (0, 2, 3, 1))
        for n_i in range(len(targets)):
            this_image1 = image1[n_i]
            this_image2 = image2[n_i]
            target_img = flow_vis.flow_to_color(targets[n_i], convert_to_bgr=False)
            pred_img = list()
            for p_i in range(len(preds)):
                this_pred = preds[p_i].detach().cpu().numpy()[n_i]
                this_pred = np.transpose(this_pred, (1, 2, 0))
                pred_img.append(flow_vis.flow_to_color(this_pred, convert_to_bgr=False))
            pred_img = np.concatenate(pred_img, axis=1)
            image = np.concatenate((this_image1, this_image2, target_img, pred_img), axis=1)

            image = image.astype(np.uint8)

            self.writer.add_image("Image_{:02d}".format(n_i + 1), image, self.total_steps, dataformats='HWC')

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    if args.stage != 'chairs':
        model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    # VAL_FREQ = 100
    IMAGE_FREQ = 100
    add_noise = True

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            flow_predictions = model(image1, image2, iters=args.iters)
            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            logger.push(metrics)
            if total_steps % IMAGE_FREQ == IMAGE_FREQ - 1:
                logger.write_images(image1, image2, flow, flow_predictions[0])

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)

                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'chairs':
                        results.update(evaluate.validate_chairs(model.module, iters=args.iters))
                    elif val_dataset == 'sintel':
                        results.update(evaluate.validate_sintel(model.module, iters=args.iters))
                    elif val_dataset == 'kitti':
                        results.update(evaluate.validate_kitti(model.module, iters=args.iters))

                logger.write_dict(results)
                
                model.train()
                if args.stage != 'chairs':
                    model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    parser.add_argument('--iters', type=int, default=3)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(2022)
    np.random.seed(2022)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)