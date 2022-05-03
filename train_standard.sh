#!/bin/bash
mkdir -p checkpoints
# 0: base
# 1: base
# 2: base
# 3: base
python -u train.py --name chairs-base --stage chairs --validation chairs --gpus 0 1 2 3 --num_steps 1000000 --batch_size 8 --lr 0.0002 --image_size 352 480 --wdecay 0.0001
#python -u train.py --name raft-things --stage things --validation sintel --restore_ckpt checkpoints/raft-chairs.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 400 720 --wdecay 0.0001
#python -u train.py --name sintel-base --stage sintel --validation sintel --gpus 0 --num_steps 1000000 --batch_size 16 --lr 0.0001 --image_size 352 768 --wdecay 0.00001 --gamma=0.85
#python -u train.py --name raft-sintel --stage sintel --validation sintel --restore_ckpt checkpoints/raft-things.pth --gpus 0 1 --num_steps 100000 --batch_size 6 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma=0.85
#python -u train.py --name kitti-base --stage kitti --validation kitti --gpus 0 --num_steps 1000000 --batch_size 16 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85
#python -u train.py --name raft-kitti --stage kitti --validation kitti --restore_ckpt checkpoints/raft-sintel.pth --gpus 0 1 --num_steps 50000 --batch_size 6 --lr 0.0001 --image_size 288 960 --wdecay 0.00001 --gamma=0.85
