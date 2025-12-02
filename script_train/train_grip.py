import os
from os.path import join
import argparse
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.GRIP import GRIP
from configs.model.model_parameters import ModelParameters
from configs.train.DiT_config import DiTConfig
from configs.train.Cross_attn_config import RDTconfig
from models.dataloader import GRIP_backbone_dataset, GRIP_head_dataset





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone", type=str, default='DiT', help="backbone model type: DiT or RDT")
    parser.add_argument("--device", type=str, default='cuda', help="device for training")
    parser.add_argument("--ptv3", type=str, default='ptv3_scannet', help="ptv3 model type")
    parser.add_argument("--use_clip", type=bool, default=False, help="whether to use clip features")
    parser.add_argument("--epoch", type=int, default=100, help="epoch number for testing")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for testing")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for testing")
    parser.add_argument("--dtype", type=str, default='float32', help="data type for training")
    parser.add_argument("--mmap", type=bool, default=True, help="whether to use mmap for dataset loading")
    parser.add_argument("--shuffle", type=bool, default=True, help="whether to shuffle the dataset during training")
    parser.add_argument("--num_worker_dataloader", type=int, default=4, help="number of workers for data loader")
    parser.add_argument("--num_worker", type=int, default=1, help="number of gpus for training")
    args = parser.parse_args()
    
    
    if args.backbone == 'DiT':
        config = DiTConfig()
    elif args.backbone == 'RDT':
        config = RDTconfig()
    else:
        assert False, "Unknown backbone model"
    
    if args.ptv3 == 'ptv3_scannet':
        ptv3_config = 'ckpt_ptv3/scannet200/config.py'
        ptv3_weight = 'ckpt_ptv3/scannet200/model_best.pth'
    else:
        assert False, "Unknown ptv3 model"
    
    if args.dtype == 'float32':
        dtype = torch.float32
    else:
        assert False, "Unknown data type"
    
    # grip = GRIP(args.device, ptv3_config, ptv3_weight, args.use_clip)
    
    grip_head_dataset = GRIP_head_dataset(
        dataset_path='/mnt/bigai_ml/rongpeng/cross_attn_dataset/gpd/train',
        gripper_feat_path='/mnt/bigai_ml/rongpeng/grip_dataset/gripper_mesh/franka_hand_point_feat.pt',
        dtype=dtype,
        device=args.device,
        mmap=args.mmap
    )
    
    train_dataset = DataLoader(grip_head_dataset, args.batch_size, args.shuffle, num_workers=args.num_worker_dataloader, pin_memory=True)
    
    for batch_idx, (contact, grasp, point_feat, gripper_feat) in enumerate(train_dataset):
        from IPython import embed; embed()
    
    

