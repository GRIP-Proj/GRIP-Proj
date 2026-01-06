import torch
import os
from os.path import join
from datetime import datetime
current_time = datetime.now()
time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")


class Cross_attn_config:
    dataset = 'gpd'
    ptv3_model = 'ptv3'

    # wandb config 
    wandb_project = 'semantic_grasp2'
    wandb_name = f'{dataset}_{ptv3_model}_cross_attn_b64_lr1e-4_DDP'
    wandb_id = None
    wandb_mode = 'offline'
    wandb_resume = None

    # training super parameters config
    max_epoch = 1000
    ckpt_every = 20
    batch_size = 64
    learning_rate = 1e-5
    weight_decay = 0
    seed = 42
    shuffle = True
    start = 1
    debug = False
    
    # multi-gpu training config
    num_worker = 1
    device = 'cuda'
    dtype = torch.float32
    
    # model setting 
    dim = 64
    num_head = 8
    in_feat = 9
    out_feat = 12
    num_layer = 4

    # output config
    checkpoint_dir = f'./ckpt_{dataset}_cross_attn_b64_lr1e-4'
    log_file = f'./output/output_{dataset}_cross_attn_b64_lr1e-4.log'
    num_worker_dataloader = 2
    