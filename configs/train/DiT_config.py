import torch
import os
from os.path import join
from datetime import datetime
current_time = datetime.now()
time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")


class RDTconfig:
    base_output = os.getcwd() # replace with your desired base output directory
    # base config
    dataset = 'graspnet'
    ptv3_model = 'ptv3'
    model = 'DiT'

    # training super parameters config
    max_epoch = 3000
    ckpt_every = 20
    batch_size = 64
    learning_rate = 1e-4
    weight_decay = 0
    seed = 42
    shuffle = True
    start = 1
    debug = False
    num_worker_dataloader = 4 # num cpus for data loader
    
    # wandb config 
    wandb_project = 'semantic_grasp2'
    wandb_name = f'{dataset}_{ptv3_model}_{model}_b{batch_size}_lr{learning_rate}_{time_str}'
    wandb_id = None
    wandb_mode = 'offline'
    wandb_resume = None
    
    # multi-gpu training config
    num_worker = 1         # num gpus for training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # diffusion config
    hidden_size = 64
    num_head = 8
    train_diffusion_step = 1000
    val_diffusion_step = 250

    # output config
    output_folder = join(base_output, wandb_name)
    checkpoint_dir = join(output_folder, 'ckpts')
    log_file = join(output_folder, 'log.txt')
    