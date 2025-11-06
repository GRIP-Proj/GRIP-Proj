import os
from os.path import join
import json

from diffusion import create_diffusion
from cross_grasp.models.semantic_grasp_backbone import Semantic_Grasp, Dataset_SemanticGrasp
from cross_grasp.models.DiT_model import DiT

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset, DataLoader
import wandb


wandb.init(project="semantic_grasp", name='graspgen_eval', mode='offline')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
semantic_grasp = Semantic_Grasp(device, './ckpt/scannet200/config.py', './ckpt/scannet200/model_best.pth')
diffusion = create_diffusion(str(250)) 
model = DiT(text_size=512, in_channels=9, hidden_size=64, depth=28, num_heads=16)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
model.to(device)
model.to(dtype)
model.eval()

ckpt_every = 5
checkpoint_dir = './save_ckpt'

graspGen_dataset_path = '/media/raid/workspace/surongpeng/ws/grasp_dataset/graspGen_dataset'
mesh_path = '/media/raid/workspace/surongpeng/ws/mesh_dataset/objaverse_data_mesh_obj'

x_path = './eval_dataset/x.pt'
y_path = './eval_dataset/y.pt'
point_feat_path = './eval_dataset/point_feat.pt'

x_all = torch.load(x_path).to(device)
y_all = torch.load(y_path).to(device)
point_feat_all = torch.load(point_feat_path).to(device)

dataset_semanticgrasp = Dataset_SemanticGrasp(x_all.detach(), y_all.detach(), point_feat_all.detach())
dataloader = DataLoader(dataset_semanticgrasp, batch_size=1, shuffle=True, num_workers=1)


for epoch in range(5, 1000, 5):
    model.load_state_dict(torch.load(f'./save_ckpt/epoch{epoch:07d}.pt')['model'])
    
    epoch_loss = 0.0
    num = 0
    
    model_kwargs = dict(y=y_all, pd=point_feat_all)
    result = diffusion.p_sample_loop(model, [x_all.shape[0], 9], \
            clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
    loss = ((result - x_all) ** 2).sum()
    print(f'eval epoch {epoch} loss is {loss.item()}')
    wandb.log({"eval/loss": loss.item(), "epoch": epoch})
    
        

    
    
        
        
        
        