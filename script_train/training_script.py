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


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
        
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
semantic_grasp = Semantic_Grasp(device, './ckpt/scannet200/config.py', './ckpt/scannet200/model_best.pth')
diffusion = create_diffusion(timestep_respacing="") 
model = DiT(text_size=512, in_channels=9, hidden_size=64, depth=28, num_heads=16)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
model.to(device)
model.to(dtype)
model.train()

ckpt_every = 5
checkpoint_dir = './save_ckpt'

graspGen_dataset_path = '/media/raid/workspace/surongpeng/ws/grasp_dataset/graspGen_dataset'
mesh_path = '/media/raid/workspace/surongpeng/ws/mesh_dataset/objaverse_data_mesh_obj'



output_dir = './dataset/'

num = 0

for uuid in os.listdir(graspGen_dataset_path):
    cur_output_dir = join(output_dir, uuid)
    if not os.path.exists(cur_output_dir):
        os.makedirs(cur_output_dir)
    if os.path.exists(join(cur_output_dir, 'x_all.pt')):
        continue
    
    x_all = []
    y_all = []
    point_feat_all = []
    
    uuid_mesh_path = join(mesh_path, uuid)
    with open(join(uuid_mesh_path, 'T_move_mesh_to_origin.json'), 'r') as json_file:
        T_move_mesh_to_origin = json.load(json_file)
    json_file.close()
    T_move_mesh_to_origin = np.array(T_move_mesh_to_origin)
    object_pd = np.load(join(uuid_mesh_path, 'pd_rgb_normal_1000.npy'))
    
    grasp_json_path = join(join(graspGen_dataset_path, uuid), 'grasp.json')
    with open(grasp_json_path, 'r') as json_file:
        grasp_json = json.load(json_file)
    json_file.close()
    
    scale = grasp_json['scale']
    object_pd[:, :3] = (object_pd[:, :3] - T_move_mesh_to_origin[:3, 3]) * scale
    object_pd[:, :3] -= np.mean(object_pd[:, :3], axis=0)
    object_pd = torch.from_numpy(object_pd).to(device).type(dtype)
    
    grasp = grasp_json['grasp']
    
    
    
    for key in grasp.keys():
        grasp_item = grasp[key]
        transform = np.array(grasp_item['transform'])
        contact_point_l = np.array(grasp_item['contact_point_l'])
        
        text = grasp_item['grasp_text']
        text_embedding = semantic_grasp.encoder_text([text]).type(dtype).to('cpu')
        text_embedding = text_embedding.squeeze(0)
        
        
        semantic_grasp.create_ptv3_input_dict(object_pd[:, :3], object_pd[:, 6:])
        output = semantic_grasp.ptv3_forward()
        point_feat = (output.feat).type(dtype)
        point_feat = point_feat.type(dtype).to('cpu')
        
        x = np.concatenate([transform[:3, 0], transform[:3, 1], contact_point_l])
        x = torch.from_numpy(x).type(dtype)
        
        x_all.append(x)
        y_all.append(text_embedding)
        point_feat_all.append(point_feat)
    
    if len(x_all) == 0:
        continue
    
    x_all = torch.stack(x_all, dim=0)
    y_all = torch.stack(y_all, dim=0)
    point_feat_all = torch.stack(point_feat_all, dim=0)

    torch.save(x_all, join(cur_output_dir, 'x_all.pt'))
    torch.save(y_all, join(cur_output_dir, 'y_all.pt'))
    torch.save(point_feat_all, join(cur_output_dir, 'point_feat_all.pt'))

x_list = []
y_list = []
point_feat_list = []

for uuid in os.listdir(output_dir):
    uuid_dir = join(output_dir, uuid)
    x_all_path = join(uuid_dir, 'x_all.pt')
    if not os.path.exists(x_all_path):
        continue
    y_all_path = join(uuid_dir, 'y_all.pt')
    point_feat_all_path = join(uuid_dir, 'point_feat_all.pt')
    x_all = torch.load(x_all_path)
    y_all = torch.load(y_all_path)
    point_feat_all = torch.load(point_feat_all_path)

    x_list.append(x_all)
    y_list.append(y_all)
    point_feat_list.append(point_feat_all)
    
x_all = torch.cat(x_list, dim=0)
y_all = torch.cat(y_list, dim=0)
point_feat_all = torch.cat(point_feat_list, dim=0)

dataset_semanticgrasp = Dataset_SemanticGrasp(x_all.detach(), y_all.detach(), point_feat_all.detach())
dataloader = DataLoader(dataset_semanticgrasp, batch_size=4096, shuffle=True, num_workers=1)

print('training start')
for epoch in range(1000):
    for batch_idx, (x, y, pd) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        pd = pd.to(device)
        
        # from IPython import embed; embed()
        
        model_kwargs = dict(y=y, pd=pd)

        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device).type(torch.int)
    
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict['loss'].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    if epoch % ckpt_every == 0 and epoch > 0:
        checkpoint = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
        }
        checkpoint_path = f"{checkpoint_dir}/epoch{epoch:07d}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f'save checkpoint {checkpoint_dir}/epoch{epoch:07d}.pt')
print('training end')
    
    
        
        
        
        
        
        
        
    