import os
from os.path import join
import json

from diffusion import create_diffusion
from cross_grasp.models.semantic_grasp_backbone import Semantic_Grasp, Dataset_SemanticGrasp
from cross_grasp.models.DiT_model import DiT
from cross_grasp.models.RDT_model import RDTBlock
from Contact_point_dection.contact_detector import ContactDetector

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset, DataLoader
import wandb
import open3d as o3d
from cross_grasp.models.cross_attn import Gripper_AutoFit_Attn

# wandb.init(project="semantic_grasp", name='graspgen_eval', mode='offline')
detector = ContactDetector()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
semantic_grasp = Semantic_Grasp(device, './ckpt/scannet200/config.py', './ckpt/scannet200/model_best.pth')
diffusion = create_diffusion(str(250)) 
model = DiT(text_size=512, in_channels=9, hidden_size=64, depth=28, num_heads=16)
# model = RDTBlock(hidden_size=64, num_heads=8)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
model.to(device)
model.to(dtype)
model.eval()

# model_path = './save_ckpt/epoch0000995.pt'
model_path = './model_ckpt/ckpt_DiT_b64_lr1e-4_DDP/epoch0000550.pt'
ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt['model'], strict=True)


graspGen_dataset_path = '/home/diligent/Desktop/semantic_grasp_DiT/training_dataset/grasp_dataset/graspGen_dataset'
mesh_path = '/home/diligent/Desktop/semantic_grasp_DiT/training_dataset/mesh_dataset/objaverse_data_mesh_obj'

best_cross_attn_ckpt = 'cross_attn_ckpt/ckpt_graspGen_cross_attn_b64_lr1e-4_DDP/best_graspGen.pth'
cross_attn = Gripper_AutoFit_Attn(hidden_size=64, num_heads=8, in_feat=9, out_feat=12, num_layers=4)
cross_attn.to(device)
cross_attn.load_state_dict(torch.load(best_cross_attn_ckpt, map_location=device)['model_state_dict'], strict=True)
gripper_feat_path = './gripper_mesh/franka_hand_point_feat.pt'
gripper_feat = torch.load(gripper_feat_path)
gripper_feat = gripper_feat.to(device=device, dtype=dtype)
gripper_feat = gripper_feat.unsqueeze(0)

for uuid in os.listdir(graspGen_dataset_path):
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
    
    from IPython import embed; embed()
    
    scale = grasp_json['scale']
    object_pd[:, :3] = (object_pd[:, :3]) * scale
    # object_pd[:, :3] = (object_pd[:, :3] - T_move_mesh_to_origin[:3, 3]) * scale
    # object_pd[:, :3] -= np.mean(object_pd[:, :3], axis=0)
    object_pd_torch = torch.from_numpy(object_pd).to(device).type(dtype)
    
    grasp = grasp_json['grasp']
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_pd[:, :3])
    
    for key in grasp.keys():
        grasp_item = grasp[key]
        gt_transform = np.array(grasp_item['transform'])
        
        # from IPython import embed; embed()
        
        contact_point_l = np.array(grasp_item['contact_point_l'])
        
        text = grasp_item['grasp_text']
        print(text)
        
        text_embedding = semantic_grasp.encoder_text([text]).type(dtype).to(device)
        
        semantic_grasp.create_ptv3_input_dict(object_pd_torch[:, :3], object_pd_torch[:, 6:])
        output = semantic_grasp.ptv3_forward()
        point_feat = (output.feat).type(dtype)
        point_feat = point_feat.type(dtype).to('cpu')
        
        x = np.concatenate([gt_transform[:3, 0], gt_transform[:3, 1], contact_point_l])
        x = torch.from_numpy(x).type(dtype)
        
        x = x.unsqueeze(0).to(device)
        point_feat = point_feat.unsqueeze(0).to(device)
        
        model_kwargs = dict(y=text_embedding, pd=point_feat)
        
        result = diffusion.p_sample_loop(model, [x.shape[0], 9], \
            clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device)
        
        result = cross_attn(gripper_feat, point_feat, result)
        
        result = result.to('cpu').detach().numpy()
        
        transform = np.eye(4)
        
        norm = np.linalg.norm(result[0, :3])
        transform[:3, 0] = result[0, :3] / norm
        norm = np.linalg.norm(result[0, 3:6])
        transform[:3, 1] = result[0, 3:6] / norm
        norm = np.linalg.norm(result[0, 6:9])
        transform[:3, 2] = result[0, 6:9] / norm
        transform[:3, 3] = result[0, 9:]
        print(f'result translation: {transform[:3, 3]}')
        
        contact_point = detector.create_radius(transform[:3, 3], radius=0.005, color=[1, 0, 0])
        gripper = detector.plot_gripper_pro_max(center=transform[:3, 3], R=transform[:3, :3], width=0.08, depth=0.06, score=0.8, color=(1, 0, 0))
        
        gt_contact_point = detector.create_radius(contact_point_l, radius=0.005, color=[0, 0, 1])
        gt_gripper = detector.plot_gripper_pro_max(center=gt_transform[:3, 3], R=gt_transform[:3, :3], width=0.08, depth=0.06, score=0.8, color=(0, 0, 1))
        
        o3d.visualization.draw_geometries([pcd, gt_gripper])
        break
    
        
        
           
        