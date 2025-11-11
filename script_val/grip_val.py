import os
from os.path import join 
import sys

sys.path.append(os.getcwd())

import torch
from diffusion import create_diffusion
import numpy as np
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset, DataLoader
import open3d as o3d

from utils.server_client import SocketServer


from models.semantic_grasp_backbone import Semantic_Grasp, Dataset_SemanticGrasp
from models.DiT_model import DiT
from models.RDT_model import RDTBlock
from models.cross_attn import Gripper_AutoFit_Attn, Gripper_AutoFit_Dataset
from script.visualize_open3d_unit import (
    load_and_process_mesh,
    load_and_process_mesh_with_texture,
    load_and_process_mesh_with_texture_raw,
    create_gripper_geometry
)
from utils.grasp_utils import plot_gripper_pro_max, create_radius


device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
num_eval = 10
model_type = 'RDT'
dataset_type = 'gpd'

def R_x(angle_deg):
    """Rotation matrix about X-axis."""
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def R_y(angle_deg):
    """Rotation matrix about Y-axis."""
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def R_z(angle_deg):
    """Rotation matrix about Z-axis."""
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
def farthest_point_sampling(points, num_samples):
    """
    最远点采样 (Farthest Point Sampling, FPS)
    参数:
        points: (N, 3) numpy 数组，表示点云
        num_samples: 要采样的点数
    返回:
        采样点索引列表
    """
    N = points.shape[0]
    sampled_idx = np.zeros(num_samples, dtype=np.int64)
    # 随机选择第一个点
    sampled_idx[0] = np.random.randint(0, N)
    # 记录所有点与已选点集的最近距离
    distances = np.linalg.norm(points - points[sampled_idx[0]], axis=1)
    
    for i in range(1, num_samples):
        # 选择距离当前集合最远的点
        next_idx = np.argmax(distances)
        sampled_idx[i] = next_idx
        # 更新所有点到最近已选点的距离
        new_dist = np.linalg.norm(points - points[next_idx], axis=1)
        distances = np.minimum(distances, new_dist)
    
    return sampled_idx

def main(cross_attn_config, dit_config):
    torch.manual_seed(dit_config.seed)
    
    semantic_grasp = Semantic_Grasp(device, './ckpt_ptv3/scannet200/config.py', './ckpt_ptv3/scannet200/model_best.pth')
    
    cross_model = Gripper_AutoFit_Attn(hidden_size=cross_attn_config.dim, num_heads=cross_attn_config.num_head, \
        in_feat=cross_attn_config.in_feat, out_feat=cross_attn_config.out_feat, num_layers=cross_attn_config.num_layer)

    diffusion = create_diffusion(timestep_respacing="") 
    
    if model_type == 'DiT':
        model = DiT(text_size=512, in_channels=9, hidden_size=64, depth=28, num_heads=dit_config.num_head)
    else:
        model = RDTBlock(hidden_size=dit_config.hidden_size, num_heads=dit_config.num_head)
    
    model.to(dtype=dtype, device=device)
    cross_model.to(dtype=dtype, device=device)
    
    model_pth_path = f'/home/diligent/Desktop/semantic_grasp_DiT/GRIP/ckpts/{model_type}_{dataset_type}/epoch0001600.pt'
    cross_model_pth_path = f'/home/diligent/Desktop/semantic_grasp_DiT/GRIP/ckpts/{model_type}_{dataset_type}/best_{dataset_type}.pth'

    model.load_state_dict(torch.load(model_pth_path, map_location=device)['model'])
    cross_model.load_state_dict(torch.load(cross_model_pth_path, map_location=device)['model_state_dict'])
    
    cross_model.eval()
    model.eval()
    
    ply_path = './objects/banana.ply'
    text = 'Grasp body of banana from downward direction'
    pcd = o3d.io.read_point_cloud(ply_path)
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30)
    )
    
    points = np.asarray(pcd.points).copy()
    points_mean = np.mean(points, axis=0)
    points -= points_mean
    normals = np.asarray(pcd.normals)
    
    pcd_indices = farthest_point_sampling(points, num_samples=1000)
    selected_points = points[pcd_indices]
    selected_normals = normals[pcd_indices]
    
    semantic_grasp.create_ptv3_input_dict(selected_points, selected_normals)
    ptv3_feat = semantic_grasp.ptv3_forward().feat.to(device=device, dtype=dtype).unsqueeze(0) # 1 * 1000 * 64
    text_feat = semantic_grasp.encoder_text([text]).to(device=device, dtype=dtype) # 1 * 512
    if model_type == 'DiT':
        model_kwargs = dict(y=text_feat.repeat_interleave(num_eval, dim=0), pd=ptv3_feat.repeat_interleave(num_eval, dim=0))
    else:
        model_kwargs = dict(c=text_feat.repeat_interleave(num_eval, dim=0), pd=ptv3_feat.repeat_interleave(num_eval, dim=0))
    output = diffusion.p_sample_loop(model, [num_eval, 9], clip_denoised=False, \
        model_kwargs=model_kwargs, progress=False, device=device)
    
    gripper_pt = torch.load('./gripper_mesh/franka_hand_point_feat.pt').to(device=device, dtype=dtype) # 1000 * 64
    
    result = cross_model(gripper_pt.unsqueeze(0).repeat_interleave(num_eval, dim=0), \
        ptv3_feat.repeat_interleave(num_eval, dim=0), output).detach().cpu().numpy()
    
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.1,  # length of each axis
    origin=[0, 0, 0]
)
    
    for i in range(num_eval):
        cur_output = output[i].detach().cpu().numpy()
        contact_point = cur_output[6:] + points_mean
        contact_radius = create_radius(contact_point, radius=0.005, color=[1, 0, 0])
        rot_1 = (cur_output[0:3]) / np.linalg.norm(cur_output[0:3])
        rot_2 = cur_output[3:6] / np.linalg.norm(cur_output[3:6])
        rot_3 = np.cross(rot_1, rot_2)
        rot = np.stack([rot_1, rot_2, rot_3], axis=1)
        rot = rot @ (R_x(90) @ R_y(90))
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = contact_point
        gripper_mesh_raw = create_gripper_geometry(transform=transform, color=[0, 1, 0])
        
        grasp_rot = result[i, :9].reshape(3, 3, order='F')
        grasp_tran = result[i, 9:].reshape(3,)
        
        grasp_rot = grasp_rot # @ (R_x(90) @ R_y(90))
        grasp_tran = grasp_tran # - grasp_rot[:, 2] * 0.06
        transform = np.eye(4)
        transform[:3, :3] = grasp_rot
        transform[:3, 3] = grasp_tran + points_mean
        gripper_mesh = create_gripper_geometry(transform)
        
        new_grasp_rot = grasp_rot @ R_x(180)
        transform[:3, :3] = new_grasp_rot
        transform[:3, 3] = grasp_tran + points_mean - new_grasp_rot[:, 2] * 0.10
        gripper_mesh_pro = create_gripper_geometry(transform, color=[1, 0, 1])
        
        o3d.visualization.draw_geometries([pcd, contact_radius, axis] + gripper_mesh_pro)
    
    
    


if __name__ == "__main__":
    from configs.val.Cross_attn_config import Cross_attn_config
    from configs.val.DiT_config import DiTconfig
    cross_attn_config = Cross_attn_config()
    dit_config = DiTconfig()
    main(cross_attn_config, dit_config)



