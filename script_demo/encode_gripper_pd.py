import os
from os.path import join
import json

import numpy as np
import torch
import open3d as o3d


from cross_grasp.models.semantic_grasp_backbone import Semantic_Grasp, Dataset_SemanticGrasp, Semantic_Grasp_object

franka_mesh_path = './gripper_mesh/franka_hand_toward_minus_z.obj'

mesh = o3d.io.read_triangle_mesh(franka_mesh_path)
if mesh.is_empty():
    raise FileNotFoundError(f"Failed to load mesh: {franka_mesh_path}")

mesh.compute_vertex_normals()
pcd = mesh.sample_points_poisson_disk(number_of_points=1000)
pcd_point_xyz = np.asarray(pcd.points)
pcd_point_normals = np.asarray(pcd.normals)
xyz_normals = np.concatenate([pcd_point_xyz, pcd_point_normals], axis=-1)  # N*6
np.save('./gripper_mesh/franka_hand_point_normal.npy', xyz_normals)

# o3d.visualization.draw_geometries([pcd])



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

semantic_grasp = Semantic_Grasp(device, './ckpt/scannet200/config.py', './ckpt/scannet200/model_best.pth')
semantic_grasp.create_ptv3_input_dict(pcd_point_xyz, pcd_point_normals)
output = semantic_grasp.ptv3_forward()
point_feat = (output.feat)
point_feat = point_feat.type(dtype).to('cpu').detach()

torch.save(point_feat, './gripper_mesh/franka_hand_point_feat.pt')







