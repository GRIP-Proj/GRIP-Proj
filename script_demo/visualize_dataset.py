import json
import os
from os.path import join
import numpy as np
import open3d as o3d
from utils.grasp_utils import plot_gripper_pro_max, create_radius
from script.visualize_open3d_unit import (
    load_and_process_mesh,
    load_and_process_mesh_with_texture,
    load_and_process_mesh_with_texture_raw,
    create_gripper_geometry
)

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
    
''' visualize graspgen train grasps '''
train_graspgen_dataset = './dataset_grasp/grasp_dataset/graspGen/train'
objaverse_data_mesh_obj = './dataset_grasp/mesh_dataset/objaverse_data_mesh_obj'

z_axis = np.array([0, 0, 1])

for uuid in os.listdir(train_graspgen_dataset):
    uuid_path = join(train_graspgen_dataset, uuid)
    mesh_path = join(join(objaverse_data_mesh_obj, uuid), 'pd_rgb_normal_5000.npy')
    mesh = np.load(mesh_path)
    
    grasps_path = join(uuid_path, 'grasp.json')
    with open(grasps_path, 'r') as f:
        grasps = json.load(f)
    f.close()
    scale = grasps['scale']
    grasps = grasps['grasp']
    
    mesh[:, :3] = mesh[:, :3] * scale
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(mesh[:, 3:6])
    pcd.normals = o3d.utility.Vector3dVector(mesh[:, 6:9])
    

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1,   # 坐标轴长度
        origin=[1, 0, 0]  # 原点位置
    )
    
    for key in grasps.keys():
        cur_grasp = grasps[key]
        transform = np.array(cur_grasp['transform'])
        
        rot = transform[:3, :3]
        
        contact_point_l = np.array(cur_grasp['contact_point_l'])
        contact_point_r = np.array(cur_grasp['contact_point_r'])

        radius_l = create_radius(contact_point_l, color=[1, 0, 0])
        radius_r = create_radius(contact_point_r)
        radius_center = create_radius(transform[:3, 3], radius=0.002, color=[0, 1, 0])
        
        gripper = create_gripper_geometry(transform)

        o3d.visualization.draw_geometries([pcd, axis, radius_l, radius_r, radius_center] + gripper)
        break


''' visualize gpd train grasps '''
# id_to_name = json.load(open('./dataset_grasp/mesh_dataset/partnet_mesh/id2name.json', 'r'))

# train_gpd_dataset = './dataset_grasp/grasp_dataset/gpd/train'

# for object_id in os.listdir(train_gpd_dataset):
#     object_path = join(train_gpd_dataset, object_id)
#     grasps_path = join(object_path, 'grasp.json')
#     with open(grasps_path, 'r') as f:
#         grasps = json.load(f)
#     f.close()
    
#     for scale_grasp in grasps:
#         scale = scale_grasp['scale']
#         cur_grasps = scale_grasp['grasp']
#         obj_name = id_to_name[object_id]
#         mesh_path = f'./dataset_grasp/mesh_dataset/partnet_mesh/{obj_name}/{object_id}/pcd_label.npy'
#         mesh_data = np.load(mesh_path)

#         xyz = mesh_data[:, :3]
#         xyz = xyz * scale   
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(xyz)
#         pcd.colors = o3d.utility.Vector3dVector(mesh_data[:, 3:6] / 255.0)
        
#         for key in cur_grasps.keys():
#             cur_grasp = cur_grasps[key]
#             rot = np.array(cur_grasp['rotation'])
#             tran = np.array(cur_grasp['translation'])
            
#             rot = rot @ (R_x(90) @ R_y(90))
            
#             hand_tile = tran - (rot[:, 2] * 0.06).reshape(-1)
            
#             transform = np.eye(4)
#             transform[:3, :3] = rot
#             transform[:3, 3] = hand_tile
            
#             contact_point_l = np.array(cur_grasp['final_contact_point_l'])
#             contact_point_r = np.array(cur_grasp['final_contact_point_r'])

#             radius_l = create_radius(contact_point_l, color=[1, 0, 0])
#             radius_r = create_radius(contact_point_r)
#             center = create_radius(tran, radius=0.002, color=[0, 1, 0])
#             hand_tile = create_radius(hand_tile, radius=0.002, color=[0, 0, 1])
#             gripper = create_gripper_geometry(transform)

#             o3d.visualization.draw_geometries([pcd, radius_l, radius_r, center] + gripper)
#             break
#         break

''' visualize graspnet train/val grasps '''

# id_to_name = json.load(open('./dataset_grasp/mesh_dataset/partnet_mesh/id2name.json', 'r'))

# train_dataset = './dataset_grasp/grasp_dataset/graspnet/train'

# for object_id in os.listdir(train_dataset):
#     object_path = join(train_dataset, object_id)
#     grasps_path = join(object_path, 'grasp.json')
#     with open(grasps_path, 'r') as f:
#         grasps = json.load(f)
#     f.close()
    
#     for scale_grasp in grasps:
#         scale = scale_grasp['scale']
#         cur_grasps = scale_grasp['grasp']
#         obj_name = id_to_name[object_id]
#         mesh_path = f'./dataset_grasp/mesh_dataset/partnet_mesh/{obj_name}/{object_id}/pcd_label.npy'
#         mesh_data = np.load(mesh_path)

#         xyz = mesh_data[:, :3]

#         xyz = xyz * scale
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(xyz)
#         pcd.colors = o3d.utility.Vector3dVector(mesh_data[:, 3:6] / 255.0)
        
#         for key in cur_grasps.keys():
#             cur_grasp = cur_grasps[key]
#             rot = np.array(cur_grasp['rotation_matrix'])
#             tran = np.array(cur_grasp['translation'])
            
#             rot = rot @ (R_x(90) @ R_y(90))
#             hand_tile = tran - (rot[:, 2] * 0.06).reshape(-1)
            
#             transform = np.eye(4)
#             transform[:3, :3] = rot
#             transform[:3, 3] = hand_tile
            
#             contact_point_l = np.array(cur_grasp['contact_point_left'])
#             contact_point_r = np.array(cur_grasp['contact_point_right'])

#             radius_l = create_radius(contact_point_l, color=[1, 0, 0])
#             radius_r = create_radius(contact_point_r)
#             gripper = create_gripper_geometry(transform)

#             o3d.visualization.draw_geometries([pcd, radius_l, radius_r] + gripper)
            
#             break
#         break