import os
from os.path import join
import sys
sys.path.append(os.getcwd())

import json

import open3d as o3d
import numpy as np

from utils.grasp_visualize import create_gripper_geometry
from wrench_resistance_calculation.evaluation.calculate_wrench_resistance import calculate_wrench_resistance_contact
from wrench_resistance_calculation.configs.config import WrenchResistanceConfig

config = WrenchResistanceConfig()

dataset_type = 'graspGen'
type_list = ['val']
mesh_path = '/mnt/bigai_ml/rongpeng/graspGen/objaverse_mesh'
pd_path = '/home/diligent/Desktop/semantic_grasp_DiT/dataset_grasp/mesh_dataset/objaverse_data_mesh_obj/'
grasp_pose_path = '/home/diligent/Desktop/semantic_grasp_DiT/dataset_grasp/selected_grasp_dataset/graspGen'

score_record = {}
for train_type in type_list:
    score_list = []
    grasp_num = 0
    
    grasp_folder = join(grasp_pose_path, train_type)
    for uuid in os.listdir(grasp_folder):
        if not os.path.exists(join(grasp_folder, uuid, 'grasp.json')):
            continue
        json_data = json.load(open(join(grasp_folder, uuid, 'grasp.json'), 'r'))[0]
        
        scale = json_data['scale']
        grasps = json_data['grasp']
        
        
        cur_pd_path = join(pd_path, uuid, f'pd_rgb_normal_5000.npy')
        pd_rgb_normal = np.load(cur_pd_path)
        xyz = pd_rgb_normal[:, :3]
        xyz = xyz * scale
        pd_center = np.mean(xyz, axis=0)
        # xyz = xyz - np.mean(xyz, axis=0)
        # pointcloud = o3d.geometry.PointCloud()
        # pointcloud.points = o3d.utility.Vector3dVector(xyz)
        # pointcloud.colors = o3d.utility.Vector3dVector(pd_rgb_normal[:, 3:6])
        # pointcloud.normals = o3d.utility.Vector3dVector(pd_rgb_normal[:, 6:9])
        
        # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        #     pointcloud, depth=9
        # )

        cur_mesh_path = join(mesh_path, f'{uuid}.ply')
        mesh = o3d.io.read_triangle_mesh(cur_mesh_path)
        # num_points = 5000
        # pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        # points = np.asarray(pcd.points)
        # point = points * scale
        # point = point - np.mean(point, axis=0)
        # pcd.points = o3d.utility.Vector3dVector(point)

        mesh.scale(scale, center=(0, 0, 0))
        
        mesh.translate(-pd_center)
        
        for grasp_key in grasps.keys():
            contact_point_l = np.array(grasps[grasp_key]['contact_point_l'])
            contact_point_r = np.array(grasps[grasp_key]['contact_point_r'])
            try:
                score = calculate_wrench_resistance_contact(contact_point_l, contact_point_r, mesh, config=config)
            except:
                continue
            score_list.append(score)
            print(f'score is {score:6f}')
            grasp_num += 1
    print(f'Object type: {train_type} - Average Wrench Resistance Score: {np.mean(score_list)} over {grasp_num} grasps')
    with open("output_graspGen.txt", "w") as f:
        for item in score_list:
            f.write(str(item) + "\n")





