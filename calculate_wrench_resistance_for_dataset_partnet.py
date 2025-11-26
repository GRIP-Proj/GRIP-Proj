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


type_list = ['val']
mesh_path = '/home/diligent/Desktop/semantic_grasp_DiT/dataset_grasp/mesh_dataset/partnet_mesh/'
grasp_pose_path = '/home/diligent/Desktop/semantic_grasp_DiT/dataset_grasp/selected_grasp_dataset/graspnet'
id2name_path = '/home/diligent/Desktop/semantic_grasp_DiT/dataset_grasp/mesh_dataset/partnet_mesh/id2name.json'
id2name = json.load(open(id2name_path, 'r'))
score_record = {}
for train_type in type_list:
    grasp_folder = join(grasp_pose_path, train_type)
    
    score_list = []
    grasp_num = 0
    
    for uuid in os.listdir(join(grasp_folder)):

        if not os.path.exists(join(grasp_folder, uuid, 'grasp.json')):
            continue
        object_type = id2name[uuid]
        
        json_data = json.load(open(join(grasp_folder, uuid, 'grasp.json'), 'r'))
        
        for item in json_data:
            scale = item['scale']
            grasps = item['grasp']

            cur_mesh_path = join(mesh_path, object_type, uuid, f'merged_mesh.ply')
            mesh = o3d.io.read_triangle_mesh(cur_mesh_path)
            center = mesh.get_center()
            mesh.scale(scale, center=(0, 0, 0))
            # mesh.translate(-center)

            # grasp_num = len(grasps)
            
            for grasp_key in grasps.keys():
                contact_point_l = np.array(grasps[grasp_key]['contact_point_left'])
                contact_point_r = np.array(grasps[grasp_key]['contact_point_right'])
                try:
                    score = calculate_wrench_resistance_contact(contact_point_l, contact_point_r, mesh, config=config)
                except:
                    continue
                score_list.append(score)
                print(f'score is {score:6f}')
                grasp_num += 1
    print(f'Object type: {train_type} - Average Wrench Resistance Score: {np.mean(score_list)} over {grasp_num} grasps')
    with open("output_graspnet.txt", "w") as f:
        for item in score_list:
            f.write(str(item) + "\n")
            





