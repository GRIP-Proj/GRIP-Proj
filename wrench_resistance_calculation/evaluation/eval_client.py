import open3d as o3d
import numpy as np
import sys
import os
import json
import shutil
from os.path import join
sys.path.append(os.getcwd())
sys.path.append(join(os.getcwd(), 'wrench_resistance_calculation'))
from graspnetAPI.utils.dexnet.grasping.meshpy import Mesh3D
from graspnetAPI.utils.dexnet.grasping.graspable_object import GraspableObject3D
from graspnetAPI.utils.dexnet.grasping.meshpy import Mesh3D, ObjFile, SdfFile
from graspnetAPI.utils.dexnet.grasping.contacts import Contact3D
from graspnetAPI.utils.dexnet.grasping.grasp import ParallelJawPtGrasp3D
from graspnetAPI.utils.dexnet.grasping.quality import PointGraspMetrics3D
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfig
from utils.grasp_visualize import create_gripper_geometry, R_x, R_y, R_z, create_radius
from utils.grasp_visualize import convert_graspnet, convert_gpd, convert_m2t2
from utils.server_client import SocketClient
from wrench_resistance_calculation.configs.config import WrenchResistanceConfig
from wrench_resistance_calculation.evaluation.calculate_wrench_resistance import calculate_wrench_resistance


client = SocketClient(host='10.1.119.63', port=12345)
client.connect()
config = WrenchResistanceConfig()

grasp_pose = None
receive_flag = False
@client.on_message
def handle_message(data):
    global grasp_pose
    global receive_flag
    print(f"📨 Received from server: {type(data)} - {data}")
    grasp_pose = data
    receive_flag = True

object_type = ['objaverse', 'partnet']
mesh_folder = './wrench_resistance_dataset/'
score_list = []
for cur_type in object_type:
    for uuid in os.listdir(join(mesh_folder, cur_type)):
        print(f'current object: {cur_type} - {uuid}')
        mesh_path = join(mesh_folder, cur_type, uuid, 'rescaled_mesh.ply')
        if not os.path.exists(mesh_path):
            print(f'Mesh not found: {mesh_path}')
            continue
        
        if config.model_type == 'gpd':
            dst = join('/home/diligent/pc_data/object.pcd')
        else:
            dst = join('/home/diligent/pc_data/object.ply')
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        pcd = mesh.sample_points_poisson_disk(number_of_points=config.num_sample_points)
        o3d.io.write_point_cloud(dst, pcd)
        
        client.send('hello')
        
        exec_flag = False
        vis_tran, vis_rot = None, None
        while not exec_flag:
            if receive_flag:
                if grasp_pose:
                    tran = grasp_pose['tran']
                    rot = grasp_pose['rot']
                    if tran is None:
                        vis_tran, vis_rot = None, None
                        o3d.visualization.draw_geometries([pcd])
                    else:
                        if config.vis:
                            if config.model_type == 'graspnet':
                                vis_tran, vis_rot = convert_graspnet(tran, rot)
                            elif config.model_type == 'gpd':
                                vis_tran, vis_rot = convert_gpd(tran, rot)
                            elif config.model_type == 'm2t2':
                                vis_tran, vis_rot = convert_m2t2(tran, rot)
                            
                            # pcd = o3d.io.read_point_cloud("/home/pc/workspace/grip_exp/debug/object.ply")
                            transform = np.eye(4)
                            transform[:3,:3] = np.array(vis_rot)
                            transform[:3,3] = np.array(vis_tran)
                            # ee_base_point = create_radius(ee_base_point, radius=0.005, color=[0,0,1])
                            gripper_geom = create_gripper_geometry(transform=transform, gripper_name="franka_panda", color=[0,1,0], cylinder_radius=0.002)
                            # o3d.visualization.draw_geometries([pcd] + gripper_geom)
                        
                    receive_flag = False
                    grasp_pose = None
                    exec_flag = True
        
        if vis_tran is None:
            score_list.append(0.0)
            print('No valid grasp received, skip wrench resistance calculation.')
        else:
            ee_base_point = vis_tran + vis_rot[:, 2] * 0.06
            grasp_config = {}
            grasp_config['ee_base_point'] = ee_base_point
            grasp_config['translation'] = vis_tran
            grasp_config['rotation'] = vis_rot
            grasp_config['approach'] = vis_rot[:, 2]
            grasp_config['binormal'] = vis_rot[:, 0]
            grasp_config['width'] = 0.08
            grasp_config['depth'] = 0.06
            score = calculate_wrench_resistance(grasp_config, mesh, config)
            print(f'Wrench resistance score: {score}')
            score_list.append(score)
    
        # from IPython import embed; embed()    

from IPython import embed; embed()            
                        
        
        