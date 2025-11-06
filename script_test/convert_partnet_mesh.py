import numpy as np
import open3d as o3d
import os
from os.path import join
import shutil
import concurrent.futures
from multiprocessing import Process

folder_path = './dataset_grasp/mesh_dataset/partnet_mesh/'
for obj in os.listdir(folder_path):
    obj_path = join(folder_path, obj)
    if not os.path.isdir(obj_path):
        continue
    
    for obj_uuid in os.listdir(obj_path):
        
        obj_uuid_path = join(obj_path, obj_uuid)
        save_path = join(obj_uuid_path, 'pd_rgb_normal_1000.npy')
        if os.path.exists(save_path):
            continue
        
        ply_path = join(obj_uuid_path, 'merged_mesh.ply')
        mesh = o3d.io.read_triangle_mesh(ply_path)
        
        pcd = mesh.sample_points_uniformly(number_of_points=1000)
        
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15))
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        rgbs = np.zeros_like(points)
        
        point_rgb_normal =  np.concatenate([points, rgbs, normals], axis=1)
        np.save(save_path, point_rgb_normal)
        print(f'save {save_path}')