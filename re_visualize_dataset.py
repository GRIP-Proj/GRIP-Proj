import open3d as o3d
import os
from os.path import join
import numpy as np

dataset_type = ['partnet', 'objaverse']
dataset_path = './wrench_resistance_dataset/'

for cur_type in dataset_type:
    for uuid in os.listdir(join(dataset_path, cur_type)):
        print(uuid)
        # mesh_path = join(dataset_path, cur_type, uuid, f'{uuid}.ply')
        # mesh = o3d.io.read_triangle_mesh(mesh_path)
        # o3d.visualization.draw_geometries([mesh])
        
        
        mesh_path = join(dataset_path, cur_type, uuid, 'rescaled_mesh.ply')
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        o3d.visualization.draw_geometries([mesh])

        pd_path = join(dataset_path, cur_type, uuid, 'pd_rgb_normal_1000.npy')
        pd_data = np.load(pd_path)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pd_data[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(pd_data[:,3:6])
        o3d.visualization.draw_geometries([pcd])


        
        
