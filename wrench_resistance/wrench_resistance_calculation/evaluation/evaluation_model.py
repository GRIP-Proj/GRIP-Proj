import open3d as o3d
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from graspnetAPI.utils.dexnet.grasping.meshpy import Mesh3D
from graspnetAPI.utils.dexnet.grasping.graspable_object import GraspableObject3D
from graspnetAPI.utils.dexnet.grasping.meshpy import Mesh3D, ObjFile, SdfFile
from graspnetAPI.utils.dexnet.grasping.contacts import Contact3D
from graspnetAPI.utils.dexnet.grasping.grasp import ParallelJawPtGrasp3D
from graspnetAPI.utils.dexnet.grasping.quality import PointGraspMetrics3D
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfig

import json
import sys
sys.path.append('/home/diligent/Desktop/semantic_grasp_DiT')
sys.path.append('/home/diligent/Desktop/semantic_grasp_DiT/script')
from grasp_utils import plot_gripper_pro_max, create_radius
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
    
def apply_scale_to_mesh(mesh, scale):
    vertices = np.asarray(mesh.vertices)
    vertices *= scale
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh

mesh = o3d.io.read_triangle_mesh('../pd_data/merged_mesh.ply')

# o3d.visualization.draw_geometries([mesh])

mesh.compute_vertex_normals()

grasps = json.load(open('../pd_data/grasp.json', 'r'))

# configuration part
mass = 0.1
friction_coef = 0.7
num_cone_faces = 16
wrench_norm_thresh = 1.0
force_limits = 120.0
vis = False

for scale_grasp in grasps:
    scale = scale_grasp['scale']
    mesh = apply_scale_to_mesh(mesh, scale)
    
    for grasp_key in scale_grasp['grasp'].keys():
        grasp = scale_grasp['grasp'][grasp_key]
        rotation = np.array(grasp['rotation'])
        translation = np.array(grasp['translation'])
        
        rot = rotation @ (R_x(90) @ R_y(90))
        hand_tile = translation - (rot[:, 2] * 0.06).reshape(-1)
        
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = hand_tile
        
        gripper = create_gripper_geometry(transform)
        
        contact_point_l = np.array(grasp['final_contact_point_l'])
        contact_point_r = np.array(grasp['final_contact_point_r'])
        
        radius_l = create_radius(contact_point_l, 0.002)
        radius_r = create_radius(contact_point_r, 0.002)
        
        if vis:
            o3d.visualization.draw_geometries([mesh, radius_l, radius_r] + gripper)

        center_mass = np.asarray(mesh.vertices).mean(axis=0)
        
        dexnet_mesh = Mesh3D(
            vertices = mesh.vertices,
            triangles = np.asarray(mesh.triangles),
            normals = mesh.vertex_normals,
            center_of_mass=center_mass,
        )
        
        
        obj = GraspableObject3D(mesh=dexnet_mesh, sdf=None, mass=mass)
        
        c1 = Contact3D(
            graspable = obj,
            contact_point = contact_point_l, # (3,)
            in_direction = contact_point_r - contact_point_l, # inward facing grasp axis
            sdf_flag = False,
        )
        
        c2 = Contact3D(
            graspable = obj,
            contact_point = contact_point_r, # (3,)
            in_direction = contact_point_l - contact_point_r, # inward facing grasp axis
            sdf_flag = False,
        )
        
        grasp = ParallelJawPtGrasp3D.grasp_from_endpoints(c1.point, c2.point)
        
        c1_torque_scaling = 1.0 / np.linalg.norm(c1.point - center_mass)
        c2_torque_scaling = 1.0 / np.linalg.norm(c2.point - center_mass)
        torque_scaling = (c1_torque_scaling + c2_torque_scaling) / 2
        
        config = GraspQualityConfig(
            config={
                "quality_method": "wrench_resistance",
                "friction_coef": friction_coef,
                "num_cone_faces": num_cone_faces,
                "soft_fingers": False,
                "check_approach": True,
                # target wrench must be 6x1 column; here we request a small downward force only
                "target_wrench": np.array([[0.0], [0.0], [mass * 9.81], [0.0], [0.0], [0.0]]),
                # per-finger L1 limit; increase if your gripper can apply more normal force
                "force_limits": force_limits,
                # normalize torque vs force contributions in QP cost
                "torque_scaling": torque_scaling,
                "all_contacts_required": True,
                'wrench_norm_thresh': wrench_norm_thresh
            }
        )
        
        quality = PointGraspMetrics3D.grasp_quality(grasp, obj, config, [c1, c2])
        print(f'Grasp quality: {quality}')
        
        from IPython import embed; embed()
        
        
        
        
    


