import open3d as o3d
import numpy as np
import sys
import os
import sys
from os.path import join
sys.path.append(os.getcwd())
sys.path.append(join(os.getcwd(), 'wrench_resistance_calculation'))
sys.path.insert(0, join(os.getcwd(), 'wrench_resistance_calculation/graspnetAPI'))
from graspnetAPI.utils.dexnet.grasping.meshpy import Mesh3D
from graspnetAPI.utils.dexnet.grasping.graspable_object import GraspableObject3D
from graspnetAPI.utils.dexnet.grasping.meshpy import Mesh3D, ObjFile, SdfFile
from graspnetAPI.utils.dexnet.grasping.contacts import Contact3D
from graspnetAPI.utils.dexnet.grasping.grasp import ParallelJawPtGrasp3D
from graspnetAPI.utils.dexnet.grasping.quality import PointGraspMetrics3D
from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfig
from Contact_point_dection.contact_detector import ContactDetector
from utils.grasp_visualize import create_gripper_geometry, R_x, R_y, R_z, create_radius

def calculate_wrench_resistance(grasp_config, mesh, config):
    mesh.compute_vertex_normals()
    
    center_mass = np.asarray(mesh.vertices).mean(axis=0)
    
    transform = np.eye(4)
    transform[:3,:3] = grasp_config['rotation']
    transform[:3,3] = grasp_config['translation']
    gripper_geom = create_gripper_geometry(transform=transform, gripper_name="franka_panda", color=[0,1,0], cylinder_radius=0.002)
    
    contact_detector = ContactDetector()
    contact_points_l, dis_l, contact_points_r, dis_r = contact_detector.find_lr_contact(mesh, grasp_config)
    dis_l = dis_l.reshape(-1)
    dis_r = dis_r.reshape(-1)
    total_dis = dis_l + dis_r
    idx_final = np.argmin(total_dis)
    
    contact_point_l = contact_points_l[idx_final]
    contact_point_r = contact_points_r[idx_final]
    
    point_l = create_radius(contact_point_l, radius=0.005, color=[1,0,0])
    point_r = create_radius(contact_point_r, radius=0.005, color=[0,1,0])
    o3d.visualization.draw_geometries([mesh, point_l, point_r] + gripper_geom)
    
    # from IPython import embed; embed()
    
    dexnet_mesh = Mesh3D(
        vertices = mesh.vertices,
        triangles = np.asarray(mesh.triangles),
        normals = mesh.vertex_normals,
        center_of_mass=center_mass,
    )
    
    obj = GraspableObject3D(mesh=dexnet_mesh, sdf=None, mass=config.mass)
    
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
            "friction_coef": config.friction_coef,
            "num_cone_faces": config.num_cone_faces,
            "soft_fingers": False,
            "check_approach": True,
            # target wrench must be 6x1 column; here we request a small downward force only
            "target_wrench": np.array([[0.0], [0.0], [config.mass * 9.81], [0.0], [0.0], [0.0]]),
            # per-finger L1 limit; increase if your gripper can apply more normal force
            "force_limits": config.force_limits,
            # normalize torque vs force contributions in QP cost
            "torque_scaling": torque_scaling,
            "all_contacts_required": True,
            'wrench_norm_thresh': config.wrench_norm_thresh
        }
    )
        
    quality = PointGraspMetrics3D.grasp_quality(grasp, obj, config, [c1, c2])
    
    return quality


def calculate_wrench_resistance_contact(contact_point_l, contact_point_r, mesh, config):
    mesh.compute_vertex_normals()
    
    # point_l = create_radius(contact_point_l, radius=0.005, color=[1,0,0])
    # point_r = create_radius(contact_point_r, radius=0.005, color=[0,1,0])
    # o3d.visualization.draw_geometries([mesh, point_l, point_r])
    
    # return
    
    center_mass = np.asarray(mesh.vertices).mean(axis=0)
    
    dexnet_mesh = Mesh3D(
        vertices = mesh.vertices,
        triangles = np.asarray(mesh.triangles),
        normals = mesh.vertex_normals,
        center_of_mass=center_mass,
    )
    
    obj = GraspableObject3D(mesh=dexnet_mesh, sdf=None, mass=config.mass)
    
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
            "friction_coef": config.friction_coef,
            "num_cone_faces": config.num_cone_faces,
            "soft_fingers": False,
            "check_approach": True,
            # target wrench must be 6x1 column; here we request a small downward force only
            "target_wrench": np.array([[0.0], [0.0], [config.mass * 9.81], [0.0], [0.0], [0.0]]),
            # per-finger L1 limit; increase if your gripper can apply more normal force
            "force_limits": config.force_limits,
            # normalize torque vs force contributions in QP cost
            "torque_scaling": torque_scaling,
            "all_contacts_required": True,
            'wrench_norm_thresh': config.wrench_norm_thresh
        }
    )
        
    quality = PointGraspMetrics3D.grasp_quality(grasp, obj, config, [c1, c2])
    
    return quality

    

