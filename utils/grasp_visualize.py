
import numpy as np
import open3d as o3d
import trimesh.transformations as tra
from typing import List, Optional, Tuple, Union, Any


control_points_franka = np.array([
        [ 0.05268743, -0.00005996, 0.05900000],
        [-0.05268743,  0.00005996, 0.05900000],
        [ 0.05268743, -0.00005996, 0.10527314],
        [-0.05268743,  0.00005996, 0.10527314]
    ])

control_points_robotiq2f140 = np.array([
    [ 0.06801729, -0, 0.0975],
    [-0.06801729,  0, 0.0975],
    [ 0.06801729, -0, 0.1950],
    [-0.06801729,  0, 0.1950]
])

control_points_suction = np.array([
    [ 0, 0, -0.10],
    [ 0, 0, -0.05],
    [ 0, 0, 0],
])

control_points_data = {
    "franka_panda": control_points_franka,
    "robotiq_2f_140": control_points_robotiq2f140,
    "single_suction_cup_30mm": control_points_suction,
}

def generate_circle_points(center: List[float], radius: float = 0.007, N: int = 30) -> np.ndarray:
    """
    Generate points forming a circle in 2D space.
    
    Args:
        center (List[float]): Center coordinates [x, y] of the circle
        radius (float): Radius of the circle
        N (int): Number of points to generate around the circle
    
    Returns:
        np.ndarray: Array of shape (N, 2) containing the circle points
    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x_points = center[0] + radius * np.cos(angles)
    y_points = center[1] + radius * np.sin(angles)
    points = np.stack((x_points, y_points), axis=1)
    return points

def get_gripper_control_points(gripper_name: str = 'franka_panda') -> np.ndarray:
    """
    Get the control points for a specific gripper.
    
    Args:
        gripper_name (str): Name of the gripper ("franka_panda", "robotiq_2f_140", "single_suction_cup_30mm")
    
    Returns:
        np.ndarray: Array of control points for the specified gripper
    
    Raises:
        NotImplementedError: If the specified gripper is not implemented
    """
    if gripper_name in control_points_data:
        return control_points_data[gripper_name]
    else:
        raise NotImplementedError(f"Gripper {gripper_name} is not implemented.")
    return control_points

def get_gripper_depth(gripper_name: str) -> float:
    """
    Get the depth parameter for a specific gripper type.
    
    Args:
        gripper_name (str): Name of the gripper ("franka_panda", "robotiq_2f_140", "single_suction_cup_30mm")
    
    Returns:
        float: Depth parameter for the specified gripper
    
    Raises:
        NotImplementedError: If the specified gripper is not implemented
    """
    # TODO: Use register module. Don't have this if-else name lookup
    pts, d = None, None
    if gripper_name in ["franka_panda", "robotiq2f140"]:
        pts = get_gripper_control_points(gripper_name)
    elif gripper_name == "suction":
        return 0.069
    else:
        raise NotImplementedError(f"Control points for gripper {gripper_name} not implemented!")
    d = pts[-1][-1] if pts is not None else d
    return d


def load_visualize_control_points_suction() -> np.ndarray:
    """
    Load visualization control points specific to the suction gripper.
    
    Returns:
        np.ndarray: Array of control points for suction gripper visualization
    """
    h = 0
    pts = [
        [0.0, 0],
    ]
    pts = [generate_circle_points(c, radius=0.005) for c in pts]
    pts = np.stack(pts)
    ptsz = h * np.ones([pts.shape[0], pts.shape[1], 1])
    pts = np.concatenate([pts, ptsz], axis=2)
    return pts

def get_gripper_offset(gripper_name: str) -> np.ndarray:
    """
    Get the offset transform for a specific gripper type.
    
    Args:
        gripper_name (str): Name of the gripper
    
    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix representing the gripper offset
    """
    return np.eye(4)

def get_gripper_visualization_control_points(gripper_name: str = 'franka_panda') -> List[np.ndarray]:
    """
    Get control points for visualizing a specific gripper type.
    
    Args:
        gripper_name (str): Name of the gripper ("franka_panda", "robotiq_2f_140", "single_suction_cup_30mm")
    
    Returns:
        List[np.ndarray]: List of control point arrays for gripper visualization
    """
    if gripper_name == "suction":
        control_points = load_visualize_control_points_suction()
        offset = get_gripper_offset('suction')
        ctrl_pts = [tra.transform_points(cpt, offset) for cpt in control_points]
        d = get_gripper_depth(gripper_name)
        line_pts = np.array([[0,0,0], [0,0,d]])
        line_pts = np.expand_dims(line_pts, 0)
        line_pts = [tra.transform_points(cpt, offset) for cpt in line_pts]
        line_pts = line_pts[0]
        ctrl_pts.append(line_pts)
        return ctrl_pts
    else:
        control_points = get_gripper_control_points(gripper_name)
        mid_point = (control_points[0] + control_points[1]) / 2
        control_points = [
            control_points[-2], control_points[0], mid_point,
            [0, 0, 0], mid_point, control_points[1], control_points[-1]
        ]
        return [control_points, ]

def load_visualization_gripper_points(gripper_name: str = "franka_panda") -> List[np.ndarray]:
    """
    Load control points for gripper visualization.
    
    Args:
        gripper_name (str): Name of the gripper to visualize
    
    Returns:
        List[np.ndarray]: List of control point arrays, each of shape [4, N]
        where N is the number of points for that segment
    """
    ctrl_points = []
    for ctrl_pts in get_gripper_visualization_control_points(gripper_name):
        ctrl_pts = np.array(ctrl_pts, dtype=np.float32)
        ctrl_pts = np.hstack([ctrl_pts, np.ones([len(ctrl_pts),1])])
        ctrl_pts = ctrl_pts.T
        ctrl_points.append(ctrl_pts)
    return ctrl_points

def create_gripper_geometry(transform, gripper_name="franka_panda", color=[1, 0, 0], cylinder_radius=0.002):
    """使用小圆柱体创建gripper几何体"""
    grasp_vertices = load_visualization_gripper_points(gripper_name)
    geometries = []
    
    for grasp_vertex in grasp_vertices:
        transformed_vertices = transform @ grasp_vertex
        points_3d = transformed_vertices[:3, :].T
        
        if len(points_3d) > 1:
            # 为每条线段创建圆柱体
            for j in range(len(points_3d)-1):
                start_point = points_3d[j]
                end_point = points_3d[j+1]
                
                # 计算线段长度和中心点
                line_vector = end_point - start_point
                line_length = np.linalg.norm(line_vector)
                center_point = (start_point + end_point) / 2
                
                # 创建圆柱体
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=cylinder_radius, 
                    height=line_length
                )
                
                # 计算旋转矩阵（将Z轴对齐到线段方向）
                z_axis = np.array([0, 0, 1])
                line_direction = line_vector / line_length
                
                # 计算旋转轴和角度
                rotation_axis = np.cross(z_axis, line_direction)
                rotation_angle = np.arccos(np.clip(np.dot(z_axis, line_direction), -1, 1))
                
                if np.linalg.norm(rotation_axis) > 1e-6:  # 避免平行向量的情况
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                        rotation_axis * rotation_angle
                    )
                else:
                    rotation_matrix = np.eye(3)
                
                # 应用变换
                cylinder.rotate(rotation_matrix, center=[0, 0, 0])
                cylinder.translate(center_point)
                cylinder.paint_uniform_color(color)
                
                geometries.append(cylinder)
    
    return geometries

def create_direction_arrow(translation, direction_vec, arrow_length=0.1, color=[1, 0, 0]):
    arrow_direction = direction_vec# * arrow_length

    # Open3D's create_arrow creates an arrow along +Z, so we need to align it
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.003, cone_radius=0.006,
        cylinder_height=arrow_length * 0.8, cone_height=arrow_length * 0.2
    )
    arrow.paint_uniform_color(color)  # red color

    # Compute rotation matrix to align +Z to arrow_direction
    z = np.array([0, 0, 1])
    v = np.cross(z, arrow_direction)
    c = np.dot(z, arrow_direction)
    if np.linalg.norm(v) < 1e-8:
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
    arrow.rotate(R, center=np.zeros(3))
    arrow.translate(translation)
    
    return arrow

def create_radius(center, radius=0.01, color=[0, 1, 0]):
    z_radius = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    z_radius.translate(center)
    z_radius.paint_uniform_color(color)  # green color
    return z_radius


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
def convert_graspnet(tran, rot):
    rot = rot @ (R_x(90) @ R_y(90))
    tran = tran - (rot[:, 2] * 0.06).reshape(-1)
    return tran, rot

def convert_gpd(tran, rot):
    # rot = rot @ (R_x(90) @ R_y(90))
    tran = tran - (rot[:, 2] * 0.06).reshape(-1)
    return tran, rot

def convert_m2t2(tran, rot):
    return tran, rot


