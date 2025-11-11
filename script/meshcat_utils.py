# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
'''
Utility functions for visualization using meshcat.

Installation:
    pip install trimesh==4.5.3 objaverse==0.1.7 meshcat==0.0.12 webdataset==0.2.111

NOTE: Start meshcat server (in a different terminal) before running this script:
    meshcat-server
'''

import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf
import trimesh
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

def get_gripper_offset(gripper_name: str) -> np.ndarray:
    """
    Get the offset transform for a specific gripper type.
    
    Args:
        gripper_name (str): Name of the gripper
    
    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix representing the gripper offset
    """
    return np.eye(4)

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

def get_color_from_score(labels: Union[float, np.ndarray], use_255_scale: bool = False) -> np.ndarray:
    """
    Convert score labels to RGB colors for visualization.
    
    Args:
        labels (Union[float, np.ndarray]): Score values between 0 and 1
        use_255_scale (bool): If True, output colors in [0-255] range, else [0-1]
    
    Returns:
        np.ndarray: RGB colors corresponding to the input scores
    """
    scale = 255.0 if use_255_scale else 1.0
    if type(labels) in [np.float32, float]:
        return scale * np.array([1 - labels, labels, 0])
    else:
        scale = 255.0 if use_255_scale else 1.0
        score = scale * np.stack(
            [np.ones(labels.shape[0]) - labels, labels, np.zeros(labels.shape[0])],
            axis=1,
        )
        return score.astype(np.int)

def trimesh_to_meshcat_geometry(mesh: trimesh.Trimesh) -> g.TriangularMeshGeometry:
    """
    Convert a trimesh mesh to meshcat geometry format.
    
    Args:
        mesh (trimesh.Trimesh): Input mesh in trimesh format
    
    Returns:
        g.TriangularMeshGeometry: Mesh in meshcat geometry format
    """
    return meshcat.geometry.TriangularMeshGeometry(mesh.vertices, mesh.faces)


def visualize_mesh(
    vis: meshcat.Visualizer,
    name: str,
    mesh: trimesh.Trimesh,
    color: Optional[List[int]] = None,
    transform: Optional[np.ndarray] = None
) -> None:
    """
    Visualize a mesh in meshcat with optional color and transform.
    
    Args:
        vis (meshcat.Visualizer): Meshcat visualizer instance
        name (str): Name/path for the mesh in the visualizer scene
        mesh (trimesh.Trimesh): Mesh to visualize
        color (Optional[List[int]]): RGB color values [0-255]. Random if None
        transform (Optional[np.ndarray]): 4x4 homogeneous transform matrix
    """
    if vis is None:
        return

    if color is None:
        color = np.random.randint(low=0, high=256, size=3)

    mesh_vis = trimesh_to_meshcat_geometry(mesh)
    color_hex = rgb2hex(tuple(color))
    material = meshcat.geometry.MeshPhongMaterial(color=color_hex)
    vis[name].set_object(mesh_vis, material)

    if transform is not None:
        vis[name].set_transform(transform)


def rgb2hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB color values to hexadecimal string.
    
    Args:
        rgb (Tuple[int, int, int]): RGB color values (0-255)
    
    Returns:
        str: Hexadecimal color string (format: "0xRRGGBB")
    """
    return "0x%02x%02x%02x" % (rgb)


def create_visualizer(clear: bool = True) -> meshcat.Visualizer:
    """
    Create a meshcat visualizer instance.
    
    Args:
        clear (bool): If True, clear the visualizer scene upon creation first
    
    Returns:
        meshcat.Visualizer: Initialized meshcat visualizer
    """
    print(
        "Waiting for meshcat server... have you started a server? Run `meshcat-server` to start a server"
    )
    vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
    if clear:
        vis.delete()
    return vis


def visualize_pointcloud(
    vis: meshcat.Visualizer,
    name: str,
    pc: np.ndarray,
    color: Optional[Union[List[int], np.ndarray]] = None,
    transform: Optional[np.ndarray] = None,
    **kwargs: Any
) -> None:
    """
    Args:
        vis: meshcat visualizer object
        name: str
        pc: Nx3 or HxWx3
        color: (optional) same shape as pc[0 - 255] scale or just rgb tuple
        transform: (optional) 4x4 homogeneous transform
    """
    if vis is None:
        return
    if pc.ndim == 3:
        pc = pc.reshape(-1, pc.shape[-1])

    if color is not None:
        if isinstance(color, list):
            color = np.array(color)
        color = np.array(color)
        # Resize the color np array if needed.
        if color.ndim == 3:
            color = color.reshape(-1, color.shape[-1])
        if color.ndim == 1:
            color = np.ones_like(pc) * np.array(color)

        # Divide it by 255 to make sure the range is between 0 and 1,
        color = color.astype(np.float32) / 255
    else:
        color = np.ones_like(pc)

    vis[name].set_object(
        meshcat.geometry.PointCloud(position=pc.T, color=color.T, **kwargs)
    )

    if transform is not None:
        vis[name].set_transform(transform)


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


def visualize_grasp(
    vis: meshcat.Visualizer,
    name: str,
    transform: np.ndarray,
    color: List[int] = [255, 0, 0],
    gripper_name: str = "franka_panda",
    **kwargs: Any
) -> None:
    """
    Visualize a gripper grasp pose in meshcat.
    
    Args:
        vis (meshcat.Visualizer): Meshcat visualizer instance
        name (str): Name/path for the grasp in the visualizer scene
        transform (np.ndarray): 4x4 homogeneous transform matrix for the grasp pose
        color (List[int]): RGB color values [0-255] for the grasp visualization
        gripper_name (str): Name of the gripper to visualize
        **kwargs: Additional arguments passed to MeshBasicMaterial
    """
    if vis is None:
        return
    grasp_vertices = load_visualization_gripper_points(gripper_name)
    for i, grasp_vertex in enumerate(grasp_vertices):
        vis[name + f"/{i}"].set_object(
            g.Line(
                g.PointsGeometry(grasp_vertex),
                g.MeshBasicMaterial(color=rgb2hex(tuple(color)), **kwargs),
            )
        )
        vis[name].set_transform(transform.astype(np.float64))

def interactive_grasp_viewer(
    vis: meshcat.Visualizer,
    grasps: list,
    transform: np.ndarray = None,
    color: List[int] = [0, 255, 0],
    gripper_name: str = "franka_panda",
    **kwargs: Any
) -> None:
    """
    Interactive grasp visualization - press Enter to show next grasp.
    
    Args:
        vis (meshcat.Visualizer): Meshcat visualizer instance
        grasps (list[np.ndarray]): List of 4x4 grasp transform matrices
        transform (np.ndarray): Base transform for all grasps (optional)
        color (List[int]): RGB color values [0-255] for the grasp visualization
        gripper_name (str): Name of the gripper to visualize
        **kwargs: Additional arguments passed to MeshBasicMaterial
    """
    if vis is None or grasps is None or len(grasps) == 0:
        print("No grasps to display!")
        return
    
    if transform is None:
        transform = np.eye(4)
    
    current_grasp_name = "current_grasp"
    total_grasps = len(grasps)
    
    print(f"\n=== Interactive Grasp Viewer ===")
    print(f"Total grasps: {total_grasps}")
    print(f"Press Enter to show next grasp, 'q' + Enter to quit\n")
    
    for i, grasp in enumerate(grasps):
        # Clear previous grasp
        vis[current_grasp_name].delete()
        
        # Show current grasp
        final_transform = transform @ grasp.astype(float)
        visualize_single_grasp(vis, current_grasp_name, final_transform, color, gripper_name, **kwargs)
        
        # Display info
        print(f"Showing grasp {i + 1}/{total_grasps}")
        
        # Wait for user input
        if i < total_grasps - 1:  # Not the last grasp
            user_input = input("Press Enter for next grasp (or 'q' to quit): ").strip().lower()
            if user_input == 'q':
                print("Exiting grasp viewer...")
                break
        else:
            print("All grasps displayed!")
            input("Press Enter to finish: ")
    
    # Clean up - remove the last grasp
    vis[current_grasp_name].delete()
    print("Grasp viewer finished.")


def visualize_single_grasp(
    vis: meshcat.Visualizer,
    name: str,
    transform: np.ndarray,
    color: List[int] = [255, 0, 0],
    gripper_name: str = "franka_panda",
    **kwargs: Any
) -> None:
    """
    Visualize a single gripper grasp pose in meshcat.
    
    Args:
        vis (meshcat.Visualizer): Meshcat visualizer instance
        name (str): Name/path for the grasp in the visualizer scene
        transform (np.ndarray): 4x4 homogeneous transform matrix for the grasp pose
        color (List[int]): RGB color values [0-255] for the grasp visualization
        gripper_name (str): Name of the gripper to visualize
        **kwargs: Additional arguments passed to MeshBasicMaterial
    """
    if vis is None:
        return
    
    grasp_vertices = load_visualization_gripper_points(gripper_name)
    for i, grasp_vertex in enumerate(grasp_vertices):
        vis[name + f"/{i}"].set_object(
            g.Line(
                g.PointsGeometry(grasp_vertex),
                g.MeshBasicMaterial(color=rgb2hex(tuple(color)), **kwargs),
            )
        )
    vis[name].set_transform(transform.astype(np.float64))