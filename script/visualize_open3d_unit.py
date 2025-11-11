import open3d as o3d
import numpy as np
import json
import os
import glob
import trimesh
import time
import sys
from os.path import join
sys.path.append(join(os.getcwd(), 'script'))
from meshcat_utils import load_visualization_gripper_points

def load_grasp_json(grasp_file_path):
    """Load grasp pose from JSON file"""
    with open(grasp_file_path, 'r') as f:
        data = json.load(f)
    return np.array(data['transform'])

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

def load_and_process_mesh(glb_path, mesh_scale):
    """Load and process mesh with proper coordinate system handling"""
    trimesh_mesh = trimesh.load(glb_path)
    if isinstance(trimesh_mesh, trimesh.Scene):
        trimesh_mesh = trimesh_mesh.dump(concatenate=True)
    
    trimesh_mesh.apply_scale(mesh_scale)
    
    # 创建转移矩阵
    T_move_mesh_to_origin = np.eye(4)
    T_move_mesh_to_origin[:3, 3] = -trimesh_mesh.centroid
    
    # 应用转移到 trimesh（保持一致性）
    trimesh_mesh.apply_transform(T_move_mesh_to_origin)
    
    # 转换到 Open3D（不再额外平移）
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    
    o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    o3d_mesh.compute_vertex_normals()
    
    return o3d_mesh, T_move_mesh_to_origin

# 3. Compute barycentric coords for sampled points
def barycentric_coords(triangles, points):
    # triangles: (N, 3, 3), points: (N, 3)
    v0 = triangles[:, 1] - triangles[:, 0]
    v1 = triangles[:, 2] - triangles[:, 0]
    v2 = points - triangles[:, 0]
    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    bary_coords = np.stack([u, v, w], axis=1)
    return np.clip(bary_coords, 0, 1)

'''
points should be sampled points
face_indices should be sampled faces
'''
def cal_uvs(mesh, points, face_indices):
    triangles = mesh.triangles[face_indices]  # (N, 3, 3)
    bary_coords = barycentric_coords(triangles, points)  # (N, 3)

    # 2. Get the UVs of vertices for the sampled faces
    uvs = mesh.visual.uv  # (num_vertices, 2)
    if hasattr(mesh.visual, 'uv_index'):
        face_uvs = uvs[mesh.visual.uv_index[face_indices]]  # correct UV mapping
    else:
        face_uvs = uvs[mesh.faces[face_indices]] 

    # from IPython import embed; embed()
    
    # 4. Interpolate UV coords
    sampled_uv = (face_uvs * bary_coords[:, :, None]).sum(axis=1)  # (N, 2)
    return sampled_uv

def load_and_process_mesh_with_texture(glb_path, mesh_scale, N=100000):
    """Load and process mesh with proper coordinate system handling"""
    mesh = trimesh.load(glb_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    # N = 100000
    
    points, face_indices = trimesh.sample.sample_surface(mesh, N)

    
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        if mesh.visual.uv is None:
            uvs = None
        else:
            uvs = cal_uvs(mesh, points, face_indices)
        
        # print(f'vertex colors kind: texture')
        
        basecolorfactor = mesh.visual.material.baseColorFactor
        
        if basecolorfactor is not None:
            basecolorfactor_np = np.array(basecolorfactor, dtype=np.float32)
        else:
            basecolorfactor_np = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        basecolortexture = mesh.visual.material.baseColorTexture
        # basecolortexture.show()
        
        if basecolortexture is None:
            texture_np = None
        else:
            texture_np = np.array(basecolortexture, dtype=np.float32)
            if len(texture_np.shape) == 2:
                texture_np = np.stack([texture_np]*4, axis=-1)
            elif len(texture_np.shape) == 3:
                if texture_np.shape[2] == 3:
                    texture_np = np.concatenate([texture_np, 255.0 * np.ones((texture_np.shape[0], texture_np.shape[1], 1))], axis=2)

        if texture_np is not None and uvs is not None:
            texture_np = basecolorfactor_np * texture_np
            
            h, w = texture_np.shape[:2]
            
            uvs[:, 1] = 1 - uvs[:, 1]
            uv_pixels = (uvs * [w - 1, h - 1]).astype(int)

            pix_y = np.clip(uv_pixels[:,1], 0, h-1)
            pix_x = np.clip(uv_pixels[:, 0], 0, w-1)
            points_colors = texture_np[pix_y, pix_x]
            
        else:
            points_colors = np.tile(basecolorfactor_np, (points.shape[0], 1))
        
    elif isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        color = np.array(mesh.visual.vertex_colors, dtype=np.float32)
        points_colors = np.tile(color[0], (points.shape[0], 1))
        
    rgbas = np.array(points_colors, dtype=np.float32)
    rgbs = rgbas[:, :3] / 255.0
    rgbs = np.clip(rgbs, 0.0, 0.8)

    
    points = points * mesh_scale
    points_center = np.mean(points, axis=0)
    points = points - points_center
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    
    T_move_mesh_to_origin = np.eye(4)
    T_move_mesh_to_origin[:3, 3] = -points_center # -mesh.centroid
    
    return pcd, T_move_mesh_to_origin

def load_and_process_mesh_with_texture_raw(glb_path, mesh_scale, N=100000):
    """Load and process mesh with proper coordinate system handling"""
    mesh = trimesh.load(glb_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    # N = 100000
    
    points, face_indices = trimesh.sample.sample_surface(mesh, N)

    
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        if mesh.visual.uv is None:
            uvs = None
        else:
            uvs = cal_uvs(mesh, points, face_indices)
        
        # print(f'vertex colors kind: texture')
        
        basecolorfactor = mesh.visual.material.baseColorFactor
        
        if basecolorfactor is not None:
            basecolorfactor_np = np.array(basecolorfactor, dtype=np.float32)
        else:
            basecolorfactor_np = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        basecolortexture = mesh.visual.material.baseColorTexture
        # basecolortexture.show()
        
        if basecolortexture is None:
            texture_np = None
        else:
            texture_np = np.array(basecolortexture, dtype=np.float32)
            if len(texture_np.shape) == 2:
                texture_np = np.stack([texture_np]*4, axis=-1)
            elif len(texture_np.shape) == 3:
                if texture_np.shape[2] == 3:
                    texture_np = np.concatenate([texture_np, 255.0 * np.ones((texture_np.shape[0], texture_np.shape[1], 1))], axis=2)

        if texture_np is not None and uvs is not None:
            texture_np = basecolorfactor_np * texture_np
            
            h, w = texture_np.shape[:2]
            
            uvs[:, 1] = 1 - uvs[:, 1]
            uv_pixels = (uvs * [w - 1, h - 1]).astype(int)

            pix_y = np.clip(uv_pixels[:,1], 0, h-1)
            pix_x = np.clip(uv_pixels[:, 0], 0, w-1)
            points_colors = texture_np[pix_y, pix_x]
            
        else:
            points_colors = np.tile(basecolorfactor_np, (points.shape[0], 1))
        
    elif isinstance(mesh.visual, trimesh.visual.color.ColorVisuals):
        color = np.array(mesh.visual.vertex_colors, dtype=np.float32)
        points_colors = np.tile(color[0], (points.shape[0], 1))
        
    rgbas = np.array(points_colors, dtype=np.float32)
    rgbs = rgbas[:, :3] / 255.0
    rgbs = np.clip(rgbs, 0.0, 0.8)

    points_center = np.mean(points, axis=0)
    points = points - points_center
    points = points * mesh_scale
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    
    T_move_mesh_to_origin = np.eye(4)
    T_move_mesh_to_origin[:3, 3] = -points_center # -mesh.centroid
    
    return pcd, T_move_mesh_to_origin


def setup_camera_views(mesh_center, distance=0.5):
    """Setup three camera views: front, right, top"""
    views = {}
    
    # Front view
    views['front'] = {
        'front': [0, 0, 1],
        'lookat': mesh_center,
        'up': [0, 1, 0]
    }
    
    # Right view  
    views['right'] = {
        'front': [1, 0, 0],
        'lookat': mesh_center,
        'up': [0, 1, 0]
    }
    
    # Top view
    views['top'] = {
        'front': [0, 1, 0],
        'lookat': mesh_center,
        'up': [0, 0, -1]
    }
    
    return views

def render_only(geometries, view_config, width=512, height=512):
    """Render scene and save image using visible window"""
    # Create visible window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=True)
    
    # Add all geometries
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Setup camera
    ctr = vis.get_view_control()
    ctr.set_front(view_config['front'])
    ctr.set_lookat(view_config['lookat'])
    ctr.set_up(view_config['up'])
    
    bounds = geometries[0].get_axis_aligned_bounding_box()
    for g in geometries[1:]:
        bounds += g.get_axis_aligned_bounding_box()

    center = bounds.get_center()
    extent = bounds.get_extent()       # [dx, dy, dz]
    max_extent = max(extent)           # 最大跨度
    print(f'max_extent: {max_extent}')

    ctr.set_zoom(6.0 * max_extent)

    # Update the visualization
    vis.poll_events()
    vis.update_renderer()
    
    # Wait a moment for the rendering to complete
    time.sleep(0.5)  
    
    # Close window
    vis.destroy_window()
    
def render_and_save_visible(geometries, output_path, view_config, width=512, height=512):
    """Render scene and save image using visible window"""
    # Create visible window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=True)
    
    # Add all geometries
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Setup camera
    ctr = vis.get_view_control()
    ctr.set_front(view_config['front'])
    ctr.set_lookat(view_config['lookat'])
    ctr.set_up(view_config['up'])
    
    bounds = geometries[0].get_axis_aligned_bounding_box()
    for g in geometries[1:]:
        bounds += g.get_axis_aligned_bounding_box()

    center = bounds.get_center()
    extent = bounds.get_extent()       # [dx, dy, dz]
    max_extent = max(extent)           # 最大跨度
    print(f'max_extent: {max_extent}')

    ctr.set_zoom(6.0 * max_extent)

    # Update the visualization
    vis.poll_events()
    vis.update_renderer()
    
    # Wait a moment for the rendering to complete
    time.sleep(0.5)
    
    # Capture and save
    vis.capture_screen_image(output_path)
    print(f"Saved image: {output_path}")
    
    # Close window
    vis.destroy_window()

def find_grasp_files(uuid, grasp_dir):
    """Find all grasp JSON files for a given UUID"""
    uuid_prefix = uuid[:6]
    pattern = os.path.join(grasp_dir, f"{uuid_prefix}_grasp*.json")
    grasp_files = glob.glob(pattern)
    return grasp_files

def create_visualization_scene(mesh, grasp_transform, gripper_name="franka_panda"):
    """
    Create complete visualization scene with mesh and gripper
    
    Args:
        mesh: Open3D mesh object
        grasp_transform: Final grasp transform matrix
    
    Returns:
        all_geometries: List of all geometry objects for visualization
        mesh_center: Center point of the mesh for camera positioning
    """
    # Create gripper geometry
    gripper_geometries = create_gripper_geometry(grasp_transform, gripper_name)

    # Combine all geometries
    all_geometries = [mesh] + gripper_geometries
    
    # Get mesh center for camera positioning
    mesh_center = mesh.get_center()
    
    return all_geometries, mesh_center

def create_visualization_scene_grasps(mesh, grasp_transform, gripper_name="franka_panda"):
    """
    Create complete visualization scene with mesh and gripper
    
    Args:
        mesh: Open3D mesh object
        grasp_transform: Final grasp transform matrix
    
    Returns:
        all_geometries: List of all geometry objects for visualization
        mesh_center: Center point of the mesh for camera positioning
    """
    all_geometries = [mesh]
    # Create gripper geometry
    for i in range(grasp_transform.shape[0]):
        
        gripper_geometries = create_gripper_geometry(grasp_transform[i], gripper_name)
        # from IPython import embed; embed()
        all_geometries += gripper_geometries
    
    # Get mesh center for camera positioning
    mesh_center = mesh.get_center()
    
    return all_geometries, mesh_center