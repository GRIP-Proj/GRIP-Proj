import os
from os.path import join
import json

import numpy as np
import torch

from cross_grasp.models.semantic_grasp_backbone import Semantic_Grasp, Dataset_SemanticGrasp, Semantic_Grasp_object

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
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
# print('start load semantic model')
semantic_grasp = Semantic_Grasp(device, './ckpt/scannet200/config.py', './ckpt/scannet200/model_best.pth')

''' graspGen dataset converter '''

# graspGen_folder = './dataset_grasp/selected_grasp_dataset/graspGen/'
# mesh_folder = './dataset_grasp/mesh_dataset/objaverse_data_mesh_obj'
# output_folder = './dataset_grasp/cross_attn_dataset/graspGen'
# type_list = ['train', 'val']

# for cur_type in type_list:
#     type_folder = join(graspGen_folder, cur_type)
#     for uuid in os.listdir(type_folder):
#         uuid_path = join(type_folder, uuid)
#         if not os.path.exists(join(uuid_path, 'grasp.json')):
#             print(f'skip {uuid_path} no grasp.json')
#             continue

#         if os.path.exists(join(output_folder, cur_type, uuid, 'grasp_config.pt')):
#             print(f'skip {uuid_path}')
#             continue
    
#         grasp_path = join(uuid_path, 'grasp.json')
#         grasps = json.load(open(grasp_path, 'r'))
        
#         if isinstance(grasps, list):
#             grasps = grasps[0]
        
#         uuid = grasps['uuid']
#         scale = grasps['scale']
#         all_grasp = grasps['grasp']
        
#         mesh_path = join(mesh_folder, uuid, 'pd_rgb_normal_1000.npy')
#         T_move_mesh_to_origin = np.array(json.load(open(join(mesh_folder, uuid, 'T_move_mesh_to_origin.json'), 'r')))
#         object_pd = np.load(mesh_path)
#         object_pd[:, :3] = (object_pd[:, :3] - T_move_mesh_to_origin[:3, 3]) * scale
#         object_pd[:, :3] -= np.mean(object_pd[:, :3], axis=0)
#         object_pd = torch.from_numpy(object_pd).to(device).type(dtype)
        
#         semantic_grasp.create_ptv3_input_dict(object_pd[:, :3], object_pd[:, 6:])
#         output = semantic_grasp.ptv3_forward()
#         point_feat = (output.feat).type(dtype)
#         point_feat = point_feat.type(dtype).to('cpu').detach()
        
#         point_feat_all = []
#         contact_config = []
#         grasp_config = []
        
#         for key in sorted(all_grasp.keys()):
#             cur_grasp = all_grasp[key]
#             transform = np.array(cur_grasp['transform'])
#             contact_point_l = np.array(cur_grasp['contact_point_l'])
#             grasp_text = cur_grasp['grasp_text']
            
#             text_embedding = semantic_grasp.encoder_text([grasp_text]).type(dtype).to('cpu')
#             text_embedding = text_embedding.squeeze(0)
            
#             x = np.concatenate([transform[:3, 0], transform[:3, 1], contact_point_l])
#             x = torch.from_numpy(x).type(dtype)

#             y = np.concatenate([transform[:3, 0], transform[:3, 1], transform[:3, 2], transform[:3, 3]])
#             y = torch.from_numpy(y).type(dtype)

#             grasp_config.append(y)
#             contact_config.append(x)
#             point_feat_all.append(point_feat)
        
#         if len(grasp_config) == 0:
#             continue
        
#         grasp_config = torch.stack(grasp_config, dim=0)
#         contact_config = torch.stack(contact_config, dim=0)
#         point_feat_all = torch.stack(point_feat_all, dim=0)

#         if not os.path.exists(join(output_folder, cur_type, uuid)):
#             os.makedirs(join(output_folder, cur_type, uuid), exist_ok=True)
        
#         torch.save(contact_config, join(output_folder, cur_type, uuid, 'contact_config.pt'))
#         torch.save(grasp_config, join(output_folder, cur_type, uuid, 'grasp_config.pt'))
#         torch.save(point_feat_all, join(output_folder, cur_type, uuid, 'point_feat_all.pt'))
#         print(f'save {uuid}')
        
        
# ''' gpd dataset converter '''
# id2name = json.load(open('./dataset_grasp/mesh_dataset/partnet_mesh/id2name.json', 'r'))

# gpd_dataset_path = './dataset_grasp/selected_grasp_dataset/gpd/'
# train_folder = join(gpd_dataset_path, 'train')
# val_folder = join(gpd_dataset_path, 'val')
# output_folder = '/mnt/bigai_ml/rongpeng/cross_attn_dataset/gpd'

# for folder in [train_folder, val_folder]:
#     cur_type = folder.split('/')[-1]
#     for uuid in os.listdir(folder):
#         uuid_path = join(folder, uuid)

#         if not os.path.exists(join(uuid_path, 'grasp.json')):
#             print(f'skip {uuid_path} no grasp.json')
#             continue

#         if os.path.exists(join(output_folder, cur_type, uuid, 'grasp_config.pt')):
#             print(f'skip {uuid_path}')
#             continue
#         os.makedirs(join(output_folder, cur_type, uuid), exist_ok=True)

#         grasps = json.load(open(join(uuid_path, 'grasp.json'), 'r'))
        
#         contact_config = []
#         grasp_config = []
#         point_feat_all = []
        
#         for scale_grasp in grasps:
#             scale = scale_grasp['scale']
#             cur_all_grasps = scale_grasp['grasp']
            
#             obj_name = id2name[uuid]
            
#             mesh_path = join('./dataset_grasp/mesh_dataset/partnet_mesh/', obj_name, uuid, 'pd_rgb_normal_1000.npy')
#             obj_pd = np.load(mesh_path)
#             xyz = obj_pd[:, :3]
#             normal = obj_pd[:, 6:]
#             xyz = xyz * scale
#             semantic_grasp.create_ptv3_input_dict(xyz, normal)
#             output = semantic_grasp.ptv3_forward()
#             point_feat = (output.feat).type(dtype)
#             point_feat = point_feat.type(dtype).to('cpu').detach()

            
#             for cur_grasp_key in sorted(cur_all_grasps.keys()):
#                 cur_grasp = cur_all_grasps[cur_grasp_key]
#                 contact_point_l = np.array(cur_grasp['final_contact_point_l'])
#                 if contact_point_l.ndim == 0:
#                     continue
#                 rot = np.array(cur_grasp['rotation'])
#                 rot = rot @ (R_x(90) @ R_y(90))
#                 x = np.concatenate([rot[:3, 0], rot[:3, 1], contact_point_l])
#                 x = torch.from_numpy(x).type(dtype)
#                 translation = np.array(cur_grasp['translation'])
#                 y = np.concatenate([rot[:3, 0], rot[:3, 1], rot[:3, 2], translation])
#                 y = torch.from_numpy(y).type(dtype)
                
#                 contact_config.append(x)
#                 grasp_config.append(y)
#                 point_feat_all.append(point_feat)

#         if len(contact_config) == 0:
#             continue
        
#         grasp_config = torch.stack(grasp_config, dim=0)
#         contact_config = torch.stack(contact_config, dim=0)
#         point_feat_all = torch.stack(point_feat_all, dim=0)
        
#         torch.save(contact_config, join(output_folder, cur_type, uuid, 'contact_config.pt'))
#         torch.save(grasp_config, join(output_folder, cur_type, uuid, 'grasp_config.pt'))
#         torch.save(point_feat_all, join(output_folder, cur_type, uuid, 'point_feat_all.pt'))

#         print(f'save {uuid_path}')
        
                
# ''' graspnet dataset converter '''
id2name = json.load(open('./dataset_grasp/mesh_dataset/partnet_mesh/id2name.json', 'r'))

graspnet_dataset_path = './dataset_grasp/selected_grasp_dataset/graspnet/'
train_folder = join(graspnet_dataset_path, 'train')
val_folder = join(graspnet_dataset_path, 'val')
output_folder = '/mnt/bigai_ml/rongpeng/cross_attn_dataset/graspnet'

for folder in [train_folder, val_folder]:
    folder_type = folder.split('/')[-1]
    for uuid in os.listdir(folder):
        uuid_path = join(folder, uuid)
        cur_output_path = join(output_folder, folder_type, uuid)

        if not os.path.exists(join(uuid_path, 'grasp.json')):
            print(f'skip {uuid_path} no grasp.json')
            continue

        if os.path.exists(join(cur_output_path, 'grasp_config.pt')):
            print(f'skip {uuid_path}')
            continue
        os.makedirs(cur_output_path, exist_ok=True)
        
        grasps = json.load(open(join(uuid_path, 'grasp.json'), 'r'))
        
        contact_config = []
        grasp_config = []
        point_feat_all = []

        for scale_grasp in grasps:
            scale = scale_grasp['scale']
            cur_all_grasps = scale_grasp['grasp']
            
            obj_name = id2name[uuid]
            
            mesh_path = join('./dataset_grasp/mesh_dataset/partnet_mesh/', obj_name, uuid, 'pd_rgb_normal_1000.npy')
            obj_pd = np.load(mesh_path)
            xyz = obj_pd[:, :3]
            rgb = obj_pd[:, 3:6]
            normal = obj_pd[:, 6:]
            xyz = xyz * scale
            semantic_grasp.create_ptv3_input_dict(xyz, normal, rgb)
            output = semantic_grasp.ptv3_forward()
            point_feat = (output.feat).type(dtype)
            point_feat = point_feat.type(dtype).to('cpu').detach()

            local_grasp_config = []
            local_point_feat = []
            local_contact_config = []
            for cur_grasp_key in sorted(cur_all_grasps.keys()):
                cur_grasp = cur_all_grasps[cur_grasp_key]
                contact_point_l = np.array(cur_grasp['contact_point_left'])
                if contact_point_l.ndim == 0:
                    continue
                    
                rot = np.array(cur_grasp['rotation_matrix'])
                rot = rot @ (R_x(90) @ R_y(90))
                x = np.concatenate([rot[:3, 0], rot[:3, 1], contact_point_l])
                x = torch.from_numpy(x).type(dtype)

                translation = cur_grasp['translation']
                y = np.concatenate([rot[:3, 0], rot[:3, 1], rot[:3, 2], translation])
                y = torch.from_numpy(y).type(dtype)
                
                local_grasp_config.append(y)
                local_contact_config.append(x)
                local_point_feat.append(point_feat)

            if len(local_contact_config) == 0:
                continue
            local_contact_config = torch.stack(local_contact_config, dim=0)
            local_grasp_config = torch.stack(local_grasp_config, dim=0)
            local_point_feat = torch.stack(local_point_feat, dim=0)
            contact_config.append(local_contact_config)
            grasp_config.append(local_grasp_config)
            point_feat_all.append(local_point_feat)
                
        if len(contact_config) == 0:
            continue
        
        contact_config = torch.cat(contact_config, dim=0)
        grasp_config = torch.cat(grasp_config, dim=0)
        point_feat_all = torch.cat(point_feat_all, dim=0)

        torch.save(contact_config, join(cur_output_path, 'contact_config.pt'))
        torch.save(grasp_config, join(cur_output_path, 'grasp_config.pt'))
        torch.save(point_feat_all, join(cur_output_path, 'point_feat_all.pt'))
        
        print(f'save {uuid_path}')
        
            
        
    
                    
                    
                    
        
    


