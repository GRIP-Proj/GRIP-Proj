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
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
# print('start load semantic model')
semantic_grasp = Semantic_Grasp_object(device, './ckpt/ptv3-object/config.py', './ckpt/ptv3-object/ptv3-object.pth')
print(f'finish semantic grasp loading')
''' graspGen dataset converter '''

# graspGen_folder = './dataset_grasp/grasp_dataset/graspGen/'
# mesh_folder = './dataset_grasp/mesh_dataset/objaverse_data_mesh_obj'
# train_folder = join(graspGen_folder, 'train')
# val_folder = join(graspGen_folder, 'val')
# output_folder = '../dataset_grasp/torch_tensor_ptv3object/graspGen'

# type_list = ['train', 'val']
# for cur_type in type_list:
#     cur_folder = join(graspGen_folder, cur_type)
    
#     for uuid in os.listdir(cur_folder):
#         cur_output_folder = join(output_folder, cur_type, uuid)
#         if os.path.exists(join(cur_output_folder, 'x_all.pt')):
#             print(f'skip {uuid}')
#             continue
#         os.makedirs(cur_output_folder, exist_ok=True)
        
#         uuid_path = join(cur_folder, uuid)
#         grasp_path = join(uuid_path, 'grasp.json')
#         grasps = json.load(open(grasp_path, 'r'))
        
        
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
#         x_all = []
#         y_all = []
        
#         if os.path.exists(join(uuid_path, 'x_all.pt')):
#             continue
        
#         for key in sorted(all_grasp.keys()):
#             cur_grasp = all_grasp[key]
#             transform = np.array(cur_grasp['transform'])
#             contact_point_l = np.array(cur_grasp['contact_point_l'])
#             grasp_text = cur_grasp['grasp_text']
            
#             text_embedding = semantic_grasp.encoder_text([grasp_text]).type(dtype).to('cpu')
#             text_embedding = text_embedding.squeeze(0)
            
#             x = np.concatenate([transform[:3, 0], transform[:3, 1], contact_point_l])
#             x = torch.from_numpy(x).type(dtype)
            
#             x_all.append(x)
#             y_all.append(text_embedding)
#             point_feat_all.append(point_feat)
        
#         if len(x_all) == 0:
#             continue
        
#         x_all = torch.stack(x_all, dim=0)
#         y_all = torch.stack(y_all, dim=0)
#         point_feat_all = torch.stack(point_feat_all, dim=0)

#         torch.save(x_all, join(cur_output_folder, 'x_all.pt'))
#         torch.save(y_all, join(cur_output_folder, 'y_all.pt'))
#         torch.save(point_feat_all, join(cur_output_folder, 'point_feat_all.pt'))
#         print(f'save {uuid}')
        
        
# ''' gpd dataset converter '''
id2name = json.load(open('./dataset_grasp/mesh_dataset/partnet_mesh/id2name.json', 'r'))

gpd_dataset_path = './dataset_grasp/grasp_dataset/gpd/'
train_folder = join(gpd_dataset_path, 'train')
val_folder = join(gpd_dataset_path, 'val')
output_folder = './dataset_grasp/torch_tensor_ptv3object/gpd'
for folder in [train_folder, val_folder]:
    folder_type = folder.split('/')[-1]
    for uuid in os.listdir(folder):
        uuid_path = join(folder, uuid)
        if not os.path.exists(join(uuid_path, 'grasp.json')):
            print(f'skip no grasp file {uuid_path}')
            continue
        grasps = json.load(open(join(uuid_path, 'grasp.json'), 'r'))
        
        x_all = []
        y_all = []
        point_feat_all = []
        
        for scale_grasp in grasps:
            scale = scale_grasp['scale']
            cur_all_grasps = scale_grasp['grasp']
            
            obj_name = id2name[uuid]
            
            cur_output_path = join(output_folder, folder_type, uuid)
        
            if os.path.exists(join(cur_output_path, 'x_all.pt')):
                print(f'skip {uuid_path}')
                continue
            os.makedirs(cur_output_path, exist_ok=True)

            
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

            text_list = []
            local_point_feat = []
            local_x = []
            for cur_grasp_key in sorted(cur_all_grasps.keys()):
                cur_grasp = cur_all_grasps[cur_grasp_key]
                contact_point_l = np.array(cur_grasp['final_contact_point_l'])
                if contact_point_l.ndim == 0:
                    continue
                rot = np.array(cur_grasp['rotation'])
                rot = rot @ (R_x(90) @ R_y(90))
                x = np.concatenate([rot[:3, 0], rot[:3, 1], contact_point_l])
                x = torch.from_numpy(x).type(dtype)
                
                semantic_items = cur_grasp['contact_l_semantic_text']['text'].split(',')
                direction = cur_grasp['direction']
                
                grasp_text = f'grasp {semantic_items[-1]} of {semantic_items[0]} from {direction} direction'
                text_list.append(grasp_text)
                # text_embedding = semantic_grasp.encoder_text([grasp_text]).type(dtype).to('cpu')
                # text_embedding = text_embedding.squeeze(0)
                
                local_x.append(x)
                local_point_feat.append(point_feat)
                # x_all.append(x)
                # y_all.append(text_embedding)
                # point_feat_all.append(point_feat)
            if len(text_list) == 0:
                continue
            local_x = torch.stack(local_x, dim=0)
            local_point_feat = torch.stack(local_point_feat, dim=0)
            print(f'process {uuid_path}')
            text_embedding = semantic_grasp.encoder_text(text_list).type(dtype).to('cpu')
            y_all.append(text_embedding)
            x_all.append(local_x)
            point_feat_all.append(local_point_feat)
                
        if len(x_all) == 0:
            continue
        
        # x_all = torch.stack(x_all, dim=0)
        # y_all = torch.stack(y_all, dim=0)
        # point_feat_all = torch.stack(point_feat_all, dim=0)
        x_all = torch.cat(x_all, dim=0)
        y_all = torch.cat(y_all, dim=0)
        point_feat_all = torch.cat(point_feat_all, dim=0)
        
        torch.save(x_all, join(cur_output_path, 'x_all.pt'))
        torch.save(y_all, join(cur_output_path, 'y_all.pt'))
        torch.save(point_feat_all, join(cur_output_path, 'point_feat_all.pt'))
        
        print(f'save {uuid_path}')
        
                
''' graspnet dataset converter '''
# id2name = json.load(open('./dataset_grasp/mesh_dataset/partnet_mesh/id2name.json', 'r'))

# graspnet_dataset_path = './dataset_grasp/selected_grasp_dataset/graspnet/'
# train_folder = join(graspnet_dataset_path, 'train')
# val_folder = join(graspnet_dataset_path, 'val')
# output_folder = './dataset_grasp/torch_tensor_ptv3object/graspnet'
# # print('start convert')
# for folder in [train_folder, val_folder]:
#     folder_type = folder.split('/')[-1]
#     for uuid in os.listdir(folder):
#         uuid_path = join(folder, uuid)
#         cur_output_path = join(output_folder, folder_type, uuid)
        
#         if os.path.exists(join(cur_output_path, 'x_all.pt')):
#             print(f'skip {uuid_path}')
#             continue
#         os.makedirs(cur_output_path, exist_ok=True)
        
#         if not os.path.exists(join(uuid_path, 'grasp.json')):
#             print(f'skip no grasp file {uuid_path}')
#             continue
        
#         grasps = json.load(open(join(uuid_path, 'grasp.json'), 'r'))
        
#         x_all = []
#         y_all = []
#         point_feat_all = []

#         for scale_grasp in grasps:
#             scale = scale_grasp['scale']
#             cur_all_grasps = scale_grasp['grasp']
            
#             obj_name = id2name[uuid]
            
#             mesh_path = join('./dataset_grasp/mesh_dataset/partnet_mesh/', obj_name, uuid, 'pd_rgb_normal_1000.npy')
#             obj_pd = np.load(mesh_path)
#             xyz = obj_pd[:, :3]
#             rgb = obj_pd[:, 3:6]
#             normal = obj_pd[:, 6:]
#             xyz = xyz * scale
#             semantic_grasp.create_ptv3_input_dict(xyz, normal, rgb)
#             output = semantic_grasp.ptv3_forward()
#             point_feat = (output.feat).type(dtype)
#             point_feat = point_feat.type(dtype).to('cpu').detach()

#             text_list = []
#             local_point_feat = []
#             local_x = []
#             for cur_grasp_key in sorted(cur_all_grasps.keys()):
#                 cur_grasp = cur_all_grasps[cur_grasp_key]
#                 contact_point_l = np.array(cur_grasp['contact_point_left'])
#                 if contact_point_l.ndim == 0:
#                     continue
                    
#                 rot = np.array(cur_grasp['rotation_matrix'])
#                 rot = rot @ (R_x(90) @ R_y(90))
#                 x = np.concatenate([rot[:3, 0], rot[:3, 1], contact_point_l])
#                 x = torch.from_numpy(x).type(dtype)
                
#                 if not isinstance(cur_grasp['contact_l_semantic_text'], dict):
#                     continue
                
#                 semantic_items = cur_grasp['contact_l_semantic_text']['text'].split(',')
#                 direction = cur_grasp['direction']
                
#                 grasp_text = f'grasp {semantic_items[-1]} of {semantic_items[0]} from {direction} direction'
#                 text_list.append(grasp_text)
#                 # text_embedding = semantic_grasp.encoder_text([grasp_text]).type(dtype).to('cpu')
#                 # text_embedding = text_embedding.squeeze(0)
                
#                 # x_all.append(x)
#                 local_x.append(x)
#                 # y_all.append(text_embedding)
#                 local_point_feat.append(point_feat)

#             if len(text_list) == 0:
#                 continue
#             local_x = torch.stack(local_x, dim=0)
#             local_point_feat = torch.stack(local_point_feat, dim=0)
#             print(f'process {uuid_path}')
#             text_embedding = semantic_grasp.encoder_text(text_list).type(dtype).to('cpu')
#             y_all.append(text_embedding)
#             x_all.append(local_x)
#             point_feat_all.append(local_point_feat)
                
#         if len(x_all) == 0:
#             continue
        
#         # x_all = torch.stack(x_all, dim=0)
#         # y_all = torch.stack(y_all, dim=0)
#         # point_feat_all = torch.stack(point_feat_all, dim=0)
#         x_all = torch.cat(x_all, dim=0)
#         y_all = torch.cat(y_all, dim=0)
#         point_feat_all = torch.cat(point_feat_all, dim=0)

#         # from IPython import embed; embed()

#         torch.save(x_all, join(cur_output_path, 'x_all.pt'))
#         torch.save(y_all, join(cur_output_path, 'y_all.pt'))
#         torch.save(point_feat_all, join(cur_output_path, 'point_feat_all.pt'))
        
#         print(f'save {uuid_path}')
        
            
        
    
                    
                    
                    
        
    


