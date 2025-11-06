import os
from os.path import join
import json


# objects = json.load(open('./dataset_grasp/grasp_dataset/all_labeled_dataset/selected_object.json'))['objects']

# graspnet_path = './dataset_grasp/grasp_dataset/all_labeled_dataset/graspnet/'
# output_path = './dataset_grasp/grasp_dataset/graspnet_all/'

# for object in objects:
#     object_path = join(graspnet_path, object)
#     scale = ['0.1', '0.2', '0.3', '0.4', '0.15', '0.25', '0.35']
#     for uuid in os.listdir(object_path):
#         grasps_list = []
#         for s in scale:
#             cur_grasp = dict()
#             scale_path = join(object_path, uuid, f'final_{s}_scale.json')
#             if not os.path.exists(scale_path):
#                 continue
#             grasps = json.load(open(scale_path))
#             grasps_dict = dict()
            
#             for item in grasps:
#                 grasp_id = item['grasp_id']
#                 del item['grasp_id']
#                 if not item['has_contact_points']:
#                     continue
#                 grasps_dict[grasp_id] = item
            
#             scale_grasp = dict()
#             scale_grasp['uuid'] = uuid
#             scale_grasp['scale'] = float(s)
#             scale_grasp['grasp'] = grasps_dict
#             grasps_list.append(scale_grasp)
#         if len(grasps_list) == 0:
#             continue
#         if not os.path.exists(join(output_path, uuid)):
#             os.makedirs(join(output_path, uuid))
        
#         json.dump(grasps_list, open(join(output_path, uuid, f'grasp.json'), 'w'), indent=4)

import numpy as np

z_axis = np.array([0, 0, 1])
split_scale = 0.8
gpd_path = './dataset_grasp/grasp_dataset/graspnet_all/'
output_gpd_path = './dataset_grasp/grasp_dataset/graspnet/'
if not os.path.exists(output_gpd_path):
    os.makedirs(output_gpd_path)
train_path = join(output_gpd_path, 'train')
val_path = join(output_gpd_path, 'val')
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(val_path):
    os.makedirs(val_path)

for item in os.listdir(gpd_path):
    item_path = join(gpd_path, item)
    grasp_path = join(item_path, 'grasp.json')
    if not os.path.exists(grasp_path):
        continue
    
    grasps = json.load(open(grasp_path, 'r'))
    train_grasps = []
    val_grasps = []
    for grasp in grasps:
        uuid = grasp['uuid']
        scale = grasp['scale']
        cur_grasps = grasp['grasp']
        if len(cur_grasps.keys()) == 0:
            continue
        keys = list(cur_grasps.keys())
        split_index = int(len(keys) * split_scale)
        train_keys = keys[:split_index]
        val_keys = keys[split_index:]
        train_cur_grasp = {}
        val_cur_grasp = {}
        for key in train_keys:
            train_cur_grasp[key] = cur_grasps[key]
            rotation_matrix = np.array(train_cur_grasp[key]['rotation_matrix'])
            v_transformed = rotation_matrix @ z_axis
            cos_theta = np.dot(z_axis, v_transformed) / (np.linalg.norm(v_transformed) * np.linalg.norm(z_axis))
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = np.arccos(cos_theta)
            if theta <= np.pi / 3:
                direction = 'upward'
            elif theta <= np.pi * 2 / 3:
                direction = 'horizontal'
            else:
                direction = 'downward'
            train_cur_grasp[key]['direction'] = direction
                
        for key in val_keys:
            val_cur_grasp[key] = cur_grasps[key]
            rotation_matrix = np.array(val_cur_grasp[key]['rotation_matrix'])
            v_transformed = rotation_matrix @ z_axis
            cos_theta = np.dot(z_axis, v_transformed) / (np.linalg.norm(v_transformed) * np.linalg.norm(z_axis))
            cos_theta = np.clip(cos_theta, -1, 1)
            theta = np.arccos(cos_theta)
            if theta <= np.pi / 3:
                direction = 'upward'
            elif theta <= np.pi * 2 / 3:
                direction = 'horizontal'
            else:
                direction = 'downward'
            val_cur_grasp[key]['direction'] = direction
            
        train_grasp_dict = {}
        train_grasp_dict['uuid'] = uuid
        train_grasp_dict['scale'] = scale
        train_grasp_dict['grasp'] = train_cur_grasp
        train_grasps.append(train_grasp_dict)
        val_grasp_dict = {}
        val_grasp_dict['uuid'] = uuid
        val_grasp_dict['scale'] = scale
        val_grasp_dict['grasp'] = val_cur_grasp
        val_grasps.append(val_grasp_dict)
    os.makedirs(join(output_gpd_path, 'train', item), exist_ok=True)
    os.makedirs(join(output_gpd_path, 'val', item), exist_ok=True)
    with open(join(output_gpd_path, 'train', item, 'grasp.json'), 'w') as f:
        json.dump(train_grasps, f, indent=4)
    f.close()
    with open(join(output_gpd_path, 'val', item, 'grasp.json'), 'w') as f:
        json.dump(val_grasps, f, indent=4)
    f.close()
    
            
        