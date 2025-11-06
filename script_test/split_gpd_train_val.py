import os
from os.path import join
import json

# gpd_path = './dataset_grasp/grasp_dataset/all_labeled_dataset/gpd/'

# for scale in os.listdir(gpd_path):
#     scale_path = join(gpd_path, scale)
#     scale_number = float(scale.split('_')[-1]) / 10.0
#     for obj in os.listdir(scale_path):
#         obj_path = join(scale_path, obj)
#         for item in os.listdir(obj_path):
#             item_path = join(obj_path, item)
#             if not os.path.exists(join(item_path, 'contact_points_final.json')):
#                 continue
#             contact_points_final = json.load(open(join(item_path, 'contact_points_final.json')))
#             cur_dict = {}
#             for key in contact_points_final.keys():
#                 cur_grasp = contact_points_final[key]
#                 if cur_grasp['contact_l_semantic_text'] == 'unknown' or cur_grasp['contact_r_semantic_text'] == 'unknown':
#                     continue
#                 if cur_grasp['contact_l_semantic_text'] != cur_grasp['contact_r_semantic_text']:
#                     continue
#                 cur_dict[key] = cur_grasp
            
#             with open(join(item_path, 'contact_points_final_process.json'), 'w') as f:
#                 json.dump(cur_dict, f, indent=4)
#             f.close()
            
            

# gpd_path = './dataset_grasp/grasp_dataset/all_labeled_dataset/gpd/'
# output_path = './dataset_grasp/grasp_dataset/gpd/'

# for scale in os.listdir(gpd_path):
#     scale_path = join(gpd_path, scale)
#     scale_number = float(scale.split('_')[-1]) / 10.0
    
#     for obj in os.listdir(scale_path):
#         obj_path = join(scale_path, obj)
        
#         for item in os.listdir(obj_path):
#             item_path = join(obj_path, item)
            
#             cur_output_path = join(output_path, item)
#             if not os.path.exists(cur_output_path):
#                 os.makedirs(cur_output_path)
            
#             if not os.path.exists(join(item_path, 'contact_points_final_process.json')):
#                 continue
#             with open(join(item_path, 'contact_points_final_process.json')) as f:
#                 contact_points = json.load(f)
#             f.close()
            
#             if os.path.exists(join(cur_output_path, 'grasp.json')):
#                 output_grasps = json.load(open(join(cur_output_path, 'grasp.json')))
#             else:
#                 output_grasps = []
            
#             cur_grasp_dict = {}
#             cur_grasp_dict['uuid'] = item
#             cur_grasp_dict['scale'] = scale_number
#             cur_grasp_dict['grasp'] = contact_points
#             output_grasps.append(cur_grasp_dict)
#             with open(join(cur_output_path, 'grasp.json'), 'w') as f:
#                 json.dump(output_grasps, f, indent=4)
#             f.close()    



# calculate how many grasp there are for training in gpd
# path = './dataset_grasp/grasp_dataset/gpd/'

# total_grasp_num = 0
# for item in os.listdir(path):
#     item_path = join(path, item)
#     grasp_path = join(item_path, 'grasp.json')
#     if not os.path.exists(grasp_path):
#         continue
#     with open(grasp_path) as f:
#         grasps = json.load(f)
#     f.close()
#     for grasp in grasps:
#         total_grasp_num += len(grasp['grasp'].keys())
# print('total grasp num in gpd:', total_grasp_num)
            
            
# split train and val
import numpy as np
z_axis = np.array([0, 0, 1])

split_scale = 0.8
gpd_path = './dataset_grasp/grasp_dataset/gpd_all/'
output_gpd_path = './dataset_grasp/grasp_dataset/gpd/'
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
        keys = list(cur_grasps.keys())
        split_index = int(len(keys) * split_scale)
        train_keys = keys[:split_index]
        val_keys = keys[split_index:]
        train_cur_grasp = {}
        val_cur_grasp = {}
        
        for key in train_keys:
            train_cur_grasp[key] = cur_grasps[key]
            rot = np.array(train_cur_grasp[key]['rotation'])
            v_transformed = rot @ z_axis
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
            rot = np.array(val_cur_grasp[key]['rotation'])
            v_transformed = rot @ z_axis
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
    

