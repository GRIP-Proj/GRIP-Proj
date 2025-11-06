# import numpy as np
# import os
# import json
# from os.path import join

# dataset_name = 'graspnet'

# graspgen_path = f'./dataset_grasp/grasp_dataset/{dataset_name}/'
# train_path = join(graspgen_path, 'train')
# val_path = join(graspgen_path, 'val')
# if not os.path.exists(train_path):
#     print('train path not exist')
# if not os.path.exists(val_path):
#     print('val path not exist')
    
# total_train_grasp_num = 0
# for item in os.listdir(train_path):
#     item_path = join(train_path, item)
#     grasp_path = join(item_path, 'grasp.json')
#     if not os.path.exists(grasp_path):
#         print(f'grasp json is not found, {item}')
#         os.rmdir(item_path)
#         continue
#     with open(grasp_path) as f:
#         grasps = json.load(f)
#     f.close()
#     for grasp in grasps:
#         total_train_grasp_num += len(grasp['grasp'].keys())
# total_val_grasp_num = 0
# for item in os.listdir(val_path):
#     item_path = join(val_path, item)
#     grasp_path = join(item_path, 'grasp.json')
#     if not os.path.exists(grasp_path):
#         print(f'grasp json is not found, {item}')
#         os.rmdir(item_path)
#         continue
#     with open(grasp_path) as f:
#         grasps = json.load(f)
#     f.close()
#     for grasp in grasps:
#         total_val_grasp_num += len(grasp['grasp'].keys())
# print('total grasp num in graspGen train:', total_train_grasp_num)
# print('total grasp num in graspGen val:', total_val_grasp_num)




import os
from os.path import join
import json
import torch
import shutil

# grasp_folder = './dataset_grasp/grasp_filter_dataset/'
# tensor_folder = './dataset_grasp/torch_tensor/'
# model_list = ['gpd', 'graspGen', 'graspnet']
# type_list = ['train', 'val']
# for model in model_list:
#     model_path = join(grasp_folder, model)
#     for cur_type in type_list:
#         type_path = join(model_path, cur_type)
#         total_grasp_num = 0
#         total_torch_tensor_num = 0
#         for uuid in os.listdir(type_path):
#             grasps = json.load(open(join(type_path, uuid, 'grasp.json'), 'r'))
#             if isinstance(grasps, dict):
#                 total_grasp_num += len(grasps['grasp'].keys())
#             else:
#                 for grasp in grasps:
#                     total_grasp_num += len(grasp['grasp'].keys())
        
#             tensor_path = join(tensor_folder, model, cur_type, uuid, 'y_all.pt')
#             if not os.path.exists(tensor_path):
#                 # print(f'missing tensor for {tensor_path}')
#                 # shutil.rmtree(join(tensor_folder, model, cur_type, uuid))
#                 continue
#             y_all = torch.load(tensor_path)
#             total_torch_tensor_num += y_all.shape[0]
            
#         print(f'{model} {cur_type}: total grasp num: {total_grasp_num}, total torch tensor num: {total_torch_tensor_num}')



# grasp_folder = './dataset_grasp/grasp_dataset/'
# tensor_folder = './dataset_grasp/torch_tensor/'
# model_list = ['gpd', 'graspGen', 'graspnet']
# type_list = ['train', 'val']
# for model in model_list:
#     model_path = join(grasp_folder, model)
#     for cur_type in type_list:
#         type_path = join(model_path, cur_type)
#         total_grasp_num = 0
#         total_torch_tensor_num = 0
#         for uuid in os.listdir(type_path):
#             cur_tensor_folder = join(tensor_folder, model, cur_type, uuid)
#             if not os.path.exists(cur_tensor_folder):
#                 continue
#             if not os.listdir(cur_tensor_folder):
#                 os.rmdir(cur_tensor_folder)
#                 print(f'cleaned {model} {cur_type}')


# grasp_folder = './dataset_grasp/grasp_filter_dataset/'
# tensor_folder = './dataset_grasp/torch_tensor/'
# model_list = ['gpd']
# type_list = ['train', 'val']
# for model in model_list:
#     model_path = join(tensor_folder, model)
#     for cur_type in type_list:
#         type_path = join(model_path, cur_type)
#         total_grasp_num = 0
#         total_torch_tensor_num = 0
#         for uuid in os.listdir(type_path):
#             if not os.path.exists(join(grasp_folder, model, cur_type, uuid, 'grasp.json')):
#                 print(f'missing grasp json for {model} {cur_type} {uuid}')
#                 # shutil.rmtree(join(tensor_folder, model, cur_type, uuid))
#                 continue
#             grasps = json.load(open(join(grasp_folder, model, cur_type, uuid, 'grasp.json'), 'r'))
#             if isinstance(grasps, dict):
#                 total_grasp_num += len(grasps['grasp'].keys())
#             else:
#                 for grasp in grasps:
#                     total_grasp_num += len(grasp['grasp'].keys())
        
#             tensor_path = join(tensor_folder, model, cur_type, uuid, 'y_all.pt')
#             y_all = torch.load(tensor_path)
#             total_torch_tensor_num += y_all.shape[0]
            
#         print(f'{model} {cur_type}: total grasp num: {total_grasp_num}, total torch tensor num: {total_torch_tensor_num}')




# grasp_folder = './dataset_grasp/selected_grasp_dataset/'
# tensor_folder = './dataset_grasp/torch_tensor/'
# model_list = ['gpd', 'graspGen', 'graspnet']
# type_list = ['train', 'val']
# for model in model_list:
#     model_path = join(grasp_folder, model)
#     for cur_type in type_list:
#         type_path = join(model_path, cur_type)
#         total_grasp_num = 0
#         total_torch_tensor_num = 0
#         for uuid in os.listdir(type_path):
#             if not os.path.exists(join(grasp_folder, model, cur_type, uuid, 'grasp.json')):
#                 # print(f'missing grasp json for {model} {cur_type} {uuid}')
#                 # shutil.rmtree(join(tensor_folder, model, cur_type, uuid))
#                 continue
#             grasps = json.load(open(join(grasp_folder, model, cur_type, uuid, 'grasp.json'), 'r'))
#             if isinstance(grasps, dict):
#                 total_grasp_num += len(grasps['grasp'].keys())
#             else:
#                 for grasp in grasps:
#                     total_grasp_num += len(grasp['grasp'].keys())
            
#         print(f'{model} {cur_type}: total grasp num: {total_grasp_num}')




grasp_folder = './dataset_grasp/selected_grasp_dataset/'
tensor_folder = '/mnt/bigai_ml/rongpeng/cross_attn_dataset/'
model_list = ['gpd', 'graspGen', 'graspnet']
type_list = ['train', 'val']
for model in model_list:
    model_path = join(tensor_folder, model)
    for cur_type in type_list:
        type_path = join(model_path, cur_type)
        total_grasp_num = 0
        total_torch_tensor_num = 0
        for uuid in os.listdir(type_path):
            if not os.path.exists(join(tensor_folder, model, cur_type, uuid, 'grasp_config.pt')):
                continue

            grasp_config = torch.load(join(tensor_folder, model, cur_type, uuid, 'grasp_config.pt'))

            total_torch_tensor_num += grasp_config.shape[0]

        print(f'{model} {cur_type}: total torch tensor num: {total_torch_tensor_num}')