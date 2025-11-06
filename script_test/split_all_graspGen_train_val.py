import os
from os.path import join
import torch

model_list = ['graspGen']
type_list = ['train', 'val']

path = './dataset_grasp/cross_attn_dataset/'
output_path = '/mnt/bigai_ml/rongpeng/'
total_id = 0
for model in model_list:
    model_path = join(path, model)
    for t in type_list:
        type_path = join(model_path, t)
        save_path = join(path, model+'_'+t)
        for uuid in sorted(os.listdir(type_path)):
            
            if not os.path.exists(join(type_path, uuid, 'grasp_config.pt')):
                continue
                
            grasp_config = torch.load(join(type_path, uuid, 'grasp_config.pt'))
            contact_config = torch.load(join(type_path, uuid, 'contact_config.pt'))
            point_feat = torch.load(join(type_path, uuid, 'point_feat_all.pt'))

            for i in range(grasp_config.shape[0]):
                if os.path.exists(join(save_path, f'grasp_config_{total_id:06d}.pt')):
                    total_id += 1
                    continue
                cur_grasp_config = grasp_config[i].squeeze(0)
                cur_contact_config = contact_config[i].squeeze(0)
                cur_point_feat = point_feat[i].squeeze(0)

                torch.save(cur_grasp_config, join(save_path, f'grasp_config_{total_id:06d}.pt'))
                torch.save(cur_contact_config, join(save_path, f'contact_config_{total_id:06d}.pt'))
                torch.save(cur_point_feat, join(save_path, f'point_feat_{total_id:06d}.pt'))
                total_id += 1