import torch
import os
from os.path import join

folder = './dataset'
x_list = []
y_list = []
point_feat_list = []

for uuid in os.listdir(folder):
    uuid_folder = join(folder, uuid)
    x_all_path = join(uuid_folder, 'x_all.pt')
    y_all_path = join(uuid_folder, 'y_all.pt')
    point_feat_all_path = join(uuid_folder, 'point_feat_all.pt')
    if not os.path.exists(x_all_path):
        continue
    
    x_all = torch.load(x_all_path)
    y_all = torch.load(y_all_path)
    point_feat_all = torch.load(point_feat_all_path)
    
    idx = torch.randperm(x_all.shape[0])[:50]
    x_list.append(x_all[idx])
    y_list.append(y_all[idx])
    point_feat_list.append(point_feat_all[idx])

x = torch.cat(x_list, dim=0)
y = torch.cat(y_list, dim=0)
point_feat = torch.cat(point_feat_list, dim=0)

print(x.shape)
print(y.shape)
print(point_feat.shape)

torch.save(x, join('./eval_dataset', 'x.pt'))
torch.save(y, join('./eval_dataset', 'y.pt'))
torch.save(point_feat, join('./eval_dataset', 'point_feat.pt'))
    