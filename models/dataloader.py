import os
from os.path import join
import json

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F

class GRIP_backbone_dataset_raw(Dataset):
    def __init__(self, dataset_path=None, dtype=torch.float32, device='cpu', mmap=False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.mmap = mmap
        if mmap == False:
            x_list = []
            y_list = []
            point_feat_list = []
            for uuid in os.listdir(dataset_path):
                uuid_dir = join(dataset_path, uuid)
                x_path = join(uuid_dir, 'x_all.pt')
                if not os.path.exists(x_path):
                    continue
                y_path = join(uuid_dir, 'y_all.pt')
                point_feat_all_path = join(uuid_dir, 'point_feat_all.pt')
                x = torch.load(x_path)
                y = torch.load(y_path)
                point_feat = torch.load(point_feat_all_path)

                x_list.append(x)
                y_list.append(y)
                point_feat_list.append(point_feat)
                
            self.x_all = torch.cat(x_list, dim=0)
            self.y_all = torch.cat(y_list, dim=0)
            self.point_feat_all = torch.cat(point_feat_list, dim=0)
            
            self.x_all.to(device=device, dtype=dtype)
            self.y_all.to(device=device, dtype=dtype)
            self.point_feat_all.to(device=device, dtype=dtype)
            
            self.x_all.requires_grad = False
            self.y_all.requires_grad = False
            self.point_feat_all.requires_grad = False
        
        else:
            self.x_list = []
            self.y_list = []
            self.point_feat_list = []
            self.offset_list = []
            sum = 0
            for uuid in os.listdir(dataset_path):
                uuid_dir = join(dataset_path, uuid)
                x_path = join(uuid_dir, 'x_all.pt')
                if not os.path.exists(x_path):
                    continue
                y_path = join(uuid_dir, 'y_all.pt')
                point_feat_path = join(uuid_dir, 'point_feat_all.pt')
                x = torch.load(x_path, mmap=mmap)
                y = torch.load(y_path, mmap=mmap)
                point_feat = torch.load(point_feat_path, mmap=mmap)
                
                cur_len = x.shape[0]
                self.x_list.append(x)
                self.y_list.append(y)
                self.point_feat_list.append(point_feat)
                
                sum += cur_len
                self.offset_list.append(sum)
    def __len__(self):
        if self.mmap:
            return self.offset_list[-1]
        else:
            return self.x_all.shape[0]
    def __getitem__(self, idx):
        if not self.mmap:
            return self.x_all[idx].detach(), self.y_all[idx].detach(), self.point_feat_all[idx].detach()
        else:
            for i in range(len(self.offset_list)):
                if idx < self.offset_list[i]:
                    if i == 0:
                        local_idx = idx
                    else:
                        local_idx = idx - self.offset_list[i-1]
                    return self.x_list[i][local_idx].to(self.device, self.dtype).detach(), \
                           self.y_list[i][local_idx].to(self.device, self.dtype).detach(), \
                           self.point_feat_list[i][local_idx].to(self.device, self.dtype).detach()

class GRIP_backbone_dataset(Dataset):
    def __init__(self, dataset_path=None, dtype=torch.float32, device='cpu', mmap=False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.mmap = mmap
        if mmap == False:
            x_list = []
            y_list = []
            point_feat_list = []
            for uuid in os.listdir(dataset_path):
                uuid_dir = join(dataset_path, uuid)
                x_path = join(uuid_dir, 'x_all.pt')
                if not os.path.exists(x_path):
                    continue
                y_path = join(uuid_dir, 'y_all.pt')
                point_feat_all_path = join(uuid_dir, 'point_feat_all.pt')
                x = torch.load(x_path)
                y = torch.load(y_path)
                point_feat = torch.load(point_feat_all_path)

                x_list.append(x)
                y_list.append(y)
                point_feat_list.append(point_feat)
                
            self.x_all = torch.cat(x_list, dim=0)
            self.y_all = torch.cat(y_list, dim=0)
            self.point_feat_all = torch.cat(point_feat_list, dim=0)
            
            self.x_all.to(device=device, dtype=dtype)
            self.y_all.to(device=device, dtype=dtype)
            self.point_feat_all.to(device=device, dtype=dtype)
            
            self.x_all.requires_grad = False
            self.y_all.requires_grad = False
            self.point_feat_all.requires_grad = False
        
        else:
            self.x_list = []
            self.y_list = []
            self.point_feat_list = []
            self.offset_list = []
            sum = 0
            for uuid in os.listdir(dataset_path):
                uuid_dir = join(dataset_path, uuid)
                x_path = join(uuid_dir, 'x_all.pt')
                if not os.path.exists(x_path):
                    continue
                y_path = join(uuid_dir, 'y_all.pt')
                point_feat_path = join(uuid_dir, 'point_feat_all.pt')
                x = torch.load(x_path, mmap=mmap)
                y = torch.load(y_path, mmap=mmap)
                point_feat = torch.load(point_feat_path, mmap=mmap)
                
                cur_len = x.shape[0]
                self.x_list.append(x)
                self.y_list.append(y)
                self.point_feat_list.append(point_feat)
                
                sum += cur_len
                self.offset_list.append(sum)
    def __len__(self):
        if self.mmap:
            return self.offset_list[-1]
        else:
            return self.x_all.shape[0]
    def __getitem__(self, idx):
        if not self.mmap:
            return self.x_all[idx].detach(), self.y_all[idx].detach(), self.point_feat_all[idx].detach()
        else:
            for i in range(len(self.offset_list)):
                if idx < self.offset_list[i]:
                    if i == 0:
                        local_idx = idx
                    else:
                        local_idx = idx - self.offset_list[i-1]
                    return self.x_list[i][local_idx].to(self.device, self.dtype).detach(), \
                           self.y_list[i][local_idx].to(self.device, self.dtype).detach(), \
                           self.point_feat_list[i][local_idx].to(self.device, self.dtype).detach()
    
class GRIP_head_dataset(Dataset):
    def __init__(self, dataset_path, gripper_feat_path, dtype=torch.float32, device='cpu', mmap=False):
        super().__init__()
        
        self.dataset_path = dataset_path
        self.gripper_feat = torch.load(gripper_feat_path).to(device=device, dtype=dtype) # N_gripper * feat_dim
        print(f'finish load gripper feature from {gripper_feat_path}')
        self.dtype = dtype
        self.device = device
        self.mmap = mmap
        
        if not mmap:
            contact_all = []
            grasp_all = []
            point_feat_all = []
            
            for uuid in os.listdir(join(self.dataset_path)):
                contact_path = join(self.dataset_path, uuid, 'contact_config.pt')
                grasp_path = join(self.dataset_path, uuid, 'grasp_config.pt')
                point_feat_path = join(self.dataset_path, uuid, 'point_feat_all.pt')
                
                if not os.path.exists(contact_path):
                    continue
                
                contact_config = torch.load(contact_path).to(device=device, dtype=dtype) # N_sample * 9
                grasp_config = torch.load(grasp_path).to(device=device, dtype=dtype) # N_sample * 12
                point_feat = torch.load(point_feat_path).to(device=device, dtype=dtype) # N_object * feat_dim
                
                contact_all.append(contact_config)
                grasp_all.append(grasp_config)
                point_feat_all.append(point_feat)
                
            self.contact_all = torch.cat(contact_all, dim=0)
            self.grasp_all = torch.cat(grasp_all, dim=0)
            self.point_feat_all = torch.cat(point_feat_all, dim=0)
        
        else:
            self.contact_list = []
            self.grasp_list = []
            self.point_feat_list = []
            self.offset_list = []
            
            sum = 0
            for uuid in os.listdir(join(self.dataset_path)):
                contact_path = join(self.dataset_path, uuid, 'contact_config.pt')
                grasp_path = join(self.dataset_path, uuid, 'grasp_config.pt')
                point_feat_path = join(self.dataset_path, uuid, 'point_feat_all.pt')
                
                if not os.path.exists(contact_path):
                    continue
                
                contact_config = torch.load(contact_path, mmap=mmap)
                grasp_config = torch.load(grasp_path, mmap=mmap)
                point_feat = torch.load(point_feat_path, mmap=mmap)
                
                cur_len = contact_config.shape[0]
                
                self.contact_list.append(contact_config)
                self.grasp_list.append(grasp_config)
                self.point_feat_list.append(point_feat)
                
                sum += cur_len
                self.offset_list.append(sum)
    
    def __len__(self):
        if self.mmap:
            return self.offset_list[-1]
        else:
            return self.contact_all.shape[0]
    
    def __getitem__(self, idx):
        if not self.mmap:
            return self.contact_all[idx].detach(), self.grasp_all[idx].detach(), self.point_feat_all[idx].detach(), self.gripper_feat.detach()
        else:
            for i in range(len(self.offset_list)):
                if idx < self.offset_list[i]:
                    if i == 0:
                        local_idx = idx
                    else:
                        local_idx = idx - self.offset_list[i-1]
                    return self.contact_list[i][local_idx].to(self.device, self.dtype).detach(), \
                           self.grasp_list[i][local_idx].to(self.device, self.dtype).detach(), \
                           self.point_feat_list[i][local_idx].to(self.device, self.dtype).detach(), \
                           self.gripper_feat.detach()

    def get_all_data(self,):
        if self.mmap:
            return None, None, None, None
        return self.contact_all.detach(), self.grasp_all.detach(), self.point_feat_all.detach(), \
            self.gripper_feat.unsqueeze(0).expand(self.contact_all.shape[0], -1, -1).detach()

