import os
from os.path import join
import json


import torch
import torch.nn as nn
import inspect
import numpy as np
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


from models.ptv3 import PointTransformerV3
from models.PTv3Object import PointTransformerV3Object
from models.utils.fun_utils import default_argument_parser, default_config_parser, load_weight
from models.DiT_model import DiT
from models.RDT_model import RDT
from models.cross_attn import Gripper_AutoFit_Attn
from diffusion import create_diffusion

import torch
from torch.utils.data import Dataset, DataLoader
from configs.model.model_parameters import ModelParameters

class GRIP(nn.Module):
    def __init__(self, device, ptv3_config_file, ptv3_ckpt, clip=False, grid_size=0.001):
        super().__init__()
        self.device = device
        
        self.ptv3_input_dict = {}
        self.ptv3_input_dict['grid_size'] = np.array([grid_size, grid_size, grid_size])
        
        if clip:
            self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
            self.text_model.to(self.device)
        
        if self.text_model:
            for param in self.text_model.parameters():
                param.requires_grad = False
        
        self.ptv3_model = None
        self.import_ptv3_model(ptv3_config_file, ptv3_ckpt)
        self.ptv3_model.to(self.device) if self.ptv3_model is not None else None
        
        if self.ptv3_model:
            for param in self.ptv3_model.parameters():
                param.requires_grad = False
        
        self.grid_size = np.array([grid_size, grid_size, grid_size])
        
        self.model_params = ModelParameters()
        backbone_config = self.model_params.backbone
        head_config = self.model_params.head
        
        self.backbone = DiT(backbone_config.text_size, backbone_config.in_channels, \
                            backbone_config.hidden_size, backbone_config.depth, backbone_config.num_heads)
        
        self.head = Gripper_AutoFit_Attn(head_config.hidden_size, head_config.num_heads, \
                                         head_config.in_feat, head_config.out_feat, head_config.num_layers)
        
        if self.backbone:
            for param in self.backbone.parameters():
                param.requires_grad = True
                
        if self.head:
            for param in self.head.parameters():
                param.requires_grad = True
        
        self.diffusion = create_diffusion(timestep_respacing=str(backbone_config.diffuse_step))
        self.val_diffusion = create_diffusion(timestep_respacing=str(backbone_config.val_diffuse_step))
        self.diffusion_step = backbone_config.diffuse_step
        self.val_diffusion_step = backbone_config.val_diffuse_step
    
    ''' x is contact feature, y is text embedding, pd is point cloud feature, grip_feat is gripper feature '''
    def forward(self, x, y, pd, grip_feat):
        '''
        functionality:
            semantic grasp forward function.
        inputs:
            text: a list of text, len(text) is B
            point_cloud: a batch of point_cloud which shape is (B, N, 6) where dim 0-2 is coord, and 
                         dim 3-5 is normal
        '''
        
        pass

    '''
        y is language embedding, pd is point cloud feature, x is contact feature to be denoised.
    '''
    def backbone_forward(self, x, y, pd):
        model_kwargs = dict(y=y, pd=pd)
        t = torch.randint(0, self.diffusion_step, (x.shape[0],), device=x.device).type(torch.int)
        loss_dict = self.diffusion.training_losses(self.backbone, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()
        return loss

    def head_forward(self, contact_feat, pd_feat, grip_feat):
        pred_grasp = self.head(grip_feat, pd_feat, contact_feat)
        return pred_grasp
    
    def import_ptv3_model(self, config_file, ckpt_path):
        config_args = default_config_parser(config_file, None)
        model_args = config_args.model.backbone
        
        sig = inspect.signature(PointTransformerV3)
        
        ptv3_init_params = {}
        for name, param in sig.parameters.items():
            if name in model_args.keys():
                ptv3_init_params.update({name: model_args[name]})
            else:
                print(f'key: {name}, value: Not in model args keys')
        
        self.ptv3_model = PointTransformerV3(**ptv3_init_params)
        
        backbone_weight = load_weight(ckpt_path, head='backbone')
        self.ptv3_model.load_state_dict(backbone_weight, strict=True)
    
    def create_ptv3_input_dict(self, coord, normal, rgb=None):
        '''
        functionality:
            create pointtransformerV3's input dict from a pointcloud
        '''
        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord).to(self.device, dtype=torch.float32)
        elif isinstance(coord, torch.Tensor):
            coord.to(self.device, dtype=torch.float32)
        else:
            assert f"coord is not ndarray or torch tensor type"
            
        if isinstance(normal, np.ndarray):
            normal = torch.from_numpy(normal).to(self.device, dtype=torch.float32)
        elif isinstance(normal, torch.Tensor):
            normal.to(self.device, dtype=torch.float32)
        else:
            assert f"normal is not ndarray or torch tensor type"
        
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).to(self.device)
        elif isinstance(rgb, torch.Tensor):
            rgb.to(self.device)
        else:
            assert f"rgb is not ndarray or torch tensor type"
            
        self.ptv3_input_dict['coord'] = coord
        self.ptv3_input_dict['feat'] = torch.cat((coord, normal), dim=1)
        self.ptv3_input_dict['batch'] = torch.tensor(np.array([1] * \
            self.ptv3_input_dict['feat'].shape[0])).to(self.device)
        self.ptv3_input_dict['grid_size'] = torch.tensor(self.grid_size).to(self.device)
    
    def create_ptv3_input_dict_batch(self, pointcloud):
        '''
        functionality:
            create pointtransformerV3's input dict from a batch pointcloud
        inputs:
            pointcloud: shape(B, N, 6) dim 0-2 is coord, dim 3-5 is normal
        '''
        if isinstance(pointcloud, np.ndarray):
            pointcloud = torch.from_numpy(pointcloud).to(self.device, dtype=torch.float32)
        elif isinstance(pointcloud, torch.Tensor):
            pointcloud.to(self.device, dtype=torch.float32)
        else:
            assert f"coord is not ndarray or torch tensor type"
    
        
        B, N, channel = pointcloud.shape
        assert channel == 6
        
        self.ptv3_input_dict['coord'] = pointcloud[:, :, 3].reshape(-1, 3)
        self.ptv3_input_dict['feat'] = pointcloud[:, :, :].reshape(-1, 6)
        self.ptv3_input_dict['batch'] = torch.arange(1, B + 1).repeat_interleave(N).to(self.device)
        self.ptv3_input_dict['grid_size'] = torch.tensor(self.grid_size).to(self.device)
    
    def ptv3_forward(self):
        output = self.ptv3_model.forward(self.ptv3_input_dict)
        return output
    
    def encoder_text(self, text):
        'text is a list, len(text) is B'
        inputs = self.text_tokenizer(text, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            return outputs.pooler_output
        
        
        
        
        
        
class Semantic_Grasp_object(nn.Module):
    def __init__(self, device, ptv3_config_file, ptv3_ckpt):
        super().__init__()
        self.device = device
        
        self.ptv3_input_dict = {}
        self.ptv3_input_dict['grid_size'] = np.array([0.01, 0.01, 0.01])
        
        self.text_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.ptv3_model = None
        self.import_ptv3_model(ptv3_config_file, ptv3_ckpt)
        
        self.text_model.to(self.device)
        self.ptv3_model.to(self.device) if self.ptv3_model is not None else None
        
        self.grid_size = np.array([0.01, 0.01, 0.01])
        
        self.point_mlp = PointMLP()

    
    def forward(self, text, point_cloud):
        '''
        functionality:
            semantic grasp forward function.
        inputs:
            text: a list of text, len(text) is B
            point_cloud: a batch of point_cloud which shape is (B, N, 6) where dim 0-2 is coord, and 
                         dim 3-5 is normal
        '''
        
        text_embedding = self.encoder_text(text)
        self.create_ptv3_input_dict_batch(point_cloud)
        output = self.ptv3_forward()
        pointcloud_embedding = self.point_mlp(output.feat)

        pass
    
    def import_ptv3_model(self, config_file, ckpt_path):
        config_args = default_config_parser(config_file, None)
        model_args = config_args.model.backbone
        
        sig = inspect.signature(PointTransformerV3Object)
        
        ptv3_init_params = {}
        for name, param in sig.parameters.items():
            if name in model_args.keys():
                ptv3_init_params.update({name: model_args[name]})
            else:
                print(f'key: {name}, value: Not in model args keys')
        
        self.ptv3_model = PointTransformerV3Object(**ptv3_init_params)
        
        backbone_weight = load_weight(ckpt_path, head=None)
        self.ptv3_model.load_state_dict(backbone_weight, strict=True)
        self.ptv3_model.to(self.device)
        self.ptv3_model.eval()
    
    def create_ptv3_input_dict(self, coord, normal, rgb=None):
        '''
        functionality:
            create pointtransformerV3's input dict from a pointcloud
        '''
        if isinstance(coord, np.ndarray):
            coord = torch.from_numpy(coord).to(self.device, dtype=torch.float32)
        elif isinstance(coord, torch.Tensor):
            coord.to(self.device, dtype=torch.float32)
        else:
            assert f"coord is not ndarray or torch tensor type"
            
        if isinstance(normal, np.ndarray):
            normal = torch.from_numpy(normal).to(self.device, dtype=torch.float32)
        elif isinstance(normal, torch.Tensor):
            normal.to(self.device, dtype=torch.float32)
        else:
            assert f"normal is not ndarray or torch tensor type"
        
        if isinstance(rgb, np.ndarray):
            rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32)
        elif isinstance(rgb, torch.Tensor):
            rgb.to(self.device, dtype=torch.float32)
        else:
            assert f"rgb is not ndarray or torch tensor type"
            
        self.ptv3_input_dict['coord'] = coord
        self.ptv3_input_dict['feat'] = torch.cat((coord, normal, rgb), dim=1)
        self.ptv3_input_dict['batch'] = torch.tensor(np.array([1] * \
            self.ptv3_input_dict['feat'].shape[0])).to(self.device)
        self.ptv3_input_dict['grid_size'] = torch.tensor(self.grid_size).to(self.device)
    
    def create_ptv3_input_dict_batch(self, pointcloud):
        '''
        functionality:
            create pointtransformerV3's input dict from a batch pointcloud
        inputs:
            pointcloud: shape(B, N, 6) dim 0-2 is coord, dim 3-5 is normal, dim 6-8 is color
        '''
        if isinstance(pointcloud, np.ndarray):
            pointcloud = torch.from_numpy(pointcloud).to(self.device, dtype=torch.float32)
        elif isinstance(pointcloud, torch.Tensor):
            pointcloud.to(self.device, dtype=torch.float32)
        else:
            assert f"coord is not ndarray or torch tensor type"
    
        
        B, N, channel = pointcloud.shape
        assert channel == 9
        
        self.ptv3_input_dict['coord'] = pointcloud[:, :, 3].reshape(-1, 3)
        self.ptv3_input_dict['feat'] = pointcloud[:, :, :].reshape(-1, 9)
        self.ptv3_input_dict['batch'] = torch.arange(1, B + 1).repeat_interleave(N).to(self.device)
        self.ptv3_input_dict['grid_size'] = torch.tensor(self.grid_size).to(self.device)
    
    def ptv3_forward(self):
        output = self.ptv3_model.forward(self.ptv3_input_dict)
        return output
    
    def encoder_text(self, text):
        'text is a list, len(text) is B'
        inputs = self.text_tokenizer(text, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            return outputs.pooler_output
        