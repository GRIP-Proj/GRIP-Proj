import torch
import torch.nn as nn
import numpy as np
import math
from torch.jit import Final
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn
from typing import Optional
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from os.path import join
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    # mlp_ratio is mlp hidden dim / input feature dim
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

#################################################################################
#                          Cross Attention Layers                               #
#################################################################################
class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    """
    fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0,
            proj_drop: float = 0,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # Prepare attn mask (B, L) to mask the conditioion
        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                dropout_p=self.attn_drop.p if self.training else 0.,
                attn_mask=mask
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if mask is not None:
                attn = attn.masked_fill_(mask.logical_not(), float('-inf'))
            attn = attn.softmax(dim=-1)
            if self.attn_drop.p > 0:
                attn = self.attn_drop(attn)
            x = attn @ v
            
        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x

class Gripper_AutoFit_Dataset(Dataset):
    def __init__(self, parent_dir, model, gripper_feat_path, dataset_type='train', dtype=torch.float32, device='cpu'):
        super().__init__()
        
        self.data_folder_path = join(parent_dir, model, dataset_type)
        self.gripper_feat = torch.load(gripper_feat_path).to(device=device, dtype=dtype) # N_gripper * feat_dim
        self.dtype = dtype
        self.device = device
        
        contact_all = []
        grasp_all = []
        point_feat_all = []
        
        for uuid in os.listdir(join(self.data_folder_path)):
            contact_path = join(self.data_folder_path, uuid, 'contact_config.pt')
            grasp_path = join(self.data_folder_path, uuid, 'grasp_config.pt')
            point_feat_path = join(self.data_folder_path, uuid, 'point_feat_all.pt')
            
            contact_config = torch.load(contact_path).to(device=device, dtype=dtype) # N_sample * 9
            grasp_config = torch.load(grasp_path).to(device=device, dtype=dtype) # N_sample * 12
            point_feat = torch.load(point_feat_path).to(device=device, dtype=dtype) # N_object * feat_dim
            
            contact_all.append(contact_config)
            grasp_all.append(grasp_config)
            point_feat_all.append(point_feat)
            
        self.contact_all = torch.cat(contact_all, dim=0)
        self.grasp_all = torch.cat(grasp_all, dim=0)
        self.point_feat_all = torch.cat(point_feat_all, dim=0)
    
    def __len__(self):
        return self.contact_all.shape[0]
    
    def __getitem__(self, idx):
        return self.contact_all[idx].detach(), self.grasp_all[idx].detach(), self.point_feat_all[idx].detach(), self.gripper_feat.detach()

    def get_all_data(self,):
        return self.contact_all.detach(), self.grasp_all.detach(), self.point_feat_all.detach(), \
            self.gripper_feat.unsqueeze(0).expand(self.contact_all.shape[0], -1, -1).detach()

        
        

class Gripper_AutoFit_Attn(nn.Module):
    """
    Gripper Auto-Fit Attention Module
    """
    # input feat: rot col 0 and 1, contact point; output feat: rot matrix and translation
    def __init__(self, hidden_size=64, num_heads=8,
                 in_feat=9, out_feat=12, num_layers=4):
        super().__init__()

        self.gripper_object_cross_attn = CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                                                       qk_norm=True, attn_drop=0.0, proj_drop=0.0)
        self.gripper_object_norms = nn.LayerNorm(hidden_size)

        self.contact_cross_attn = nn.ModuleList([
            CrossAttention(hidden_size, num_heads=num_heads, qkv_bias=True,
                           qk_norm=True, attn_drop=0.0, proj_drop=0.0)
            for _ in range(num_layers)
        ])
        self.contact_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_feat, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, out_feat, bias=True),
        )
    def init_weight(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
    def forward(self, gripper_feat, object_feat, contact_config):
        B, N, _ = gripper_feat.shape

        fused_gripper_object_feat = self.gripper_object_cross_attn(gripper_feat, object_feat)
        fused_gripper_object_feat = self.gripper_object_norms(fused_gripper_object_feat)

        # concate_feat = torch.cat([gripper_feat, object_feat], dim=-1) # B*N*hidden_size -> B*(2N)*hidden_size
        contact_config_up_sampled = self.input_proj(contact_config) # B*9 -> B*hidden_size
        contact_config_up_sampled = contact_config_up_sampled.unsqueeze(1)

        for attn, norm in zip(self.contact_cross_attn, self.contact_norms):
            contact_config_up_sampled = contact_config_up_sampled + attn(
                contact_config_up_sampled, fused_gripper_object_feat)
            contact_config_up_sampled = norm(contact_config_up_sampled)
        
        out = self.output_proj(contact_config_up_sampled)  # B*N*hidden_size -> B*N*12
        out = out.squeeze(1)
        return out