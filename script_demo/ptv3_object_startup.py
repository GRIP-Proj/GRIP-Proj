from os.path import join
import os
import sys
import inspect
import torch
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import pointops
import torch.nn.functional as F


import cross_grasp.comm as comm
from cross_grasp.models.PTv3Object import PointTransformerV3Object
from cross_grasp.fun_utils import default_argument_parser, default_config_parser, load_weight
from cross_grasp.models.semantic_grasp_backbone import Semantic_Grasp_object


# args = default_argument_parser().parse_args()


# config_args = default_config_parser(args.config_file, args.options)
# model_args = config_args.model.backbone

# sig = inspect.signature(PointTransformerV3Object)

# ptv3_init_params = {}
# for name, param in sig.parameters.items():
#     # print(f"Name: {name}, Kind: {param.kind}, Default: {param.default}")
#     if name in model_args.keys():
#         # print(f'key: {name}, value: {model_args[name]}')
#         ptv3_init_params.update({name: model_args[name]})
#     else:
#         print(f'key: {name}, value: Not in model args keys')
        
# backbone_weight = load_weight('./ckpt/ptv3-object/ptv3-object.pth', head=None)
# # backbone_weight = torch.load('./ckpt/ptv3-object/ptv3-object.pth')
# # with open('ckpt_keys.txt', 'w') as f:
# #     f.write('\n'.join(backbone_weight['state_dict'].keys()))
# # f.close()

# model = PointTransformerV3Object(**ptv3_init_params)

# model.load_state_dict(backbone_weight, strict=True)

# # with open('model_param.txt', 'w') as f:
# #     for name, param in model.named_parameters():
# #         f.write(f"{name}\n")


# from IPython import embed; embed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

semantic_grasp = Semantic_Grasp_object(device, './ckpt/ptv3-object/config.py', './ckpt/ptv3-object/ptv3-object.pth')

num_points = 1000

# Random XYZ points (e.g., in range [0,1])
xyz = np.random.rand(num_points, 3)

# Random RGB colors (0-255 integers)
rgb = np.random.randint(0, 256, size=(num_points, 3), dtype=np.uint8)

# Random normals (unit vectors)
normals = np.random.randn(num_points, 3)         # Gaussian random
normals /= np.linalg.norm(normals, axis=1, keepdims=True)

semantic_grasp.create_ptv3_input_dict(xyz, normals, rgb)
output = semantic_grasp.ptv3_forward()
point_feat = (output.feat).type(dtype)
point_feat = point_feat.type(dtype).to('cpu').detach()

from IPython import embed; embed()




