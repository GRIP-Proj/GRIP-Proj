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
from cross_grasp.models.ptv3 import PointTransformerV3
from cross_grasp.fun_utils import default_argument_parser, default_config_parser, load_weight



args = default_argument_parser().parse_args()


config_args = default_config_parser(args.config_file, args.options)
model_args = config_args.model.backbone

sig = inspect.signature(PointTransformerV3)

ptv3_init_params = {}
for name, param in sig.parameters.items():
    # print(f"Name: {name}, Kind: {param.kind}, Default: {param.default}")
    if name in model_args.keys():
        # print(f'key: {name}, value: {model_args[name]}')
        ptv3_init_params.update({name: model_args[name]})
    else:
        print(f'key: {name}, value: Not in model args keys')
        
model = PointTransformerV3(**ptv3_init_params)



backbone_weight = load_weight('./ckpt/scannet200/model_best.pth', head='backbone')
model.load_state_dict(backbone_weight, strict=True)
model.to('cuda')
model.eval()
num_params = sum(p.numel() for p in model.parameters())
model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
model_size_MB = model_size_bytes / (1024 ** 2)
print(f"Model size: {model_size_MB:.2f} MB")

print(f"Number of parameters: {num_params:,}")
# from IPython import embed; embed()

seg_head = (
            nn.Linear(config_args.model.backbone_out_channels, config_args.model.num_classes)
            if config_args.model.num_classes > 0
            else nn.Identity()
        )

seg_head_weight = load_weight('./ckpt/scannet200/model_best.pth', head='seg_head')
seg_head.load_state_dict(seg_head_weight)

classes = config_args.CLASS_LABELS_200


data_path = './chair_0001.txt'
data = np.loadtxt(data_path, delimiter=",").astype(np.float32)
num_point = None
uniform_sampling = False
if num_point is not None:
    if uniform_sampling:
        with torch.no_grad():
            mask = pointops.farthest_point_sampling(
                torch.tensor(data).float().cuda(),
                torch.tensor([len(data)]).long().cuda(),
                torch.tensor([num_point]).long().cuda(),
            )
        data = data[mask.cpu()]
    else:
        data = data[: num_point]
coord, normal = data[:, 0:3], data[:, 3:6]
category = np.array(['chair'])

input_dict = {}
input_dict['coord'] = np.concatenate([coord, coord], axis=0)
input_dict['feat'] = np.concatenate([data[:, :6], data[:, :6]], axis=0)
input_dict['batch'] = np.concatenate([np.array([1] * coord.shape[0]), \
    np.array([2] * coord.shape[0])])
input_dict['grid_size'] = np.array([0.01, 0.01, 0.01])

input_dict = {k: torch.tensor(v).to('cuda') for k, v in input_dict.items()}
for k, v in input_dict.items():
    print(f'{k} shape is {v.shape}')

model.eval()
output = model.forward(input_dict)

from IPython import embed; embed()

seg_head.to('cuda')
seg_head.eval()
print(f'feat shape is {output.feat.shape}')



with torch.no_grad():
    bit_logits = seg_head.forward(output.feat)
    pred_part = F.softmax(bit_logits, -1)

    print(pred_part.shape)
    pred_part_sum = torch.sum(pred_part, dim=0)
    
    bit_logits_index = torch.argmax(pred_part_sum, dim=0)
    sorted_indices = torch.argsort(pred_part_sum, descending=True)
    print(sorted_indices)
    for i in range(5):
        print(classes[sorted_indices[i]])
                            



