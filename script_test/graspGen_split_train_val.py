import os
from os.path import join
import json


graspGen_path = './training_dataset/grasp_dataset/graspGen_dataset/'
output_path = './dataset_grasp/grasp_dataset/graspGen/'

train_path = join(output_path, 'train')
val_path = join(output_path, 'val')

split_scale = 0.8

for uuid in os.listdir(graspGen_path):
    if not os.path.exists(join(train_path, uuid)):
        os.makedirs(join(train_path, uuid))
    if not os.path.exists(join(val_path, uuid)):
        os.makedirs(join(val_path, uuid))
    else:
        continue

    graspGen_json = json.load(open(join(graspGen_path, uuid, 'grasp.json')))
    uuid = graspGen_json['uuid']
    scale = graspGen_json['scale']
    grasps = graspGen_json['grasp']
    
    keys = grasps.keys()
    
    keys = list(keys)
    split_idx = int(len(keys) * split_scale)
    
    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]

    train_grasps = {k: grasps[k] for k in train_keys}
    val_grasps = {k: grasps[k] for k in val_keys}
    
    with open(join(train_path, uuid, 'grasp.json'), 'w') as f:
        json.dump({'uuid': uuid, 'scale': scale, 'grasp': train_grasps}, f, indent=4)
    with open(join(val_path, uuid, 'grasp.json'), 'w') as f:
        json.dump({'uuid': uuid, 'scale': scale, 'grasp': val_grasps}, f, indent=4)



