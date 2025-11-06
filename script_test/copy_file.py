import os
from os.path import join
import shutil

source_dir = './cuda-11.8_source'

output_dir = './cuda-11.8'

bin_dir = join(output_dir, 'bin')
lib64_dir = join(output_dir, 'lib64')
include_dir = join(output_dir, 'include')

for cur_dir in os.listdir(source_dir):
    cur_dir_path = join(source_dir, cur_dir)
    if len(cur_dir) >= 3:
        if cur_dir[:3] == 'lib':
            src_lib = join(cur_dir_path, 'lib64')
            src_include = join(cur_dir_path, 'include')
            src_bin = join(cur_dir_path, 'bin')
            if os.path.exists(src_lib):
                shutil.copytree(src_lib, lib64_dir, dirs_exist_ok=True)
            if os.path.exists(src_include):
                shutil.copytree(src_include, include_dir, dirs_exist_ok=True)
            if os.path.exists(src_bin):
                shutil.copytree(src_bin, bin_dir, dirs_exist_ok=True)
    if len(cur_dir) >= 4:
        if cur_dir[:4] == 'cuda':
            src_lib = join(cur_dir_path, 'lib64')
            src_include = join(cur_dir_path, 'include')
            src_bin = join(cur_dir_path, 'bin')
            if os.path.exists(src_lib):
                shutil.copytree(src_lib, lib64_dir, dirs_exist_ok=True)
            if os.path.exists(src_include):
                shutil.copytree(src_include, include_dir, dirs_exist_ok=True)
            if os.path.exists(src_bin):
                shutil.copytree(src_bin, bin_dir, dirs_exist_ok=True)
        


