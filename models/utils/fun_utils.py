import sys
import argparse
import os
import os.path as osp
import warnings
from collections import abc
from importlib import import_module
from yapf.yapflib.yapf_api import FormatCode
from datetime import datetime
import importlib.util
import numpy as np
from os.path import join

from models.utils.base import get_random_seed, check_file_exist
import models.utils.comm as comm
from models.utils.class_utils import ConfigDict, Config, DictAction




def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported
def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    return parser
def default_config_parser(file_path, options):
    # config name protocol: dataset_name/model_name-exp_name
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1 :]))

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg

def load_py_config(cfg_path):
    spec = importlib.util.spec_from_file_location("config", cfg_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


def load_weight(weight_path, head=None):
    import torch
    from collections import OrderedDict
    checkpoint = torch.load(weight_path, weights_only=False)
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        # print(f'key: {key}')
        if key.startswith("module."):
            if comm.get_world_size() == 1:
                key = key[7:]  # module.xxx.xxx -> xxx.xxx
        else:
            if comm.get_world_size() > 1:
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
        weight[key] = value
        
    if head is not None:
        return_weight = {
            k.replace(f'{head}.', '', 1): v
            for k, v in weight.items()
            if k.startswith(f'{head}.')
        }
    else:
        return_weight = weight

    return return_weight


def load_data(data_path, type='scannet'):
    if type == 'scannet':
        color_name = 'color.npy'
        coord_name = 'coord.npy'
        normal_name = 'normal.npy'
        color_points = np.load(join(data_path, color_name))
        coord_points = np.load(join(data_path, coord_name))
        normal_points = np.load(join(data_path, normal_name))
        return {'color': color_points, 'coord': coord_points, 'normal': normal_points}