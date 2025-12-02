import os
from os.path import join
import json

from diffusion import create_diffusion
from cross_grasp.models.semantic_grasp_backbone import Semantic_Grasp, Dataset_SemanticGrasp
from cross_grasp.models.RDT_model import RDTBlock
from cross_grasp.models.DiT_model import DiT

import numpy as np
import torch
from diffusers.models import AutoencoderKL
from torch.utils.data import Dataset, DataLoader
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from configs.train.DiT_config import DiTConfig
from models.dataloader import GRIP_backbone_dataset, GRIP_head_dataset
from utils.logger import Logger

def train(config):
    logger = Logger(output_path=config.log_file, multi_worker=config.num_worker > 1)

    if not config.debug:
        wandb.init(project=config.wandb_project, name=config.wandb_name, id=config.wandb_id, \
                mode=config.wandb_mode, resume=config.wandb_resume)
    diffusion = create_diffusion(timestep_respacing="") 
    val_diffusion = create_diffusion(str(config.val_diffusion_step))
    torch.manual_seed(config.seed)

    smallest_val_loss = 1e10

    sampler = None
    if config.num_worker > 1:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if config.model == 'RDT':
            model = RDTBlock(hidden_size=config.hidden_size, num_heads=config.num_head)
        elif config.model == 'DiT':
            model = DiT(text_size=512, in_channels=9, hidden_size=64, depth=28, num_heads=config.num_head)
        
        config.device = torch.device('cuda', local_rank)
        model.to(dtype=config.dtype, device=config.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

        if config.start > 1:
            checkpoint_path = f"{config.checkpoint_dir}/epoch{(config.start - 1):07d}.pt"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f'Checkpoint {checkpoint_path} not found.')
            
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["opt"])
            logger.info(f'Load checkpoint from {checkpoint_path}')

        logger.info(f'Rank {dist.get_rank()} training on {config.device}')
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        train_dataset = Ptv3_dataset(dataset_type=config.dataset, ptv3_model=config.ptv3_model, \
                                     train_model='train', dtype=config.dtype, device='cpu')
        val_dataset = Ptv3_dataset(dataset_type=config.dataset, ptv3_model=config.ptv3_model, \
                                   train_model='val', dtype=config.dtype, device='cpu')
        
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                  num_replicas=dist.get_world_size(),
                                                                  rank=local_rank,
                                                                  shuffle=config.shuffle,
                                                                  seed=config.seed)
    else:
        if config.model == 'RDT':
            model = RDTBlock(hidden_size=config.hidden_size, num_heads=config.num_head)
        elif config.model == 'DiT':
            model = DiT(text_size=512, in_channels=9, hidden_size=64, depth=28, num_heads=config.num_head)
        
        model.to(dtype=config.dtype, device=config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        
        if config.start > 1:
            checkpoint_path = f"{config.checkpoint_dir}/epoch{(config.start - 1):07d}.pt"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f'Checkpoint {checkpoint_path} not found.')
            
            checkpoint = torch.load(checkpoint_path, map_location=config.device)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["opt"])
            logger.info(f'Load checkpoint from {checkpoint_path}')

        train_dataset = Ptv3_dataset(dataset_type=config.dataset, ptv3_model=config.ptv3_model, \
                                     train_model='train', dtype=config.dtype, device='cpu')
        val_dataset = Ptv3_dataset(dataset_type=config.dataset, ptv3_model=config.ptv3_model, \
                                   train_model='val', dtype=config.dtype, device='cpu')
    
    val_x, val_y, val_pd_feat = val_dataset.get_all_data()
    val_x = val_x.to(config.device)
    val_y = val_y.to(config.device)
    val_pd_feat = val_pd_feat.to(config.device)
    val_model_kwargs = dict(c=val_y, pd=val_pd_feat) if config.model == 'RDT' else dict(y=val_y, pd=val_pd_feat)

    if sampler is None:
        dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    else:
        dataloader = DataLoader(train_dataset, sampler=sampler, \
                                batch_size=config.batch_size // dist.get_world_size(), \
                                num_workers=config.num_worker,
                                pin_memory=True,
                                drop_last=True)
    
    logger.info('finish model setting and optimizer setting')
    
    if not config.debug:
        if config.num_worker == 1:
            if not os.path.exists(config.checkpoint_dir):
                os.makedirs(config.checkpoint_dir)
        else:
            if dist.get_rank() == 0:
                if not os.path.exists(config.checkpoint_dir):
                    os.makedirs(config.checkpoint_dir)
    
    logger.info('start training')
    for epoch in range(config.start, config.max_epoch + 1):
        model.train()

        if sampler is not None:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        num_batches = len(dataloader)
        log_step = 0
        for batch_idx, (x, y, pd) in enumerate(dataloader):
            
            x, y, pd = x.to(config.device), y.to(config.device), pd.to(config.device)
            
            model_kwargs = dict(c=y, pd=pd) if config.model == 'RDT' else dict(y=y, pd=pd)
            t = torch.randint(0, config.train_diffusion_step, \
                              (x.shape[0],), device=config.device).type(torch.int)
            t = t.to(config.device)
            
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict['loss'].mean()
            optimizer.zero_grad()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            log_step += 1
            
        mean_loss = total_loss / log_step
        if config.num_worker > 1:
            mean_loss = torch.tensor(mean_loss, device=config.device)
            dist.all_reduce(mean_loss, op=dist.ReduceOp.SUM)
            mean_loss = mean_loss.item() / dist.get_world_size()
        
        logger.info(f'Epoch {epoch}, Loss: {loss.item()}')
        
        if not config.debug:
            if (config.num_worker > 1 and dist.get_rank() == 0) or config.num_worker == 1:
                wandb.log({"train/loss": mean_loss, "epoch": epoch})

        model.eval()
        with torch.no_grad():
            val_result = val_diffusion.p_sample_loop(model, [val_x.shape[0], 9], \
                clip_denoised=False, model_kwargs=val_model_kwargs, progress=False, device=config.device)
            
            val_loss = torch.norm(val_result - val_x, dim=-1).mean()
            if not config.debug:
                if (config.num_worker > 1 and dist.get_rank() == 0) or config.num_worker == 1:
                    wandb.log({"val/loss": val_loss.item(), "epoch": epoch})
                    logger.info(f'Epoch {epoch}, Val Loss: {val_loss.item()}')
                    if val_loss.item() < smallest_val_loss:
                        smallest_val_loss = val_loss.item()
                        # save best model
                        if config.num_worker > 1 and dist.get_rank() == 0:
                            checkpoint = {
                                "model": model.state_dict(),
                                "opt": optimizer.state_dict(),
                            }
                            checkpoint_path = f"{config.checkpoint_dir}/best.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f'save best checkpoint {config.checkpoint_dir}/best.pt')
                            dist.barrier() # other gpus should wait for gpu 0 write down.
                        else:
                            checkpoint = {
                                "model": model.state_dict(),
                                "opt": optimizer.state_dict(),
                            }
                            checkpoint_path = f"{config.checkpoint_dir}/best.pt"
                            torch.save(checkpoint, checkpoint_path)
                            logger.info(f'save best checkpoint {config.checkpoint_dir}/best.pt')
        

        if epoch % config.ckpt_every == 0 and epoch > 0 and not config.debug:
            if config.num_worker > 1 and dist.get_rank() == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                }
                checkpoint_path = f"{config.checkpoint_dir}/epoch{epoch:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f'save checkpoint {config.checkpoint_dir}/epoch{epoch:07d}.pt')
                dist.barrier() # other gpus should wait for gpu 0 write down.
            else:
                checkpoint = {
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                }
                checkpoint_path = f"{config.checkpoint_dir}/epoch{epoch:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f'save checkpoint {config.checkpoint_dir}/epoch{epoch:07d}.pt')

    dist.destroy_process_group() # end DDP training

if __name__ == "__main__":
    config = training_setting_DiT()
    train(config)
    

        
        
        
    