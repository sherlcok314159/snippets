import os
import random

import torch
import numpy as np
import torch.distributed as dist


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def is_main_process(rank):
    return rank == 0

def cleanup():
    dist.destroy_process_group()

def dist_setup_launch(args):
    set_seed(args.seed)
    # tell DDP available devices [NECESSARY]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group(args.backend, args.init_method,
                            rank=args.rank, world_size=args.world_size)
    # this is optional, otherwise you may need to specify the device
    # when you move something e.g., model.cuda(1) or model.to(args.rank)
    # Setting device makes things easy: model.cuda()
    torch.cuda.set_device(args.rank)
    print('The Current Rank is %d | The Total Ranks are %d' 
          %(args.rank, args.world_size))

def dist_setup_mp(args):
    set_seed(args.seed)
    # tell DDP available devices [NECESSARY]
    os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

    dist.init_process_group(args.backend, args.init_method,
                            rank=args.rank, world_size=args.world_size)
    # this is optional, otherwise you may need to specify the device
    # when you move something e.g., model.cuda(1) or model.to(args.rank)
    # Setting device makes things easy: model.cuda()
    torch.cuda.set_device(args.rank)
    print('The Current Rank is %d | The Total Ranks are %d' 
          %(args.rank, args.world_size))

@torch.no_grad()
def reduce_value(value, world_size, average=True):
    if world_size < 2:  # single gpu
        return value
    dist.all_reduce(value)
    if average:
        value /= world_size
    return value

def sync_across_gpus(t, world_size):
    gather_t_tensor = [torch.zeros_like(t) for _ in range(world_size)]
    dist.all_gather(gather_t_tensor, t)
    return torch.cat(gather_t_tensor, dim=0)

def save_checkpoint(rank, model, path):
    if is_main_process(rank):
    	# All processes should see same parameters as they all 
        # start from same random parameters and gradients are 
        # synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(model.module.state_dict(), path)
    
    # Use a barrier() to keep process 1 waiting for process 0
    dist.barrier()

def load_checkpoint(rank, model, path):
    # remap the model from cuda:0 to other devices
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    model.module.load_state_dict(
        torch.load(path, map_location=map_location)
    )