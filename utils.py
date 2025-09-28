def evaluate_policy(env, agent, turns = 1):
    total_scores = 0
    comp_full = None
    for j in range(turns):
        s, info = env.reset()
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True, mask=comp_full)
            s_next, r, dw, tr, info, comp_full = env.step(a)
            done = (dw or tr)

            total_scores += r
            s = s_next
    return int(total_scores/turns)

def getdirsize(dir):
   size = 0
   for root, dirs, files in os.walk(dir):
      size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
   return size

import torch

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print
    
    if not is_master:
        logging.disable(logging.CRITICAL)
        
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def init_distributed_mode(args):
    if hasattr(args, 'rank'): # already initialized
        return
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        logging.info('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    logging.info('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url))
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)

#You can just ignore this funciton. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        print('Wrong Input.')
        raise
    
import logging
# def setup_logging(log_file='log.txt'):
#     """Setup logging configuration
#     """
#     logging.basicConfig(level=logging.INFO,
#                         format="%(asctime)s - %(levelname)s - %(message)s",
#                         datefmt="%Y-%m-%d %H:%M:%S",
#                         filename=log_file,
#                         filemode='w')
#     console = logging.StreamHandler()
#     console.setLevel(logging.INFO)
#     formatter = logging.Formatter('%(message)s')
#     console.setFormatter(formatter)
#     logging.getLogger('').addHandler(console)
import os
def setup_logging(log_file_debug='debug_log.txt', log_file_info='info_log.txt'):
    """Setup logging configuration"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    folder_path = os.path.dirname(log_file_info)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler_debug = logging.FileHandler(log_file_debug)
    file_handler_debug.setLevel(logging.DEBUG)
    formatter_debug = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler_debug.setFormatter(formatter_debug)
    logger.addHandler(file_handler_debug)

    file_handler_info = logging.FileHandler(log_file_info)
    file_handler_info.setLevel(logging.INFO)
    formatter_info = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler_info.setFormatter(formatter_info)
    logger.addHandler(file_handler_info)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter_console = logging.Formatter('%(message)s')
    console.setFormatter(formatter_console)
    logger.addHandler(console)