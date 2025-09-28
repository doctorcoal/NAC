from utils import evaluate_policy, str2bool, setup_logging, getdirsize
import logging
import os, shutil
import utils
import argparse
import torch

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset for env setup', choices=['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet'])
parser.add_argument('--model', type=str, default='resnet_exit_quant', help='model for env setup', choices=['resnet_exit_quant', 'mobilenetV2_hira_quant', 'DeiT-tiny'])
parser.add_argument('--env_file_name', type=str, default='', help='name env files')
parser.add_argument('--suffix', type=str, default='', help='suffix of save file')
# parser.add_argument('--shallowf', action='store_true', help='shallow finetune mode (different F in comp_env, different save/load)')
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--lr_decay_b', type=int, default=4, help='lr_decay_level for backbone')
parser.add_argument('--lr_decay_e', type=int, default=4, help='lr_decay_level for exit layer')
parser.add_argument('--batch-size', type=int, default=0, help='!= 0 means reset the batch_size in env.t_conf')
parser.add_argument('--eval', action='store_true', help='evaluation mode')
parser.add_argument('--no-post-f', action='store_true', help='no post finetune')
parser.add_argument('--save-load-model', action='store_true', help='transfer a env file to model state dict')
parser.add_argument('--beta', type=float, default=0.95, help='beta value for evaluation')

# distributed
# parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
# parser.add_argument('--distributed', action='store_true', default=False, help='Enabling distributed training')

args = parser.parse_args()
args.dvc = torch.device(args.dvc) # from str to torch.device
logging.info(args)

def main():
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # torch.set_float32_matmul_precision('high')
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    model, dataset, env_file_name = args.model, args.dataset, args.env_file_name
    if args.eval:
        if args.no_post_f:
            env_name = os.path.join('/root/autodl-tmp/env_memory',model,dataset, 'default', env_file_name)
        else:
            env_name = os.path.join('/root/autodl-tmp/env_memory',model,dataset, 'best_finetune', env_file_name, 'checkpoint.pth')
        save_path = os.path.join('/root/autodl-tmp/env_memory',model,dataset, 'best_finetune', env_file_name)
    else:
        save_path = os.path.join('/root/autodl-tmp/env_memory',model,dataset, 'best_finetune',env_file_name+args.suffix)
        env_name = os.path.join('/root/autodl-tmp/env_memory',model,dataset, 'best', env_file_name)
    setup_logging(os.path.join(save_path, 'log_all.txt'), 
                  os.path.join(save_path, 'log_info.txt'))
    
    
    if os.path.isfile(env_name):   # trajectory memory hit
        logging.info(f"# load env file {env_name}.")
        env_file = torch.load(env_name)
        env = env_file['env'] # avoid load previous env's expired time
    else:
        logging.info(f"# model loading failed on {env_name}.")
    if args.save_load_model:
        utils.save_on_master(env.model, os.path.join(save_path, 'model.pth'))
        return
    if not args.eval:
        env.pt_finetune(lrdecay_b=args.lr_decay_b, lrdecay_e=args.lr_decay_e, batch_size=args.batch_size)
    # env.model.place_layer = [9, 10]
    logging.info(f"\nplace layer {env.model.place_layer}.")
    main_component = (env.model.blocklist.layers if 'DeiT' in args.model else env.model.blocklist)
    print(f"total params: {sum([sum(p.numel() for p in l.parameters() if p.requires_grad) for l in main_component])}")
    env.full_inference(model, dataset, args.beta)
    if not args.eval:
        utils.save_on_master({'env': env}, os.path.join(save_path, 'checkpoint.pth'))
    # env.close()
    # eval_env.close()

if __name__ == '__main__':
    main()
