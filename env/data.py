import os
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from .preprocess import get_transform
import time
import utils
import logging
# run main by: nohup python main.py 2>&1 &
MAX_Q_LEVEL = {
    'resnet_exit_quant':{
        'cifar10': 3,
        'cifar100': 3,
        'tiny-imagenet': 3,
        },
    'mobilenetV2_hira_quant':{
        'cifar10': 3,
        'cifar100': 3,
        'tiny-imagenet': 3,
        },
    'DeiT-tiny': {
        'cifar10': 3,
        'cifar100': 3,
        'tiny-imagenet': 3,
        'imagenet': 3
    },
    }

EXIT_LAYER_RATIO = {
    'resnet_exit_quant': 0.136,
    'mobilenetV2_hira_quant': 0.136,
    'DeiT-tiny': 0.012
    }

DEPTH = {
    'resnet_exit_quant': 34,
    'mobilenetV2_hira_quant': 19,
    'DeiT-tiny': 12
    }

BATCH_SIZE = {
    'resnet_exit_quant':{
        'cifar10': 256,
        'cifar100': 256,
        'tiny-imagenet': 256,
        },
    'mobilenetV2_hira_quant':{
        'cifar10': 256,
        'cifar100': 256,
        'tiny-imagenet': 128,
        },
    'DeiT-tiny': {
        'cifar10': 256,
        'cifar100': 256,
        'tiny-imagenet': 256,
        'imagenet': 256
    },
    }

ORIGINAL_ACC = {
    'resnet_exit_quant':{
        'cifar10': 92.40,
        'cifar100': 79.08,
        'tiny-imagenet': 67.20,
        },
    'mobilenetV2_hira_quant':{
        'cifar10': 95.15,
        'cifar100': 77.71,
        'tiny-imagenet': 64.76,
        },
    'DeiT-tiny': {
        'cifar10': 91.66,
        'cifar100': 69.95,
        'tiny-imagenet': 56.28,
        'imagenet': 72.2
    },
    }

BAD_ACC_D = {  # this is the curial hyperparameter for CCS, it determines the frequency of finetune. 
    'resnet_exit_quant':{ # high bad_acc reduce finetune frequency, make search faster. however, when able to finetune, model already loss much informantion
        'cifar10': 10.0,  # low bad_acc increase finetune frequency, make accuracy of the model higher. however, when able to finetune
        'cifar100': 20.0,
        'tiny-imagenet': 20.0,
        },
    'mobilenetV2_hira_quant':{
        'cifar10': 10.0,
        'cifar100': 20.0,
        'tiny-imagenet': 20.0,
        },
    'DeiT-tiny': {
        'cifar10': 3.0,
        'cifar100': 4.0,
        'tiny-imagenet': 4.0,
        'imagenet': 4.0
    },
    }

BETAS = {
    'resnet_exit_quant':{
        'cifar10': 0.95,
        'cifar100': 0.95,
        'tiny-imagenet': 0.7,
        },
    'mobilenetV2_hira_quant':{
        'cifar10': 0.97,
        'cifar100': 0.97,
        'tiny-imagenet': 0.8,
        },
    'DeiT-tiny': {
        'cifar10': 0.995,
        'cifar100': 0.35,
        'tiny-imagenet': 0.35, 
        'imagenet': 0.7
    },
    }

PTFS = {
    'resnet_exit_quant':{
        'cifar10':'result/resnet_exit/cifar10/Ewd4e-5-resnet_exit34_place_cifar10_200_best.pt',
        'cifar100':'result/resnet_exit/cifar100/Ewd0realonlyexit-resnet_exit34_place_cifar100_200_best.pt',
        'tiny-imagenet':'result/resnet_exit/tiny-imagenet/Ewd0realonlyexit-resnet_exit34_place_tiny-imagenet_200_best.pt',
        },
    'mobilenetV2_hira_quant':{
        'cifar10':'result/mobilenetV2_hira/cifar10/Ewd0realonlyexit-mobilenetV2_hira_place_cifar10_200_best.pt',
        'cifar100':'result/mobilenetV2_hira/cifar100/Ewd0realonlyexit-mobilenetV2_hira_place_cifar100_200_best.pt',
        'tiny-imagenet':'result/mobilenetV2_hira/tiny-imagenet/Ewd0realonlyexit-mobilenetV2_hira_place_tiny-imagenet_200_best.pt',
    },
    'DeiT-tiny': {
        'cifar10':'result/DeiT/cifar10/LG_cifar10.pth',
        'cifar100': 'result/DeiT/cifar100/LG_cifar100.pth',
        'tiny-imagenet': 'result/DeiT/tiny-imagenet/LG_tiny-imagenet.pth',
        'imagenet': 'result/DeiT/imagenet/LG_imagenet.pth',
    },
    }

ADAMW_LEVEL = {
    'cifar10': 4,
    'cifar100': 3,
    'tiny-imagenet': 3,
    'imagenet': 4,
    }

P_RATIO = {
    'resnet_exit_quant':{
        'cifar10': 0.025,
        'cifar100': 0.025,
        'tiny-imagenet': 0.025,
        },
    'mobilenetV2_hira_quant':{
        'cifar10': 0.025,
        'cifar100': 0.025,
        'tiny-imagenet': 0.025,
        },
    'DeiT-tiny': {
        'cifar10': 0.025,
        'cifar100': 0.025,
        'tiny-imagenet': 0.025,
        'imagenet': 0.025
    },
    }

class T_conf():
    '''
    train configuration, a data structure to efficiently pass numerous parameters between function calls
    '''
    def __init__(self, model='resnet_exit_quant', dataset='cifar100', suffix='', pt_finetune=False, **kwargs):
        self.__dict__.update(kwargs)
        self.dist_url = 'env://'
        self.dist_eval = True
        self.distributed = False
        self.name = suffix # time.asctime()
        # if pt_finetune:
        #     self.name = os.path.join(self.name, 'pt_finetune')
        self.suffix=suffix
        self.pt_finetune=pt_finetune
        self.F_merge = True # warn: enable F merge saving mode, a F traj hit may load a multiple finetuned model. memory saving, faster search but bring a little train bias
        self.F_once = True # only finetune once for F action
        if self.F_once:
            self.F_merge = False
        ### dataset configuration ###
        self.dataset=dataset
        self.inf_data_size = 1000
        self.print_freq=10
        self.pruning_ratio = P_RATIO[model][dataset]
        if self.dataset == "cifar100":
            self.num_class = 100
            self.input_size = 32
            self.print_freq = 2*self.print_freq
        elif self.dataset == "imagenet":
            self.inf_data_size = 10000
            self.num_class = 1000
            self.input_size = 224
            self.print_freq = 20*self.print_freq
        elif self.dataset == "tiny-imagenet":
            self.inf_data_size = 2000
            self.num_class = 200
            self.input_size = 64
            self.print_freq = 4*self.print_freq
        else:
            if self.dataset == 'svhn':
                self.print_freq = 3*self.print_freq
            elif self.dataset == 'cinic10':
                self.print_freq = 9*self.print_freq
            else:
                self.print_freq = 2*self.print_freq
            self.input_size = 32
            self.num_class = 10
        self.batch_size=BATCH_SIZE[model][dataset]
        self.num_workers=32
        ### train_configuration ###
        # hyperpara
        self.sgdlr = 0.1*self.batch_size*utils.get_world_size()/256 
        self.adamlr = 5e-3*self.batch_size*utils.get_world_size()/256 # better than 5e-4
        self.adamwlr = 5e-3*self.batch_size*utils.get_world_size()/256
        self.adamwlr_level = ADAMW_LEVEL[dataset]
        self.lr = [self.sgdlr, self.adamlr, self.adamwlr]
        self.lr_decay = 'loss' 
        self.decay_level = [1.0, 5e-1, 1e-1, 5e-2, 1e-2, 5e-3]
        self.good_acc_loss = 0.3
        self.bad_acc_loss = 1.0
        self.acc_change_limitation = [-10.0, -5.0, -3.0, -1.0, -0.5, -0.3]
        self.broken_acc_loss = BAD_ACC_D[model][dataset]
        self.sgd_wd = 5e-4
        self.adam_wd = 0 # wd of adam + exit layer should be low
        self.adamw_wd = 0.05 # support wd filter
        self.model = model
        # self.quant_model = "resnet_exit_quant"
        if not pt_finetune:
            self.save_path = os.path.join('result/NAS',self.model,self.dataset,self.name)
        else:
            self.save_path = os.path.join('/data/autodl-tmp/env_memory',self.model,self.dataset, 'best_finetune')
        self.env_path = os.path.join('/data/autodl-tmp/env_memory',self.model,self.dataset,'default' if not pt_finetune else 'best')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if not os.path.exists(self.env_path):
            os.makedirs(self.env_path)
        # quant level setting
        self.max_q_level = MAX_Q_LEVEL[self.model][self.dataset]
        if dataset == 'imagenet':
            self.wa_list = [(32,32), (8,8), (6,8), (1,8)]
            self.bcr_mul = [1, 16, 4/3, 4]
            self.mcr_mul = [1, 4, 4/3, 4]
        else:
            self.wa_list = [(32,32), (8,8), (4,8), (1,8)]
            self.bcr_mul = [1, 16, 2, 4]
            self.mcr_mul = [1, 4, 2, 4]
        self.wa_list=self.wa_list[:self.max_q_level]
        self.bcr_mul=self.bcr_mul[:self.max_q_level]
        self.mcr_mul=self.mcr_mul[:self.max_q_level]
        self.exit_layer_ratio = EXIT_LAYER_RATIO[self.model]
        
        # util hyperpara
        self.gpus = '0'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.type = 'torch.cuda.FloatTensor' # 'type of tensor - e.g torch.cuda.HalfTensor'
        # unimportant para
        self.beta = BETAS[model][dataset] # beta for placement inference
        self.max_iter_for_cos_lr_adjust = 200
        self.momentum = 0.9
        self.criterion = torch.nn.CrossEntropyLoss() # for exit and original train
        self.criterion.type(self.type)
        self.gamma = 0.1 # for lr_decay, LR is multiplied by gamma on schedule.
        self.warmup = False
        self.schedule = [150,225] # decrease learning rate at these epochs.

        self.m_conf = {
            'depth': DEPTH[model],
            'w': 32, 
            'a': 32,
            'width': 1.0,
            'scale': model[model.rfind('-')+1:]
        }

        self.m_conf['input_size'] = self.input_size
        self.m_conf['dataset'] = self.dataset
        self.m_conf['num_classes'] = self.num_class
        self.ptf = PTFS[self.model][self.dataset]
        print(self.model)
        
    def add_dataloader(self):
        self.train_loader, self.val_loader, self.test_loader, self.example_inputs = self.dataloader(
            self.dataset, self.batch_size, self.num_workers, self.input_size
        )
        
    def dataloader(self, name, batch_size, num_workers, input_size):
        transform = default_transform(name, input_size)
        val_data = get_dataset(name, 'test', transform['eval'])
        train_data = get_dataset(name, 'train', transform['train'])
        
        if self.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            if self.dist_eval:
                if len(val_data) % num_tasks != 0:
                    logging.info('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. \n \
                        This will slightly alter validation results as extra duplicate entries are added to achieve \n \
                        equal num of samples per-process.')
                sampler_val = torch.utils.data.DistributedSampler(
                    val_data, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(val_data)
        else:
            sampler_train = torch.utils.data.RandomSampler(train_data)
            sampler_val = torch.utils.data.SequentialSampler(val_data)
        
        train_loader = torch.utils.data.DataLoader(
            train_data, sampler=sampler_train,
            batch_size=batch_size, shuffle = False,
            num_workers=num_workers, pin_memory=True)
        
        val_loader = torch.utils.data.DataLoader(
            val_data, sampler=sampler_val,
            batch_size=int(batch_size if name!='imagenet' and batch_size!=1 else batch_size), shuffle=False,
            num_workers=int(num_workers), pin_memory=True)
        
        test_loader = torch.utils.data.DataLoader(
            val_data, sampler=sampler_val,
            batch_size=1, shuffle=False,   # do not change
            num_workers=0, pin_memory=True)
        return train_loader, val_loader, test_loader, train_data[0][0].unsqueeze(0)

    def reset_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.train_loader, self.val_loader, self.test_loader, self.example_inputs = self.dataloader(
            self.dataset, self.batch_size, self.num_workers, self.input_size
        )


HIRA = {'resnet_hira_quant': [7, 15, 23], 'resnet_quant_hira': [7, 15, 23], 'resnet_exit_hira': [7, 15, 23], 'resnet_exit': [7, 15, 23], 'vgg_exit': [5, 10, 15]}
EXITCOST = 0.136 # theoritical cost of a exit layer over a cnn layer, used in inf_acc_costcheck
SINGLE_MUL_RATIO = 2 # see inf_acc_costcheck
MULTI_ADD_RATIO = 0.1 # see inf_acc_costcheck

# _DATASETS_MAIN_PATH = '/data/Datasets'
_DATASETS_MAIN_PATH = '~/autodl-tmp/dataset' # modify this path to your datasets path

def default_transform(dataset, input_size):
    default_t = {
        'train': get_transform(dataset,
                                input_size=input_size, augment=True),
        'eval': get_transform(dataset,
                                input_size=input_size, augment=False)
    }
    return default_t

_dataset_path = {
    'cifar10': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR10'),
    'cifar100': os.path.join(_DATASETS_MAIN_PATH, 'CIFAR100'),
    'stl10': os.path.join(_DATASETS_MAIN_PATH, 'STL10'),
    'mnist': os.path.join(_DATASETS_MAIN_PATH, 'MNIST'),
    'svhn': os.path.join(_DATASETS_MAIN_PATH, 'SVHN'),
    'cinic10': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'CINIC10/train'),
        'test': os.path.join(_DATASETS_MAIN_PATH, 'CINIC10/test')
    },
    'imagenet': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'IMAGENET/train'),
        'test': os.path.join(_DATASETS_MAIN_PATH, 'IMAGENET/val')
    },
    'tiny-imagenet': {
        'train': os.path.join(_DATASETS_MAIN_PATH, 'TINY-IMAGENET/train'),
        'test': os.path.join(_DATASETS_MAIN_PATH, 'TINY-IMAGENET/val')
    }
}

cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]
_transform = {
    'cinic10': {
        'train': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)]),
        'test': transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])
    },
    'svhn': {
        'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    },
    'tiny-imagenet': {
        'train': transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.44785526394844055, 0.41693055629730225, 0.36942949891090393],
        std = [0.2928885519504547, 0.28230994939804077, 0.2889912724494934])]),
        'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.44785526394844055, 0.41693055629730225, 0.36942949891090393],
        std = [0.2928885519504547, 0.28230994939804077, 0.2889912724494934])])
    },
}

def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True):
    train = (split == 'train')
    if name == 'cifar10':
        return datasets.CIFAR10(root=_dataset_path['cifar10'],
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=_dataset_path['cifar100'],
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'svhn':
        return datasets.SVHN(root=_dataset_path['svhn'],
                                 split=split,
                                 transform=_transform[name][split],
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'cinic10':
        path = _dataset_path[name][split]
        return datasets.ImageFolder(root=path, transform=_transform[name][split], target_transform=target_transform)
    elif name == 'imagenet':
        path = _dataset_path[name][split]
        print('data loaded from', path)
        return datasets.ImageFolder(root=path,
                                    transform=transform,
                                    target_transform=target_transform)
    elif name == 'tiny-imagenet':
        path = _dataset_path[name][split]
        print('data loaded from', path)
        return datasets.ImageFolder(root=path,
                                    transform=_transform[name][split],
                                    target_transform=target_transform)
