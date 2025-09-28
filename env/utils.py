from models import ResNet_exit
from models.vit_pytorch.distill_comp import Deit_Comp
from collections import deque
import torch
import models
import matplotlib as plt
from matplotlib.ticker import MaxNLocator
import csv

import pandas as pd
import matplotlib.pyplot as plt
import collections
import os

class ResultsLog:
    def __init__(self, path='results.csv', plot_path=None):
        # Initialize the results and plot paths
        self.path = path
        self.plot_path = plot_path or (self.path + '.png')
        self.results = None
        self.empty = True

    def save(self, title='Training Results'):
        """
        Save the data as a CSV file.
        """
        self.results.to_csv(self.path, index=False)

    def plot(self, x, y, title="Line Plot", xlabel="X-axis", ylabel="Y-axis", suffix=''):
        """
        Plot a single line graph (step vs. y-value).
        :param x: Column name for x-axis.
        :param y: Column name for y-axis.
        :param title: Title of the plot.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self.results[x], self.results[y], label=y, color="blue", linewidth=2)
        plt.xlabel(xlabel, fontsize=16, fontname="Times New Roman")
        plt.ylabel(ylabel, fontsize=16, fontname="Times New Roman")
        plt.title(title, fontsize=18, fontname="Times New Roman")
        plt.legend(fontsize=14)
        plt.grid(True)

        # Save the plot
        if os.path.isfile(self.plot_path):
            os.remove(self.plot_path)
        plt.savefig(self.plot_path+'_'+str(y)+suffix, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_m(self, x, y_list, labels, title=None, xlabel="X-axis", ylabel="Y-axis"):
        """
        Plot multiple lines on the same figure.
        :param x: Column name for x-axis data.
        :param y_list: List of column names for y-axis data.
        :param labels: List of labels for the lines.
        :param title: Title of the plot.
        :param xlabel: Label for the x-axis.
        :param ylabel: Label for the y-axis.
        """
        plt.figure(figsize=(8, 6))
        colors = ["blue", "green", "red", "orange", "purple", "brown"]  # Add more colors as needed

        # Plot each line
        for i, (y, label) in enumerate(zip(y_list, labels)):
            plt.plot(self.results[x], self.results[y], label=label, color=colors[i % len(colors)], linewidth=2)

        # Set axis labels, title, and legend
        plt.xlabel(xlabel, fontsize=16, fontname="Times New Roman")
        plt.ylabel(ylabel, fontsize=16, fontname="Times New Roman")
        plt.title(title, fontsize=18, fontname="Times New Roman")
        plt.legend(fontsize=14)
        plt.grid(True)

        # Save the plot
        if os.path.isfile(self.plot_path):
            os.remove(self.plot_path)
        plt.savefig(f"{self.plot_path}_{title}_combined.png", dpi=300, bbox_inches="tight")
        plt.close()

    def add(self, **kwargs):
        """
        Add a new row of data to the results.
        :param kwargs: Keyword arguments representing column names and their values.
        """
        self.empty = False
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = pd.concat([self.results, df], ignore_index=True)

    def load(self, path=None):
        """
        Load the results from a CSV file.
        :param path: Path to the CSV file.
        """
        path = path or self.path
        if os.path.isfile(path):
            self.results = pd.read_csv(path)

    def show(self):
        """
        Display the plot (for interactive environments like Jupyter).
        """
        if self.results is not None:
            self.plot(self.results.columns[0], self.results.columns[1])  # Display first two columns by default
            
def get_model(args):
    '''
    get the model for inference from the pretrained model
    1. initialize a model according to args.model_name
    2. copy the parameters to model for inference 
    3. return the model
    '''
    # load model parameters (the trained file must be a gpu trained file)
    if 'cuda' in args.type:
        trained_model = torch.load(args.ptf)
    else:
        trained_model = torch.load(args.ptf, map_location=lambda storage, loc: storage)
    if "DeiT" in args.model:
        model = models.__dict__["DeiT"]
        trained_model = trained_model['model']
    else:
        model = models.__dict__[args.model]   # here model is config function
    model = model(**args.m_conf)   # here model becomes the correct class
    model.type(args.type)

    model.transfer_copy(trained_model, True)
    # model.direct_transfer(trained_model, past_model = True)
    state_dict = trained_model if type(trained_model) is collections.OrderedDict else trained_model.state_dict()
    # state_dict['agn.noise'] = torch.tensor(0.0, dtype=torch.float32)
    if 'agn.noise' in state_dict:
        state_dict.pop('agn.noise')
    removed_key = []
    if 'mobilenetV2' in args.model:
        for key, _ in state_dict.items():
            if key not in model.state_dict().keys():
                removed_key.append(key)
    print('removed key: ', removed_key)
    for key in removed_key:
        state_dict.pop(key)
    model.load_state_dict(state_dict)
    del trained_model
    return model

def exit_order(model):
    # current support: resnet18/34/50/101/152
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right

    def build_bst(arr):
        if not arr: return None
        mid = len(arr) // 2
        root = TreeNode(arr[mid])
        root.left = build_bst(arr[:mid]) 
        root.right = build_bst(arr[mid+1:]) 
        return root

    def bfs_traversal(root):
        if not root: return []
        result = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            result.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        return result
    
    if type(model) is Deit_Comp:
        order = sorted(model.init_place_layer, reverse=True)
    else:
        root = build_bst(model.init_place_layer)
        order = bfs_traversal(root)
    return order
    
def get_channel_list(model: ResNet_exit):
    # output the channel num of each layers [for obeservation]
    c_list = []
    for b in model.blocklist:
        c_list.append(b.seq[0].in_channels)
    c_list.append(model.blocklist[-1].seq[0].out_channels)
    return c_list

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name='meter', fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    
from math import cos, pi
def adjust_learning_rate(optimizers, epoch, iteration, num_iter, args):
    # warmup_epoch = 5 if args.warmup else 0
    # warmup_iter = warmup_epoch * num_iter
    # current_iter = iteration + epoch * num_iter
    # max_iter = args.max_iter_for_cos_lr_adjust * num_iter # we assume is the max consecutive epoch num
    for i, optimizer in enumerate(optimizers):
        lr = optimizer.param_groups[0]['lr']

        # if args.lr_decay == 'cos':
        #     lr = args.lr[i] * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
        # elif args.lr_decay == 'step':
        #     lr = args.lr[i] * (args.gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
        # elif args.lr_decay == 'linear':
        #     lr = args.lr[i] * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
        # elif args.lr_decay == 'schedule':
        #     count = sum([1 for s in args.schedule if s <= epoch])
        #     lr = args.lr[i] * pow(args.gamma, count)
        # else:
        #     raise ValueError('Unknown lr mode {}'.format(args.lr_decay))
        # if epoch < warmup_epoch:
        #     lr = args.lr[i] * current_iter / warmup_iter

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = (correct[:k]).reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res