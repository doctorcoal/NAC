import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Reduce
from .binarymodule import Quant_Conv2d, IRlinear
import logging
import time
import math
from .binaryfunction import T_quantize
import collections
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )
    
def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, Quant_Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear) or isinstance(m, IRlinear):
            nn.init.xavier_normal_(m.weight)
            if m.bias != None:
                nn.init.constant_(m.bias, 0)

class Comp_Base(nn.Module):
    def __init__(self, num_classes=10, depth=19, w=32, a=32, exit_mode='original', exit_type=None, place_layer = {}):
        # print("prototype loaded")
        super(Comp_Base, self).__init__()
        self.onlyexit = False
        # stocastic NN for mutinfo
        self.depth = depth
        self.mutinfo_forward = False
        # self.dropout = torch.nn.Dropout(0.1)
        # self.agn = AdditiveGaussianNoise(sigma = 1e-3, enabled_on_inference=True)

        self.activation = a
        self.w_atten = w
        self.f_atten = w
        logging.info('model activation bit: '+str(self.activation))
        logging.info('model attention bit: '+str(self.w_atten))
        self.mode = exit_mode
        self.exit_type = exit_type
        
        # multi prediction part
        self.prediction_w = [1, 2, 3]
        self.prediction_num = len(self.prediction_w)
        self.prediction_list = [0] * self.prediction_num
        self.multi_in_accuracy_f = False
        self.conv1 = None

        # block
        self.num_classes = num_classes
        self.litelast = True # small classifier is more friendly for Repdistiller
        self.blocklist = nn.ModuleList([])
        self.first = torch.nn.Identity()
        self.last = torch.nn.Identity()

        # time part
        self.pred_time = 0
        self.exit_time = 0
        self.inf_time = 0
        # self.pred_list = []
        # self.dvfs_time = 0
        self.dvfs_list = []

        self.inf_layer = 0
        self.exit_count = 0
        self.pred_count = 0

        # do not do reprediction
        self.single_jump = False
        self.max_round = 8

        # threshold for switching between layers
        self.exit_num = 0
        self.num_early_exit_list = [0]*self.exit_num
        self.original = 0

        # inference
        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
        self.target_layer = 6
        self.start_layer = 6
        self.layer_store = [0] * self.exit_num

        # accuracy forward
        self.jumpstep_store = []
        self.correctratio_store = []
        self.prediction_store = []
        self.predictratio_store = []

        # early exits
        self.Bof_layer_index = []
        self.place_layer = place_layer
        self.num_exit = len(self.place_layer)
        self.exit_heads = nn.ModuleList([])
        self.mlp_head = nn.Identity()

    def init_exit(self, channels, num_classes, image_size):##子类搭建模型后显式调用
        if not self.place_layer:
            self.exit_heads = nn.Identity()
            self.cls_flops = 2*image_size/32*image_size/32*channels[-2]*channels[-1]+2*channels[-1]*num_classes##改不改？
            self.mlp_head = nn.Sequential(
                conv_1x1_bn(channels[-2], channels[-1]),
                Reduce('b c h w -> b c', 'mean'),
                nn.Linear(channels[-1], num_classes, bias=False)
            )
        else:
            self.exit_heads = nn.ModuleList()
            self.exit_flops = []
            self.mlp_head = nn.Identity() 
            for layer_idx in range(self.depth):
                if layer_idx in self.place_layer:
                    exit_type = self.place_layer[layer_idx]
                    if exit_type == 'cnn_gap':  # CNN
                        c = channels[layer_idx] if layer_idx < len(channels) else channels[-1]
                        exit_head = nn.Sequential(
                            conv_1x1_bn(c, channels[-1]),
                            Reduce('b c h w -> b c', 'mean'),
                            nn.Linear(channels[-1], num_classes, bias=False)
                        )
                        # 计算FLOPs: 1x1卷积 + 全连接
                        flops = 2 * (image_size // (2 ** (layer_idx + 2))) * (image_size // (2 ** (layer_idx + 2))) * c * channels[-1] + 2 * channels[-1] * num_classes               
                    elif exit_type == 'transformer_cls': # Transformer
                        dim = channels[layer_idx] if layer_idx < len(channels) else channels[-1]
                        exit_head = nn.Linear(dim, num_classes)
                        flops = 2 * dim * num_classes    # 计算FLOPs: 仅全连接
                    elif exit_type == 'mlp':
                        dim = channels[layer_idx] if layer_idx < len(channels) else channels[-1]
                        hidden_dim = dim // 2
                        exit_head = nn.Sequential(
                            nn.Linear(dim, hidden_dim),
                            nn.ReLU(),
                            nn.Linear(hidden_dim, num_classes)
                        )
                        flops = 2 * dim * hidden_dim + 2 * hidden_dim * num_classes   # 计算FLOPs: 两个全连接层
                    else:
                        exit_head = nn.Identity()
                        flops = 0
                        
                    self.exit_heads.append(exit_head)
                    self.exit_flops.append(flops)
                else:
                    self.exit_heads.append(nn.Identity())
                    self.exit_flops.append(0)
        self.num_exit = len(self.place_layer)
        self.exit_params = [sum(p.numel() for p in l.parameters() if p.requires_grad) for l in self.exit_heads]
        self.exit_params_dict = dict([(i, l) for i, l in zip(range(self.depth), self.exit_params)])
        logging.info(f"exit params: {self.exit_params_dict}")
        self.exit_flops_dict = dict([(i, l) for i, l in zip(range(self.depth), self.exit_flops)])
        logging.info(f"exit flops: {self.exit_flops_dict}")
        self.main_list = [self.conv1, self.blocklist, self.mlp_head]
        self.exit_list = [self.exit_heads]
        init_model(self)
        self.initcount()
        
    def forward(self, x):
        # self.inf_start()
        x = self.conv1(x)
        x = T_quantize(x, self.activation-1)
        # self.inf_record()
        if self.mode == 'original':
            x = self.forward_original(x)
        elif 'exit_t' in self.mode:
            x = self.forward_exit_t(x)
        elif 'exit_i' in self.mode:
            x = self.forward_exit_i(x)
        return x
        
    def block_forward(self, x, i, block): # block推理
        self.inf_start()
        x = block(x)
        self.inf_record()
        return x
        
    def exit_forward(self, x, i, block): # exit推理
        self.exit_start()
        exit_x = self.exit_heads[i](x)
        self.exit_record()
        return exit_x
    
    def forward_original(self, x): # 普通推理
        for i, block in enumerate(self.blocklist):
            x = self.block_forward(x, i, block)
        x = T_quantize(x, self.activation-1)
        return self.mlp_head(x)
    
    def forward_exit_t(self, x, dominant_list): # exit 训练
        out_list = []
        for i, block in enumerate(self.blocklist):
            x = self.block_forward(x, i, block)
            if i in self.place_layer:
                if dominant_list is not None and i not in dominant_list:
                    # 不更新dominant函数
                    exit_x = x.detach()
                    # exit_x.requires_grad = True
                    exit_x = T_quantize(self.blocklist.norm(exit_x), self.activation-1)
                else:
                    exit_x = T_quantize(self.blocklist.norm(x), self.activation-1)
                out = self.exit_forward(exit_x, i, block)
                out_list.append(out)
        return out_list

    def forward_exit_i(self, x): # exit推理
        for i, block in enumerate(self.blocklist):
            x = self.block_forward(x, i, block)
            if i in self.place_layer:
                exit_x = T_quantize(self.blocklist.norm(x), self.activation-1)
                out = self.exit_forward(exit_x, i, block)
                if i == self.place_layer[-1] or torch.max(F.log_softmax(out, dim=1, dtype=torch.double)).item() >= math.log(self.beta):
                    self.num_early_exit_list[i] += 1
                    return i, out
            i += 1
    
    def initcount(self):
        self.pred_time = 0
        self.exit_time = 0
        self.inf_time = 0

        self.inf_layer = 0
        self.exit_count = 0
        self.pred_count = 0

        self.dvfs_list = []
        self.jumpstep_store = []
        self.correctratio_store = []
        self.prediction_store = []
        self.predictratio_store = []
        self.num_early_exit_list = [0]*self.exit_num
        self.original = 0

    def direct_transfer(self, model, past_model = False):
        if past_model:
            del model.exit_list
            model.exit_list = [model.BoF_list, model.fc0_list, model.fc1_list]
            model.main_list = [model.first if hasattr(model, 'first') else None, model.blocklist, model.last]
        # for i, _ in enumerate(self.exit_list):
        #     temp = self.exit_list[i]
        #     self.exit_list[i] = model.exit_list[i]
        #     del temp
        # for i, _ in enumerate(self.main_list):
        #     temp = self.main_list[i]
        #     self.main_list[i] = model.main_list[i]
        #     del temp
        self.BoF_list = model.exit_list[0]
        self.fc0_list = model.exit_list[1]
        self.fc1_list = model.exit_list[2]
        self.first = model.main_list[0]
        self.blocklist = model.main_list[1]
        self.last = model.main_list[2]
            
    def BNnoquant(self, doing):
        for i in self.blocklist:
            i.BNnoquant = doing
            
    def change_quant(self, w, a):
        self.activation_bit = a
        self.main_weight_bit = w
        self.bof_weight_bit = w
        self.fc_weight_bit = w
        for i in self.blocklist:
            i.bit = w
            i.seq[0].bit = w
            i.activation_bit = a
        for i in self.BoF_list:
            for j in i:
                j.bit = w
                j.codebook.bit = w
        for i in self.fc0_list:
            i.bit = w
        for i in self.fc1_list:
            i.bit = w

    def transfer_copy(self, model, transfer_exit):
        mystate_dict = self.state_dict()
        state_dict = model if type(model) is collections.OrderedDict else model.state_dict()
        for name, parameter in state_dict.items():
            if name in mystate_dict.keys():
                if not transfer_exit:
                    if '_list' in name: # don't load the exit layer of these model
                        continue
                mystate_dict[name].copy_(parameter)
        self.load_state_dict(mystate_dict)

    def BNnoquant(self, doing):
        for i in self.blocklist:
            i.BNnoquant = doing

    def _calculate_max_activation(self, param):
        '''
        return the maximum activation item in [param]
        '''

        # top2_values, _ = torch.topk(param,2)
        # a, b = torch.exp(top2_values[0])
        # return (a-b)/a
        return torch.max(param)
    
    def get_specific_exit_number(self, iterate):
        return self.num_early_exit_list[iterate]

    def _calculate_max_activation_pred(self, param):
        return torch.stack([torch.max(i) for i in param])

    def print_exit_percentage(self, log = False):
        total_inference = sum(self.num_early_exit_list)+ self.original
        for i in range(self.exit_num):
            text = 'Early Exit' + str(i) + ': ' + "{:.2f}".format(100*self.num_early_exit_list[i]/total_inference)
            if not log:
                print(text)
            else:
                logging.debug(text)
        text =  f'original: {100*self.original/total_inference:.3f}% ({self.original}/{total_inference})' 
        if not log:
                print(text)
        else:
            logging.debug(text)

    def simple_conv1d(self, param):
        copy = param.copy()
        temp = []
        copy.insert(0,0)
        copy.append(0)
        for i in range(len(copy)-2):
            temp.append(copy[i]+copy[i+1]+copy[i+2])
        return temp
    
    def np_log_softmax(self, x):
        c = x.max()
        logsumexp = np.log(np.exp(x - c).sum())
        return x - c - logsumexp

    def softXEnt(self, input, target, select = None):
        # input values are logits
        # target values are "soft" probabilities that sum to one (for each sample in batch)
        # print(torch.max(target*select, dim = 1))
        return F.kl_div(F.log_softmax(input, dim=1), target * select, reduction='batchmean')

    def output_time(self):
        return [self.inf_time, self.exit_time, self.pred_time, sum(self.dvfs_list)]
    
    def output_count(self):
        return [self.inf_layer, self.exit_count, self.pred_count]
    
    def output_pred_error(self):
        jump_layer = sum(self.jumpstep_store)
        pred_error_layer = sum([abs(i-j) for i, j in zip(self.jumpstep_store, self.prediction_store)])
        return [pred_error_layer, jump_layer, pred_error_layer/len(self.jumpstep_store), jump_layer/len(self.jumpstep_store)]

    def set_mode(self, mode = 'original', onlyexit = False):
        self.mode = mode
        self.onlyexit = onlyexit

    def set_eval(self, mode = 'normal_forward'):
        self.mode = 'exit_i'

    def set_train(self, exit_layer = 'original', onlyexit = False):
        self.mode = 'original' if exit_layer == 'original' else 'exit_t'
        self.onlyexit = onlyexit
        for element in self.main_list:
            if type(element) is torch.nn.Parameter:
                element.requires_gred = not onlyexit
            else:
                for name, value in element.named_parameters():
                        value.requires_grad = not onlyexit

    def set_start_layer(self, layer):
        self.target_layer = layer  # start from 0, meanning exit decision 
        self.start_layer = layer   # after finish this layer

    def set_beta(self, beta):
        self.beta = beta

    def set_eval_pred(self, layer, forward_mode, trained_pred = False, max_round = 8):
        if type(layer) == list:    # place layer
            self.place_layer = layer
        else:    # start layer
            self.set_start_layer(layer)
        self.set_eval(forward_mode)
        self.trained_pred = trained_pred
        self.max_round = max_round

    def inf_start(self):
        self.start_inf = time.perf_counter()

    def inf_record(self):
        self.inf_layer += 1
        self.inf_time += time.perf_counter() - self.start_inf

    def exit_start(self):
        self.start_exit = time.perf_counter()

    def exit_record(self):
        self.exit_count += 1
        self.exit_time += time.perf_counter() - self.start_exit

    def forward( self, x):
        return x

    def del_exit_layer(self, idx):
        if idx < 0 or idx >= len(self.exit_heads) or idx not in self.place_layer:
            raise ValueError(f"Index {idx} out of range for exit_heads (length {len(self.exit_heads)})")
        self.exit_heads[idx] = nn.Identity()
        # del self.exit_heads[idx]
        self.exit_params[idx] = 0
        self.exit_flops[idx] = 0
        self.exit_params_dict[idx] = 0
        self.exit_flops_dict[idx] = 0
        self.place_layer.remove(idx)
        self.num_exit -= 1
        self._modules['exit_heads'] = self.exit_heads 
        torch.cuda.empty_cache() 
        
    def set_place(self, new_place = []):
        self.past_place = self.place_layer
        self.place_layer = new_place
        self.num_exit = len(self.place_layer)
        
    def recover_place(self):
        self.place_layer = self.past_place
        self.num_exit = len(self.place_layer)