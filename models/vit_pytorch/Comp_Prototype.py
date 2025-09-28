import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import time
import collections

class Comp_Prototype(nn.Module):
    def __init__(self, num_classes=10, depth=19):
        # print("prototype loaded")
        super(Comp_Prototype, self).__init__()
        self.onlyexit = False
        # stocastic NN for mutinfo
        self.depth = depth
        self.mutinfo_forward = False
        # self.dropout = torch.nn.Dropout(0.1)
        # self.agn = AdditiveGaussianNoise(sigma = 1e-3, enabled_on_inference=True)

        # multi prediction part
        self.prediction_w = [1, 2, 3]
        self.prediction_num = len(self.prediction_w)
        self.prediction_list = [0] * self.prediction_num
        self.multi_in_accuracy_f = False

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
        self.place_layer = list(range(depth))
        self.num_exit = len(self.place_layer)
        self.exit_heads = nn.ModuleList([])
        self.mlp_head = nn.Identity()

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