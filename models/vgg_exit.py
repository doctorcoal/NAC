import torch
import torch.nn as nn
import time
from .bof_utils import LogisticConvBoF
import torch.nn.functional as F
import numpy as np
from .gpusetting import GPULEVELNUM, GPUFREQLEVELS, GPUPATH, POLICYLIST
import math
import logging
# from mutinfo.torch.layers import AdditiveGaussianNoise

class VGG_exit(nn.Module):
    def __init__(self, num_classes=10, depth=19, input_size = 32):
        # print("prototype loaded")
        super(VGG_exit, self).__init__()
        # stocastic NN for mutinfo
        self.mutinfo_forward = False
        self.onlyexit = False
        # self.dropout = torch.nn.Dropout(0.1)
        # self.agn = AdditiveGaussianNoise(sigma = 1e-3, enabled_on_inference=True)

        # multi prediction part
        self.prediction_w = [1, 2, 3]
        self.prediction_num = len(self.prediction_w)
        self.prediction_list = [0] * self.prediction_num
        self.multi_in_accuracy_f = False

        # block
        self.num_classes = num_classes
        # self.litelast = (input_size == 32 or input_size == 64)
        self.litelast = True # small classifier is more friendly for Repdistiller
        if depth==19:
            self.blocks = [2, 2, 4, 4, 4]
        elif depth==16:
            self.blocks = [2, 2, 3, 3, 3]
        elif depth==13:
            self.blocks = [2, 2, 2, 2, 2]
        elif depth==11:
            self.blocks = [1, 1, 2, 2, 2]
        else:
            raise NotImplementedError
        self.blockdivide = [sum(self.blocks[0:i+1]) for i in range(len(self.blocks))]
        self.blocklist = nn.ModuleList([])
        inplane = 3
        plane = 64
        self.first = torch.nn.Identity()
        for j, i in enumerate(self.blocks):
            if j:
                self.blocklist.append(nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(inplane, plane, kernel_size=3, stride=1, padding=1,
                            bias=True),
                    nn.BatchNorm2d(plane),
                    nn.ReLU(),
                ))
            else:
                self.blocklist.append(nn.Sequential(
                    nn.Conv2d(inplane, plane, kernel_size=3, stride=1, padding=1,
                            bias=True),
                    nn.BatchNorm2d(plane),
                    nn.ReLU(),
                ))
            for _ in range(1, i):
                self.blocklist.append(nn.Sequential(
                    nn.Conv2d(plane, plane, kernel_size=3, stride=1, padding=1,
                            bias=True),
                    nn.BatchNorm2d(plane),
                    nn.ReLU(),
                ))
            inplane = plane
            if plane < 512:
                plane = plane*2

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

        self.num_classes = num_classes
        ### exit layer added until block16
        ### last block must add right after the last exit layer
        ### and combine all last things
        if self.litelast:
            self.last = nn.Sequential(
            # nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            # nn.Sequential(
                # nn.Dropout(),
                # nn.Linear(512, 512),
                # nn.ReLU(True),
                # nn.Dropout(),
                # nn.Linear(512, 512),
                # nn.ReLU(True),
            nn.Linear(512, num_classes)
                # )
            )
        else:
            self.last = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )

        # threshold for switching between layers
        self.activation_threshold_list = []
        self.exit_num = depth-3
        self.num_early_exit_list = [0]*self.exit_num
        self.original = 0

        # inference
        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0
        self.target_layer = 6
        self.start_layer = 6
        self.layer_store = [0] * self.exit_num

        # accuracy forward
        self.ratio_interval = [0] * 11
        self.statics = [([0] * self.exit_num) for i in range(11)]
        self.jumpstep_store = []
        self.correctratio_store = []
        self.prediction_store = []
        self.predictratio_store = []

        # early exits
        self.dvfs = 'none'
        self.Bof_layer_index = []
        self.place_layer = list(range(self.exit_num))
        for pl in self.place_layer:
            for j, i in enumerate(self.blockdivide):
                if i > pl:
                    self.Bof_layer_index.append(j)
                    break
        self.bof_in_channel_list = [64, 128, 256, 512, 512]
        # print('hira exit layer index in this model:', self.place_layer, 'on group: ', self.Bof_layer_index)
        if self.litelast:
            self.BoF_list = nn.ModuleList([LogisticConvBoF(self.bof_in_channel_list[i], 64,  
                                                        avg_horizon=2) for i in self.Bof_layer_index])
            self.fc0_list = nn.ModuleList([nn.Linear(256, 64) for _ in self.place_layer])
            self.fc1_list = nn.ModuleList([nn.Linear(64, num_classes) for _ in self.place_layer])
        else:
            self.BoF_list = nn.ModuleList([LogisticConvBoF(self.bof_in_channel_list[i], 64,  
                                                        avg_horizon=7) for i in self.Bof_layer_index])
            self.fc0_list = nn.ModuleList([nn.Linear(64*7*7,2048) for _ in self.place_layer])
            self.fc1_list = nn.ModuleList([nn.Linear(2048, num_classes) for _ in self.place_layer])

        self.exit_list = [self.BoF_list, self.fc0_list, self.fc1_list]
        self.main_list = [self.first, self.blocklist, self.last]
        self.trained_pred = False
        self.regime = None

    def direct_transfer(self, model, past_model = False):
        if past_model:
            del model.exit_list
            model.exit_list = [model.BoF_list, model.fc0_list, model.fc1_list]
            model.main_list = [model.first if hasattr(model, 'first') else None, model.blocklist, model.last]

        self.BoF_list = model.exit_list[0]
        self.fc0_list = model.exit_list[1]
        self.fc1_list = model.exit_list[2]
        self.first = model.main_list[0]
        self.blocklist = model.main_list[1]
        self.last = model.main_list[2]
            
    def level_determined(self, jump, current_layer, dvfs):
        remain = self.exit_num - current_layer
        if dvfs == 'tx2cpu':
            jump = min(jump * self.cpu_jump_ratio, remain)
        elif dvfs == 'tx2cpulite':
            jump = jump * self.cpulite_jump_ratio
        target = jump*GPUFREQLEVELS[dvfs][-1]/(remain)
        previous = 0
        for i, freq in enumerate(GPUFREQLEVELS[dvfs]):
            if freq >= target:
                return (i if (freq - target < target - previous or not previous) else i-1)
            previous = freq
    
    def specific_hardware_setup(self, platform, gpu_level=13):
        targetfreq = GPUFREQLEVELS[platform][gpu_level]
        if targetfreq == self.curfreq[platform]:
            return
        if platform == 'tx2' or platform == 'agx':
            if targetfreq > self.curfreq[platform]:
                self.CFL['max'].write(str(targetfreq)+'\n')
                self.CFL['max'].seek(0)
                self.CFL['min'].write(str(targetfreq)+'\n')
                self.CFL['min'].seek(0)
            else:
                self.CFL['min'].write(str(targetfreq)+'\n')
                self.CFL['min'].seek(0)
                self.CFL['max'].write(str(targetfreq)+'\n')
                self.CFL['max'].seek(0)
        elif platform in ['i7-10700', 'agxcpu', 'tx2cpu', 'tx2cpulite']:
            for i in POLICYLIST[platform]:
                self.CFL['cpu'+str(i)].write(str(targetfreq)+'\n')
                self.CFL['cpu'+str(i)].seek(0)
        elif platform == 'agxmem' or platform == 'tx2mem':
            self.CFL['mem'].write(str(targetfreq)+'\n')
            self.CFL['mem'].seek(0)
        self.curfreq[platform] = targetfreq
    
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

    def init_clockfile_header(self):
        self.CFL = {}  # self.clockfilelist
        self.curfreq = {self.dvfs: 0, self.dvfs+'cpu': 0, self.dvfs+'mem': 0, self.dvfs+'cpulite': 0}
        if self.dvfs == 'tx2':
            if self.exit_num <= 20:  
                self.cpu_jump_ratio = 3 
                self.cpulite_jump_ratio = 1
            else:
                self.cpu_jump_ratio = (4 if len(self.np_scw) == 3 else 2)
                self.cpulite_jump_ratio = 1
            print('set cpu jump ratio: ', self.cpu_jump_ratio)
        if self.dvfs == 'tx2' or self.dvfs == 'agx':
            gpu_path = GPUPATH[self.dvfs]
            self.CFL['max'] = open(gpu_path + "/max_freq", "w", 1)
            self.CFL['min'] = open(gpu_path + "/min_freq", "w", 1)
            
            cpu_path = GPUPATH[self.dvfs+'cpu']
            for i in POLICYLIST[self.dvfs+'cpu']:
                self.CFL['cpu'+str(i)] = open(cpu_path+'/policy'+ str(i)+"/scaling_setspeed", "w", 1)
                
            if self.dvfs == 'tx2':
                for i in POLICYLIST[self.dvfs+'cpulite']:
                    self.CFL['cpu'+str(i)] = open(cpu_path+'/policy'+ str(i)+"/scaling_setspeed", "w", 1)
                
            mem_path = GPUPATH[self.dvfs+'mem']
            self.CFL['mem'] = open(mem_path, "w", 1)
        if self.dvfs == 'i7-10700':
            cpu_path = GPUPATH[self.dvfs]
            for i in POLICYLIST[self.dvfs]:
                self.CFL['cpu'+str(i)] = open(cpu_path+'/policy'+ str(i)+"/scaling_setspeed", "w", 1)
            
    def close_clockfile_header(self):
        for i in self.CFL.values():
            i.close()
        return
    
    def initcount(self):
        self.pred_time = 0
        self.exit_time = 0
        self.inf_time = 0

        self.inf_layer = 0
        self.exit_count = 0
        self.pred_count = 0

        self.dvfs_list = []
        self.ratio_interval = [0] * 11
        self.statics = [([0] * self.exit_num) for i in range(11)]
        self.jumpstep_store = []
        self.correctratio_store = []
        self.prediction_store = []
        self.predictratio_store = []
        self.num_early_exit_list = [0]*self.exit_num
        self.original = 0

    def BNnoquant(self, doing):
        for i in self.blocklist:
            i.BNnoquant = doing

    def set_prediction(self, singlejump=False, multi = False, weight = torch.tensor([[[1.0, 1.0, 1.0]]])):
        self.prediction_num = len(weight)
        self.prediction_w = weight
        self.single_jump = singlejump
        self.multi_in_accuracy_f = multi

    def _calculate_max_activation(self, param):
        '''
        return the maximum activation item in [param]
        '''
        return torch.max(param)
    
    def get_specific_exit_number(self, iterate):
        return self.num_early_exit_list[iterate]

    def _calculate_max_activation_pred(self, param):
        return torch.stack([torch.max(i) for i in param])

    def set_activation_thresholds( self, threshold_list:list ):
        if len(threshold_list) != self.exit_num:
            print(f'the length of the threshold_list ({len(threshold_list)}) is invalid, should be {self.exit_num}')
            raise NotImplementedError
        for i in range(len(threshold_list)):
            self.activation_threshold_list.append(abs(threshold_list[i]))

    def print_exit_percentage(self, log=False):
        total_inference = sum(self.num_early_exit_list)+ self.original
        for i in range(self.exit_num):
            logging.debug('Early Exit' + str(i) + ': ' + "{:.2f}".format(100*self.num_early_exit_list[i]/total_inference))
        logging.debug( f'original: {100*self.original/total_inference:.3f}% ({self.original}/{total_inference})' )

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

    def prediction(self, param, layer):
        if self.single_jump:
            round = self.exit_num - layer
        else:
            round = min(self.max_round, self.exit_num - layer)  # here current_layer + 1 is inputted, prediction start from next layer
        param = param[0].cpu().numpy()
        for i in range(round - 1):
            param = np.convolve(param, self.np_scw, 'same')
            ratio = self.np_log_softmax(param).max()
            if ratio > math.log(self.beta): # * temp_threshold:
                return i+1
        return round
    
    def prediction_gpu(self, param, layer):
        # layer is the next layer of current layer
        if self.single_jump:
            round = self.exit_num - layer
        else:
            round = min(self.max_round, self.exit_num - layer)  # here current_layer + 1 is inputted, prediction start from next layer
        # length = len(self.activation_threshold_list)
        param = param.reshape(-1, 1, self.num_classes)
        for i in range(round - 1):
            param = F.conv1d(param, self.simple_conv_weight, padding = 'same')
            ratio = torch.max(F.log_softmax(param, dim=2, dtype=torch.double)).item()
            if ratio > math.log(self.beta): # * temp_threshold:
                return i+1
        return round

    def multi_prediction(self, param, layer):
        # length = len(self.activation_threshold_list)
        if self.single_jump:
            round = self.exit_num - layer
        else:
            round = min(self.max_round, self.exit_num - layer)  # here current_layer + 1 is inputted, prediction start from next layer
        temp = [param[i][0].cpu().numpy() for i in range(self.prediction_num)]
        for i in range(self.prediction_num):   # do conv for previous layer
            for j in range(i):
                temp[j] = np.convolve(temp[j], self.np_scw, 'same')
        ratio_list = [0] * self.prediction_num
        for i in range(round - 1):
            # temp_thresold = self.activation_threshold_list[layer+i] if (layer+i) < length else self.activation_threshold_list[length-1]
            for j in range(self.prediction_num):
                temp[j] = np.convolve(temp[j], self.np_scw, 'same')
                ratio_list[j] = math.exp(self.np_log_softmax(temp[j]).max()) * self.prediction_w[j]
            if sum(ratio_list) > sum(self.prediction_w) * self.beta:
                return i+1
        return round

    def place_test_forward(self, x):  # multi jump + single jump
        j = 0
        for i, block in enumerate(self.blocklist):
            start_inf = time.perf_counter()
            x = block(x)
            end_inf = time.perf_counter()
            self.inf_layer += 1
            self.inf_time += end_inf - start_inf
            if i >= self.blockdivide[j]: j += 1
            if i == self.start_layer:
                start_exit = time.perf_counter()
                x_exit = self.exit_list[j](x)
                end_exit = time.perf_counter()
                self.exit_count += 1
                self.exit_time += end_exit - start_exit
                ratio = self._calculate_max_activation(F.log_softmax(x_exit, dim=1, dtype=torch.double)).item()
                if ratio >= math.log(self.beta):
                    self.num_early_exit_list[i] += 1
                    return i, x_exit
        x = self.last(x)
        self.original += 1
        self.inf_layer += 1
        return len(self.blocklist), x

    def place_forward(self, x):  # multi jump + single jump
        j = 0
        for i, block in enumerate(self.blocklist):
            start_inf = time.perf_counter()
            x = block(x)
            end_inf = time.perf_counter()
            self.inf_layer += 1
            self.inf_time += end_inf - start_inf
            if i >= self.blockdivide[j]: j += 1
            if i in self.place_layer:
                start_exit = time.perf_counter()
                x_exit = self.exit_list[j](x)
                end_exit = time.perf_counter()
                self.exit_count += 1
                self.exit_time += end_exit - start_exit
                ratio = self._calculate_max_activation(F.log_softmax(x_exit, dim=1, dtype=torch.double)).item()
                if ratio >= math.log(self.beta):
                    self.num_early_exit_list[i] += 1
                    return i, x_exit
        x = self.last(x)
        self.original += 1
        self.inf_layer += 1
        return len(self.blocklist), x

    def exits_forward(self, x):  # multi jump + single jump
        self.target_layer = self.start_layer  # the layer to begin enable exit
        dvfs_in_this_inf = 0
        j = 0
        predicted = False
        for i, block in enumerate(self.blocklist):
            start_inf = time.perf_counter()
            x = block(x)
            end_inf = time.perf_counter()
            self.inf_layer += 1
            self.inf_time += end_inf - start_inf
            if i >= self.blockdivide[j]: j += 1
            if self.target_layer == i:
                start_exit = time.perf_counter()
                x_exit = self.exit_list[j](x)
                end_exit = time.perf_counter()
                self.exit_count += 1
                self.exit_time += end_exit - start_exit
                ratio = self._calculate_max_activation(F.log_softmax(x_exit, dim=1, dtype=torch.double)).item()
                if ratio >= math.log(self.beta) or (self.single_jump and predicted):
                    self.num_early_exit_list[i] += 1
                    if self.start_layer:
                        if self.dvfs != 'none':  # should not be edge device with weak cpu, if need time performance, remove this
                            start_dvfs = time.perf_counter()
                            self.specific_hardware_setup(self.dvfs, -1)
                            if self.dvfs == 'tx2':
                            # self.specific_hardware_setup(self.dvfs+'mem', gpu_level=self.level_determined(jump, i, self.dvfs+'mem'))
                                self.specific_hardware_setup(self.dvfs+'cpu', -1)
                                self.specific_hardware_setup(self.dvfs+'cpulite', 1)
                            end_dvfs = time.perf_counter()
                            dvfs_time = end_dvfs - start_dvfs
                            self.dvfs_list.append(dvfs_time)
                            dvfs_in_this_inf += dvfs_time
                    return i, x_exit, dvfs_in_this_inf
                else:
                    predicted = True
                    if i < len(self.blocklist) - 1:  # prediction except last layer
                        start_pred = time.perf_counter()
                        jump = self.prediction(x_exit, i)  # prediction output starts from next layer
                        end_pred = time.perf_counter()
                        self.pred_count += 1
                        pred_time = end_pred - start_pred
                        self.pred_time += pred_time
                        self.target_layer += jump
                        if self.target_layer > self.exit_num:
                            self.target_layer = self.exit_num
                            jump = self.exit_num - i
                        if self.dvfs != 'none':
                            start_dvfs = time.perf_counter()
                            self.specific_hardware_setup(self.dvfs, gpu_level=self.level_determined(jump, i, self.dvfs))
                            if self.dvfs == 'tx2':
                                # self.specific_hardware_setup(self.dvfs+'mem', gpu_level=self.level_determined(jump, i, self.dvfs+'mem'))
                                self.specific_hardware_setup(self.dvfs+'cpu', gpu_level=self.level_determined(jump, i, self.dvfs+'cpu'))
                                self.specific_hardware_setup(self.dvfs+'cpulite', gpu_level=self.level_determined(jump, i, self.dvfs+'cpulite'))
                            elif self.dvfs == 'agx':
                                self.specific_hardware_setup(self.dvfs+'cpu', gpu_level=self.level_determined(jump, i, self.dvfs+'cpu'))
                            end_dvfs = time.perf_counter()
                            dvfs_time = end_dvfs - start_dvfs
                            self.dvfs_list.append(dvfs_time)
                            dvfs_in_this_inf += dvfs_time
        x = self.last(x)
        self.original += 1
        self.inf_layer += 1
        return len(self.blocklist), x, dvfs_in_this_inf

    def output_time(self):
        return [self.inf_time, self.exit_time, self.pred_time, sum(self.dvfs_list)]
    
    def output_count(self):
        return [self.inf_layer, self.exit_count, self.pred_count]
    
    def output_pred_error(self):
        jump_layer = sum(self.jumpstep_store)
        pred_error_layer = sum([abs(i-j) for i, j in zip(self.jumpstep_store, self.prediction_store)])
        return [pred_error_layer, jump_layer, pred_error_layer/len(self.jumpstep_store), jump_layer/len(self.jumpstep_store)]

    def accuracy_forward(self, x):  # multi jump accuracy + single jump accuracy
        self.target_layer = self.start_layer  # the layer to count prediction accuracy
        j = 0
        for i, block in enumerate(self.blocklist):
            start_inf = time.perf_counter()
            x = block(x)
            end_inf = time.perf_counter()
            self.inf_layer += 1
            self.inf_time += end_inf - start_inf
            if i >= self.blockdivide[j]: j += 1
            if self.start_layer <= i:
                start_exit = time.perf_counter()
                x_exit = self.exit_list[j](x)
                end_exit = time.perf_counter()
                self.exit_count += 1
                self.exit_time += end_exit - start_exit
                ratio = self._calculate_max_activation(F.log_softmax(x_exit, dim=1, dtype=torch.double)).item()
                # ratio_index = int(ratio/0.1)
                # self.ratio_interval[ratio_index] += 1
                if ratio >= math.log(self.beta):
                    if self.start_layer < i:
                        if self.single_jump:
                            self.jumpstep_store.append(i-self.target_layer) # save jump number
                        else:
                            self.jumpstep_store.append(i) # save layer number to exit
                            self.prediction_store.append(self.target_layer)
                    self.num_early_exit_list[i] += 1
                    return i, x_exit
                if self.target_layer == i:
                    start_pred = time.perf_counter()
                    jump = self.prediction(x_exit, i)
                    end_pred = time.perf_counter()
                    if self.single_jump:
                        self.prediction_store.append(jump)
                    else: 
                        self.target_layer += jump
                    self.pred_count += 1
                    self.pred_time += end_pred - start_pred
        self.original += 1
        x = self.last(x)
        self.inf_layer += 1
        self.jumpstep_store.append(i-self.target_layer)
        return len(self.blocklist), x

    def normal_forward( self, x ):
        for block in self.blocklist:
            x = block(x)
            self.inf_layer += 1
        x = self.last(x)
        self.inf_layer += 1
        self.original += 1
        return x

    def set_eval(self, mode = 'normal_forward'):
        self.eval_mode = mode
        self.train_mode = None

    def set_train(self, exit_layer = 'original', onlyexit = False):
        self.train_mode = exit_layer
        self.eval_mode = None
        self.onlyexit = onlyexit
        if onlyexit:
            for name, value in self.named_parameters():
                if 'blocklist' in name:
                    value.requires_grad = False
        else:
            for name, value in self.named_parameters():
                if 'blocklist' in name:
                    value.requires_grad = True
                    
    def forward( self, x):
        if self.eval_mode: # evaluation
            if self.eval_mode == 'accuracy_forward':
                return self.accuracy_forward(x) # only used to test accuracy
            elif self.eval_mode == 'exits_forward':
                return self.exits_forward(x)
            elif self.eval_mode == 'normal_forward':
                return self.normal_forward(x)
            elif self.eval_mode == 'place_forward':
                return self.place_forward(x)
            elif self.eval_mode == 'place_test_forward':
                return self.place_test_forward(x)
        else: # train
            if self.train_mode == 'original':
                return self.forward_original( x )
            elif self.train_mode == 'exits':
                return self.forward_exits( x )
            elif self.train_mode == 'place':
                return self.forward_place( x )

    def forward_original( self, x ):
        for block in self.blocklist:
            x = block(x)
        if self.distill:
            if self.litelast:
                f5 = self.last[0:2](x)
                x = self.last[2:](f5)
            else:
                f5 = self.last[0:3](x)
                x = self.last[3:](f5)
            return [f5], x
        else:
            x = self.last(x)
            return x

    def forward_exits( self, x ):
        if self.distill:
            raise NotImplementedError
        output_list = []
        j = 0
        for i, block in enumerate(self.blocklist):
            x = block(x)
            if i >= self.blockdivide[j]: j += 1
            x_exit = self.exit_list[j](x)
            output_list.append(x_exit)
        x = self.last(x)
        output_list.append(x)
        return output_list


def vgg_exit(**kwargs):
    num_classes, depth, input_size = map(
        kwargs.get, ['num_classes', 'depth', 'input_size'])
    num_classes = num_classes or 10
    input_size = input_size or 32
    depth = depth or 19
    print(f'num_classes {str(num_classes)}, depth {str(depth)}, input_size {str(input_size)}.')
    return VGG_exit(num_classes, depth, input_size=input_size)
