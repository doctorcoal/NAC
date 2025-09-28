import torch.nn as nn
import torch
import math
import time
from .bof_utils import LogisticConvBoF
import torch.nn.functional as F
from .vgg_exit import VGG_exit
import logging

# __all__ = ['ResNet_exit']
EXIT_NUM = 32

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

class Single_layer(nn.Module):
    def __init__(self, seq=None, layer=-1, block=-1, downsample = None, rest = None, begin_layer = False, end_layer = False):
        super().__init__()
        self.seq = seq
        self.block = block
        self.layer = layer
        self.downsample = downsample
        self.rest = rest   # processing after forward+downsample
        self.end_layer = end_layer
        self.begin_layer = begin_layer
        self.BNnoquant = True

    def forward(self, x):
        return self.seq(x)

    def residual(self, x):
        if self.downsample is not None:
            return self.downsample(x)
        else:
            return x

# from mutinfo.torch.layers import AdditiveGaussianNoise

class ResNet_exit(VGG_exit):
    def __init__(self, num_classes=10, block='BasicBlock', layers=[3, 4, 6, 3], exit_num = EXIT_NUM, input_size = 32, sigma = 1e-3):
        # print("resnet exit loaded")
        super(ResNet_exit, self).__init__(num_classes = num_classes, input_size = input_size)
        self.model_class = 'resnet'
        self.distill = False
        # self.agn = AdditiveGaussianNoise(sigma, enabled_on_inference=True)
        
        # multi prediction part
        self.prediction_w = [1, 2, 3]
        self.prediction_num = len(self.prediction_w)
        self.prediction_list = [0] * self.prediction_num
        self.multi_in_accuracy_f = False

        # time part
        self.pred_time = 0
        self.exit_time = 0
        self.inf_time = 0
        self.dvfs_list = []
        self.inf_layer = 0
        self.exit_count = 0
        self.pred_count = 0

        # do not do reprediction
        self.single_jump = False

        # adding layers
        self.cifar = (input_size == 32 or input_size == 64)
        self.num_classes = num_classes
        self.blocklist = nn.ModuleList([])
        self.inplanes = 64
        self.conv_in_singlelayer = 2 if block == 'BasicBlock' else 3
        self.expansion = 1 if block == 'BasicBlock' else 4
        self.first = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3 if self.cifar else 7,
                      stride=1 if self.cifar else 2, padding=1 if self.cifar else 3, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        )
        self.blocklist += self._make_layer(block, 64, layers[0], stride=1, layer=0)
        self.blocklist += self._make_layer(block, 128, layers[1], stride=2, layer=1)
        self.blocklist += self._make_layer(block, 256, layers[2], stride=2, layer=2)
        self.blocklist += self._make_layer(block, 512, layers[3], stride=2, layer=3)
        ### exit layer added until last block
        ### last block must add right after the last exit layer
        ### and combine all last things
        self.last = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * self.expansion, num_classes)
        )

        # threshold for switching between layers
        self.activation_threshold_list = []
        self.exit_num = exit_num   # exit_num = 32 for this model, DONT CHANGE ,32 + last layer
        self.depth = sum(layers)*self.conv_in_singlelayer
        self.max_round = self.depth
        self.num_early_exit_list = [0]*self.exit_num
        self.original = 0
        self.init_place_layer = list(range(self.conv_in_singlelayer-1, # model.conv_in_singlelayer-1 for post exit placement; 0 for pre exit placement
                                    self.depth,self.conv_in_singlelayer))
        self.place_layer = self.init_place_layer
        if block == 'Bottleneck':
            for i in range(len(self.place_layer)):
                if self.place_layer[i] % 3 == 0:
                    self.place_layer[i] -= 1
                elif self.place_layer[i] % 3 == 1:
                    self.place_layer[i] += 1

        # inference
        # the beta coefficient used for accuracy-speed trade-off, the higher the more accurate
        self.beta = 0.95
        self.target_layer = 0
        self.start_layer = 0
        self.layer_store = [0] * self.exit_num

        # accuracy forward
        self.jumpstep_store = []
        self.correctratio_store = []
        self.prediction_store = []
        self.predictratio_store = []

        # early exits
        self.dvfs = 'none'
        self.BoF_list = nn.ModuleList([])
        for i, l in enumerate(layers): # layers = [3, 4, 6, 3]
            self.BoF_list.append(nn.ModuleList(l*[LogisticConvBoF(int(self.expansion*64*(2**i)), 64, avg_horizon=2 if self.cifar else 4)]))
        self.fc0_list = nn.ModuleList([nn.Linear(256 if self.cifar else 1024, 64 if self.cifar else num_classes) for _ in layers])
        self.fc1_list = nn.ModuleList([(nn.Linear(64, num_classes) if self.cifar else torch.nn.Identity()) for _ in layers])
        self.exit_list = [self.BoF_list, self.fc0_list, self.fc1_list]
        self.main_list = [self.first, self.blocklist, self.last]
        init_model(self)

    def _make_layer(self, block, planes, blocks, stride=1, layer=-1):
        downsample = None
        if block == 'BasicBlock':
            expansion = 1
            block = self.BasicBlock
        else:
            expansion = 4
            block = self.Bottleneck
        if stride != 1 or self.inplanes != planes * expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )
        layers = nn.ModuleList([])
        layers += block(self.inplanes, planes, stride, downsample, layer, 0)
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers += block(self.inplanes, planes, stride=1, downsample=None, layer=layer, block=i)
        return layers

    def BasicBlock(self, inplanes, planes, stride=1, downsample=None, layer=-1, block=-1):
        layer1 = Single_layer(nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        ), layer, block, begin_layer=True)
        layer2 = Single_layer(nn.Sequential(
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes)
        ), layer, block, downsample, nn.ReLU(inplace=True), end_layer=True)
        # at the end of each block out += residual and go through Single_layer.rest
        # at beginning of each block the input of Single_layer.residual should be the final result of last block
        return nn.ModuleList([layer1, layer2])

    def Bottleneck(self, inplanes, planes, stride=1, downsample=None, layer=-1, block=-1):
        layer1 = Single_layer(nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        ), layer, block, begin_layer=True)
        layer2 = Single_layer(nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        ), layer, block)
        layer3 = Single_layer(nn.Sequential(
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4)
        ), layer, block, downsample, nn.ReLU(inplace=True), end_layer=True)
        # at the end of each block out += residual and go through Single_layer.rest
        # at beginning of each block the input of Single_layer.residual should be the final result of last block
        return nn.ModuleList([layer1, layer2, layer3])

    def place_test_forward(self, x):
        start_inf = time.perf_counter()
        x = self.first(x)
        end_inf = time.perf_counter()
        self.inf_layer += 1
        self.inf_time += end_inf - start_inf
        for i, block in enumerate(self.blocklist):
            start_inf = time.perf_counter()
            if block.begin_layer:
                res = x
            x = block(x)
            if block.end_layer:
                if block.downsample is not None:
                    res = block.downsample(res)
                x += res
                x = block.rest(x)
            end_inf = time.perf_counter()
            self.inf_layer += 1
            self.inf_time += end_inf - start_inf
            if i == self.start_layer:
                start_exit = time.perf_counter()
                bof_exit = self.BoF_list[block.layer][block.block](x)
                x_exit = self.fc0_list[block.layer](bof_exit)  # no this exit layer right after pooling layer
                x_exit = self.fc1_list[block.layer](x_exit)
                end_exit = time.perf_counter()
                self.exit_count += 1
                self.exit_time += end_exit - start_exit
                ratio = self._calculate_max_activation(F.log_softmax(x_exit, dim=1, dtype=torch.double)).item() # * self.activation_threshold_list[i])
                if ratio >= math.log(self.beta):
                    self.num_early_exit_list[i] += 1
                    return i, x_exit
        self.original += 1
        self.inf_layer += 1
        start_inf = time.perf_counter()
        x = self.last(x)
        end_inf = time.perf_counter()
        self.inf_time += end_inf - start_inf
        return len(self.blocklist), x

    def place_forward(self, x):
        start_inf = time.perf_counter()
        x = self.first(x)
        end_inf = time.perf_counter()
        self.inf_layer += 1
        self.inf_time += end_inf - start_inf
        for i, block in enumerate(self.blocklist):
            start_inf = time.perf_counter()
            if block.begin_layer:
                res = x
            x = block(x)
            if block.end_layer:
                if block.downsample is not None:
                    res = block.downsample(res)
                x += res
                x = block.rest(x)
            end_inf = time.perf_counter()
            self.inf_layer += 1
            self.inf_time += end_inf - start_inf
            if i in self.place_layer:
                start_exit = time.perf_counter()
                bof_exit = self.BoF_list[block.layer][block.block](x)
                x_exit = self.fc0_list[block.layer](bof_exit)  # no this exit layer right after pooling layer
                x_exit = self.fc1_list[block.layer](x_exit)
                end_exit = time.perf_counter()
                self.exit_count += 1
                self.exit_time += end_exit - start_exit
                ratio = self._calculate_max_activation(F.log_softmax(x_exit, dim=1, dtype=torch.double)).item() # * self.activation_threshold_list[i])
                if ratio >= math.log(self.beta):
                    self.num_early_exit_list[i] += 1
                    return i, x_exit
        self.original += 1
        self.inf_layer += 1
        start_inf = time.perf_counter()
        x = self.last(x)
        end_inf = time.perf_counter()
        self.inf_time += end_inf - start_inf
        return len(self.blocklist), x

    def exits_forward(self, x):
        self.target_layer = self.start_layer  # the layer to begin enable exit
        predicted = False
        dvfs_in_this_inf = 0
        start_inf = time.perf_counter()
        x = self.first(x)
        end_inf = time.perf_counter()
        # self.inf_layer += 1
        self.inf_time += end_inf - start_inf
        for i, block in enumerate(self.blocklist):
            start_inf = time.perf_counter()
            if block.begin_layer:
                res = x
            x = block(x)
            if block.end_layer:
                if block.downsample is not None:
                    res = block.downsample(res)
                x += res
                x = block.rest(x)
            end_inf = time.perf_counter()
            self.inf_layer += 1
            self.inf_time += end_inf - start_inf
            if self.target_layer == i:
                start_exit = time.perf_counter()
                bof_exit = self.BoF_list[block.layer][block.block](x)
                x_exit = self.fc0_list[block.layer](bof_exit)  # no this exit layer right after pooling layer
                x_exit = self.fc1_list[block.layer](x_exit)
                end_exit = time.perf_counter()
                self.exit_count += 1
                self.exit_time += end_exit - start_exit
                ratio = self._calculate_max_activation(F.log_softmax(x_exit, dim=1, dtype=torch.double)).item() # * self.activation_threshold_list[i])
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
                    if i < len(self.blocklist) - 1:  # prediction except last two layer
                        start_pred = time.perf_counter()
                        if self.trained_pred:
                            jump = self.trained_prediction(x_exit, i)
                        else:
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
        self.original += 1
        # self.inf_layer += 1
        start_inf = time.perf_counter()
        x = self.last(x)
        end_inf = time.perf_counter()
        self.inf_time += end_inf - start_inf
        return len(self.blocklist), x, dvfs_in_this_inf

    def accuracy_forward(self, x):
        self.target_layer = self.start_layer  # the layer to count prediction accuracy
        start_inf = time.perf_counter()
        x = self.first(x)
        end_inf = time.perf_counter()
        self.inf_layer += 1
        self.inf_time += end_inf - start_inf
        for i, block in enumerate(self.blocklist):
            start_inf = time.perf_counter()
            if block.begin_layer:
                res = x
            x = block(x)
            if block.end_layer:
                if block.downsample is not None:
                    res = block.downsample(res)
                x += res
                x = block.rest(x)
            end_inf = time.perf_counter()
            self.inf_layer += 1
            self.inf_time += end_inf - start_inf
            if self.start_layer <= i:
                start_exit = time.perf_counter()
                bof_exit = self.BoF_list[block.layer][block.block](x)
                x_exit = self.fc0_list[block.layer](bof_exit)  # no this exit layer right after pooling layer
                x_exit = self.fc1_list[block.layer](x_exit)
                end_exit = time.perf_counter()
                self.exit_count += 1
                self.exit_time += end_exit - start_exit
                ratio = self._calculate_max_activation(F.log_softmax(x_exit, dim=1, dtype=torch.double)).item() # * self.activation_threshold_list[i])
                # print(i, self._calculate_max_activation(F.softmax(x_exit, dim=1)).item())
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
                    self.prediction_store.append(jump)
                    # print('jump', jump)
        self.original += 1
        self.jumpstep_store.append(i-self.target_layer)
        self.inf_layer += 1
        start_inf = time.perf_counter()
        x = self.last(x)
        end_inf = time.perf_counter()
        self.inf_time += end_inf - start_inf
        return len(self.blocklist) - 1, x

    def normal_forward( self, x ):
        x = self.first(x)
        # self.inf_layer += 1
        for block in self.blocklist:
            self.inf_layer += 1
            # if True: # self.mutinfo_forward:
            #     x = self.agn(x)
            if block.begin_layer:
                res = x
            x = block(x)
            if block.end_layer:
                if block.downsample is not None:
                    res = block.downsample(res)
                x += res
                x = block.rest(x)
        if self.mutinfo_forward:
            return {'last conv': x}
        x = self.last(x)
        # self.inf_layer += 1
        self.original += 1
        return x

    def forward_original( self, x ):
        x = self.first(x)
        for block in self.blocklist:
            # if True:
            # # if self.mutinfo_forward:
            #     x = self.agn(x)
            if block.begin_layer:
                res = x
            x = block(x)
            if block.end_layer:
                if block.downsample is not None:
                    res = block.downsample(res)
                x += res
                x = block.rest(x)
        if self.mutinfo_forward:
            return {'last conv': x}
        if self.distill:
            x = self.last[0](x)
            f5 = self.last[1](x)
            x = self.last[2](f5)
            return [f5], x
        else:
            x = self.last(x)
            return x

    def forward_exits( self, x ):
        output_list = []
        x = self.first(x)
        for block in self.blocklist:
            if block.begin_layer:
                res = x
            x = block(x)
            if block.end_layer:
                if block.downsample is not None:
                    res = block.downsample(res)
                x += res
                x = block.rest(x)
            x_exit = self.BoF_list[block.layer][block.block](x)
            x_exit = self.fc0_list[block.layer](x_exit)  # no this exit layer right after pooling layer
            x_exit = self.fc1_list[block.layer](x_exit)
            output_list.append(x_exit)
        x = self.last(x)
        output_list.append(x)
        return output_list

    def forward_place( self, x ):
        output_list = []
        x = self.first(x)
        for i, block in enumerate(self.blocklist):
            if block.begin_layer:
                res = x
            x = block(x)
            if block.end_layer:
                if block.downsample is not None:
                    res = block.downsample(res)
                x += res
                x = block.rest(x)
            if i in self.place_layer:
                x_exit = self.BoF_list[block.layer][block.block](x)
                x_exit = self.fc0_list[block.layer](x_exit)  # no this exit layer right after pooling layer
                x_exit = self.fc1_list[block.layer](x_exit)
                output_list.append(x_exit)
                if self.onlyexit and i == max(self.place_layer):
                    return output_list
        x = self.last(x)
        output_list.append(x)
        return output_list

    def transfer_copy(self, model, exitmodel):
        mystate_dict = self.state_dict()
        for name, parameter in model.state_dict().items():
            if name in mystate_dict.keys():
                if not exitmodel:
                    if '_list' in name: # don't load the exit layer of these model
                        continue
                mystate_dict[name].copy_(parameter)
            else:
                print('transfer copy: ignoring', name)
        self.load_state_dict(mystate_dict)

def resnet_exit(**kwargs):
    num_classes, depth, input_size = map(
        kwargs.get, ['num_classes', 'depth', 'input_size'])
    num_classes = num_classes or 10
    input_size = input_size or 32
    depth = depth or 34
    logging.info(f'num_classes {str(num_classes)}, depth {str(depth)}, input_size {str(input_size)}.')
    if depth == 18:
        return ResNet_exit(num_classes=num_classes,
                                block='BasicBlock', layers=[2, 2, 2, 2], input_size=input_size)
    if depth == 34:
        return ResNet_exit(num_classes=num_classes,
                                block='BasicBlock', layers=[3, 4, 6, 3], input_size=input_size)
    if depth == 50:
        return ResNet_exit(num_classes=num_classes,
                                block='Bottleneck', layers=[3, 4, 6, 3], input_size=input_size)
    if depth == 101:
        return ResNet_exit(num_classes=num_classes,
                                block='Bottleneck', layers=[3, 4, 23, 3], input_size=input_size)
    if depth == 152:
        return ResNet_exit(num_classes=num_classes,
                                block='Bottleneck', layers=[3, 8, 36, 3], input_size=input_size)
