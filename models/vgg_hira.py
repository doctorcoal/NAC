import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from .resnet_hira import ResNet_hira
from .bof_utils import LogisticConvBoF
import time
from .binarymodule import Quant_Conv2d, IRlinear
from .bof_utils_float import LogisticConvBoF as Bof_q
from .binaryfunction import T_quantize, T_quantize_activation

__all__ = ['vgg_hira', 'vgg_hira_quant']

class FPQuant(nn.Module):
    def __init__(self, bit):
        super(FPQuant, self).__init__()
        self.a_bit = bit
    
    def forward(self, x):
        return T_quantize(x, self.a_bit-1)

class VGG_hira(ResNet_hira):
    def __init__(self, num_classes=10, depth=19, input_size = 32):
        super(VGG_hira, self).__init__(num_classes=num_classes, input_size=input_size)
        # setting of exit
        del self.first
        del self.quant
        del self.dequant

        # setting of inverted residual blocks
        self.depth = depth-3
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
        # elif depth==8:
        #     self.blocks = [1, 1, 1, 1, 1]
        # building first layer
        self.blocklist = nn.ModuleList([])
        inplane = 3
        plane = 64
        self.first = None
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
        if self.litelast:
            self.last = nn.Sequential(
                # nn.MaxPool2d(2, 2),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
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
        self.blockdivide = [sum(self.blocks[0:i+1]) for i in range(len(self.blocks))]
        self.place_layer = [int((self.depth)/4-1), int((self.depth)/2-1), int((self.depth)*3/4-1)]
        self.Bof_layer_index = []
        for pl in self.place_layer:
            for j, i in enumerate(self.blockdivide):
                if i > pl:
                    self.Bof_layer_index.append(j)
                    break
        self.bof_in_channel_list = [64, 128, 256, 512, 512]
        print('hira exit layer index in this model:', self.place_layer, 'on group: ', self.Bof_layer_index)
        del self.BoF_list, self.fc0_list, self.fc1_list, self.exit_list
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
        self._initialize_weights()

    def normal_forward( self, x ):
        for block in self.blocklist:
            self.inf_layer += 1
            x = block(x)
        x = self.last(x)
        self.original += 1
        return x

    def place_forward(self, x):
        bof_index = 0
        for i, block in enumerate(self.blocklist):
            start_inf = time.perf_counter()
            x = block(x)
            end_inf = time.perf_counter()
            self.inf_layer += 1
            self.inf_time += end_inf - start_inf
            if i in self.place_layer:
                start_exit = time.perf_counter()
                bof_exit = self.BoF_list[bof_index](x)
                x_exit = self.fc0_list[bof_index](bof_exit)  # no this exit layer right after pooling layer
                x_exit = self.fc1_list[bof_index](x_exit)
                end_exit = time.perf_counter()
                self.exit_count += 1
                self.exit_time += end_exit - start_exit
                ratio = self._calculate_max_activation(F.log_softmax(x_exit, dim=1, dtype=torch.double)).item() # * self.activation_threshold_list[i])
                if ratio >= math.log(self.beta):
                    self.num_early_exit_list[bof_index] += 1
                    return bof_index, x_exit
                bof_index += 1
        self.original += 1
        start_inf = time.perf_counter()
        x = self.last(x)
        end_inf = time.perf_counter()
        self.inf_time += end_inf - start_inf
        return bof_index, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward_original( self, x ): # early feature also not good.
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
        output_list = []
        if self.distill:
            f5_list = []
        bof_index = 0
        for i, block in enumerate(self.blocklist):
            x = block(x)
            if i in self.place_layer:
                x_exit = self.BoF_list[bof_index](x)
                if self.distill:
                    f5_list.append(x_exit)
                x_exit = self.fc0_list[bof_index](x_exit)  # no this exit layer right after pooling layer
                x_exit = self.fc1_list[bof_index](x_exit)
                output_list.append(x_exit)
                bof_index += 1
        if self.distill:
            # for i in range(len(self.last[2])):
            #     x = self.last[2][i](x)
            #     if i == 5:
            #         f5_list.append(x)
            if self.litelast:
                f5 = self.last[0:2](x)
                x = self.last[2:](f5)
            else:
                f5 = self.last[0:3](x)
                x = self.last[3:](f5)
            f5_list.append(f5)
            output_list.append(x)
            return [f5_list], output_list
        else:
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
        self.load_state_dict(mystate_dict)

    def transfer_from_prune(self, model, exitmodel):
        for i, j in zip(self.blocklist, model.blocklist):
            for k in range(len(i)):
                if isinstance(i[k], nn.Conv2d):
                    i[k] = nn.Conv2d(j[k].in_channels, j[k].out_channels, i[k].kernel_size,
                                      i[k].stride, i[k].padding, i[k].dilation, j[k].groups, True if i[k].bias is not None else False)
                elif isinstance(i[k], nn.BatchNorm2d):
                    i[k] = nn.BatchNorm2d(j[k].num_features)
        # for i in range(len(self.last[2])):
        #     if isinstance(self.last[2][i], nn.Linear):
        #         self.last[2][i] = nn.Linear(model.last[2][i].in_features, model.last[2][i].out_features)
        self.last[2] = nn.Linear(model.last[2].in_features, model.last[2].out_features)
        self.BoF_list[0] = type(self.BoF_list[0])(self.blocklist[self.place_layer[0]][-2].num_features, model.BoF_list[0].codebook.out_channels
                                                  , avg_horizon=self.BoF_list[0].avg_horizon)
        self.BoF_list[1] = type(self.BoF_list[1])(self.blocklist[self.place_layer[1]][-2].num_features, model.BoF_list[1].codebook.out_channels
                                                  , avg_horizon=self.BoF_list[1].avg_horizon)
        self.BoF_list[2] = type(self.BoF_list[2])(self.blocklist[self.place_layer[2]][-2].num_features, model.BoF_list[2].codebook.out_channels
                                                  , avg_horizon=self.BoF_list[2].avg_horizon)
        self.fc0_list[0] = type(self.fc0_list[0])(model.fc0_list[0].in_features, model.fc0_list[0].out_features)
        self.fc0_list[1] = type(self.fc0_list[1])(model.fc0_list[1].in_features, model.fc0_list[1].out_features)
        self.fc0_list[2] = type(self.fc0_list[2])(model.fc0_list[2].in_features, model.fc0_list[2].out_features)
        self.fc1_list[0] = type(self.fc1_list[0])(model.fc1_list[0].in_features, model.fc1_list[0].out_features)
        self.fc1_list[1] = type(self.fc1_list[1])(model.fc1_list[1].in_features, model.fc1_list[1].out_features)
        self.fc1_list[2] = type(self.fc1_list[2])(model.fc1_list[2].in_features, model.fc1_list[2].out_features)
        self.transfer_copy(model, exitmodel)

class VGG_hira_quant(VGG_hira):
    def __init__(self, num_classes=10, depth=19, w=8, a=8, input_size = 32):
        self.activation_bit = a
        self.main_weight_bit = w
        self.bof_weight_bit = w
        self.fc_weight_bit = w
        print('model activation bit:', self.activation_bit)
        print('model main bit:', self.main_weight_bit)
        super(VGG_hira_quant, self).__init__(num_classes=num_classes, depth=depth, input_size=input_size)
        # building inverted residual blocks
        self.blocklist = nn.ModuleList([])
        inplane = 3
        plane = 64
        for j, i in enumerate(self.blocks):
            if j:
                self.blocklist.append(nn.Sequential(
                    nn.MaxPool2d(2, 2),
                    Quant_Conv2d(inplane, plane, kernel_size=3, stride=1, padding=1,
                            bias=True, bit=self.main_weight_bit),
                    nn.BatchNorm2d(plane),
                    nn.Sequential(nn.ReLU(),FPQuant(self.activation_bit))
                ))
            else:
                self.blocklist.append(nn.Sequential(
                    nn.Conv2d(inplane, plane, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.BatchNorm2d(plane),
                    nn.Sequential(nn.ReLU(),FPQuant(self.activation_bit))
                ))
            for _ in range(1, i):
                self.blocklist.append(nn.Sequential(
                    Quant_Conv2d(plane, plane, kernel_size=3, stride=1, padding=1,
                            bias=True, bit=self.main_weight_bit),
                    nn.BatchNorm2d(plane),
                    nn.Sequential(nn.ReLU(),FPQuant(self.activation_bit))
                ))
            inplane = plane
            if plane < 512:
                plane = plane*2
        if self.litelast:
            self.BoF_list = nn.ModuleList([Bof_q(self.bof_in_channel_list[i], 64, avg_horizon=2, 
                                                bit=self.bof_weight_bit) for i in self.Bof_layer_index])
            self.fc0_list = nn.ModuleList([IRlinear(256, 64, bit=self.fc_weight_bit) for _ in self.place_layer])
            self.fc1_list = nn.ModuleList([IRlinear(64, num_classes, bit=self.fc_weight_bit) for _ in self.place_layer])
        else:
            self.BoF_list = nn.ModuleList([Bof_q(self.bof_in_channel_list[i], 64, avg_horizon=7, 
                                                bit=self.bof_weight_bit) for i in self.Bof_layer_index])
            self.fc0_list = nn.ModuleList([IRlinear(64*7*7, 2048, bit=self.fc_weight_bit) for _ in self.place_layer])
            self.fc1_list = nn.ModuleList([IRlinear(2048, num_classes, bit=self.fc_weight_bit) for _ in self.place_layer])
        self.exit_list = [self.BoF_list, self.fc0_list, self.fc1_list]
        self.main_list = [self.first, self.blocklist, self.last]
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
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

    def transfer_from_prune(self, model, exitmodel):
        for i, j in zip(self.blocklist, model.blocklist):
            for k in range(len(i)):
                if isinstance(i[k], nn.Conv2d):
                    if isinstance(i[k], Quant_Conv2d):
                        i[k] = Quant_Conv2d(j[k].in_channels, j[k].out_channels, i[k].kernel_size,
                                      i[k].stride, i[k].padding, i[k].dilation, j[k].groups, True if i[k].bias is not None else False, bit=i[k].bit)
                    else:
                        i[k] = nn.Conv2d(j[k].in_channels, j[k].out_channels, i[k].kernel_size,
                                      i[k].stride, i[k].padding, i[k].dilation, j[k].groups, True if i[k].bias is not None else False)
                elif isinstance(i[k], nn.BatchNorm2d):
                    i[k] = nn.BatchNorm2d(j[k].num_features)
        # for i in range(len(self.last[2])):
        #     if isinstance(self.last[2][i], nn.Linear):
        #         self.last[2][i] = nn.Linear(model.last[2][i].in_features, model.last[2][i].out_features)
        #     elif isinstance(self.last[2][i], IRlinear):
        #         self.last[2][i] = IRlinear(model.last[2][i].in_features, model.last[2][i].out_features, bit=self.last[2][i].bit)
        self.last[2] = nn.Linear(model.last[2].in_features, model.last[2].out_features)
        self.BoF_list[0] = type(self.BoF_list[0])(self.blocklist[self.place_layer[0]][-2].num_features, model.BoF_list[0].codebook.out_channels
                                                  , bit=self.BoF_list[0].bit, avg_horizon=self.BoF_list[0].avg_horizon)
        self.BoF_list[1] = type(self.BoF_list[1])(self.blocklist[self.place_layer[1]][-2].num_features, model.BoF_list[1].codebook.out_channels
                                                  , bit=self.BoF_list[1].bit, avg_horizon=self.BoF_list[1].avg_horizon)
        self.BoF_list[2] = type(self.BoF_list[2])(self.blocklist[self.place_layer[2]][-2].num_features, model.BoF_list[2].codebook.out_channels
                                                  , bit=self.BoF_list[2].bit, avg_horizon=self.BoF_list[2].avg_horizon)
        self.fc0_list[0] = type(self.fc0_list[0])(model.fc0_list[0].in_features, model.fc0_list[0].out_features, bit=self.fc0_list[0].bit)
        self.fc0_list[1] = type(self.fc0_list[1])(model.fc0_list[1].in_features, model.fc0_list[1].out_features, bit=self.fc0_list[1].bit)
        self.fc0_list[2] = type(self.fc0_list[2])(model.fc0_list[2].in_features, model.fc0_list[2].out_features, bit=self.fc0_list[2].bit)
        self.fc1_list[0] = type(self.fc1_list[0])(model.fc1_list[0].in_features, model.fc1_list[0].out_features, bit=self.fc1_list[0].bit)
        self.fc1_list[1] = type(self.fc1_list[1])(model.fc1_list[1].in_features, model.fc1_list[1].out_features, bit=self.fc1_list[1].bit)
        self.fc1_list[2] = type(self.fc1_list[2])(model.fc1_list[2].in_features, model.fc1_list[2].out_features, bit=self.fc1_list[2].bit)
        self.transfer_copy(model, exitmodel)

def vgg_hira(**kwargs):
    num_classes, depth, input_size = map(
        kwargs.get, ['num_classes', 'depth', 'input_size'])
    num_classes = num_classes or 10
    input_size = input_size or 32
    depth = depth or 19
    print(f'num_classes {str(num_classes)}, depth {str(depth)}, input_size {str(input_size)}.')
    return VGG_hira(num_classes=num_classes, depth=depth, input_size=input_size)

def vgg_hira_quant(**kwargs):
    num_classes, depth, w, a, input_size = map(
        kwargs.get, ['num_classes', 'depth', 'w', 'a', 'input_size'])
    num_classes = num_classes or 10
    input_size = input_size or 32
    depth = depth or 19
    w = w or 8
    a = a or 8
    print(f'num_classes {str(num_classes)}, depth {str(depth)}, w{str(w)}, a{str(a)}, input_size {str(input_size)}.')
    return VGG_hira_quant(num_classes=num_classes, depth=depth, w=w, a=a, input_size=input_size)
