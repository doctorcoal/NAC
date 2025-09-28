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

__all__ = ['mobilenetV2_hira', 'mobilenetV2_hira_quant']


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def quant_conv_1x1_bn(inp, oup, bit=32):
    return nn.Sequential(
        Quant_Conv2d(inp, oup, 1, 1, 0, bias=False, bit=bit),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class FPQuant(nn.Module):
    def __init__(self, bit):
        super(FPQuant, self).__init__()
        self.a_bit = bit
    
    def forward(self, x):
        return T_quantize(x, self.a_bit-1)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.oup = oup
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual_q(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, bit=8, a_bit=8, firstblock=False):
        self.bit = bit
        self.a_bit = a_bit
        self.oup = oup
        super(InvertedResidual_q, self).__init__()
        assert stride in [1, 2]
        self.firstblock = firstblock
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Quant_Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False, bit=bit),
                nn.BatchNorm2d(hidden_dim),
                nn.Sequential(nn.ReLU6(inplace=True),
                FPQuant(a_bit)),
                # pw-linear
                Quant_Conv2d(hidden_dim, oup, 1, 1, 0, bias=False, bit=bit),
                nn.BatchNorm2d(oup),
                FPQuant(a_bit),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                Quant_Conv2d(inp, hidden_dim, 1, 1, 0, bias=False, bit=bit),
                nn.BatchNorm2d(hidden_dim),
                nn.Sequential(nn.ReLU6(inplace=True),
                FPQuant(a_bit)),
                # dw
                Quant_Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False, bit=bit),
                nn.BatchNorm2d(hidden_dim),
                nn.Sequential(nn.ReLU6(inplace=True),
                FPQuant(a_bit)),
                # pw-linear
                Quant_Conv2d(hidden_dim, oup, 1, 1, 0, bias=False, bit=bit),
                nn.BatchNorm2d(oup),
                FPQuant(a_bit),
            )

    def forward(self, x):
        if self.firstblock: 
            x = T_quantize(x, self.a_bit-1)
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2_hira(ResNet_hira):
    def __init__(self, num_classes=10, width_mult=1., input_size = 32):
        super(MobileNetV2_hira, self).__init__(num_classes=num_classes)
        self.depth = 17
        self.conv_in_singlelayer=2
        # setting of inverted residual blocks

        # building first layer
        self.input_size = input_size
        if input_size == 224:
            self.size_factor = 2
            self.linear1 = 1024
            self.linear2 = 256
        else:
            self.size_factor = 1
            self.linear1 = 256
            self.linear2 = 64
        self.cfgs = [
            # expand_ratio, channel, num_of_blocks, stride
            [1,  16, 1, 1],
            [6,  24, 2, self.size_factor],
            [6,  32, 3, self.size_factor],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        self.first = conv_3x3_bn(3, input_channel, self.size_factor)
        # building inverted residual blocks
        layers = []
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.blocklist = nn.ModuleList(layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.last = nn.Sequential(
            conv_1x1_bn(input_channel, output_channel),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(output_channel, num_classes)
        )

        # self.size_list = []
        # temp = self.input_size/self.size_factor
        # for i in self.cfgs:
        #     temp = int(temp/i[3])
        #     self.size_list.append(temp)
        # self.bof_channel = [int(self.input_size/self.size_factor*self.input_size/self.size_factor*16/i**2) for i in self.size_list]
        if self.size_factor == 2:
            self.bof_channel = [16,16,64,64,64,256,256]
        else:
            self.size_list = [self.input_size, self.input_size, self.input_size, self.input_size//2**1, 
                          self.input_size//2**1, self.input_size//2**2, self.input_size//2**2]
            self.bof_channel = [int(self.input_size*self.input_size*16/i**2) for i in self.size_list]
        self.blockdepth = sum([i[2] for i in self.cfgs]) + 2
        
        # exit location
        if num_classes <= 100:
            exit_interval = 2
            self.init_place_layer = list(range(exit_interval-1, # model.conv_in_singlelayer-1 for post exit placement; 0 for pre exit placement
                                        self.depth,exit_interval))
        else:
            exit_interval = 4
            self.init_place_layer = [3,7,11,15]
        self.place_layer = self.init_place_layer
        # self.place_layer = list(range(len(self.blocklist)))
        self.exit_num = self.depth
        self.Bof_layer_index = []
        self.BoF_list = nn.ModuleList([])
        self.fc0_list = nn.ModuleList([])
        for pl in range(len(self.blocklist)):
            if pl in self.place_layer:
                layer_now = 0
                for i, l in enumerate([i[2] for i in self.cfgs]):
                    layer_now += 1 * l
                    if pl < layer_now:
                        self.Bof_layer_index.append(i)
                        break
                self.BoF_list.append(nn.Sequential(
                    conv_1x1_bn(_make_divisible(self.cfgs[self.Bof_layer_index[-1]][1] * width_mult, 4 if width_mult == 0.1 else 8), output_channel),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                ))
                self.fc0_list.append(nn.Linear(output_channel, num_classes))
            else:
                self.BoF_list.append(nn.Identity())
                self.fc0_list.append(nn.Identity())
        print('hira exit layer index in this model:', self.place_layer, 'on group: ', self.Bof_layer_index)
        # self.BoF_list = nn.ModuleList([LogisticConvBoF(_make_divisible(self.cfgs[i][1] * width_mult, 4 if width_mult == 0.1 else 8), self.bof_channel[i],  
        #                                                avg_horizon=int(math.sqrt(self.linear1/self.bof_channel[i]))) for i in self.Bof_layer_index])
        # self.fc0_list = nn.ModuleList([nn.Linear(self.linear1, self.linear2) for _ in self.place_layer])
        # self.fc1_list = nn.ModuleList([nn.Linear(self.linear2, num_classes) for _ in self.place_layer])
        self.fc1_list = nn.ModuleList([nn.Identity() for _ in range(len(self.blocklist))])
        self.exit_list = [self.BoF_list, self.fc0_list, self.fc1_list]
        self.main_list = [self.first, self.blocklist, self.last]
        # print(self.exit_list)

        self._initialize_weights()

    def normal_forward( self, x ):
        x = self.first(x)
        self.inf_layer += 1
        for block in self.blocklist:
            self.inf_layer += 1
            x = block(x)
        x = self.last(x)
        self.inf_layer += 1
        self.original += 1
        return x

    def place_forward(self, x):
        start_inf = time.perf_counter()
        x = self.first(x)
        end_inf = time.perf_counter()
        self.inf_layer += 1
        self.inf_time += end_inf - start_inf
        for i, block in enumerate(self.blocklist):
            start_inf = time.perf_counter()
            x = block(x)
            end_inf = time.perf_counter()
            self.inf_layer += 1
            if i in self.place_layer:
                start_exit = time.perf_counter()
                bof_exit = self.BoF_list[i](x)
                x_exit = self.fc0_list[i](bof_exit)  # no this exit layer right after pooling layer
                x_exit = self.fc1_list[i](x_exit)
                end_exit = time.perf_counter()
                self.exit_count += 1
                self.exit_time += end_exit - start_exit
                # print(F.softmax(x_exit))
                # print(torch.max(F.softmax(x_exit)))
                ratio = self._calculate_max_activation(F.log_softmax(x_exit, dim=1, dtype=torch.double)).item() # * self.activation_threshold_list[i])
                if ratio >= math.log(self.beta):
                    # print(math.exp(ratio))
                    self.num_early_exit_list[i] += 1
                    return i, x_exit
        self.original += 1
        self.inf_layer += 1
        start_inf = time.perf_counter()
        x = self.last(x)
        # print(torch.max(F.softmax(x)))
        end_inf = time.perf_counter()
        self.inf_time += end_inf - start_inf
        return len(self.blocklist), x

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

    def forward_original( self, x ):
        x = self.first(x)
        for block in self.blocklist:
            x = block(x)
        if self.distill:
            x = self.last[0](x)
            x = self.last[1](x)
            f5 = self.last[2](x)
            x = self.last[3](f5)
            return [f5], x
        else:
            x = self.last(x)
            return x
    
    def forward_place( self, x):
        return self.forward_exits(x)
    
    def forward_exits( self, x ):
        output_list = []
        if self.distill:
            f5_list = []
        x = self.first(x)
        for i, block in enumerate(self.blocklist):
            x = block(x)
            if i in self.place_layer:
                x_exit = self.BoF_list[i](x)
                if self.distill:
                    f5_list.append(x_exit)
                x_exit = self.fc0_list[i](x_exit)  # no this exit layer right after pooling layer
                x_exit = self.fc1_list[i](x_exit)
                output_list.append(x_exit)
                if self.onlyexit and i == max(self.place_layer):
                    return output_list
        if self.distill:
            x = self.last[0](x)
            x = self.last[1](x)
            f5 = self.last[2](x)
            f5_list.append(f5)
            x = self.last[3](f5)
            output_list.append(x)
            return [f5_list], output_list
        else:
            x = self.last(x)
            output_list.append(x)
            return output_list
        
    def transfer_copy(self, model, exitmodel):
        mystate_dict = self.state_dict()
        # print(model.fc0_list[0].weight.shape)
        # print(self.fc0_list[0].weight.shape)
        # print(mystate_dict['fc0_list.0.weight'].shape)
        for name, parameter in model.state_dict().items():
            if name in mystate_dict.keys():
                if not exitmodel:
                    if '_list' in name: # don't load the exit layer of these model
                        continue
                mystate_dict[name].copy_(parameter)
        self.load_state_dict(mystate_dict)

class MobileNetV2_hira_quant(MobileNetV2_hira):
    def __init__(self, num_classes=10, width_mult=1.0, w=8, a=8, input_size = 32):
        self.activation_bit = a
        self.main_weight_bit = w
        self.bof_weight_bit = w
        self.fc_weight_bit = w
        print('model activation bit:', self.activation_bit)
        print('model main bit:', self.main_weight_bit)
        super(MobileNetV2_hira_quant, self).__init__(num_classes=num_classes, width_mult=width_mult, input_size=input_size)
        # building inverted residual blocks
        layers = []
        block = InvertedResidual_q
        firstblock = True
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, self.main_weight_bit, 
                                    self.activation_bit, firstblock=firstblock))
                firstblock = False
                input_channel = output_channel
        del self.blocklist, self.BoF_list, self.fc0_list
        self.blocklist = nn.ModuleList(layers)
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.Bof_layer_index = []
        self.BoF_list = nn.ModuleList([])
        self.fc0_list = nn.ModuleList([])
        for pl in range(len(self.blocklist)):
            if pl in self.place_layer:
                layer_now = 0
                for i, l in enumerate([i[2] for i in self.cfgs]):
                    layer_now += 1 * l
                    if pl < layer_now:
                        self.Bof_layer_index.append(i)
                        break
                self.BoF_list.append(nn.Sequential(
                    quant_conv_1x1_bn(_make_divisible(self.cfgs[self.Bof_layer_index[-1]][1] * width_mult, 4 if width_mult == 0.1 else 8), output_channel, bit=self.bof_weight_bit),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                ))
                self.fc0_list.append(IRlinear(output_channel, num_classes, bit=self.fc_weight_bit))
            else:
                self.BoF_list.append(nn.Identity())
                self.fc0_list.append(nn.Identity())
        self.fc1_list = nn.ModuleList([nn.Identity() for _ in range(len(self.blocklist))])
        self.exit_list = [self.BoF_list, self.fc0_list, self.fc1_list]
        self.main_list = [self.first, self.blocklist, self.last]
        self._initialize_weights()

    def change_quant(self, w, a):
        self.activation_bit = a
        self.main_weight_bit = w
        self.bof_weight_bit = w
        self.fc_weight_bit = w
        for i in self.blocklist:
            # print('change: ', i)
            i.bit = w
            i.a_bit = a
            for j in i.conv:
                if type(j) is Quant_Conv2d:
                    # print('    change Quant_Conv2d: ', j)
                    j.bit = w
                elif type(j) is FPQuant:
                    # print('    change FPQuant: ', j)
                    j.a_bit = a
                elif type(j) is nn.Sequential:
                    for k in j:
                        if type(k) is FPQuant:
                            # print('    change FPQuant: ', k)
                            k.a_bit = a
        for i in self.BoF_list:
            # print('    change quant_conv: ', i)
            if type(i) is not nn.Identity:
                i[0].bit = w
        for i in self.fc0_list:
            # print('    change IRlinear: ', i)
            if type(i) is not nn.Identity:
                i.bit = w

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

def mobilenetV2_hira(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    num_classes, width, input_size = map(
        kwargs.get, ['num_classes', 'width', 'input_size'])
    num_classes = num_classes or 10
    width = width or 1.0
    print(f'num_classes {str(num_classes)}, width {str(width)}, input_size {str(input_size)}.')
    return MobileNetV2_hira(num_classes=num_classes, width_mult=width, input_size=input_size)

def mobilenetV2_hira_quant(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    num_classes, width, w, a, input_size = map(
        kwargs.get, ['num_classes', 'width', 'w', 'a', 'input_size'])
    num_classes = num_classes or 10
    width = width or 1.0
    w = w or 8
    a = a or 8
    print(f'num_classes {str(num_classes)}, width {str(width)}, w{str(w)}, a{str(a)}, input_size {str(input_size)}.')
    return MobileNetV2_hira_quant(num_classes=num_classes, width_mult=width, w=w, a=a, input_size=input_size)
