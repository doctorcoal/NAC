from .resnet_exit import ResNet_exit
from .binarymodule import Quant_Conv2d, IRlinear
from .binaryfunction import T_quantize, T_quantize_activation, quantize
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bof_utils_float import LogisticConvBoF
import logging

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

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

class Single_layer(nn.Module):
    def __init__(self, in_planes, planes, stride=1, layer=-1, block=-1, downsample = None, 
                 rest = None, begin_layer = False, end_layer = False, bit = 8, activation_bit = 8, kernel_size=3, padding=1):
        super().__init__()
        self.bit = bit
        self.seq = nn.ModuleList([Quant_Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False, bit = bit),
                                 nn.BatchNorm2d(planes)])
        self.k = torch.tensor([10]).float().cuda()
        self.t = torch.tensor([0.1]).float().cuda()
        self.block = block
        self.layer = layer
        self.downsample = downsample
        self.restfunc = rest
        self.begin_layer = begin_layer
        self.end_layer = end_layer
        self.activation_bit = activation_bit
        self.BNnoquant = True

    def rest(self, x):
        return T_quantize(self.restfunc(x), self.activation_bit-1)

    def forward(self, x):
        if self.begin_layer and self.layer == 0 and self.block == 0:
            x = T_quantize(x, self.activation_bit-1)
        # out = self.seq[1](self.seq[0](x)) # faster trainning but no BN quantimization
        out = self.my_bn1(self.seq[0](x))
        if not self.end_layer:
            out = T_quantize(F.relu(out, inplace=True), self.activation_bit-1)
        return out

    # def residual(self, x):
    #     if self.downsample is not None:
    #         return self.downsample(x)
    #     else:
    #         return x

    def my_bn1(self, input):
        global var, mean
        if self.training or self.BNnoquant:  # self model.train()/eval() will also infect its elements
            return self.seq[1](input)
            # y = input
            # y = y.permute(1, 0, 2, 3)  # NCHW -> CNHW
            # y = y.contiguous().view(y.shape[0], -1)  # CNHW -> C,NHW
            # mean = y.mean(1).detach()
            # var = y.var(1).detach()
            # self.seq[1].running_mean = self.seq[1].momentum * self.seq[1].running_mean + (1 - self.seq[1].momentum) * mean
            # self.seq[1].running_var = self.seq[1].momentum * self.seq[1].running_var + (1 - self.seq[1].momentum) * var
        else:  
            mean = self.seq[1].running_mean
            var = self.seq[1].running_var
        std = torch.sqrt(var + self.seq[1].eps)
        weight = self.seq[1].weight / std
        bias = self.seq[1].bias - weight * mean
        weight = weight.view(input.shape[1], 1)

        p3d = (0, input.shape[1] - 1)
        weight = F.pad(weight, p3d, 'constant', 0)
        for i in range(input.shape[1]):
            weight[i][i] = weight[i][0]
            if i > 0:
                weight[i][0] = 0
        weight = weight.view(input.shape[1], input.shape[1], 1, 1)
        # T_a = 3 * 3 * 64
        T_a = 3 * 3 * self.seq[0].out_channels
        bw = T_quantize(weight, self.bit-1)
        activation_q = T_quantize_activation(input, self.activation_bit-1, T_a)
        bb = T_quantize(bias, self.bit-1)
        out = F.conv2d(activation_q, bw, bb, stride=1, padding=0)
        return out


class ResNet_exit_quant(ResNet_exit):
    def __init__(self, num_classes=10, block='BasicBlock', layers=[3, 4, 6, 3], w=8, a=8, input_size = 32):
        self.activation_bit = a
        self.main_weight_bit = w
        self.bof_weight_bit = w
        self.fc_weight_bit = w
        logging.info('model activation bit: '+str(self.activation_bit))
        logging.info('model main bit: '+str(self.main_weight_bit))
        # this init correctly constructs quantmized CNN, which means __init__ do simpled copied code
        super(ResNet_exit_quant, self).__init__(num_classes, block, layers, input_size=input_size)
        self.BoF_list = nn.ModuleList([])
        for i, l in enumerate(layers):
            self.BoF_list.append(nn.ModuleList(l*[LogisticConvBoF(int(self.expansion*64*(2**i)), 64, avg_horizon=2 if self.cifar else 4, bit=self.bof_weight_bit)]))
        self.fc0_list = nn.ModuleList([IRlinear(256 if self.cifar else 1024, 64 if self.cifar else num_classes, bit=self.fc_weight_bit) for _ in layers])
        self.fc1_list = nn.ModuleList([(IRlinear(64, num_classes, bit=self.fc_weight_bit) if self.cifar else torch.nn.Identity()) for _ in layers])
        self.exit_list = [self.BoF_list, self.fc0_list, self.fc1_list]
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
                Quant_Conv2d(self.inplanes, planes * expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )
        layers = nn.ModuleList([])
        layers += block(self.inplanes, planes, stride, downsample, layer, 0)
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers += block(self.inplanes, planes, stride=1, downsample=None, layer=layer, block=i)
        return layers

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

    def BasicBlock(self, inplanes, planes, stride=1, downsample=None, layer=-1, block=-1):
        layer1 = Single_layer(inplanes, planes, stride, layer, block, None, None, begin_layer=True, bit = self.main_weight_bit, activation_bit=self.activation_bit)
        layer2 = Single_layer(planes, planes, 1, layer, block, downsample, nn.ReLU(inplace=True), end_layer=True, bit = self.main_weight_bit, activation_bit=self.activation_bit)
        # at the end of each block out += residual and go through Single_layer.rest
        # at beginning of each block the input of Single_layer.residual should be the final result of last block
        return nn.ModuleList([layer1, layer2])
    
    def Bottleneck(self, inplanes, planes, stride=1, downsample=None, layer=-1, block=-1):
        layer1 = Single_layer(inplanes, planes, 1, layer, block, None, None, begin_layer=True, 
                              bit = self.main_weight_bit, activation_bit=self.activation_bit, kernel_size=1, padding=0)
        layer2 = Single_layer(planes, planes, stride, layer, block, None, None, 
                              bit = self.main_weight_bit, activation_bit=self.activation_bit, kernel_size=3, padding=1)
        layer3 = Single_layer(planes, planes*4, 1, layer, block, downsample, nn.ReLU(inplace=True), end_layer=True, 
                              bit = self.main_weight_bit, activation_bit=self.activation_bit, kernel_size=1, padding=0)
        # at the end of each block out += residual and go through Single_layer.rest
        # at beginning of each block the input of Single_layer.residual should be the final result of last block
        return nn.ModuleList([layer1, layer2, layer3])

def resnet_exit_quant(**kwargs):
    num_classes, depth, w, a, input_size = map(
        kwargs.get, ['num_classes', 'depth', 'w', 'a', 'input_size'])
    num_classes = num_classes or 10
    input_size = input_size or 32
    depth = depth or 34
    w = w or 8
    a = a or 8
    logging.info(f'num_classes {str(num_classes)}, depth {str(depth)}, input_size {str(input_size)}.')
    if depth == 18:
        return ResNet_exit_quant(num_classes=num_classes,
                                block='BasicBlock', layers=[2, 2, 2, 2], w=w, a=a, input_size=input_size)
    if depth == 34:
        return ResNet_exit_quant(num_classes=num_classes,
                                block='BasicBlock', layers=[3, 4, 6, 3], w=w, a=a, input_size=input_size)
    if depth == 50:
        return ResNet_exit_quant(num_classes=num_classes,
                                block='Bottleneck', layers=[3, 4, 6, 3], w=w, a=a, input_size=input_size)
    if depth == 101:
        return ResNet_exit_quant(num_classes=num_classes,
                                block='Bottleneck', layers=[3, 4, 23, 3], w=w, a=a, input_size=input_size)
    if depth == 152:
        return ResNet_exit_quant(num_classes=num_classes,
                                block='Bottleneck', layers=[3, 8, 36, 3], w=w, a=a, input_size=input_size)
