from .binarymodule import Quant_Conv2d, IRlinear
from .resnet_exit_quant import Single_layer, init_model
from .resnet_hira import ResNet_hira
from .bof_utils_float import LogisticConvBoF as Bof_q
import torch.nn as nn

class ResNet_quant_hira(ResNet_hira):
    def __init__(self, num_classes=10, block='BasicBlock', layers=[3, 4, 6, 3], w=8, a=8, input_size = 32):
        self.activation_bit = a
        self.main_weight_bit = w
        self.bof_weight_bit = w
        self.fc_weight_bit = w
        print('model activation bit:', self.activation_bit)
        print('model main bit:', self.main_weight_bit)
        super(ResNet_quant_hira, self).__init__(num_classes=num_classes, block=block, layers=layers, input_size=input_size)
        init_model(self)

    def transfer_from_prune(self, model, exitmodel):
        for i, j in zip(self.blocklist, model.blocklist):
            i.seq[0] = type(i.seq[0])(j.seq[0].in_channels, j.seq[0].out_channels, i.seq[0].kernel_size,
                                      i.seq[0].stride, i.seq[0].padding, i.seq[0].dilation, i.seq[0].groups, i.seq[0].bias, bit=i.seq[0].bit)
            i.seq[1] = type(i.seq[1])(j.seq[1].num_features)
            if i.downsample:
                i.downsample[0] = type(i.downsample[0])(j.downsample[0].in_channels, j.downsample[0].out_channels, i.downsample[0].kernel_size,
                                      i.downsample[0].stride, i.downsample[0].padding, i.downsample[0].dilation,
                                        i.downsample[0].groups, i.downsample[0].bias, bit=i.downsample[0].bit)
                i.downsample[1] = type(i.downsample[1])(j.downsample[1].num_features)
        self.first[0] = type(self.first[0])(model.first[0].in_channels, model.first[0].out_channels, self.first[0].kernel_size,
                                      self.first[0].stride, self.first[0].padding, self.first[0].dilation,
                                        self.first[0].groups, self.first[0].bias)
        self.first[1] = type(self.first[1])(model.first[1].num_features)
        self.last[2] = type(self.last[2])(model.last[2].in_features, model.last[2].out_features)
        self.BoF_list[0] = type(self.BoF_list[0])(self.blocklist[self.place_layer[0]].seq[0].out_channels, model.BoF_list[0].codebook.out_channels
                                                  , avg_horizon=self.BoF_list[0].avg_horizon)
        self.BoF_list[1] = type(self.BoF_list[1])(self.blocklist[self.place_layer[1]].seq[0].out_channels, model.BoF_list[1].codebook.out_channels
                                                  , avg_horizon=self.BoF_list[1].avg_horizon)
        self.BoF_list[2] = type(self.BoF_list[2])(self.blocklist[self.place_layer[2]].seq[0].out_channels, model.BoF_list[2].codebook.out_channels
                                                  , avg_horizon=self.BoF_list[2].avg_horizon)
        self.fc0_list[0] = type(self.fc0_list[0])(model.fc0_list[0].in_features, model.fc0_list[0].out_features)
        self.fc0_list[1] = type(self.fc0_list[1])(model.fc0_list[1].in_features, model.fc0_list[1].out_features)
        self.fc0_list[2] = type(self.fc0_list[2])(model.fc0_list[2].in_features, model.fc0_list[2].out_features)
        if self.cifar:
            self.fc1_list[0] = type(self.fc1_list[0])(model.fc1_list[0].in_features, model.fc1_list[0].out_features)
            self.fc1_list[1] = type(self.fc1_list[1])(model.fc1_list[1].in_features, model.fc1_list[1].out_features)
            self.fc1_list[2] = type(self.fc1_list[2])(model.fc1_list[2].in_features, model.fc1_list[2].out_features)
        self.transfer_copy(model, exitmodel)

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
                          kernel_size=1, stride=stride, bias=False, bit=self.main_weight_bit),
                nn.BatchNorm2d(planes * expansion),
            )
        layers = nn.ModuleList([])
        layers += block(self.inplanes, planes, stride, downsample, layer, 0)
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers += block(self.inplanes, planes, stride=1, downsample=None, layer=layer, block=i)
        return layers
    
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

class ResNet_hira_quant(ResNet_quant_hira):
    def __init__(self, num_classes=10, block='BasicBlock', layers=[3, 4, 6, 3], w=8, a=8, input_size = 32):
        super(ResNet_hira_quant, self).__init__(num_classes=num_classes, block=block, layers=layers, w=w, a=a, input_size=input_size)
        self.BoF_list = nn.ModuleList([Bof_q(int(self.expansion*64*(2**i)), 64, avg_horizon=2 if self.cifar else 4, bit=self.bof_weight_bit) for i in self.Bof_layer_index])
        self.fc0_list = nn.ModuleList([IRlinear(256 if self.cifar else 1024, 64 if self.cifar else num_classes, bit=self.fc_weight_bit) for _ in self.place_layer])
        self.fc1_list = nn.ModuleList([(IRlinear(64, num_classes, bit=self.fc_weight_bit) if self.cifar else nn.Identity()) for _ in self.place_layer])
        self.exit_list = [self.BoF_list, self.fc0_list, self.fc1_list]
        init_model(self)
# FIRST MODEL: only CNN quant, quant_hira
# inherit from Hira_train
# use the Single layer of resnet_exit_quant to reconstruct the CNN layers (add parameter copy function)
# SECOND MODEL: CNN + EXIT QUANT, hira_quant
# inherit from hira_PE
# replace self.BoF_list[0],[1],[2] with Quant_Conv2d from binarymodule (add parameter copy function) 
    def transfer_from_prune(self, model, exitmodel):
        for i, j in zip(self.blocklist, model.blocklist):
            i.seq[0] = type(i.seq[0])(j.seq[0].in_channels, j.seq[0].out_channels, i.seq[0].kernel_size,
                                      i.seq[0].stride, i.seq[0].padding, i.seq[0].dilation, i.seq[0].groups, i.seq[0].bias, bit=i.seq[0].bit)
            i.seq[1] = type(i.seq[1])(j.seq[1].num_features)
            if i.downsample:
                i.downsample[0] = type(i.downsample[0])(j.downsample[0].in_channels, j.downsample[0].out_channels, i.downsample[0].kernel_size,
                                      i.downsample[0].stride, i.downsample[0].padding, i.downsample[0].dilation,
                                        i.downsample[0].groups, i.downsample[0].bias, bit=i.downsample[0].bit)
                i.downsample[1] = type(i.downsample[1])(j.downsample[1].num_features)
        self.first[0] = type(self.first[0])(model.first[0].in_channels, model.first[0].out_channels, self.first[0].kernel_size,
                                      self.first[0].stride, self.first[0].padding, self.first[0].dilation,
                                        self.first[0].groups, self.first[0].bias)
        self.first[1] = type(self.first[1])(model.first[1].num_features)
        self.last[2] = type(self.last[2])(model.last[2].in_features, model.last[2].out_features)
        self.BoF_list[0] = type(self.BoF_list[0])(self.blocklist[self.place_layer[0]].seq[0].out_channels, model.BoF_list[0].codebook.out_channels
                                                  , bit=self.BoF_list[0].bit, avg_horizon=self.BoF_list[0].avg_horizon)
        self.BoF_list[1] = type(self.BoF_list[1])(self.blocklist[self.place_layer[1]].seq[0].out_channels, model.BoF_list[1].codebook.out_channels
                                                  , bit=self.BoF_list[1].bit, avg_horizon=self.BoF_list[1].avg_horizon)
        self.BoF_list[2] = type(self.BoF_list[2])(self.blocklist[self.place_layer[2]].seq[0].out_channels, model.BoF_list[2].codebook.out_channels
                                                  , bit=self.BoF_list[2].bit, avg_horizon=self.BoF_list[2].avg_horizon)
        self.fc0_list[0] = type(self.fc0_list[0])(model.fc0_list[0].in_features, model.fc0_list[0].out_features, bit=self.fc0_list[0].bit)
        self.fc0_list[1] = type(self.fc0_list[1])(model.fc0_list[1].in_features, model.fc0_list[1].out_features, bit=self.fc0_list[1].bit)
        self.fc0_list[2] = type(self.fc0_list[2])(model.fc0_list[2].in_features, model.fc0_list[2].out_features, bit=self.fc0_list[2].bit)
        if self.cifar:
            self.fc1_list[0] = type(self.fc1_list[0])(model.fc1_list[0].in_features, model.fc1_list[0].out_features, bit=self.fc1_list[0].bit)
            self.fc1_list[1] = type(self.fc1_list[1])(model.fc1_list[1].in_features, model.fc1_list[1].out_features, bit=self.fc1_list[1].bit)
            self.fc1_list[2] = type(self.fc1_list[2])(model.fc1_list[2].in_features, model.fc1_list[2].out_features, bit=self.fc1_list[2].bit)
        self.transfer_copy(model, exitmodel)

def resnet_quant_hira(**kwargs):
    num_classes, depth, w, a, input_size = map(
        kwargs.get, ['num_classes', 'depth', 'w', 'a', 'input_size'])
    num_classes = num_classes or 10
    input_size = input_size or 32
    depth = depth or 34
    w = w or 8
    a = a or 8
    print(f'num_classes {str(num_classes)}, depth {str(depth)}, w{str(w)}, a{str(a)}, input_size {str(input_size)}.')
    if depth >= 50:
        block='Bottleneck'
    else:
        block='BasicBlock'
    layers = {10:[1, 1, 1, 1], 12:[1, 1, 2, 1], 14:[1, 2, 2, 1], 16:[1, 2, 2, 2], 
              18:[2, 2, 2, 2], 20:[2, 2, 3, 2], 22:[2, 2, 4, 2], 24:[2, 3, 4, 2], 
              26:[2, 3, 5, 2], 28:[2, 3, 6, 2], 30:[2, 4, 6, 2], 32:[2, 4, 6, 3],
              34:[3, 4, 6, 3], 50:[3, 4, 6, 3], 101:[3, 4, 23, 3], 152:[3, 8, 36, 3]}
    return ResNet_quant_hira(num_classes=num_classes, block=block, layers=layers[depth], w=w, a=a,input_size=input_size)

def resnet_hira_quant(**kwargs):
    num_classes, depth, w, a, input_size = map(
        kwargs.get, ['num_classes', 'depth', 'w', 'a', 'input_size'])
    num_classes = num_classes or 10
    input_size = input_size or 32
    depth = depth or 34
    w = w or 8
    a = a or 8
    print(f'num_classes {str(num_classes)}, depth {str(depth)}, w{str(w)}, a{str(a)}, input_size {str(input_size)}.')
    if depth >= 50:
        block='Bottleneck'
    else:
        block='BasicBlock'
    layers = {10:[1, 1, 1, 1], 12:[1, 1, 2, 1], 14:[1, 2, 2, 1], 16:[1, 2, 2, 2], 
              18:[2, 2, 2, 2], 20:[2, 2, 3, 2], 22:[2, 2, 4, 2], 24:[2, 3, 4, 2], 
              26:[2, 3, 5, 2], 28:[2, 3, 6, 2], 30:[2, 4, 6, 2], 32:[2, 4, 6, 3],
              34:[3, 4, 6, 3], 50:[3, 4, 6, 3], 101:[3, 4, 23, 3], 152:[3, 8, 36, 3]}
    return ResNet_hira_quant(num_classes=num_classes, block=block, layers=layers[depth], w=w, a=a,input_size=input_size)
