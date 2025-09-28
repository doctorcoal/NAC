from .resnet_exit import ResNet_exit
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bof_utils import LogisticConvBoF
import time

EXIT_NUM=3

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


class ResNet_hira(ResNet_exit):
    def __init__(self, num_classes=10, block='BasicBlock', layers=[3, 4, 6, 3], input_size = 32):
        super(ResNet_hira, self).__init__(num_classes, block, layers, EXIT_NUM, input_size=input_size)
        self.name = "resnet_hira"
        self.Bof_layer_index = []
        self.place_layer = [int((self.depth)/4-1), int((self.depth)/2-1), int((self.depth)*3/4-1)]
        for pl in self.place_layer:
            layer_now = 0
            for i, l in enumerate(layers):
                layer_now += self.conv_in_singlelayer * l
                if pl < layer_now:
                    self.Bof_layer_index.append(i)
                    break
        print('hira exit layer index in this model:', self.place_layer, 'on group: ', self.Bof_layer_index)
        self.BoF_list = nn.ModuleList([LogisticConvBoF(int(self.expansion*64*(2**i)), 64, avg_horizon=2 if self.cifar else 4) for i in self.Bof_layer_index])
        self.fc0_list = nn.ModuleList([nn.Linear(256 if self.cifar else 1024, 64 if self.cifar else num_classes) for _ in self.place_layer])
        self.fc1_list = nn.ModuleList([(nn.Linear(64, num_classes) if self.cifar else torch.nn.Identity()) for _ in self.place_layer])
        self.exit_list = [self.BoF_list, self.fc0_list, self.fc1_list]
        self.main_list = [self.first, self.blocklist, self.last]
        self.eval_mode = None
        init_model(self)

    def transfer_from_fullexit(self, model):
        # if not isinstance(model, ResNet_train):
        #     raise NotImplementedError
        mystate_dict = self.state_dict()
        transfer_from = ["BoF_list.1.0", "BoF_list.2.0", "BoF_list.2.4","fc0_list.1", "fc0_list.2",
                          "fc1_list.1", "fc1_list.2"]
        transfer_to = [["BoF_list.0"], ["BoF_list.1"], ["BoF_list.2"],["fc0_list.0"], ["fc0_list.1", "fc0_list.2"],
                       ["fc1_list.0"], ["fc1_list.1", "fc1_list.2"]]
        for name, parameter in model.state_dict().items():
            if "BoF_list" in name or "fc0_list" in name or "fc1_list" in name or "exit_list" in name:
                for i in range(len(transfer_from)):
                    if transfer_from[i] in name:
                        for to_name in transfer_to[i]:
                            mystate_dict[to_name + name[len(transfer_from[i]):]].copy_(parameter)
            elif name in mystate_dict.keys():
                mystate_dict[name].copy_(parameter)
        self.load_state_dict(mystate_dict)

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
            i.seq[0] = type(i.seq[0])(j.seq[0].in_channels, j.seq[0].out_channels, i.seq[0].kernel_size,
                                      i.seq[0].stride, i.seq[0].padding, i.seq[0].dilation, i.seq[0].groups, i.seq[0].bias)
            i.seq[1] = type(i.seq[1])(j.seq[1].num_features)
            if i.downsample:
                i.downsample[0] = type(i.downsample[0])(j.downsample[0].in_channels, j.downsample[0].out_channels, i.downsample[0].kernel_size,
                                      i.downsample[0].stride, i.downsample[0].padding, i.downsample[0].dilation,
                                        i.downsample[0].groups, i.downsample[0].bias)
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
        
    def place_forward(self, x):
        start_inf = time.perf_counter()
        x = self.first(x)
        end_inf = time.perf_counter()
        # self.inf_layer += 1
        self.inf_time += end_inf - start_inf
        bof_index = 0
        for i, block in enumerate(self.blocklist):
            start_inf = time.perf_counter()
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
                    if self.mutinfo_forward:
                        return {'last conv': bof_exit}
                    return bof_index, x_exit
                bof_index += 1
        self.original += 1
        # self.inf_layer += 1
        start_inf = time.perf_counter()
        x = self.last(x)
        end_inf = time.perf_counter()
        self.inf_time += end_inf - start_inf
        return bof_index, x

    def forward_exits( self, x ):
        output_list = []
        if self.distill:
            f5_list = []
        x = self.first(x)
        bof_index = 0
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
                x_exit = self.BoF_list[bof_index](x)
                if self.distill:
                    f5_list.append(x_exit)
                x_exit = self.fc0_list[bof_index](x_exit)  # no this exit layer right after pooling layer
                x_exit = self.fc1_list[bof_index](x_exit)
                output_list.append(x_exit)
                bof_index += 1
        # if self.mutinfo_forward:
        #     return {'last conv': x}
        if self.distill:
            x = self.last[0](x)
            f5 = self.last[1](x)
            f5_list.append(f5)
            x = self.last[2](f5)
            output_list.append(x)
            return [f5_list], output_list
        else:
            x = self.last(x)
            output_list.append(x)
            return output_list
    
def resnet_hira(**kwargs):
    num_classes, depth, input_size = map(
        kwargs.get, ['num_classes', 'depth', 'input_size'])
    num_classes = num_classes or 10
    input_size = input_size or 32
    depth = depth or 34
    print(f'num_classes {str(num_classes)}, depth {str(depth)}, input_size {str(input_size)}.')
    if depth >= 50:
        block='Bottleneck'
    else:
        block='BasicBlock'
    layers = {10:[1, 1, 1, 1], 12:[1, 1, 2, 1], 14:[1, 2, 2, 1], 16:[1, 2, 2, 2], 
              18:[2, 2, 2, 2], 20:[2, 2, 3, 2], 22:[2, 2, 4, 2], 24:[2, 3, 4, 2], 
              26:[2, 3, 5, 2], 28:[2, 3, 6, 2], 30:[2, 4, 6, 2], 32:[2, 4, 6, 3],
              34:[3, 4, 6, 3], 50:[3, 4, 6, 3], 101:[3, 4, 23, 3], 152:[3, 8, 36, 3]}
    return ResNet_hira(num_classes=num_classes, block=block, layers=layers[depth],input_size=input_size)
