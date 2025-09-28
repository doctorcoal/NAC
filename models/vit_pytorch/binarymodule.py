import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from .binaryfunction import T_quantize, BinaryQuantize
import torch

class Quant_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bit = 8):
        super(Quant_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).float().cpu()
        self.t = torch.tensor([0.1]).float().cpu()
        self.bit = bit
        self.bw = 0

    def change_quant(self, w, a):
        self.bit = w

    def forward(self, input):
        if self.bit >= 31:
            output = F.conv2d(input, self.weight, self.bias,
                self.stride, self.padding,
                self.dilation, self.groups)
            return output
        if self.bit == 1:
            w = self.weight
            bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
            bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
            bw = BinaryQuantize().apply(bw, self.k, self.t)
            bb = None
        else:
            bw = T_quantize(self.weight, self.bit-1)
            bb = T_quantize(self.bias, self.bit-1) if (self.bias!=None) else None
        
        output = F.conv2d(input, bw, bb,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output

class IRlinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, bit = 8) -> None:
        super(IRlinear, self).__init__(in_features, out_features, bias)
        self.k = torch.tensor([10]).float().cpu()
        self.t = torch.tensor([0.1]).float().cpu()
        self.bit = bit
        
    def change_quant(self, w, a):
        self.bit = w
        
    def forward(self, input):
        if self.bit >= 31:
            output = F.linear(input, self.weight, self.bias)
            return output
        if self.bit == 1:
            w = self.weight
            bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1)
            bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1)
            bw = BinaryQuantize().apply(bw, self.k, self.t)
            bb = None
        else:
            bw = T_quantize(self.weight, self.bit-1)
            bb = T_quantize(self.bias, self.bit-1) if (self.bias!=None) else None
        # activation
        output = F.linear(input, bw, bb)
        return output
