from .vgg_hira import *
from .vgg_exit import *
from .resnet_exit import *
from .resnet_exit_quant import *
from .resnet_hira import * # special use, divide unified exit layer
from .mobilenetv2_hira import *
from .resnet_hira_quant import *
from .vit_pytorch.distill_comp import DeiT
# and delete all layers thet hira don't require

total_round = 8
