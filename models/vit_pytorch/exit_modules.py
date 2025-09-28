import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit import Transformer
from .binarymodule import Quant_Conv2d, IRlinear
from .binaryfunction import T_quantize, T_quantize_activation, quantize

class LogisticConvBoF(nn.Module):
    def __init__(self, input_features, n_codewords, bit = 32, avg_horizon=2):
        super(LogisticConvBoF, self).__init__()
        self.input_features = input_features
        self.n_codewords = n_codewords
        self.bit = bit
        self.codebook = Quant_Conv2d(input_features, n_codewords, kernel_size=1, bias=True, bit=self.bit)

        self.a = nn.Parameter(torch.FloatTensor(data=[1]))
        self.c = nn.Parameter(torch.FloatTensor(data=[0]))
        self.avg_horizon = avg_horizon

    def forward(self, input, eps=5e-16 ):
        x = self.codebook(input)
        x = torch.tanh(x*self.a + self.c)
        x = (x + 1) / 2.0
        x = (x / (torch.sum(x, dim=1, keepdim=True)+ eps))
        x = F.adaptive_avg_pool2d(x, self.avg_horizon)
        x = x.reshape((x.size(0), -1))    
        return x

from .configuration_deit import DeiTConfig
from .modeling_deit import DeiTPooler, DeiTLayer, DeiTAttention

class highway_conv_normal(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.GELU(),
            nn.BatchNorm2d(in_features, eps=1e-5),
        )
    
    def forward(self,x,H,W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x

# Local perception head
class highway_conv1_1(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bit=32):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            Quant_Conv2d(in_features, hidden_features, 1, 1, 0, bias=True, bit=bit),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        
        self.proj = Quant_Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bit=bit)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            Quant_Conv2d(hidden_features, out_features, 1, 1, 0, bias=True, bit=bit),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def change_quant(self, w, a):
        self.conv1[0].change_quant(w, a)
        self.proj.change_quant(w, a)
        self.conv2[0].change_quant(w, a)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x0 = x
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x)
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1) + x0
        x = self.drop(x)
        return x

# Local perception head
class highway_conv2_1(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bit=32):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            Quant_Conv2d(in_features, hidden_features, 1, 1, 0, bias=True, bit=bit),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.conv2 = nn.Sequential(
            Quant_Conv2d(hidden_features, out_features, 1, 1, 0, bias=True, bit=bit),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)
        
    def change_quant(self, w, a):
        self.conv1[0].change_quant(w, a)
        self.conv2[0].change_quant(w, a)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        x0 = x
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1) + x0
        x = self.drop(x)
        return x

# Global aggregation head
class GlobalSparseAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,  sr_ratio=1 ,bit=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = IRlinear(dim, dim * 3, bias=qkv_bias, bit=bit)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = IRlinear(dim, dim, bit=bit)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr= sr_ratio
        if self.sr > 1:
            self.sampler = nn.AvgPool2d(1, sr_ratio)
        else:
            self.sampler = nn.Identity()
            self.upsample = nn.Identity()
            self.norm = nn.Identity()

    def change_quant(self, w, a):
        self.qkv.change_quant(w, a)
        self.proj.change_quant(w, a)        

    def forward(self, x, H:int, W:int):
        B, N, C = x.shape
        if self.sr > 1.:
            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.sampler(x)
            x = x.flatten(2).transpose(1, 2)
            
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,  sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    
    def forward(self,x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        return x

highway_classes = {
    "conv_normal": highway_conv_normal,
    "conv1_1": highway_conv1_1,
    "conv2_1": highway_conv2_1,
    "attention": GlobalSparseAttn,
    "self_attention" : DeiTAttention,
}

class ViTHighway(nn.Module):
    r'''
    A module to provide a shortcut from
    the output of one non-final DeiTLayer in DeiTEncoder to
    cross-entropy computation in DeiTForImageClassification
    '''

    def __init__(self, hidden_size, num_classes, dropout) -> None:
        super(ViTHighway, self).__init__()
        self.pooler = DeiTPooler(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.classifier = nn.Linear(hidden_size, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward(self, encoder_outputs):
        # Pooler
        pooler_input = encoder_outputs
        pooler_output = self.pooler(pooler_input)
        # 'return' pooler_output  

        # DeiTModel
        # dmodle_output = (pooler_input, pooler_output) + encoder_outputs[1:]
        # 'return' bmodel_output

        # Dropout and classification
        # pooled_output = dmodle_output[1]
        pooled_output = pooler_output
        

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits #, pooled_output


class DeiTHighway(nn.Module):
    def __init__(self, config: DeiTConfig) -> None:
        super(DeiTHighway, self).__init__()
        self.config = config
        self.layernorm = self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.cls_classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        self.distillation_classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()
        
    def forward(self, encoder_outputs):
        # sequence_output
        sequence_output = encoder_outputs
        sequence_output = self.layernorm(sequence_output)
        
        # logits, pooler_output
        cls_logits = self.cls_classifier(sequence_output[:, 0, :])
        distillation_logits = self.distillation_classifier(sequence_output[:, 1, :])
        logits = (cls_logits + distillation_logits) / 2
        pooled_output = sequence_output[:, 0:2, :]
        
        return logits #, pooled_output


class DeiTHighway_v2(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_r, backbone='vit', layer_type=f'conv1_1', weight=32):
        super(DeiTHighway_v2, self).__init__()
        self.weight = weight
        self.backbone = backbone
        self.layer_type = layer_type
        if "attention" in layer_type:
            sr_ratio = eval(layer_type[-1])
            self.mlp = GlobalSparseAttn(dim=hidden_size, sr_ratio=sr_ratio, bit=weight)
        # elif highway_type == 'self_attention':
        #     self.mlp = DeiTAttention(self.config)
        else:
            self.mlp = highway_classes[layer_type](hidden_size, bit=weight)

        self.pooler = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_r)

        if self.backbone == 'vit':
            self.classifier = IRlinear(hidden_size,
                                        num_classes, bit=weight) if num_classes > 0 else nn.Identity()
        elif self.backbone == 'deit':
            self.cls_classifier = IRlinear(hidden_size,
                                            num_classes, bit=weight) if num_classes > 0 else nn.Identity()
            self.distillation_classifier = IRlinear(hidden_size,
                                                     num_classes, bit=weight) if num_classes > 0 else nn.Identity()
        else:
            raise ValueError("Please select one of the backbones: ViT, DeiT")

    def change_quant(self, w, a):
        self.weight = w
        self.mlp.change_quant(w, a)
        if self.backbone == 'vit':
            self.classifier.change_quant(w, a)
        elif self.backbone == 'deit':
            self.cls_classifier.change_quant(w, a)
            self.distillation_classifier.change_quant(w, a)

    def forward(self, encoder_outputs):
        
        hidden_states = encoder_outputs
        cls_embeddings = hidden_states[:, 0, :]
        if self.backbone == 'deit':
            distillation_embeddings = hidden_states[:, 1, :]
            patch_embeddings = hidden_states[:, 2:, :]
        elif self.backbone == 'vit':
            patch_embeddings = hidden_states[:, 1:, :]
        if self.layer_type == 'self_attention':
            x = self.mlp(patch_embeddings)[0]
        else:
            h = w = int(math.sqrt(patch_embeddings.size()[1]))  # sequence_length
            x = self.mlp(patch_embeddings, h, w)
        hidden_states = x
        pooled_output = self.pooler(x.transpose(1, 2)).transpose(1, 2).squeeze(1)
        if self.backbone == 'vit':
            logits = self.classifier(pooled_output + cls_embeddings)
        elif self.backbone == 'deit': 
            cls_logits = self.cls_classifier(pooled_output + cls_embeddings)
            distillation_logits = self.distillation_classifier(pooled_output + distillation_embeddings)
            logits = (cls_logits + distillation_logits) / 2

        return logits #, hidden_states

class ViT_EE(nn.Module):
    # for ViT-EE
    def __init__(self, num_classes, dim, heads, dim_head, mlp_dim, dropout = 0., pool = 'cls'):
        super(ViT_EE, self).__init__()
        self.layer = Transformer(dim, 1, heads, dim_head, mlp_dim, dropout)
        self.mlp_head = ViTHighway(dim, num_classes, dropout)
        self.pool = pool
        
    def forward(self, x):
        x = self.layer(x)
        # x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)
