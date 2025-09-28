from .Comp_Prototype import Comp_Prototype
from .vit import *
from .exit_modules import *
import torch.nn.functional as F
from .binarymodule import Quant_Conv2d, IRlinear
from .binaryfunction import T_quantize, T_quantize_activation, quantize
import math
import logging

def exists(val):
    return val is not None

def l_f(module):
    # linear flops
    return 2 * module.in_features * module.out_features

def c_f(module, h, w=None):
    # conv flops
    if not w:
        w = h
    return 2 * module.in_channels * module.out_channels * h * w
# the backbone model is based on google research:
# the flops computation part is based on deepmind: Training Compute-Optimal Large Language Models
# the exit design is a modified veresion of LG-Vit/Vit-EE/: 
class Deit_Comp(Comp_Prototype):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., exit_type = 'LG-deit-8', exit_mode = 'original', w=32, a=32):
        super().__init__(num_classes=num_classes, depth=depth)
        self.activation = a
        self.w_atten = w
        self.f_atten = w
        logging.info('model activation bit: '+str(self.activation))
        logging.info('model attention bit: '+str(self.w_atten))
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.mode = exit_mode
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim), # neglact the flops of patch embedding
            nn.LayerNorm(dim),
        )
        # self.first = nn.ModuleList([nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, dim),
        #     nn.LayerNorm(dim),
        # ), nn.Parameter(torch.randn(1, num_patches + 1, dim)), nn.Parameter(torch.randn(1, 1, dim))])
        # self.to_patch_embedding = self.first[0]

        # self.pos_embedding = self.first[1]
        # self.cls_token = self.first[2]
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        # self.first = nn.ModuleList([self.to_patch_embedding, nn.ParameterList([self.pos_embedding, self.cls_token])])
        self.dropout = nn.Dropout(emb_dropout)
        self.depth = depth
        self.blocklist = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, self.w_atten, self.activation)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.model_class = 'deit'
        self.exit_type = exit_type
        # self.gated_batch_eval = False
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.heads=heads
        n = num_patches + 1
        flops_embedding = 2*num_patches*patch_dim*dim
        flops_one_atten_and_ff = (2*n*3*dim*dim+2*n*n*dim+3*heads*n*n+2*n*n*dim+2*n*dim*dim) + 4*n*dim*mlp_dim
        self.main_flops = [flops_one_atten_and_ff for _ in range(depth)]
        self.main_flops[0] += flops_embedding
        self.main_flops_dict = dict([(i,l) for i,l in zip(range(depth),self.main_flops)])
        logging.info(f"main flops: {self.main_flops_dict}")
        self.main_params = [sum(p.numel() for p in l.parameters() if p.requires_grad) for l in self.blocklist.layers]
        self.main_params_dict = dict([(i,l) for i,l in zip(range(depth),self.main_params)])
        logging.info(f"main params: {self.main_params_dict}")
        self.cls_flops = 2*dim*num_classes
        if not exit_type:
            self.mlp_head = nn.Linear(dim, num_classes)
        else:
            if exit_type == 'fc':
                self.place_layer = [4, 5, 6, 7, 8, 9, 10, 11]
                self.exit_heads = nn.ModuleList([nn.Identity()]*4) + nn.ModuleList([nn.Linear(dim, num_classes) for _ in self.place_layer])
                self.exit_flops = [0]*4+[2*dim*num_classes for _ in self.place_layer]
            elif exit_type == 'mlp':
                self.place_layer = [4, 5, 6, 7, 8, 9, 10, 11]
                self.exit_heads = nn.ModuleList([nn.Identity()]*4) + nn.ModuleList([ViTHighway(dim, num_classes, dropout) for _ in self.place_layer])
                self.exit_flops = [0]*4+[2*dim*dim+2*dim*num_classes for l in self.exit_heads]
            if exit_type == 'fc-12':
                self.place_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                self.exit_heads = nn.ModuleList([nn.Linear(dim, num_classes) for _ in self.place_layer])
                self.exit_flops = [2*dim*num_classes for _ in self.place_layer]
            elif exit_type == 'mlp-12':
                self.place_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                self.exit_heads = nn.ModuleList([ViTHighway(dim, num_classes, dropout) for _ in self.place_layer])
                self.exit_flops = [2*dim*dim+2*dim*num_classes for l in self.exit_heads]
            elif exit_type == 'vit':
                self.place_layer = [4, 5, 6, 7, 8, 9, 10, 11]
                self.exit_heads = nn.ModuleList([nn.Identity()]*4) + nn.ModuleList([ViT_EE(num_classes, dim, heads, dim_head, mlp_dim, dropout, pool) for _ in self.place_layer])
                self.exit_flops = [0]*4+[flops_one_atten_and_ff+2*dim*dim+2*dim*num_classes for _ in self.place_layer]
            elif exit_type == 'vit-12':
                self.place_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                self.exit_heads = nn.ModuleList([ViT_EE(num_classes, dim, heads, dim_head, mlp_dim, dropout, pool) for _ in self.place_layer])
                self.exit_flops = [flops_one_atten_and_ff+2*dim*dim+2*dim*num_classes for _ in self.place_layer]
            elif exit_type == 'LG-deit-8':
                self.place_layer = [4, 5, 6, 7, 8, 9, 10, 11]
                self.exit_heads = nn.ModuleList([nn.Identity()]*4) + nn.ModuleList([
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'conv1_1', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'conv1_1', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'conv2_1', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'conv2_1', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'attention_r2', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'attention_r2', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'attention_r3', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'attention_r3', weight=self.w_atten),
                ])
                self.exit_flops = []
                self.exit_flops += [0]*4
                h = int(math.sqrt(num_patches))
                self.exit_flops += [2*dim*dim*h*h+2*3*3*dim*h*h+2*dim*dim*h*h+2*dim*num_classes]*2
                self.exit_flops += [2*dim*dim*h*h+2*dim*dim*h*h+2*dim*num_classes]*2
                pool_h2 = int((h-1)/2)+1
                self.exit_flops += [(2*pool_h2*3*dim*dim+2*pool_h2*pool_h2*dim+3*8*pool_h2*pool_h2
                                     +2*pool_h2*pool_h2*dim+2*pool_h2*dim*dim)+2*dim*num_classes]*2
                pool_h3 = int((h-1)/3)+1
                self.exit_flops += [(2*pool_h3*3*dim*dim+2*pool_h3*pool_h3*dim+3*8*pool_h3*pool_h3
                                     +2*pool_h3*pool_h3*dim+2*pool_h3*dim*dim)+2*dim*num_classes]*2
            elif exit_type == 'LG-deit-12':
                self.place_layer = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                self.exit_heads = nn.ModuleList([
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'conv1_1', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'conv1_1', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'conv1_1', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'conv2_1', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'conv2_1', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'conv2_1', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'attention_r2', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'attention_r2', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'attention_r2', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'attention_r3', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'attention_r3', weight=self.w_atten),
                    DeiTHighway_v2(num_classes=num_classes, dropout_r=dropout, hidden_size=dim, backbone='vit', layer_type=f'attention_r3', weight=self.w_atten),
                ])
                self.exit_flops = []
                h = int(math.sqrt(num_patches))
                self.exit_flops += [2*dim*dim*h*h+2*3*3*dim*h*h+2*dim*dim*h*h+2*dim*num_classes]*3
                self.exit_flops += [2*dim*dim*h*h+2*dim*dim*h*h+2*dim*num_classes]*3
                pool_h2 = int((h-1)/2)+1
                self.exit_flops += [(2*pool_h2*3*dim*dim+2*pool_h2*pool_h2*dim+3*8*pool_h2*pool_h2
                                     +2*pool_h2*pool_h2*dim+2*pool_h2*dim*dim)+2*dim*num_classes]*3
                pool_h3 = int((h-1)/3)+1
                self.exit_flops += [(2*pool_h3*3*dim*dim+2*pool_h3*pool_h3*dim+3*8*pool_h3*pool_h3
                                     +2*pool_h3*pool_h3*dim+2*pool_h3*dim*dim)+2*dim*num_classes]*3
            else:
                self.exit_heads = nn.ModuleList([nn.Identity() for _ in range(self.num_exit - 1)]+[nn.Linear(dim, num_classes)])
                
            # main/exit division
            self.exit_list = [self.exit_heads]
            self.main_list = [self.to_patch_embedding, self.pos_embedding, self.cls_token, self.blocklist]
            if not exit_type:
                self.main_list += self.mlp_head
            self.place_layer = self.place_layer[:-1]
            self.init_place_layer = self.place_layer
            self.exit_num = self.depth
            
            self.num_exit = len(self.place_layer)
            self.exit_params = [sum(p.numel() for p in l.parameters() if p.requires_grad) for l in self.exit_heads]
            self.exit_params_dict = dict([(i, l) for i, l in zip(range(self.depth), self.exit_params)])
            logging.info(f"exit params: {self.exit_params_dict}")
            self.exit_flops_dict = dict([(i, l) for i, l in zip(range(self.depth), self.exit_flops)])
            logging.info(f"exit flops: {self.exit_flops_dict}")
        
    def update_main_flops_params(self, verbose = False):
        
        n = self.num_patches + 1
        flops_embedding = 2*self.num_patches*self.to_patch_embedding[2].in_features*self.to_patch_embedding[2].out_features
        self.main_flops = []
        for attn, ff in self.blocklist.layers:   # to_qkv, q mul k, softmax, attn mul v, to_out
            q_dim = attn.to_qkv.out_features/3
            flops_one_atten_and_ff = (n*l_f(attn.to_qkv)+2*n*n*q_dim+3*self.heads*n*n+2*n*n*q_dim+n*l_f(attn.to_out[0])) + (n*l_f(ff.net[1])+n*l_f(ff.net[4]))
            self.main_flops.append(int(flops_one_atten_and_ff))
        self.main_flops[0] += flops_embedding
        
        self.main_params = [sum(p.numel() for p in l.parameters() if p.requires_grad) for l in self.blocklist.layers]
        
        self.main_flops_dict = dict([(i,l) for i,l in zip(range(self.depth),self.main_flops)])
        self.main_params_dict = dict([(i,l) for i,l in zip(range(self.depth),self.main_params)])
        if verbose:
            logging.info(f"main flops: {self.main_flops_dict}")
            logging.info(f"main params: {self.main_params_dict}")
        return sum(self.main_flops), sum(self.main_params)
    
    def update_flops_params(self, verbose=False):
        flops, params = self.update_main_flops_params(verbose)
        
        if self.exit_type == 'LG-deit-8':
            h = int(math.sqrt(self.num_patches))
            e = self.exit_heads[4]
            self.exit_flops[4] = c_f(e.mlp.conv1[0], h)+2*3*3*e.mlp.proj.in_channels*h*h+c_f(e.mlp.conv2[0], h)+l_f(e.classifier)
            e = self.exit_heads[5]
            self.exit_flops[5] = c_f(e.mlp.conv1[0], h)+2*3*3*e.mlp.proj.in_channels*h*h+c_f(e.mlp.conv2[0], h)+l_f(e.classifier)
            e = self.exit_heads[6]
            self.exit_flops[6] = c_f(e.mlp.conv1[0], h)+c_f(e.mlp.conv2[0], h)+l_f(e.classifier)
            e = self.exit_heads[7]
            self.exit_flops[7] = c_f(e.mlp.conv1[0], h)+c_f(e.mlp.conv2[0], h)+l_f(e.classifier)
        else:
            raise NotImplementedError
        self.exit_params = [sum(p.numel() for p in l.parameters() if p.requires_grad) for l in self.exit_heads]
        
        self.exit_params_dict = dict([(i, l) for i, l in zip(range(self.depth), self.exit_params)])
        self.exit_flops_dict = dict([(i, l) for i, l in zip(range(self.depth), self.exit_flops)])
        if verbose:
            logging.info(f"exit params: {self.exit_params_dict}")
            logging.info(f"exit flops: {self.exit_flops_dict}")
        flops += sum(self.exit_flops)
        params += sum(self.exit_params)
        return flops, params
        
        
    def forward(self, img, distill_token = None, gated_batch_eval = False, layer_pruning = False, dominant_list=None):
        # first
        self.inf_start()
        distilling = exists(distill_token)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]

        if distilling:
            distill_tokens = repeat(distill_token, '1 n d -> b n d', b = b)
            x = torch.cat((x, distill_tokens), dim = 1)

        x = self.dropout(x)
        x = T_quantize(x, self.activation-1)
        # self.inf_record()
        if self.mode == 'original':
            return self.forward_original(x, distilling)
        elif 'exit_t' in self.mode:
            if gated_batch_eval:
                return self.forward_exit_tv(x, distilling, enhance=True if layer_pruning else False)
            else:
                return self.forward_exit_t(x, distilling, dominant_list)
        elif 'exit_i' in self.mode:
            return self.forward_exit_i(x, distilling)

    def forward_original(self, x, distilling):
        # blocklist
        for attn, ff in self.blocklist.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = T_quantize(self.blocklist.norm(x), self.activation-1)
        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]
        if not self.exit_type or 'fc' in self.exit_type:
            x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
        out = self.exit_heads[-1](x) if self.exit_type else self.mlp_head(x)
        if distilling:
            return out, distill_tokens
        return out
    
    def forward_exit_t(self, x, distilling, dominant_list):
        out_list = []
        if distilling:
            distill_list = []
        i = 0
        for attn, ff in self.blocklist.layers:
            x = attn(x) + x
            x = ff(x) + x
            if i in self.place_layer:
                if dominant_list is not None and i not in dominant_list:
                    # 不更新dominant函数
                    exit_x = x.detach()
                    # exit_x.requires_grad = True
                    exit_x = T_quantize(self.blocklist.norm(exit_x), self.activation-1)
                else:
                    exit_x = T_quantize(self.blocklist.norm(x), self.activation-1)
                if distilling:
                    exit_x, distill_tokens = exit_x[:, :-1], exit_x[:, -1]
                if not self.exit_type or 'fc' in self.exit_type:
                    exit_x = exit_x.mean(dim = 1) if self.pool == 'mean' else exit_x[:, 0]
                    exit_x = self.to_latent(exit_x)
                out = self.exit_heads[i](exit_x)
                out_list.append(out)
                if distilling:
                    distill_list.append(distill_tokens)
                if self.onlyexit and i == max(self.place_layer):
                    return out_list
            i += 1
        x = T_quantize(self.blocklist.norm(x), self.activation-1)
        out = self.exit_heads[-1](x)
        out_list.append(out)
        if distilling:
            return ((o, d) for o, d in zip(out_list, distill_list))
        return out_list
    
    def forward_exit_i(self, x, distilling):
        i = 0
        for attn, ff in self.blocklist.layers:
            self.inf_start()
            x = attn(x) + x
            x = ff(x) + x
            self.inf_record()
            if i in self.place_layer:
                self.exit_start()
                exit_x = T_quantize(self.blocklist.norm(x), self.activation-1)
                if distilling:
                    exit_x, distill_tokens = exit_x[:, :-1], exit_x[:, -1]
                if not self.exit_type or 'fc' in self.exit_type:
                    exit_x = exit_x.mean(dim = 1) if self.pool == 'mean' else exit_x[:, 0]
                    exit_x = self.to_latent(exit_x)
                out = self.exit_heads[i](exit_x)
                self.exit_record()
                # print(math.exp(torch.max(F.log_softmax(exit_x, dim=1, dtype=torch.double)).item()))
                if torch.max(F.log_softmax(exit_x, dim=1, dtype=torch.double)).item() >= math.log(self.beta):
                    self.num_early_exit_list[i] += 1
                    if distilling:
                        return i, out, distill_tokens
                    return i, out
            i += 1
        exit_x = T_quantize(self.blocklist.norm(x), self.activation-1)
        out = self.exit_heads[-1](exit_x)
        self.num_early_exit_list[-1] += 1
        return i, out
        
    def change_quant(self, w, a):
        logging.info(f'change quant bit to weight {w}, activation {a}')
        self.activation = a
        self.w_atten = w
        self.f_atten = w
        self.blocklist.change_quant(w, a)
        # for e in self.exit_heads:
        #     if type(e) is not nn.Identity:
        #         e.change_quant(w, a)
                

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

from .deit_config import get_config

def DeiT(**kwargs):
    num_classes, input_size, scale = map(
        kwargs.get, ['num_classes', 'input_size', 'scale'])
    scale = scale or 'tiny'
    num_classes = num_classes or 10
    input_size = input_size or 32
    if input_size == 32:
        patch_size = 4 
    elif input_size == 64:
        patch_size = 8
    elif input_size == 224:
        patch_size = 16
    config = get_config(scale)
    
    return Deit_Comp(
        image_size = input_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = config.embed_dim, # = hidden_size = embed_dim (e)
        depth = config.depth,
        heads = config.num_heads, # heads * dim_head(default 64) is the inner dim of attention
        mlp_dim = config.embed_dim*config.mlp_ratio, # dim of the middle layer of mlp
        dropout = config.transformer.attention_dropout_rate, # = hidden_drop_out (e)
        emb_dropout = config.transformer.dropout_rate,
        exit_type = 'LG-deit-8',
        exit_mode = 'exit_t_only'
    )