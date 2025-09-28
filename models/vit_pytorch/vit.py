import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .binarymodule import Quant_Conv2d, IRlinear
from .binaryfunction import T_quantize
# (float simulation ver. (for QAT) of Int quantization)

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., weight = 32, activation = 32):
        super().__init__()
        self.weight = weight
        self.activation = activation
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            IRlinear(dim, hidden_dim, bit=weight),
            nn.GELU(),
            nn.Dropout(dropout),
            IRlinear(hidden_dim, dim, bit=weight),
            nn.Dropout(dropout)
        )

    def change_quant(self, w, a):
        self.weight = w
        self.activation = a
        self.net[1].change_quant(w, a)
        self.net[4].change_quant(w, a)

    def forward(self, x):
        x = T_quantize(self.net[0](x), self.activation-1)
        x = T_quantize(self.net[1:3](x), self.activation-1)
        return self.net[3:](x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., weight = 32, activation = 32):
        super().__init__()
        self.weight = weight
        self.activation = activation
        inner_dim = dim_head *  heads
        self.head_dim = dim_head
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = IRlinear(dim, inner_dim * 3, bias = False, bit=weight)

        self.to_out = nn.Sequential(
            IRlinear(inner_dim, dim, bit=weight),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def change_quant(self, w, a):
        self.weight = w
        self.activation = a
        self.to_qkv.change_quant(w, a)
        if type(self.to_out) is not nn.Identity:
            self.to_out[0].change_quant(w, a)

    def forward(self, x):
        B, N, C = x.shape
        x = T_quantize(self.norm(x), self.activation-1)
        qkv = self.to_qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if True:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = T_quantize(attn.softmax(dim=-1), self.activation-1)
            attn = self.dropout(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, -1) # original implementation: x = x.transpose(1, 2).reshape(B, N, C)
        x = self.to_out(x)
        return x

    # def forward(self, x):
    #     x = T_quantize(self.norm(x), self.activation-1)

    #     qkv = self.to_qkv(x).chunk(3, dim = -1)
    #     q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

    #     dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

    #     attn = T_quantize(self.attend(dots), self.activation-1)
    #     attn = self.dropout(attn)

    #     out = torch.matmul(attn, v)
    #     out = rearrange(out, 'b h n d -> b n (h d)')
    #     return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., weight=32, activation=32):
        super().__init__()
        self.weight = weight
        self.activation = activation
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, weight = weight, activation = activation),
                FeedForward(dim, mlp_dim, dropout = dropout, weight = weight, activation = activation)
            ]))

    def change_quant(self, w, a):
        self.weight = w
        self.activation = a
        for attn, ff in self.layers:
            attn.change_quant(w, a)
            ff.change_quant(w, a)

    def forward(self, x):
        x = T_quantize(x, self.activation-1)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return T_quantize(self.norm(x), self.activation-1)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
