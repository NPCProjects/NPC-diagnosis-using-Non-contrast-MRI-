import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


class LayerNorm(nn.Module):
    """Layer Normalization that supports channels_first and channels_last formats."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class SelfAttention(nn.Module):
    """Self-attention mechanism with learnable query, key, and value projections."""

    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

        # Initialize weights
        self.apply(self.init_weights)

    def init_weights(self, layer):
        """Initialize weights of layers."""
        if isinstance(layer, (nn.Linear, nn.Conv3d)):
            xavier_uniform_(layer.weight)
            if layer.bias is not None:
                constant_(layer.bias, 0)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class HyperConv(nn.Module):
    """Hyper-parameterized convolution."""

    def __init__(self, style_dim, dim_in, dim_out, ksize, stride=1, padding=None, bias=True, dilation=1, groups=1,
                 weight_dim=8, ndims=2):
        super().__init__()
        assert ndims in [2, 3], "Only 2D and 3D convolutions are supported."
        self.ndims = ndims
        self.fc = nn.Linear(style_dim, weight_dim)
        self.param = nn.Parameter(
            torch.randn(dim_out, dim_in // groups, ksize, ksize) if self.ndims == 2 else torch.randn(dim_out,
                                                                                                     dim_in // groups,
                                                                                                     ksize, ksize,
                                                                                                     ksize))
        nn.init.kaiming_normal_(self.param, a=0, mode='fan_in')

        self.fc_bias = nn.Linear(style_dim, weight_dim) if bias else None
        self.b = nn.Parameter(torch.randn(dim_out, weight_dim).type(torch.float32)) if bias else None

        self.dilation = dilation
        self.stride = stride
        self.padding = (ksize - 1) // 2 if padding is None else padding
        self.groups = groups

        self.conv = getattr(F, f'conv{ndims}d')

    def forward(self, x, s):
        kernel = torch.matmul(self.param, self.fc(s).view(-1, 1)).view(*self.param.shape)
        if self.fc_bias:
            bias = torch.matmul(self.b, self.fc_bias(s).view(-1, 1)).view(self.param.shape[0])
            return self.conv(x, weight=kernel, bias=bias, stride=self.stride, padding=self.padding,
                             dilation=self.dilation, groups=self.groups)
        else:
            return self.conv(x, weight=kernel, stride=self.stride, padding=self.padding, dilation=self.dilation,
                             groups=self.groups)


class HyperBlock(nn.Module):
    """A block consisting of a depthwise convolution, norm, and 1x1 convolutions."""

    def __init__(self, dim, style_dim, latent_dim=8, kernel_size=7, padding=3, layer_scale_init_value=1e-6):
        super().__init__()
        self.pad = nn.ZeroPad3d(padding)
        self.dwconv = HyperConv(style_dim, dim, dim, kernel_size, groups=dim, weight_dim=latent_dim, ndims=3)
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.pwconv1 = HyperConv(style_dim, dim, dim * 4, kernel_size=1, weight_dim=latent_dim, ndims=3)
        self.act = nn.GELU()
        self.pwconv2 = HyperConv(style_dim, dim * 4, dim, kernel_size=1, weight_dim=latent_dim, ndims=3)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, s):
        input = x
        x = self.pad(x)
        x = self.dwconv(x, s)
        x = self.norm(x)
        x = self.pwconv1(x, s)
        x = self.act(x)
        x = self.pwconv2(x, s)

        if self.gamma is not None:
            x = self.gamma * x

        return input + x


class HyperAttnResBlock(nn.Module):
    """Hyper Attention Residual Block with optional attention mechanism."""

    def __init__(self, dim, style_dim, n_layer, latent_dim=8, kernel_size=7, padding=3, layer_scale_init_value=1e-6,
                 use_attn=False):
        super().__init__()

        self.use_attn = use_attn
        self.n_layer = n_layer
        self.block1 = HyperBlock(dim, style_dim, latent_dim, kernel_size, padding,
                                 layer_scale_init_value) if n_layer > 0 else nn.Identity()

        self.attnblocks = nn.ModuleList(
            [CrossAttnBlock(dim, style_dim) for _ in range(n_layer - 1)]) if use_attn else nn.ModuleList()
        self.resblocks = nn.ModuleList(
            [HyperBlock(dim, style_dim, latent_dim, kernel_size, padding, layer_scale_init_value) for _ in
             range(n_layer - 1)])

    def forward(self, x, s):
        x = self.block1(x, s)
        if self.use_attn:
            for attn, res in zip(self.attnblocks, self.resblocks):
                x = attn(x, s.unsqueeze(1))
                x = res(x, s)
        else:
            for res in self.resblocks:
                x = res(x, s)
        return x


class CrossAttnBlock(nn.Module):
    """Cross Attention Block with query, key, and value projections."""

    def __init__(self, in_channels, in_channels_style, heads=1):
        super().__init__()
        self.dim_heads = in_channels // heads
        self.norm = LayerNorm(in_channels, eps=1e-6, data_format='channels_first')
        self.norm_style = nn.LayerNorm(in_channels_style, eps=1e-6)

        self.q = nn.Conv3d(in_channels, self.dim_heads * heads, kernel_size=1, stride=1, padding=0, bias=False)
        self.k = nn.Linear(in_channels_style, self.dim_heads * heads, bias=False)
        self.v = nn.Linear(in_channels_style, self.dim_heads * heads, bias=False)
        self.proj_out = nn.Conv3d(self.dim_heads * heads, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, s):
        x = self.norm(x)
        s = self.norm_style(s)

        q = self.q(x).view(x.shape[0], self.dim_heads, -1, x.shape[2] * x.shape[3] * x.shape[4]).permute(0, 2, 3, 1)
        k = self.k(s).view(s.shape[0], s.shape[2], -1)
        v = self.v(s).view(s.shape[0], s.shape[2], -1)

        w_ = torch.matmul(q, k.transpose(-2, -1)) * (self.dim_heads ** -0.5)
        w_ = torch.nn.functional.softmax(w_, dim=-1)

        h_ = torch.matmul(w_, v)
        h_ = h_.view(x.shape[0], x.shape[1], *x.shape[2:]).contiguous()

        return self.proj_out(h_)
