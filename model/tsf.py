import torch
from torch import nn
import torch.nn.functional as F



class TSF_attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.num_channel = 3
        self.hyper_dim = 8
        self.latent_space_dim = 128
        # self.latent_space_dim = 32 # Ablation Study Remove Module 2
        self.style_dim = 3 * 2

        self.fc_w = nn.Sequential(
            nn.Linear(self.style_dim, self.num_channel),
            nn.Softmax(dim=1)
        )

        self.attn_layer = hyperAttnResBlock(
            self.latent_space_dim * self.num_channel, self.style_dim, 2, self.hyper_dim, 3, 1, use_attn=True)

        self.out_layer = nn.Sequential(
            LayerNorm(self.latent_space_dim * self.num_channel, eps=1e-6, data_format="channels_first"),
            nn.Conv3d(self.latent_space_dim * self.num_channel, out_channels=self.latent_space_dim, kernel_size=3,
                      padding=1, stride=1, padding_mode='zeros'),
        )

    def forward(self, zs, s, eps=1e-5):
        res = zs
        seq_in = s[:, :self.num_channel].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        param = self.fc_w(s).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + eps  # + eps to avoid dividing 0
        x = torch.stack(zs.chunk(self.num_channel, dim=1), dim=1)
        x = torch.sum(x * seq_in * param, dim=1) / torch.sum(seq_in * param)

        res = self.attn_layer(res, s)
        res = self.out_layer(res)
        finetune = x + res
        return x

class hyperConv(nn.Module):
    def __init__(
            self,
            style_dim,
            dim_in,
            dim_out,
            ksize,
            stride=1,
            padding=None,
            bias=True,
            dilation=1,
            groups=1,
            weight_dim=8,
            ndims=2,
    ):
        super().__init__()
        assert ndims in [2, 3]
        self.ndims = ndims
        self.dim_out = dim_out
        self.stride = stride
        self.bias = bias
        self.weight_dim = weight_dim
        self.fc = nn.Linear(style_dim, weight_dim)
        self.kshape = [dim_out, dim_in // groups, ksize, ksize] if self.ndims == 2 else [dim_out, dim_in // groups,
                                                                                         ksize, ksize, ksize]
        self.padding = (ksize - 1) // 2 if padding is None else padding
        self.groups = groups
        self.dilation = dilation

        self.param = nn.Parameter(torch.randn(*self.kshape, weight_dim).type(torch.float32))
        nn.init.kaiming_normal_(self.param, a=0, mode='fan_in')

        if self.bias is True:
            self.fc_bias = nn.Linear(style_dim, weight_dim)
            self.b = nn.Parameter(torch.randn(self.dim_out, weight_dim).type(torch.float32))
            nn.init.constant_(self.b, 0.0)

        self.conv = getattr(F, 'conv%dd' % self.ndims)

    def forward(self, x, s):
        if s.shape[0] == 1:
            return self.forwart_bs1(x, s)
        elif s.shape[0] == x.shape[0]:
            out = []
            for i in range(s.shape[0]):
                out.append(self.forwart_bs1(x[i:i + 1], s[i:i + 1]))
            out = torch.cat(out, dim=0)
            return out

    def forwart_bs1(self, x, s):
        kernel = torch.matmul(self.param, self.fc(s).view(self.weight_dim, 1)).view(*self.kshape)
        if self.bias is True:
            bias = torch.matmul(self.b, self.fc_bias(s).view(self.weight_dim, 1)).view(self.dim_out)
            return self.conv(x, weight=kernel, bias=bias, stride=self.stride, padding=self.padding,
                             dilation=self.dilation, groups=self.groups)
        else:
            return self.conv(x, weight=kernel, stride=self.stride, padding=self.padding, dilation=self.dilation,
                             groups=self.groups)


class hyperBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, style_dim, latent_dim=8, kernel_size=7, padding=3, layer_scale_init_value=1e-6):
        super().__init__()
        # self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.pad = nn.ZeroPad3d(padding)  # nn.ReflectionPad3d(padding)
        self.dwconv = hyperConv(style_dim, dim, dim, ksize=kernel_size, padding=0, groups=dim, weight_dim=latent_dim,
                                ndims=3)
        self.norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.pwconv1 = hyperConv(style_dim, dim, dim * 4, ksize=1, padding=0, weight_dim=latent_dim, ndims=3)
        self.act = nn.GELU()
        self.pwconv2 = hyperConv(style_dim, dim * 4, dim, ksize=1, padding=0, weight_dim=latent_dim, ndims=3)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, s):
        input = x
        x = self.pad(x)
        x = self.dwconv(x, s)
        # x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x, s)
        x = self.act(x)
        x = self.pwconv2(x, s)

        if self.gamma is not None:
            x = self.gamma * x

        x = input + x
        return x

class hyperAttnResBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, style_dim, n_layer, latent_dim=8, kernel_size=7, padding=3, layer_scale_init_value=1e-6,
                 use_attn=False):
        super().__init__()

        self.n_layer = n_layer
        self.use_attn = use_attn

        if n_layer > 0:
            self.block1 = hyperBlock(dim, style_dim, latent_dim=latent_dim, kernel_size=kernel_size, padding=padding,
                                     layer_scale_init_value=layer_scale_init_value)
        else:
            self.block1 = nn.Identity()

        if use_attn:
            self.attnblocks = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for _ in range(n_layer - 1):
            if use_attn:
                self.attnblocks.append(CrossAttnBlock(dim, style_dim))
            self.resblocks.append(
                hyperBlock(dim, style_dim, latent_dim=latent_dim, kernel_size=kernel_size, padding=padding,
                           layer_scale_init_value=layer_scale_init_value))

    def forward(self, x, s):
        if self.n_layer == 0:
            x = self.block1(x)
        else:
            x = self.block1(x, s)

        if self.use_attn:
            for attn, res in zip(self.attnblocks, self.resblocks):
                x = attn(x, s.unsqueeze(1))
                x = res(x, s)
        else:
            for res in self.resblocks:
                x = res(x, s)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

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

class CrossAttnBlock(nn.Module):
    def __init__(self, in_channels, in_channels_style, heads=1):
        super().__init__()
        self.in_channels = in_channels
        self.heads = heads
        self.dim_heads = in_channels//heads

        self.norm = LayerNorm(in_channels, eps=1e-6, data_format='channels_first')
        self.norm_style = nn.LayerNorm(in_channels_style, eps=1e-6)
        self.q = torch.nn.Conv3d(in_channels,
                                 self.dim_heads*heads,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0, bias=False)
        self.k = torch.nn.Linear(in_channels_style, self.dim_heads*heads, bias=False)
        self.v = torch.nn.Linear(in_channels_style, self.dim_heads*heads, bias=False)
        self.proj_out = torch.nn.Conv3d(self.dim_heads*heads,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, s):
        h_ = x
        h_ = self.norm(h_)
        s_ = self.norm_style(s)
        q = self.q(h_)
        k = self.k(s_).permute(0,2,1)
        v = self.v(s_).permute(0,2,1)

        # compute attention
        b,c,d,w,h = q.shape
        sn = k.shape[-1]
        q = q.reshape(b*self.heads,self.dim_heads,d*w*h)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b*self.heads,self.dim_heads,sn) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(self.dim_heads)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b*self.heads,self.dim_heads,sn)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,d,w,h)

        h_ = self.proj_out(h_)

        return x+h_