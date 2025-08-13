import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn1, fn2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn1 = fn1
        self.fn2 = fn2

    def forward(self, x):
        x = self.norm(x)
        return self.fn1(x) + self.fn2(x)


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn1, fn2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn1 = fn1
        self.fn2 = fn2
        self.byconv = Convpass()

    def forward(self, x):
        x = self.norm(x)
        y1 = self.fn1(x)
        y2 = self.fn2(x)
        return self.dropout(y1 + y2)


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Convpass(nn.Module):
    def __init__(self, in_dim=512, dim=8, xavier_init=True):
        super().__init__()

        self.adapter_conv = nn.Conv3d(dim, dim, 3, 1, 1)
        self._initialize_conv_weights(xavier_init)
        self.adapter_down = nn.Linear(in_dim, dim)
        self.adapter_up = nn.Linear(dim, in_dim)
        self._initialize_linear_weights()
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def _initialize_conv_weights(self, xavier_init):
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

    def _initialize_linear_weights(self):
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        x_down = self.act(x_down)

        x_patch = x_down.reshape(B, 8, 8, 8, self.dim).permute(0, 4, 1, 2, 3)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 4, 1).reshape(B, 8 * 8 * 8, self.dim)

        x_down = self.act(x_patch)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)
        return x_up


class CTrans(nn.Module):
    def __init__(self, embedding_dim=512, dim=8, depth=1, heads=8, mlp_dim=4096, dropout_rate=0.1, n_levels=1, n_points=4):
        super(CTrans, self).__init__()
        self.depth = depth
        self.cross_attention_list = nn.ModuleList([Residual(PreNormDrop(embedding_dim, dropout_rate,
                                                                      SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate),
                                                                      Convpass(embedding_dim, dim))) for _ in range(self.depth)])
        self.cross_ffn_list = nn.ModuleList([Residual(PreNorm(embedding_dim, FeedForward(embedding_dim, mlp_dim, dropout_rate),
                                                              Convpass(embedding_dim, dim))) for _ in range(self.depth)])

    def forward(self, x, pos):
        for i in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[i](x)
            x = self.cross_ffn_list[i](x)
        return x


class CMF(nn.Module):
    def __init__(self, embedding_dim, dim):
        super(CMF, self).__init__()
        self.transformer_basic_dims = embedding_dim
        self.conv_trans = CTrans1(embedding_dim=embedding_dim, dim=dim)
        self.conv_pass = Convpass1(embedding_dim*3, dim * 4)
        self.decode_conv = nn.Conv3d(embedding_dim * 3, 384, kernel_size=1, padding=0)

    def forward(self, x1, x2, x3, pos1, pos2, pos3):
        t1ce_token = x1.permute(0, 2, 3, 4, 1).contiguous().view(x1.size(0), -1, self.transformer_basic_dims)
        t1_token = x2.permute(0, 2, 3, 4, 1).contiguous().view(x1.size(0), -1, self.transformer_basic_dims)
        t2_token = x3.permute(0, 2, 3, 4, 1).contiguous().view(x1.size(0), -1, self.transformer_basic_dims)

        multi_token = torch.cat((t1ce_token, t1_token, t2_token), dim=1)

        convpass_token = torch.cat((t1ce_token, t1_token, t2_token), dim=2)
        convpass_out = self.conv_pass(convpass_token)

        multi_pos = torch.cat((pos1, pos2, pos3), dim=1)
        out_token = self.conv_trans(multi_token, multi_pos)

        out = out_token.view(out_token.size(0), 8, 8, 8, self.transformer_basic_dims * 3).permute(0, 4, 1, 2, 3).contiguous()
        out = self.decode_conv(out)
        return out + convpass_out


if __name__ == '__main__':
    x1 = torch.rand(1, 512, 8, 8, 8)
    x2 = torch.rand(1, 512, 8, 8, 8)
    x3 = torch.rand(1, 512, 8, 8, 8)
    pos1 = torch.rand(1, 512, 512)
    pos2 = torch.rand(1, 512, 512)
    pos3 = torch.rand(1, 512, 512)

    model = CMF(embedding_dim=512, dim=8)
    y = model(x1, x2, x3, pos1, pos2, pos3)

    print(y.shape)
