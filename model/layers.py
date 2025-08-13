import torch
import torch.nn as nn


def normalization(planes, norm='bn'):
    """Returns normalization layer based on specified type."""
    if norm == 'bn':
        return nn.BatchNorm3d(planes)
    elif norm == 'gn':
        return nn.GroupNorm(4, planes)
    elif norm == 'in':
        return nn.InstanceNorm3d(planes)
    else:
        raise ValueError(f'Normalization type {norm} is not supported')


class GeneralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros'):
        super(GeneralConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride,
                              padding=padding, padding_mode=pad_type, bias=True)

    def forward(self, x):
        return self.conv(x)


class GeneralConv3dPreNorm(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', act_type='lrelu', relufactor=0.2):
        super(GeneralConv3dPreNorm, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride,
                              padding=padding, padding_mode=pad_type, bias=True)
        self.norm = normalization(in_ch, norm)
        self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True) if act_type == 'lrelu' else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.activation(x)
        return self.conv(x)


class GeneralConv3d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size=3, stride=1, padding=1, pad_type='zeros', norm='in', act_type='lrelu', relufactor=0.2):
        super(GeneralConv3d, self).__init__()
        self.conv = nn.Conv3d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride,
                              padding=padding, padding_mode=pad_type, bias=True)
        self.norm = normalization(out_ch, norm)
        self.activation = nn.LeakyReLU(negative_slope=relufactor, inplace=True) if act_type == 'lrelu' else nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class PrmGeneratorLastStage(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=3):
        super(PrmGeneratorLastStage, self).__init__()
        self.embedding_layer = nn.Sequential(
            GeneralConv3d(in_channel * 4, in_channel // 4, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel // 4, in_channel // 4, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel // 4, in_channel, k_size=1, padding=0, stride=1)
        )

        self.prm_layer = nn.Sequential(
            GeneralConv3d(in_channel, 16, k_size=1, stride=1, padding=0),
            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.prm_layer(self.embedding_layer(x))


class PrmGenerator(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=3):
        super(PrmGenerator, self).__init__()
        self.embedding_layer = nn.Sequential(
            GeneralConv3d(in_channel * 4, in_channel // 4, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel // 4, in_channel // 4, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel // 4, in_channel, k_size=1, padding=0, stride=1)
        )
        self.prm_layer = nn.Sequential(
            GeneralConv3d(in_channel * 2, 16, k_size=1, stride=1, padding=0),
            nn.Conv3d(16, num_cls, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        concatenated_input = torch.cat((x1, self.embedding_layer(x2)), dim=1)
        return self.prm_layer(concatenated_input)


class ModalFusion(nn.Module):
    def __init__(self, in_channel=64):
        super(ModalFusion, self).__init__()
        self.weight_layer = nn.Sequential(
            nn.Conv3d(4 * in_channel + 1, 128, 1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(128, 4, 1, padding=0, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prm):
        B, K, C, H, W, Z = x.size()
        prm_avg = torch.mean(prm, dim=(3, 4, 5), keepdim=False) + 1e-7
        feat_avg = torch.mean(x, dim=(3, 4, 5), keepdim=False) / prm_avg
        feat_avg = feat_avg.view(B, K * C, 1, 1, 1)
        feat_avg = torch.cat((feat_avg, prm_avg[:, 0, 0, ...].view(B, 1, 1, 1, 1)), dim=1)
        weight = torch.reshape(self.weight_layer(feat_avg), (B, K, 1))
        weight = self.sigmoid(weight).view(B, K, 1, 1, 1, 1)

        return torch.sum(x * weight, dim=1)


class RegionFusionLastStage(nn.Module):
    def __init__(self, in_channel=64, num_cls=3):
        super(RegionFusionLastStage, self).__init__()
        self.fusion_layer = nn.Sequential(
            GeneralConv3d(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel, in_channel, k_size=1, padding=0, stride=1)
        )

    def forward(self, x):
        B, _, H, W, Z = x.size()
        x = x.view(B, -1, H, W, Z)
        return self.fusion_layer(x)


class RegionFusion(nn.Module):
    def __init__(self, in_channel=64, num_cls=3):
        super(RegionFusion, self).__init__()
        self.fusion_layer = nn.Sequential(
            GeneralConv3d(in_channel * num_cls, in_channel, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1)
        )

    def forward(self, x):
        return self.fusion_layer(x)


class FusionPrenorm(nn.Module):
    def __init__(self, in_channel=64, num_modals=3):
        super(FusionPrenorm, self).__init__()
        self.fusion_layer = nn.Sequential(
            GeneralConv3dPreNorm(in_channel * num_modals, in_channel, k_size=1, padding=0, stride=1),
            GeneralConv3dPreNorm(in_channel, in_channel, k_size=3, padding=1, stride=1),
            GeneralConv3dPreNorm(in_channel, in_channel, k_size=1, padding=0, stride=1)
        )

    def forward(self, x):
        return self.fusion_layer(x)


class RegionAwareModalFusion(nn.Module):
    def __init__(self, in_channel=64, norm='in', num_cls=3):
        super(RegionAwareModalFusion, self).__init__()
        self.num_cls = num_cls
        self.modal_fusion = nn.ModuleList([ModalFusion(in_channel=in_channel) for _ in range(num_cls)])
        self.region_fusion = RegionFusion(in_channel=in_channel, num_cls=num_cls)
        self.short_cut = nn.Sequential(
            GeneralConv3d(in_channel * 4, in_channel, k_size=1, padding=0, stride=1),
            GeneralConv3d(in_channel, in_channel, k_size=3, padding=1, stride=1),
            GeneralConv3d(in_channel, in_channel // 2, k_size=1, padding=0, stride=1)
        )
        self.clsname_list = ['BG', 'NCR', 'ED', 'NET', 'ET']

    def forward(self, x, prm):
        B, _, H, W, Z = x.size()
        y = x.view(B, 4, -1, H, W, Z)
        B, K, C, H, W, Z = y.size()

        prm = torch.unsqueeze(prm, 2).repeat(1, 1, C, 1, 1, 1)
        flair = y[:, 0:1, ...] * prm
        t1ce = y[:, 1:2, ...] * prm
        t1 = y[:, 2:3, ...] * prm
        t2 = y[:, 3:4, ...] * prm

        modal_feat = torch.stack((flair, t1ce, t1, t2), dim=1)
        region_feat = [modal_feat[:, :, i, :, :] for i in range(self.num_cls)]

        region_fused_feat = [self.modal_fusion[i](region_feat[i], prm[:, i:i+1, ...], self.clsname_list[i]) for i in range(self.num_cls)]
        region_fused_feat = torch.stack(region_fused_feat, dim=1)

        final_feat = torch.cat((self.region_fusion(region_fused_feat), self.short_cut(y.view(B, -1, H, W, Z))), dim=1)
        return final_feat
