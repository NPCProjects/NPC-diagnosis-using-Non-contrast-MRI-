import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
import numpy as np
from model.layers import general_conv3d_prenorm, fusion_prenorm
from model.ctrans import CMF, CTrans
# from model.DeiT_ViT import CMF, CTrans  # revision
# from model.cmf_swin import CMF, CTrans  # revision
# from model.T2T_ViT import CMF, CTrans  # revision
# from model.cvt import CMF, CTrans  # revision


from model.tsf import LayerNorm, hyperAttnResBlock, TSF_attention





basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 3
patch_size = 8


class Encoder(nn.Module):
    def __init__(self, in_channels=1):
        super(Encoder, self).__init__()

        self.e1_c1 = nn.Conv3d(in_channels=in_channels, out_channels=basic_dims, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims)
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims)

        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims * 2, stride=2)
        self.e2_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2)
        self.e2_c3 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2)

        self.e3_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims * 4, stride=2)
        self.e3_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4)
        self.e3_c3 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4)

        self.e4_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 8, stride=2)
        self.e4_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8)
        self.e4_c3 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8)

        self.e5_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 16, stride=2)
        self.e5_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16)
        self.e5_c3 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 16)

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))

        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))

        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))

        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))

        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))

        return x1, x2, x3, x4, x5


class Decoder(nn.Module):
    def __init__(self, num_cls=3):
        super(Decoder, self).__init__()

        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8)
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8)
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0)

        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4)
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)

        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2)
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)

        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims)
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims)
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0)

        # self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
        #                            bias=True)
        # self.average_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.classification_layer = nn.Linear(in_features=basic_dims * 16, out_features=2)
        # self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(128 * 8 * 8 * 8, 128 * 8 * 8)
        self.fc2 = nn.Linear(128 * 8 * 8, 128 * 8)
        self.fc3 = nn.Linear(128 * 8, 1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))

        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))

        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))

        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))

        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))

        # logits = self.classification_layer(torch.mean(x5, dim=(2,3,4)))
        # out = torch.flatten(x5, 1)  # flatten all dimensions except batch
        out = F.relu(self.fc1(x5.view(x5.size(0), -1)))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # pred = self.softmax(logits)

        return out


class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()

        self.d4_c1 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8)
        self.d4_c2 = general_conv3d_prenorm(basic_dims * 16, basic_dims * 8)
        self.d4_out = general_conv3d_prenorm(basic_dims * 8, basic_dims * 8, k_size=1, padding=0)

        self.d3_c1 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4)
        self.d3_c2 = general_conv3d_prenorm(basic_dims * 8, basic_dims * 4)
        self.d3_out = general_conv3d_prenorm(basic_dims * 4, basic_dims * 4, k_size=1, padding=0)

        self.d2_c1 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2)
        self.d2_c2 = general_conv3d_prenorm(basic_dims * 4, basic_dims * 2)
        self.d2_out = general_conv3d_prenorm(basic_dims * 2, basic_dims * 2, k_size=1, padding=0)

        self.d1_c1 = general_conv3d_prenorm(basic_dims * 2, basic_dims)
        self.d1_c2 = general_conv3d_prenorm(basic_dims * 2, basic_dims)
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0)

        # self.seg_d4 = nn.Conv3d(in_channels=basic_dims * 16, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
        #                         bias=True)
        # self.seg_d3 = nn.Conv3d(in_channels=basic_dims * 8, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
        #                         bias=True)
        # self.seg_d2 = nn.Conv3d(in_channels=basic_dims * 4, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
        #                         bias=True)
        # self.seg_d1 = nn.Conv3d(in_channels=basic_dims * 2, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
        #                         bias=True)
        self.class_d4 = nn.Linear(in_features=basic_dims * 16, out_features=2, bias=True)
        self.class_d3 = nn.Linear(in_features=basic_dims * 8, out_features=2, bias=True)
        self.class_d2 = nn.Linear(in_features=basic_dims * 4, out_features=2, bias=True)
        self.class_d1 = nn.Linear(in_features=basic_dims * 2, out_features=2, bias=True)
        # self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0,
        #                            bias=True)
        # self.softmax = nn.Softmax(dim=1)

        self.average_pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.classification_layer = nn.Linear(in_features=basic_dims*16*3, out_features=2, bias=True)
        self.fc1 = nn.Linear(384 * 8 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 1)
        # self.fc3 = nn.Linear(128, 1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims * 16, num_modals=num_modals)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims * 8, num_modals=num_modals)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims * 4, num_modals=num_modals)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims * 2, num_modals=num_modals)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims * 1, num_modals=num_modals)

    def forward(self, x1, x2, x3, x4, x5):
        B = x1.shape[0]
        de_x5 = self.RFM5(x5)
        pred4 = self.class_d4(self.average_pooling(de_x5).view((B, -1)))
        de_x5 = self.d4_c1(self.up2(de_x5))

        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.class_d3(self.average_pooling(de_x4).view((B, -1)))
        de_x4 = self.d3_c1(self.up2(de_x4))

        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.class_d2(self.average_pooling(de_x3).view((B, -1)))
        de_x3 = self.d2_c1(self.up2(de_x3))

        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.class_d1(self.average_pooling(de_x2).view((B, -1)))
        de_x2 = self.d1_c1(self.up2(de_x2))

        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))

        # logits = self.seg_layer(de_x1)

        # logits = self.classification_layer(torch.mean(x5, dim=(2,3,4)))
        # pred = self.softmax(logits)
        out = torch.flatten(x5, 1)  # flatten all dimensions except batch
        out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        out = self.fc2(out)

        return out, (pred1, pred2, pred3, pred4)


class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()

    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()  # K corresponds to the number of stacked modalities.
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        x = y.view(B, -1, H, W, Z)
        return x


class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()

        # self.t1ce_encoder = Encoder()
        # self.t1_encoder = Encoder()
        # self.t2_encoder = Encoder()
        self.t_encoder = Encoder()

        self.all_encoder = Encoder(in_channels=3)

        ########### channel exchange

        self.t1ce_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims * 16, transformer_basic_dims, kernel_size=1, stride=1, padding=0)


        self.t1ce_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t1_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)
        self.t2_decode_conv = nn.Conv3d(transformer_basic_dims, basic_dims * 16, kernel_size=1, stride=1, padding=0)



        self.t1ce_trans = CTrans(embedding_dim=transformer_basic_dims, depth=8)
        self.t1_trans = CTrans(embedding_dim=transformer_basic_dims, depth=8)
        self.t2_trans = CTrans(embedding_dim=transformer_basic_dims, depth=8)




        self.cmf = CMF(embedding_dim=512, dim=8)
        # self.cmf = CMF(embedding_dim=128, dim=8) #Ablation Study, remove MCWB


        self.t1ce_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.t1_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))
        self.t2_pos = nn.Parameter(torch.zeros(1, patch_size ** 3, transformer_basic_dims))

        self.masker = MaskModal()

        self.classifier_sep = nn.Linear(128 * 8 * 8 * 8, 2)

        self.classifier_fuse = nn.Linear(128 * 8 * 8 * 8, 2)
        # self.classifier_fuse = nn.Linear(96 * 8 * 8 * 8, 2) #Ablation Study Remove Module 2

        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

        self.tsf = TSF_attention()

    def forward(self, x, mask=None):
        # extract feature from different layers
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t_encoder(x[:, 0:1, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t_encoder(x[:, 1:2, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t_encoder(x[:, 2:3, :, :, :])



        ########### IntraFormer
        ### channels 128 >> 384

        t1ce_token_x5 = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                                transformer_basic_dims)
        t1_token_x5 = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                          transformer_basic_dims)
        t2_token_x5 = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(x.size(0), -1,
                                                                                          transformer_basic_dims)



        t1ce_intra_token_x5 = self.t1ce_trans(t1ce_token_x5, self.t1ce_pos)
        t1_intra_token_x5 = self.t1_trans(t1_token_x5, self.t1_pos)
        t2_intra_token_x5 = self.t2_trans(t2_token_x5, self.t2_pos)


        t1ce_intra_x5 = t1ce_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size,
                                                 transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size,
                                             transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token_x5.view(x.size(0), patch_size, patch_size, patch_size,
                                             transformer_basic_dims).permute(0, 4, 1, 2, 3).contiguous()

        t1ce_trans_x5 = self.t1ce_decode_conv(t1ce_intra_x5)
        t1_trans_x5 = self.t1_decode_conv(t1_intra_x5)
        t2_trans_x5 = self.t2_decode_conv(t2_intra_x5)

        t1ce_pred = self.classifier_sep(torch.flatten(t1ce_trans_x5, 1))
        t1_pred = self.classifier_sep(torch.flatten(t1_trans_x5, 1))
        t2_pred = self.classifier_sep(torch.flatten(t2_trans_x5, 1))

        # if self.is_training:

        #     t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
        #     t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
        #     t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
        ########### IntraFormer

        # x1 = self.masker(torch.stack((t1ce_x1, t1_x1, t2_x1), dim=1), mask)  # Bx4xCxHWZ
        # x2 = self.masker(torch.stack((t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        # x3 = self.masker(torch.stack((t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        # x4 = self.masker(torch.stack((t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1), mask)
        # x5_intra = self.masker(torch.stack((t1ce_x5, t1_x5, t2_x5), dim=1), mask) # Ablation Study, remove module2


        all_x1, all_x2, all_x3, all_x4, all_x5 = self.all_encoder(x)

        t1ce_intra_x5, t1_intra_x5, t2_intra_x5 = torch.chunk(x5_intra, num_modals, dim=1)


        x5_inter = self.cmf(t1ce_intra_x5, t1_intra_x5, t2_intra_x5,  self.t1ce_pos,
                            self.t1_pos, self.t2_pos)
        '''tsf start'''
        if mask is None:
            mask = torch.tensor([False, True, True])
            mask = torch.unsqueeze(mask, dim=0).repeat(x.shape[0], 1).cuda()
        tgt_code = F.one_hot(torch.zeros(x.shape[0]).long(), num_classes=3).to('cuda', dtype=torch.float, non_blocking=True)
        tsf_tgt_code = torch.cat([mask.float(), tgt_code], dim=1)
        latent_tsf_tgt = self.tsf(x5_inter, tsf_tgt_code)
        '''tsf end'''

        fuse_pred = self.classifier_fuse(torch.flatten(latent_tsf_tgt, 1))
        # fuse_pred = self.classifier_fuse(torch.flatten(x5_inter, 1)) # Ablation Study, remove module2
        pred_all = self.classifier_sep(torch.flatten(all_x5, 1)) # teacher model


        t1ce = (t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
        t1 = (t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
        t2 = (t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)
        all = (all_x1, all_x2, all_x3, all_x4, all_x5)

        aux = (t1ce_pred, t1_pred, t2_pred)


        return fuse_pred, pred_all, t1ce, t1, t2, all, aux, latent_tsf_tgt.view(latent_tsf_tgt.size(1), -1)





