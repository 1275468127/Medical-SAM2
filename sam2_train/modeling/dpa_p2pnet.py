# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys
sys.path.insert(0, '/data/hhb/project1/project/Medical-SAM2/sam2_train/modeling')
import timm
print(timm.__file__)
import copy
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from sam2_train.modeling.fpn import FPN
#from .image_encoder import ImageEncoderViT

class Backbone(nn.Module):
    def __init__(
            self,
            cfg
    ):
        super(Backbone, self).__init__()

        backbone = timm.create_model(
            **cfg.prompter.backbone
        )

        self.backbone = backbone

        self.neck = FPN(
            **cfg.prompter.neck
        )

        new_dict = copy.copy(cfg.prompter.neck)
        new_dict['num_outs'] = 1
        self.neck1 = FPN(
            **new_dict
        )

    def forward(self, images):
        x = self.backbone(images)
        return list(self.neck(x)), self.neck1(x)[0]


class AnchorPoints(nn.Module):
    def __init__(self, space=16):
        super(AnchorPoints, self).__init__()
        self.space = space

    def forward(self, images):
        bs, _, h, w = images.shape
        anchors = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / self.space)),
                np.arange(np.ceil(h / self.space))),
            -1) * self.space

        origin_coord = np.array([w % self.space or self.space, h % self.space or self.space]) / 2
        anchors += origin_coord

        anchors = torch.from_numpy(anchors).float().to(images.device)
        return anchors.repeat(bs, 1, 1, 1)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, drop=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList()

        for n, k in zip([input_dim] + h, h):
            self.layers.append(nn.Linear(n, k))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(drop))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

class SR_PFO(nn.Module):
    """ This is the SAM-Guided Self Refinement Point Feature Optimization Module """
    ''' input: point_feature & sam_semantic_feature'''
    def __init__(
            self,
            dropout=0.1,
            input_dim: int = 256,
            hidden_dim: int = 512,
            num_layers:int = 3,
            output_dim:int = 256
    ):
        """
            Initializes the model.
        """
        super().__init__()


        self.mlp_p = MLP(input_dim, hidden_dim, num_layers, output_dim)
        self.mlp_s = MLP(input_dim, hidden_dim, num_layers, output_dim)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        )
        # self.scale = nn.Parameter(torch.rand(256, 1, 1) )
        self.scale = nn.Parameter(torch.ones(1))  # 可训练的缩放因子
        
        # self.scale1 = nn.Parameter(torch.ones(1)) 
        # self.scale2 = nn.Parameter(torch.ones(1)) 
        # self.scale3 = nn.Parameter(torch.ones(1)) 
        
        # self.head_layer = nn.Sequential(
        #     nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(output_dim),
        #     nn.GELU(),
        #     nn.Conv2d(output_dim, output_dim, kernel_size=3, stride=1, padding=1)
        # )
        # self.attention_layer = Attention(output_dim)
        # self.mlp = MLPBlock(output_dim, output_dim)
        # self.bn = nn.BatchNorm2d(output_dim)
        
    def through_mlp(self,feature):
        batch_size, channels, height, width = feature.shape
        flattened_feature_input = feature.permute(0, 2, 3, 1).reshape(-1, channels)
        flattened_feature_output = self.mlp_p(flattened_feature_input)
        feature_mlp_output = flattened_feature_output.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
        return feature_mlp_output
    # # 256token
    # def forward(self,
    #             point_feature,feats1,sam_semantic_feature):
    #     # 让语义特征乘以倍数
    #     point_feature_mlp_output_0 = self.through_mlp(point_feature[0]) #b,256,64,64
    #     point_feature_mlp_output_1 = self.through_mlp(point_feature[1]) #b,256,32,32
    #     point_feature_mlp_output_2 = self.through_mlp(point_feature[2]) #b,256,16,16
    #     point_feature_mlp_output_3 = self.through_mlp(point_feature[3]) #b,256,8,8
    #     feats1_mlp_output = self.through_mlp(feats1) #b,256,64,64

    #     sam_semantic_feature *=self.scale
    #     sam_semantic_feature_mlp_output = self.through_mlp(sam_semantic_feature) #b,256,16,16

    #     sam_semantic_feature_conv_64x64 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output, size=(64, 64), mode='bilinear', align_corners=False))
    #     sam_semantic_feature_conv_32x32 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output, size=(32, 32), mode='bilinear', align_corners=False))
    #     sam_semantic_feature_conv_16x16 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output, size=(16, 16), mode='bilinear', align_corners=False))
    #     sam_semantic_feature_conv_8x8 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output, size=(8, 8), mode='bilinear', align_corners=False))

    #     refined_point_feature_0 = point_feature_mlp_output_0 + sam_semantic_feature_conv_64x64
    #     refined_point_feature_1 = point_feature_mlp_output_1 + sam_semantic_feature_conv_32x32
    #     refined_point_feature_2 = point_feature_mlp_output_2 + sam_semantic_feature_conv_16x16
    #     refined_point_feature_3 = point_feature_mlp_output_3 + sam_semantic_feature_conv_8x8

    #     # refined_feats1 = feats1_mlp_output + sam_semantic_feature_conv_64x64

    #     refined_feats1 = feats1 #不改
    #     return [refined_point_feature_0,refined_point_feature_1,refined_point_feature_2,refined_point_feature_3],refined_feats1
    

    # # 普通的k_2
    def forward(self,
                point_feature,feats1,sam_semantic_feature):
        #point_feature list 顺序处理
        
        point_feature_mlp_output_0 = self.through_mlp(point_feature[0]) #b,256,64,64
        point_feature_mlp_output_1 = self.through_mlp(point_feature[1]) #b,256,32,32
        point_feature_mlp_output_2 = self.through_mlp(point_feature[2]) #b,256,16,16
        point_feature_mlp_output_3 = self.through_mlp(point_feature[3]) #b,256,8,8
        feats1_mlp_output = self.through_mlp(feats1) #b,256,64,64

        sam_semantic_feature *=self.scale
        sam_semantic_feature_mlp_output = self.through_mlp(sam_semantic_feature) #b,256,16,16

        sam_semantic_feature_conv_64x64 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output, size=(64, 64), mode='bilinear', align_corners=False))
        sam_semantic_feature_conv_32x32 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output, size=(32, 32), mode='bilinear', align_corners=False))
        sam_semantic_feature_conv_16x16 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output, size=(16, 16), mode='bilinear', align_corners=False))
        sam_semantic_feature_conv_8x8 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output, size=(8, 8), mode='bilinear', align_corners=False))

        refined_point_feature_0 = point_feature_mlp_output_0 + sam_semantic_feature_conv_64x64
        refined_point_feature_1 = point_feature_mlp_output_1 + sam_semantic_feature_conv_32x32
        refined_point_feature_2 = point_feature_mlp_output_2 + sam_semantic_feature_conv_16x16
        refined_point_feature_3 = point_feature_mlp_output_3 + sam_semantic_feature_conv_8x8

        # refined_feats1 = feats1_mlp_output + sam_semantic_feature_conv_64x64

        refined_feats1 = feats1
        return [refined_point_feature_0,refined_point_feature_1,refined_point_feature_2,refined_point_feature_3],refined_feats1

    # 0123
    # def forward(self,
    #             point_feature,feats1,sam_semantic_feature):
    #     #point_feature list 顺序处理
        
    #     point_feature_mlp_output_0 = self.through_mlp(point_feature[0]) #b,256,64,64
    #     point_feature_mlp_output_1 = self.through_mlp(point_feature[1]) #b,256,32,32
    #     point_feature_mlp_output_2 = self.through_mlp(point_feature[2]) #b,256,16,16
    #     point_feature_mlp_output_3 = self.through_mlp(point_feature[3]) #b,256,8,8

    #     sam_semantic_feature_0 = sam_semantic_feature.clone()
    #     sam_semantic_feature_1 = sam_semantic_feature.clone()
    #     sam_semantic_feature_2 = sam_semantic_feature.clone()
    #     sam_semantic_feature_3 = sam_semantic_feature.clone()
    #     sam_semantic_feature_0 *=self.scale
    #     sam_semantic_feature_1 *=self.scale1
    #     sam_semantic_feature_2 *=self.scale2
    #     sam_semantic_feature_3 *=self.scale3


    #     sam_semantic_feature_mlp_output_0 = self.through_mlp(sam_semantic_feature_0) #b,256,16,16
    #     sam_semantic_feature_mlp_output_1 = self.through_mlp(sam_semantic_feature_1) #b,256,16,16
    #     sam_semantic_feature_mlp_output_2 = self.through_mlp(sam_semantic_feature_2) #b,256,16,16
    #     sam_semantic_feature_mlp_output_3 = self.through_mlp(sam_semantic_feature_3) #b,256,16,16

    #     sam_semantic_feature_conv_64x64 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output_0, size=(64, 64), mode='bilinear', align_corners=False))
    #     sam_semantic_feature_conv_32x32 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output_1, size=(32, 32), mode='bilinear', align_corners=False))
    #     sam_semantic_feature_conv_16x16 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output_2, size=(16, 16), mode='bilinear', align_corners=False))
    #     sam_semantic_feature_conv_8x8 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output_3, size=(8, 8), mode='bilinear', align_corners=False))

    #     refined_point_feature_0 = point_feature_mlp_output_0 + sam_semantic_feature_conv_64x64
    #     refined_point_feature_1 = point_feature_mlp_output_1 + sam_semantic_feature_conv_32x32
    #     refined_point_feature_2 = point_feature_mlp_output_2 + sam_semantic_feature_conv_16x16
    #     refined_point_feature_3 = point_feature_mlp_output_3 + sam_semantic_feature_conv_8x8

    #     # refined_feats1 = feats1_mlp_output + sam_semantic_feature_conv_64x64

    #     refined_feats1 = feats1
    #     return [refined_point_feature_0,refined_point_feature_1,refined_point_feature_2,refined_point_feature_3],refined_feats1

    # # 给feats[0]接cross-attention 注意scale用的是啥
    # def forward(self,
    #             point_feature,feats1,sam_semantic_feature):
    #     #point_feature list 顺序处理
        
    #     # point_feature_mlp_output_0 = self.through_mlp(point_feature[0]) #b,256,64,64

    #     sam_semantic_feature *=self.scale
    #     # sam_semantic_feature_mlp_output = self.through_mlp(sam_semantic_feature) #b,256,16,16
    #     point_feature_conv_16x16 = self.conv(nn.functional.interpolate(point_feature[0], size=(16, 16), mode='bilinear', align_corners=False))

    #     ie = point_feature_conv_16x16.clone()
    #     de = sam_semantic_feature.clone()

    #     ie_one = self.bn(self.head_layer(ie) + ie + de)
    #     de_one = self.bn(self.head_layer(de) + de + ie)

    #     # Flatten the tensor to (batch_size, seq_len, dim) before attention
    #     b, c, h, w = ie_one.shape
    #     ie_one_flat = ie_one.view(b, c, -1).transpose(1, 2)  # (batch_size, seq_len, dim)
    #     de_one_flat = de_one.view(b, c, -1).transpose(1, 2)  # (batch_size, seq_len, dim)

    #     ie_two_flat = self.attention_layer(ie_one_flat, ie_one_flat, ie_one_flat)
    #     de_two_flat = self.attention_layer(de_one_flat, de_one_flat, de_one_flat)

    #     # Reshape back to original shape after attention
    #     ie_two = ie_two_flat.transpose(1, 2).view(b, c, h, w)
    #     de_two = de_two_flat.transpose(1, 2).view(b, c, h, w)

    #     ie_three = self.bn(ie_one + ie_two + de_one)
    #     de_three = self.bn(de_one + de_two + ie_one)

    #     ie_four = self.mlp(ie_three.view(b, c, -1).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)
    #     de_four = self.mlp(de_three.view(b, c, -1).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)

    #     ie_out = self.bn(ie_three + ie_four + de_three)
    #     refined_point_feature_0 = self.conv(nn.functional.interpolate(ie_out, size=(64,64), mode='bilinear', align_corners=False))

    #     refined_feats1 = feats1
    #     return [refined_point_feature_0,point_feature[1],point_feature[2],point_feature[3]],refined_feats1

    # 给feats[0]接cross-attention ，给其他层+不同scale后的特征
    # def forward(self,
    #             point_feature,feats1,sam_semantic_feature):
    #     #point_feature list 顺序处理
        
    #     point_feature_mlp_output_1 = self.through_mlp(point_feature[1]) #b,256,64,64
    #     point_feature_mlp_output_2 = self.through_mlp(point_feature[2]) #b,256,16,16
    #     point_feature_mlp_output_3 = self.through_mlp(point_feature[3]) #b,256,8,8
    #     sam_semantic_feature_1 = sam_semantic_feature.clone()
    #     sam_semantic_feature_2 = sam_semantic_feature.clone()
    #     sam_semantic_feature_3 = sam_semantic_feature.clone()
    #     sam_semantic_feature_1  *= self.scale1
    #     sam_semantic_feature_2  *= self.scale2
    #     sam_semantic_feature_3  *= self.scale3
    #     sam_semantic_feature_mlp_output_1 = self.through_mlp(sam_semantic_feature_1) #b,256,16,16
    #     sam_semantic_feature_mlp_output_2 = self.through_mlp(sam_semantic_feature_2) #b,256,16,16
    #     sam_semantic_feature_mlp_output_3 = self.through_mlp(sam_semantic_feature_3) #b,256,16,16

    #     sam_semantic_feature_conv_32x32 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output_1, size=(32, 32), mode='bilinear', align_corners=False))
    #     sam_semantic_feature_conv_16x16 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output_2, size=(16, 16), mode='bilinear', align_corners=False))
    #     sam_semantic_feature_conv_8x8 = self.conv(nn.functional.interpolate(sam_semantic_feature_mlp_output_3, size=(8, 8), mode='bilinear', align_corners=False))

    #     refined_point_feature_1 = point_feature_mlp_output_1 + sam_semantic_feature_conv_32x32
    #     refined_point_feature_2 = point_feature_mlp_output_2 + sam_semantic_feature_conv_16x16
    #     refined_point_feature_3 = point_feature_mlp_output_3 + sam_semantic_feature_conv_8x8
    #     sam_semantic_feature *=self.scale
        
    #     point_feature_conv_16x16 = self.conv(nn.functional.interpolate(point_feature[0], size=(16, 16), mode='bilinear', align_corners=False))

    #     ie = point_feature_conv_16x16.clone()
    #     de = sam_semantic_feature.clone()

    #     ie_one = self.bn(self.head_layer(ie) + ie + de)
    #     de_one = self.bn(self.head_layer(de) + de + ie)

    #     # Flatten the tensor to (batch_size, seq_len, dim) before attention
    #     b, c, h, w = ie_one.shape
    #     ie_one_flat = ie_one.view(b, c, -1).transpose(1, 2)  # (batch_size, seq_len, dim)
    #     de_one_flat = de_one.view(b, c, -1).transpose(1, 2)  # (batch_size, seq_len, dim)

    #     ie_two_flat = self.attention_layer(ie_one_flat, ie_one_flat, ie_one_flat)
    #     de_two_flat = self.attention_layer(de_one_flat, de_one_flat, de_one_flat)

    #     # Reshape back to original shape after attention
    #     ie_two = ie_two_flat.transpose(1, 2).view(b, c, h, w)
    #     de_two = de_two_flat.transpose(1, 2).view(b, c, h, w)

    #     ie_three = self.bn(ie_one + ie_two + de_one)
    #     de_three = self.bn(de_one + de_two + ie_one)

    #     ie_four = self.mlp(ie_three.view(b, c, -1).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)
    #     de_four = self.mlp(de_three.view(b, c, -1).transpose(1, 2)).transpose(1, 2).view(b, c, h, w)

    #     ie_out = self.bn(ie_three + ie_four + de_three)
    #     refined_point_feature_0 = self.conv(nn.functional.interpolate(ie_out, size=(64,64), mode='bilinear', align_corners=False))

    #     refined_feats1 = feats1


    #     return [refined_point_feature_0,refined_point_feature_1,refined_point_feature_2,refined_point_feature_3],refined_feats1

class DPAP2PNet(nn.Module):
    """ This is the Proposal-aware P2PNet module that performs cell recognition """

    def __init__(
            self,
            backbone,
            sr_pfo,
            # sam_img_encoder,
            num_levels,
            num_classes,
            dropout=0.1,
            space: int = 16,
            hidden_dim: int = 256,
            with_mask=False
    ):
        """
            Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        # self.sam_img_encoder = sam_img_encoder
        self.get_aps = AnchorPoints(space)
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.with_mask = with_mask
        self.strides = [2 ** (i + 2) for i in range(self.num_levels)]

        self.deform_layer = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)

        self.reg_head = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)
        self.cls_head = MLP(hidden_dim, hidden_dim, 2, num_classes + 1, drop=dropout)

        self.conv = nn.Conv2d(hidden_dim * num_levels, hidden_dim, kernel_size=3, padding=1)

        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SyncBatchNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=1)
        )
        self.sr_pfo = sr_pfo

    def forward(self,
                images,semantic_feature):
        # extract features 我感觉就在这。。尺寸截图了 b c h w
        (feats, feats1) = self.backbone(images)
        if semantic_feature !=None:
            feats,feats1= self.sr_pfo(feats,feats1,semantic_feature)

        proposals =  self.get_aps(images) 
        embedding = feats
        feats_origin = feats
        feat_sizes = [torch.tensor(feat.shape[:1:-1], dtype=torch.float, device=proposals.device) for feat in feats]

        # DPP deformable point proposals
        grid = (2.0 * proposals / self.strides[0] / feat_sizes[0] - 1.0) #跟proposal长得一样 应该是归一化到-1到1之间
        roi_features = F.grid_sample(feats[0], grid, mode='bilinear', align_corners=True) #torch.Size([8, 32, 32, 2])
        deltas2deform = self.deform_layer(roi_features.permute(0, 2, 3, 1))
        deformed_proposals = proposals + deltas2deform

        # MSD  multi-scale decoding
        roi_features = []
        for i in range(self.num_levels):
            grid = (2.0 * deformed_proposals / self.strides[i] / feat_sizes[i] - 1.0)
            roi_features.append(F.grid_sample(feats[i], grid, mode='bilinear', align_corners=True))
        roi_features = torch.cat(roi_features, 1)   #torch.Size([8, 1024, 32, 32]) 这里做的就是作者说的，把point放进去
        #使用 bilinear 插值在特征图 feats[0] 上进行采样，获取感兴趣区域对应的特征。

        roi_features = self.conv(roi_features).permute(0, 2, 3, 1)
        deltas2refine = self.reg_head(roi_features)  #所以最后传给mlp的是这个
        pred_coords = deformed_proposals + deltas2refine

        pred_logits = self.cls_head(roi_features)

        output = {
            'pred_coords': pred_coords.flatten(1, 2),
            'pred_logits': pred_logits.flatten(1, 2),
            'pred_masks': F.interpolate(
                self.mask_head(feats1), size=images.shape[2:], mode='bilinear', align_corners=True)
        }

        return output,feats_origin,embedding,feats
'''
class DPAP2PNet(nn.Module):
    """ This is the Proposal-aware P2PNet module that performs cell recognition """

    def __init__(
            self,
            backbone,
            # sam_img_encoder,
            num_levels,
            num_classes,
            dropout=0.1,
            space: int = 16,
            hidden_dim: int = 256,
            with_mask=False
    ):
        """
            Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        # self.sam_img_encoder = sam_img_encoder
        self.get_aps = AnchorPoints(space)
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.with_mask = with_mask
        self.strides = [2 ** (i + 2) for i in range(self.num_levels)]

        self.deform_layer = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)

        self.reg_head = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)
        self.cls_head = MLP(hidden_dim, hidden_dim, 2, num_classes + 1, drop=dropout)

        self.conv = nn.Conv2d(hidden_dim * num_levels, hidden_dim, kernel_size=3, padding=1)

        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SyncBatchNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=1)
        )

    def forward(self,
                images):
        # extract features 我感觉就在这。。尺寸截图了 b c h w
        (feats, feats1), proposals = self.backbone(images), self.get_aps(images) 
        # img_embedding = self.sam_img_encoder(images)
        # embedding = []
        embedding = feats
        # embedding = [tensor.cuda() for tensor in embedding]
        feats_origin = feats
        conv0 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=4, padding=0).cuda() 
        conv1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0).cuda()  
        conv3 = nn.Conv2d(256, 256, kernel_size=2, stride=2, padding=0).cuda() 
        # embedding.append(conv0(img_embedding)) # torch.Size([8, 256, 128, 128])
        # embedding.append(conv1(img_embedding)) #torch.Size([8, 256, 64, 64])
        # embedding.append(img_embedding)         # torch.Size([8, 256, 32, 32])
        # embedding.append(conv3(img_embedding))  # torch.Size([8, 256, 16, 16])
        # feats[0]+=embedding[0]
        # feats[1]+=embedding[1]
        # feats[2]+=embedding[2]
        # feats[3]+=embedding[3]
        feat_sizes = [torch.tensor(feat.shape[:1:-1], dtype=torch.float, device=proposals.device) for feat in feats]

        # DPP
        grid = (2.0 * proposals / self.strides[0] / feat_sizes[0] - 1.0) #跟proposal长得一样的梯度
        roi_features = F.grid_sample(feats[0], grid, mode='bilinear', align_corners=True) #torch.Size([8, 32, 32, 2])
        deltas2deform = self.deform_layer(roi_features.permute(0, 2, 3, 1))
        deformed_proposals = proposals + deltas2deform

        # MSD
        roi_features = []
        for i in range(self.num_levels):
            grid = (2.0 * deformed_proposals / self.strides[i] / feat_sizes[i] - 1.0)
            roi_features.append(F.grid_sample(feats[i], grid, mode='bilinear', align_corners=True))
        roi_features = torch.cat(roi_features, 1)   #torch.Size([8, 1024, 32, 32]) 这里做的就是作者说的，把point放进去
        #使用 bilinear 插值在特征图 feats[0] 上进行采样，获取感兴趣区域对应的特征。

        roi_features = self.conv(roi_features).permute(0, 2, 3, 1)
        deltas2refine = self.reg_head(roi_features)  #所以最后传给mlp的是这个
        pred_coords = deformed_proposals + deltas2refine

        pred_logits = self.cls_head(roi_features)

        output = {
            'pred_coords': pred_coords.flatten(1, 2),
            'pred_logits': pred_logits.flatten(1, 2),
            'pred_masks': F.interpolate(
                self.mask_head(feats1), size=images.shape[2:], mode='bilinear', align_corners=True)
        }

        return output,feats_origin,embedding,feats
'''

def build_model(cfg):
    backbone = Backbone(cfg)
    sr_pfo = SR_PFO()
    # sam_img_encoder = ImageEncoderViT().cuda()
    # checkpoint = torch.load(
    #     '/data/hotaru/projects/PNS_tmp/segmentor/checkpoint/cpm17/cpm_cell_num/best.pth',
    #     map_location="cpu"
    # )
    # for name, param in sam_img_encoder.named_parameters():
    #     if 'image_encoder.'+name in checkpoint["model"]:
    #         param_data = checkpoint["model"]['image_encoder.'+name]
    #         param.data.copy_(param_data)
    
    model = DPAP2PNet(
        backbone,
        sr_pfo,
        # sam_img_encoder,
        num_levels=cfg.prompter.neck.num_outs,
        num_classes=cfg.data.num_classes,
        dropout=cfg.prompter.dropout,
        space=cfg.prompter.space,
        hidden_dim=cfg.prompter.hidden_dim
    )

    return model
 
 
class MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.view(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous().view(b, seq_len, num_heads * head_dim)
        return x

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        return out