###########################################################################
# Created by: CASIA IVA 
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample,normalize
from ...nn import PAM_Module
from ...nn import CAM_Module
from ...nn import ClusteringModule
from .base import BaseNet


__all__ = ['DANet', 'get_danet']

class ClusterDANet(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ClusterDANet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = clusterDANetHead(2048, nclass, norm_layer)

    def forward(self, x):
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)
        x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)

        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])
        return tuple(outputs)
        
class clusterDANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(clusterDANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.ca = ClusteringModule(K=100.0, num_clusters=inter_channels, fix_init=True, channels=inter_channels, V_count_init=1.0)
        self.cc = ClusteringModule(K=100.0, num_clusters=inter_channels, fix_init=True, channels=inter_channels, V_count_init=1.0)
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        # Cell feature clustering.
        B, _, H, W = x.shape
        shape = {}
        shape['B'], shape['H'], shape['W'] = B, H, W

        out_feat1_reshape = feat1.permute(0,2,3,1).contiguous().view(B*H*W, -1) 
        UV_dist1 = self.ca(out_feat1_reshape, shape)       # Shape: [B*H*W,C'].            
        feature_sideout1 = UV_dist1.view(B,H,W,-1).permute(0,3,1,2).contiguous() # Shape: [B,C',H,W]   
        # Position attention
        sa_feat = self.sa(feature_sideout1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)


        feat2 = self.conv5c(x)
        # Cell feature clustering.
        out_feat2_reshape = feat2.permute(0,2,3,1).contiguous().view(B*H*W, -1) 
        UV_dist2 = self.cc(out_feat2_reshape, shape)       # Shape: [B*H*W,C'].            
        feature_sideout2 = UV_dist2.view(B,H,W,-1).permute(0,3,1,2).contiguous() # Shape: [B,C',H,W] 
        # Channel attention
        sc_feat = self.sc(feature_sideout2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)


def get_danet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
           root='~/.encoding/models', **kwargs):
    r"""ClusterDANet model from the paper `"Dual Attention Network for Scene Segmentation" 
    <https://arxiv.org/abs/1809.02983.pdf>` with an added clustering layer.
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = ClusterDANet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model

