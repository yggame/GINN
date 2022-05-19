# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import os, pdb
from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.functional import embedding

from .DualGCN_modules import ChannelGCN, SpatialGCN
from .util import basic_con, SegmentationHead, PatchMerging, PatchExpanding_d2, PatchExpanding_d4, Up, Down, PatchExpanding
from .InvBlock import *

import torchsummary

logger = logging.getLogger(__name__)


class GINN(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.inter_channels = inter_channels

        self.base_conv = basic_con(in_channels, inter_channels)

        self.cgcn1 = ChannelGCN(inter_channels)
        self.sgcn1 = SpatialGCN(inter_channels)
        self.inn1 = InvBlock(subnet('HinResnet'), inter_channels, inter_channels)
        self.conv1 = nn.Conv2d(inter_channels * 2, inter_channels//2, kernel_size=3, stride=1, padding=1)
        self.PatchMerging1 = PatchMerging(inter_channels//2)

        self.cgcn2 = ChannelGCN(inter_channels)
        self.sgcn2 = SpatialGCN(inter_channels)
        self.inn2 = InvBlock(subnet('HinResnet'), inter_channels, inter_channels)
        self.conv2 = nn.Conv2d(inter_channels * 2, inter_channels//2, kernel_size=3, stride=1, padding=1)
        self.PatchMerging2 = PatchMerging(inter_channels//2)

        self.cgcn3 = ChannelGCN(inter_channels)
        self.sgcn3 = SpatialGCN(inter_channels)
        self.inn3 = InvBlock(subnet('HinResnet'), inter_channels, inter_channels)
        self.conv3 = nn.Conv2d(inter_channels * 2, inter_channels//2, kernel_size=3, stride=1, padding=1)
        self.PatchMerging3 = PatchMerging(inter_channels//2)

        self.inn5 = InvBlock(subnet('HinResnet'), inter_channels//2, inter_channels//2)

        self.inn7 = InvBlock(subnet('HinResnet'), inter_channels, inter_channels)
        self.PatchExpanding3 = PatchExpanding(inter_channels*2, factor=2)

        self.inn8 = InvBlock(subnet('HinResnet'), inter_channels, inter_channels)
        self.PatchExpanding2 = PatchExpanding(inter_channels*2, factor=2)

        self.inn9 = InvBlock(subnet('HinResnet'), inter_channels, inter_channels)
        self.PatchExpanding1 = PatchExpanding(inter_channels*2, factor=2)

        self.inn10 = InvBlock(subnet('HinResnet'), inter_channels, inter_channels)

        self.segmentation_head = SegmentationHead(
            in_channels=inter_channels*2,
            out_channels=out_channels,
            kernel_size=3,
        )


    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)

        x = self.base_conv(x)     
        cg1 = self.cgcn1(x)      
        sg1 = self.sgcn1(x)    
        inn1 = self.conv1(self.inn1(cg1, sg1))  
        
        x2 = self.PatchMerging1(inn1) 

        cg2 = self.cgcn2(x2)        
        sg2 = self.sgcn2(x2)        
        inn2 = self.conv2(self.inn2(cg2, sg2)) 

        x3 = self.PatchMerging2(inn2) 

        cg3 = self.cgcn3(x3)       
        sg3 = self.sgcn3(x3)      
        inn3 = self.conv3(self.inn3(cg3, sg3))  

        x4 = self.PatchMerging3(inn3)

        x6 = self.inn5(x4.narrow(1, 0, self.inter_channels//2), x4.narrow(1, self.inter_channels//2, self.inter_channels//2))   # 16 14 14

        x8 = self.inn7(x6, x4) 
        x8 = self.PatchExpanding3(x8) 

        x9 = self.inn8(x8, x3) 
        x9 = self.PatchExpanding2(x9) 

        x10 = self.inn9(x9, x2)  
        x10 = self.PatchExpanding1(x10) 

        x11 = self.inn10(x10, x) 

        logits = self.segmentation_head(x11)
        return logits
    
    # TODO VisionTransformer test
    def test(self, device='cpu'):
        input_tensor = torch.rand(1, 3, 224, 224) 
        ideal_out = torch.rand(1, 9, 224, 224)
        out = self.forward(input_tensor)
        assert out.shape == ideal_out.shape
        import torchsummaryX
        torchsummaryX.summary(self, input_tensor.to(device))


if __name__ == "__main__":
    net3 = GINN(3, 16, 9)
    net3.test()
