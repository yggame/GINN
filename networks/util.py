
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


class basic_con(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(basic_con, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
            )
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, factor=2):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Conv2d(4 * dim, factor * dim, kernel_size=3, stride=1, padding=1)
        # self.norm = norm_layer(4 * dim, eps=1e-6)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        # H, W = self.input_resolution
        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, :, 0::2, 0::2]  # B H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2]  # B H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2]  # B H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], dim=1)  # B H/2 W/2 4*C
        # x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)

        return x

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class PatchExpanding_d4(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.expand = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(dim//4, dim//4, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        assert C % 4 == 0, f'channel {C} is not Multiple of 4'

        x = self.expand(x)
        y = torch.zeros([B, C//4, 2*H, 2*W]).cuda()
        y[:, :, 0::2, 0::2] = x[:, 0:C//4, :, :]
        y[:, :, 1::2, 0::2] = x[:, C//4:C//2, :, :]
        y[:, :, 0::2, 1::2] = x[:, C//2:C*3//4, :, :]
        y[:, :, 1::2, 1::2] = x[:, C*3//4:C, :, :]
        y = self.conv(y)
        return y


class PatchExpanding_d2(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.expand = nn.Conv2d(dim, 2 * dim, kernel_size=3, stride=1, padding=1)
        # self.norm = norm_layer(dim)
        self.conv = nn.Conv2d(dim//2, dim//2, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        # H, W = self.input_resolution
        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        assert C % 2 == 0, f'channel {C} is not even'

        x = self.expand(x)
        y = torch.zeros([B, C//2, 2*H, 2*W]).cuda()
        y[:, :, 0::2, 0::2] = x[:, 0:C//2, :, :]
        y[:, :, 1::2, 0::2] = x[:, C//2:C, :, :]
        y[:, :, 0::2, 1::2] = x[:, C:C*3//2, :, :]
        y[:, :, 1::2, 1::2] = x[:, C*3//2:C*2, :, :]
        y = self.conv(y)
        return y


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels*2, out_channels, in_channels)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.up(x)
        return self.conv(x)
        

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class PatchExpanding(nn.Module):
    def __init__(self, dim, factor=4):
        super().__init__()
        self.dim = dim
        self.factor = factor
        self.expand = nn.Conv2d(dim, dim*4//factor, kernel_size=3, stride=1, padding=1)
        # self.norm = norm_layer(dim)
        self.conv = nn.Conv2d(dim//factor, dim//factor, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        # H, W = self.input_resolution
        B, C, H, W = x.shape
        # assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        assert C % self.factor == 0, f'channel {C} is not Multiple of 4'

        x = self.expand(x) 
        y = torch.zeros([B, (C*4//self.factor)//4, 2*H, 2*W]).cuda()
        # y = torch.zeros([B, (C*4//self.factor)//4, 2*H, 2*W])
        y[:, :, 0::2, 0::2] = x[:, 0:(C*4//self.factor)//4, :, :]
        y[:, :, 1::2, 0::2] = x[:, (C*4//self.factor)//4:(C*4//self.factor)//2, :, :]
        y[:, :, 0::2, 1::2] = x[:, (C*4//self.factor)//2:(C*4//self.factor)*3//4, :, :]
        y[:, :, 1::2, 1::2] = x[:, (C*4//self.factor)*3//4:C*4//self.factor, :, :]
        y = self.conv(y)
        return y