import torch
import torch.nn as nn
import torch.nn.functional as F


class HinResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(HinResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)
        self.norm = nn.InstanceNorm2d(feature // 2, affine=True)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))

        out_1, out_2 = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([self.norm(out_1), out_2], dim=1)

        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out


def subnet(net_structure, init='xavier'):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init)
            else:
                return DenseBlock(channel_in, channel_out)
        elif net_structure == 'Resnet':
            return ResBlock(channel_in, channel_out)
        elif net_structure == 'HinResnet':
            return HinResBlock(channel_in, channel_out)
        else:
            return None
    return constructor


# class InvBlock(nn.Module):
#     def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=0.8):
#         super(InvBlock, self).__init__()
#         # channel_num: 3
#         # channel_split_num: 1

#         self.split_len1 = channel_split_num  # 1
#         self.split_len2 = channel_num - channel_split_num  # 2

#         self.clamp = clamp

#         self.F = subnet_constructor(self.split_len2, self.split_len1)
#         self.G = subnet_constructor(self.split_len1, self.split_len2)
#         self.H = subnet_constructor(self.split_len1, self.split_len2)

#         #in_channels = 3
#         self.invconv = InvertibleConv1x1(channel_num, LU_decomposed=True)
#         self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

#     def forward(self, x, rev=False):
#         if not rev:
#             # invert1x1conv
#             x, logdet = self.flow_permutation(x, logdet=0, rev=False)

#             # split to 1 channel and 2 channel.
#             x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

#             y1 = x1 + self.F(x2)  # 1 channel
#             self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
#             y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
#             out = torch.cat((y1, y2), 1)
#         else:
#             # split.
#             x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
#             self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
#             y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
#             y1 = x1 - self.F(y2)

#             x = torch.cat((y1, y2), 1)

#             # inv permutation
#             out, logdet = self.flow_permutation(x, logdet=0, rev=True)

#         return out


class InvBlock(nn.Module):
    def __init__(self, subnet_constructor, channel_num1, channel_num2, clamp=0.8):
        super(InvBlock, self).__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_num1  # 1
        self.split_len2 = channel_num2  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        #in_channels = 3
        # self.invconv = InvertibleConv1x1(channel_num, LU_decomposed=True)
        # self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    # def forward(self, x, rev=False):
    def forward(self, x1, x2, rev=False):
        if not rev:
            # invert1x1conv
            # x, logdet = self.flow_permutation(x, logdet=0, rev=False)

            # split to 1 channel and 2 channel.
            # x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

            y1 = x1 + self.F(x2)  # 1 channel
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
            out = torch.cat((y1, y2), 1)
        else:
            # split.
            # x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

            x = torch.cat((y1, y2), 1)
            print("rev_inn")
            # inv permutation
            out = x

        return out


class InvBlock_2(nn.Module):
    def __init__(self, subnet_constructor, channel_num1, channel_num2, clamp=0.8):
        super().__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_num1  # 1
        self.split_len2 = channel_num2  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        #in_channels = 3
        # self.invconv = InvertibleConv1x1(channel_num, LU_decomposed=True)
        # self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    # def forward(self, x, rev=False):
    def forward(self, x, rev=False):
        if not rev:
            # invert1x1conv
            # x, logdet = self.flow_permutation(x, logdet=0, rev=False)

            # split to 1 channel and 2 channel.
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

            y1 = x1 + self.F(x2)  # 1 channel
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
            out = torch.cat((y1, y2), 1)
        else:
            # split.
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

            x = torch.cat((y1, y2), 1)
            print("rev_inn")
            # inv permutation
            out = x

        return out

class InvBlock_2(nn.Module):
    def __init__(self, subnet_constructor, channel_num1, channel_num2, clamp=0.8):
        super().__init__()
        # channel_num: 3
        # channel_split_num: 1

        self.split_len1 = channel_num1  # 1
        self.split_len2 = channel_num2  # 2

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

        #in_channels = 3
        # self.invconv = InvertibleConv1x1(channel_num, LU_decomposed=True)
        # self.flow_permutation = lambda z, logdet, rev: self.invconv(z, logdet, rev)

    # def forward(self, x, rev=False):
    def forward(self, x,rev=False):
        if not rev:
            # invert1x1conv
            # x, logdet = self.flow_permutation(x, logdet=0, rev=False)

            # split to 1 channel and 2 channel.
            x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

            y1 = x1 + self.F(x2)  # 1 channel
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)  # 2 channel
            # out = torch.cat((y1, y2), 1)
        # else:
        #     # split.
        #     x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        #     self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
        #     y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
        #     y1 = x1 - self.F(y2)

        #     x = torch.cat((y1, y2), 1)
        #     print("rev_inn")
        #     # inv permutation
        #     out = x

        return y1, y2


class PatchExpanding_INN(nn.Module):
    def __init__(self, dim, out_channels, factor=1):
        super().__init__()
        self.dim = dim
        self.factor = factor
        self.expand = nn.Conv2d(dim, dim*4//factor, kernel_size=3, stride=1, padding=1)
        # self.norm = norm_layer(dim)
        # self.conv = nn.Conv2d(dim//factor, dim//factor, kernel_size=3, stride=1, padding=1)
        self.inn = InvBlock(subnet('HinResnet'), (dim*4//factor)//4, (dim*4//factor)//4)
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(dim*2//factor, dim//factor, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        

    def forward(self, x, z):
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
        # y = torch.zeros([B, (C*4//self.factor)//4, 2*H, 2*W])       # for test
        y[:, :, 0::2, 0::2] = x[:, 0:(C*4//self.factor)//4, :, :]
        y[:, :, 1::2, 0::2] = x[:, (C*4//self.factor)//4:(C*4//self.factor)//2, :, :]
        y[:, :, 0::2, 1::2] = x[:, (C*4//self.factor)//2:(C*4//self.factor)*3//4, :, :]
        y[:, :, 1::2, 1::2] = x[:, (C*4//self.factor)*3//4:C*4//self.factor, :, :]

        y = self.inn(z, y)
        y = self.double_conv(y)
        return y


class Up_INN(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
            self.inn = InvBlock(subnet('HinResnet'), in_channels // 2, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            # self.conv = DoubleConv(in_channels, out_channels)
            self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
            self.inn = InvBlock(subnet('HinResnet'), in_channels // 2, in_channels // 2)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        x = self.inn(x2, x1)
        return self.conv(x)