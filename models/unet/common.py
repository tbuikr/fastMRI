import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2)+dilation-1, bias=bias, dilation=dilation)


def default_conv1(in_channels, out_channels, kernel_size, bias=True, groups=3):
    return nn.Conv2d(
        in_channels,out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, groups=groups)


def matrix_init():
    a = torch.linspace(1, 15, steps=15) - 8
    a = a.float()

    mat_1 = torch.mul(a, a)
    mat_1, mat_3 = torch.meshgrid([mat_1, mat_1])
    a = a.view(15, 1)
    mat_2 = torch.mul(a, a.t())
    mat_1 = mat_1.contiguous().view(1, 1, 225, 1, 1)
    mat_2 = mat_2.contiguous().view(1, 1, 225, 1, 1)
    mat_3 = mat_3.contiguous().view(1, 1, 225, 1, 1)

    return torch.cat((mat_1, mat_2, mat_3), 1).cuda()

class FDoCReLU(nn.Module):
    def __init__(self, stride = 1, in_channels = 64):
        super(FDoCReLU, self).__init__()
        self.requires_grad = True 
        self.stride = stride 
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, stride = stride, padding=1, bias=False, groups = in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size = 5, stride = stride, padding=2, bias=False, groups = in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size = 7, stride = stride, padding=3, bias=False, groups = in_channels)
         
        if stride == 2:
            self.pool1 = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out0 = x
        if self.stride == 2:
            out0 = self.pool1(x)
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))

        res1 = out0 - out1
        res2 = out1 - out2 
        res3 = out2 - out3
        res4 = out3 

        return torch.cat((res1, res2, res3, res4), 1)

 
class IDoCReLU(nn.Module) :   
    ### Need to update with stride > 1 

    def __init__(self, stride = 1, in_channels = 64)    :
        super(IDoCReLU, self).__init__()
        self.stride = stride 
        out_channels = int(in_channels // 4)

        if stride > 1:
            self.convt = nn.Upsample(scale_factor=2, mode='bicubic' , align_corners= True )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding=0, groups= out_channels)

    def forward(self, x):
        out = x 
        if self.stride > 1:
            out = F.relu(self.convt(x))
        out = self.conv1(out)

        return out

def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    del x   

    x_LL = x01[:, :, :, 0::2] + x02[:, :, :, 0::2] + x01[:, :, :, 1::2] + x02[:, :, :, 1::2]
    x_HL = -x01[:, :, :, 0::2] - x02[:, :, :, 0::2] + x01[:, :, :, 1::2] + x02[:, :, :, 1::2]
    x_LH = -x01[:, :, :, 0::2] + x02[:, :, :, 0::2] - x01[:, :, :, 1::2] + x02[:, :, :, 1::2]
    x_HH = x01[:, :, :, 0::2] - x02[:, :, :, 0::2] - x01[:, :, :, 1::2] + x02[:, :, :, 1::2]
    del x01, x02 

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    #print([in_batch, in_channel, in_height, in_width])
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r ** 2)), r * in_height, r * in_width
    
    x = x /2

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()

    h[:, :, 0::2, 0::2] = x[:, 0:out_channel, :, :] - x[:, out_channel:out_channel * 2, :, :] \
                        - x[:, out_channel * 2:out_channel * 3, :, :] + x[:, out_channel * 3:out_channel * 4, :, :]
    h[:, :, 1::2, 0::2] = x[:, 0:out_channel, :, :] - x[:, out_channel:out_channel * 2, :, :] \
                        + x[:, out_channel * 2:out_channel * 3, :, :] - x[:, out_channel * 3:out_channel * 4, :, :]
    h[:, :, 0::2, 1::2] = x[:, 0:out_channel, :, :] + x[:, out_channel:out_channel * 2, :, :] \
                        - x[:, out_channel * 2:out_channel * 3, :, :] - x[:, out_channel * 3:out_channel * 4, :, :]
    h[:, :, 1::2, 1::2] = x[:, 0:out_channel, :, :] + x[:, out_channel:out_channel * 2, :, :] \
                        + x[:, out_channel * 2:out_channel * 3, :, :] + x[:, out_channel * 3:out_channel * 4, :, :]

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

class Channel_Shuffle(nn.Module):
    def __init__(self, conv_groups):
        super(Channel_Shuffle, self).__init__()
        self.conv_groups = conv_groups
        self.requires_grad = False

    def forward(self, x):
        return channel_shuffle(x, self.conv_groups)

class SP(nn.Module):
    def __init__(self):
        super(SP, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return sp_init(x)

class Pixel_Down_Shuffle(nn.Module):
    def __init__(self):
        super(Pixel_Down_Shuffle, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return pixel_down_shuffle(x, 2)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class BBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(BBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x
        
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class DBlock_com(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=3))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_com1(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv1(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv1, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=1))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_com2(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_com2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class DBlock_inv2(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DBlock_inv2, self).__init__()
        m = []

        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias, dilation=2))
        if bn: m.append(nn.BatchNorm2d(out_channels, eps=1e-4, momentum=0.95))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x)
        return x

class ShuffleBlock(nn.Module):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,conv_groups=1):

        super(ShuffleBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        m.append(Channel_Shuffle(conv_groups))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x


class DWBlock(nn.Module):
    def __init__(
        self, conv, conv1, in_channels, out_channels, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(DWBlock, self).__init__()
        m = []
        m.append(conv(in_channels, out_channels, kernel_size, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)

        m.append(conv1(in_channels, out_channels, 1, bias=bias))
        if bn: m.append(nn.BatchNorm2d(out_channels))
        m.append(act)


        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        x = self.body(x).mul(self.res_scale)
        return x