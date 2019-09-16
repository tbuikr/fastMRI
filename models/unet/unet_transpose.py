import functools
from torch.nn import init
import torch.nn as nn
import torch

import functools
import torch
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
from torch.nn import init
from models.unet import common

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_network(net, init_weight_flag = True):
    init_weights(net)
    return net

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob, norm_layer):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=use_bias),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=use_bias),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'

class ConvResBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob, norm_layer):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.layers1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=use_bias),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob))
        self.layers2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=use_bias),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        out = self.layers1(input) 
        out = out +  self.layers2(out)
        return out

    def __repr__(self):
        return f'ConvResBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'


class ConvDilationBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, relu activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob, norm_layer):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=2, stride =1, dilation =2, bias=use_bias),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=use_bias),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return (input + self.layers(input))

    def __repr__(self):
        return f'ConvBlock(in_chans={self.in_chans}, out_chans={self.out_chans}, ' \
            f'drop_prob={self.drop_prob})'

class UnetGenerator(nn.Module):

    def __init__(self, in_chans=1, out_chans = 1, num_init_features=32,  norm_layer=nn.BatchNorm2d, drop_prob= 0.0):

        super(UnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        #--------------- Downsampling path ----------------------
        in_channel_block10, out_channel_block10 = in_chans, num_init_features
        self.block_down_10 = nn.Sequential(OrderedDict([
            ('conv_down_block_10', ConvBlock(in_channel_block10, num_init_features, drop_prob))     
        ]))   
        
        in_channel_block20, out_channel_block20 = out_channel_block10, out_channel_block10 * 2
        self.block_down_20 = nn.Sequential(OrderedDict([            
            ('conv_down_block_20', ConvBlock(in_channel_block20, out_channel_block20, drop_prob)),
            ('conv_down_block_21', nn.MaxPool2d(kernel_size=2))
        ]))   
        
        in_channel_block30, out_channel_block30 = out_channel_block20, out_channel_block20 * 2
        self.block_down_30 = nn.Sequential(OrderedDict([
             ('conv_down_block_30', ConvBlock(in_channel_block30, out_channel_block30, drop_prob)),
             ('conv_down_block_31', nn.MaxPool2d(kernel_size=2))
        ]))  
        
        in_channel_block40, out_channel_block40 = out_channel_block30, out_channel_block30 * 2
        self.block_down_40 = nn.Sequential(OrderedDict([
            ('conv_down_block_40', ConvBlock(in_channel_block40, out_channel_block40, drop_prob)),
            ('conv_down_block_41', nn.MaxPool2d(kernel_size=2))
        ]))       
        
        # --------Squeeze layer------------------
        in_channel_block_mid, out_channel_block_mid = out_channel_block40, out_channel_block40
        self.block_middle = nn.Sequential(OrderedDict([
            ('conv_down_block_middle', ConvBlock(in_channel_block_mid, out_channel_block_mid, drop_prob))           
        ]))   

        #-------------Upsampling path-------------- 
        in_channel_block41, out_channel_block41  = out_channel_block40, out_channel_block40
        self.block_up_41 = nn.Sequential(OrderedDict([
            ('conv41', nn.ConvTranspose2d(in_channel_block41, out_channel_block41, kernel_size=4, stride=2, padding=1, bias=use_bias)),
            ('norm41', norm_layer(out_channel_block41)),   
        ])) 
        in_channel_block_merge_41 = in_channel_block41 + out_channel_block30
        out_channel_block_merge_41 = in_channel_block_merge_41 // 2
        self.block_up_down_merge_41 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_41', ConvBlock(in_channel_block_merge_41, out_channel_block_merge_41, drop_prob))       
        ])) 
        
        in_channel_block31, out_channel_block31  = out_channel_block_merge_41, out_channel_block_merge_41
        self.block_up_31 = nn.Sequential(OrderedDict([
            ('conv31', nn.ConvTranspose2d(in_channel_block31, out_channel_block31, kernel_size=4, stride=2, padding=1, bias=use_bias)),
            ('norm31', norm_layer(out_channel_block31)),   
        ])) 
        in_channel_block_merge_31 = in_channel_block31 + out_channel_block20
        out_channel_block_merge_31 = in_channel_block_merge_31 // 2
        self.block_up_down_merge_31 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_31', ConvBlock(in_channel_block_merge_31, out_channel_block_merge_31, drop_prob))       
        ])) 

        in_channel_block21, out_channel_block21  = out_channel_block_merge_31, out_channel_block_merge_31
        self.block_up_21 = nn.Sequential(OrderedDict([
            ('conv21', nn.ConvTranspose2d(in_channel_block21, out_channel_block21, kernel_size=4, stride=2, padding=1, bias=use_bias)),
            ('norm21', norm_layer(out_channel_block21)),   
        ])) 
        in_channel_block_merge_21 = in_channel_block21 + out_channel_block10
        out_channel_block_merge_21 = in_channel_block_merge_31 // 2
        self.block_up_down_merge_21 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_21', ConvBlock(in_channel_block_merge_21, out_channel_block_merge_21, drop_prob))       
        ])) 
        # ----------Classifier--------------
        in_channel_block11, out_channel_block11 = out_channel_block_merge_21, out_chans
        self.block_classifier_11 = nn.Sequential(OrderedDict([             
            ('conv11', nn.Conv2d(in_channel_block11, in_channel_block11 //2 , kernel_size=1, stride=1, padding=0)),
            ('conv12', nn.Conv2d(in_channel_block11//2, out_channel_block11 , kernel_size=1, stride=1, padding=0)),
            ('conv13', nn.Conv2d(out_channel_block11, out_channel_block11, kernel_size=1, stride=1, padding=0)),
        ]))
               

    def forward(self, input):
        #------------Forward downsampling path-------------
        out_block_down_10 = self.block_down_10(input)
        out_block_down_20 = self.block_down_20(out_block_down_10)
        out_block_down_30 = self.block_down_30(out_block_down_20)
        out_block_down_40 = self.block_down_40(out_block_down_30)

        #------------Forward middle path-------------
        out_block_middle = self.block_middle(out_block_down_40)

        #------------Forward upsampling path-------------
        out_block_up_41 = self.block_up_41(out_block_middle)
        out_block_up_down_41 = torch.cat([out_block_up_41,out_block_down_30], 1)
        out_block_up_down_41_merge = self.block_up_down_merge_41(out_block_up_down_41)
        
        out_block_up_31 = self.block_up_31(out_block_up_down_41_merge)
        out_block_up_down_31 = torch.cat([out_block_up_31, out_block_down_20], 1) 
        out_block_up_down_31_merge = self.block_up_down_merge_31(out_block_up_down_31) 
        
        out_block_up_21 = self.block_up_21(out_block_up_down_31_merge)
        out_block_up_down_21 = torch.cat([out_block_up_21,out_block_down_10], 1)
        out_block_up_down_21_merge = self.block_up_down_merge_21(out_block_up_down_21) 
        out_block_classifier_11= self.block_classifier_11(out_block_up_down_21_merge)      
        return out_block_classifier_11  





"""
Squeeze and Excitation Module
*****************************
Collection of squeeze and excitation classes where each can be inserted as a block into a neural network architechture
    1. `Channel Squeeze and Excitation <https://arxiv.org/abs/1709.01507>`_
    2. `Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
    3. `Channel and Spatial Squeeze and Excitation <https://arxiv.org/abs/1803.02579>`_
"""

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class Upsample(nn.Module):
    def __init__(self,  scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)


class UnetUpsamplingGenerator(nn.Module):

    def __init__(self, in_chans=1, out_chans = 1, num_init_features=32,  norm_layer=nn.InstanceNorm2d, drop_prob= 0.0):

        super(UnetUpsamplingGenerator, self).__init__()
        #--------------- Downsampling path ----------------------
        in_channel_block10, out_channel_block10 = in_chans, num_init_features
        self.block_down_10 = nn.Sequential(OrderedDict([
            ('conv_down_block_10', ConvBlock(in_channel_block10, num_init_features, drop_prob,norm_layer )),
            #('conv_down_block_10_se', ChannelSpatialSELayer (num_init_features))    
        ]))   
        
        in_channel_block20, out_channel_block20 = out_channel_block10, out_channel_block10 * 2
        self.block_down_20 = nn.Sequential(OrderedDict([            
            ('conv_down_block_20', ConvBlock(in_channel_block20, out_channel_block20, drop_prob, norm_layer)),
            #('conv_down_block_20_se', ChannelSpatialSELayer (out_channel_block20)) 
            ('conv_down_block_21', nn.MaxPool2d(kernel_size=2))
        ]))   
        
        in_channel_block30, out_channel_block30 = out_channel_block20, out_channel_block20 * 2
        self.block_down_30 = nn.Sequential(OrderedDict([
             ('conv_down_block_30', ConvBlock(in_channel_block30, out_channel_block30, drop_prob, norm_layer)),
             ('conv_down_block_31', nn.MaxPool2d(kernel_size=2))
        ]))  
        
        in_channel_block40, out_channel_block40 = out_channel_block30, out_channel_block30 * 2
        self.block_down_40 = nn.Sequential(OrderedDict([
            ('conv_down_block_40', ConvBlock(in_channel_block40, out_channel_block40, drop_prob, norm_layer)),
            ('conv_down_block_41', nn.MaxPool2d(kernel_size=2))
        ]))

        in_channel_block50, out_channel_block50 = out_channel_block40, out_channel_block40
        self.block_down_50 = nn.Sequential(OrderedDict([
            ('conv_down_block_50', ConvBlock(in_channel_block50, out_channel_block50, drop_prob, norm_layer)),
            ('conv_down_block_51', nn.MaxPool2d(kernel_size=2))
        ]))    
        
        # --------Squeeze layer------------------
        in_channel_block_mid, out_channel_block_mid = out_channel_block50, out_channel_block50
        self.block_middle = nn.Sequential(OrderedDict([
            ('conv_down_block_middle', ConvBlock(in_channel_block_mid, out_channel_block_mid, drop_prob, norm_layer))           
        ]))  

        #-------------Upsampling path-------------- 
        in_channel_block51, out_channel_block51  = out_channel_block_mid + out_channel_block40 , (out_channel_block_mid + out_channel_block40) //4
        self.block_up_merge_51 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_51', ConvBlock(in_channel_block51, out_channel_block51, drop_prob, norm_layer))  
        ])) 

        in_channel_block41  = out_channel_block51 + out_channel_block30
        out_channel_block41 = (out_channel_block51 + out_channel_block30)//4
        self.block_up_merge_41 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_41', ConvBlock(in_channel_block41, out_channel_block41, drop_prob, norm_layer))  
        ])) 

        in_channel_block31  = out_channel_block41 + out_channel_block20
        out_channel_block31 = (out_channel_block41 + out_channel_block20)//4
        self.block_up_merge_31 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_31', ConvBlock(in_channel_block31, out_channel_block31, drop_prob, norm_layer))  
        ])) 

        in_channel_block21  = out_channel_block31 + out_channel_block10
        out_channel_block21 = (out_channel_block31 + out_channel_block10)//2
        self.block_up_merge_21 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_21', ConvBlock(in_channel_block21, out_channel_block21, drop_prob, norm_layer))  
        ])) 

        # ----------Classifier--------------
        in_channel_block11, out_channel_block11 = out_channel_block21, out_chans
        self.block_classifier_11 = nn.Sequential(OrderedDict([             
            ('conv11', nn.Conv2d(in_channel_block11, in_channel_block11 //2 , kernel_size=1, stride=1, padding=0)),
            ('conv12', nn.Conv2d(in_channel_block11//2, out_channel_block11 , kernel_size=1, stride=1, padding=0)),
            ('conv13', nn.Conv2d(out_channel_block11, out_channel_block11, kernel_size=1, stride=1, padding=0)),
        ]))
               

    def forward(self, input):
        #------------Forward downsampling path-------------
        out_block_down_10 = self.block_down_10(input)
        out_block_down_20 = self.block_down_20(out_block_down_10)
        out_block_down_30 = self.block_down_30(out_block_down_20)
        out_block_down_40 = self.block_down_40(out_block_down_30)
        out_block_down_50 = self.block_down_50(out_block_down_40)
       
        #------------Forward middle path-------------
        out_block_middle = self.block_middle(out_block_down_50)
        #------------Forward upsampling path-------------
        out_block_up_51 = F.interpolate(out_block_middle, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_51 = torch.cat([out_block_up_51,out_block_down_40], 1)
        out_block_up_down_51_merge = self.block_up_merge_51(out_block_up_down_51)

        out_block_up_41 = F.interpolate(out_block_up_down_51_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_41 = torch.cat([out_block_up_41,out_block_down_30], 1)
        out_block_up_down_41_merge = self.block_up_merge_41(out_block_up_down_41)
        
        out_block_up_31 = F.interpolate(out_block_up_down_41_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_31 = torch.cat([out_block_up_31, out_block_down_20], 1) 
        out_block_up_down_31_merge = self.block_up_merge_31(out_block_up_down_31) 
        
        out_block_up_21 = F.interpolate(out_block_up_down_31_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_21 = torch.cat([out_block_up_21,out_block_down_10], 1)
        out_block_up_down_21_merge = self.block_up_merge_21(out_block_up_down_21) 

        out_block_classifier_11= self.block_classifier_11(out_block_up_down_21_merge)      
        return out_block_classifier_11  


class UnetUpsamplingDilationGenerator(nn.Module):

    def __init__(self, in_chans=1, out_chans = 1, num_init_features=32,  norm_layer=nn.InstanceNorm2d, drop_prob= 0.0):

        super(UnetUpsamplingDilationGenerator, self).__init__()
        #--------------- Downsampling path ----------------------
        in_channel_block10, out_channel_block10 = in_chans, num_init_features
        self.block_down_10 = nn.Sequential(OrderedDict([
            ('conv_down_block_10', ConvResBlock(in_channel_block10, num_init_features, drop_prob,norm_layer )),
            #('conv_down_block_10_se', ChannelSpatialSELayer (num_init_features))    
        ]))   
        
        in_channel_block20, out_channel_block20 = out_channel_block10, out_channel_block10 * 2
        self.block_down_20 = nn.Sequential(OrderedDict([            
            ('conv_down_block_20', ConvResBlock(in_channel_block20, out_channel_block20, drop_prob, norm_layer)),
            #('conv_down_block_20_se', ChannelSpatialSELayer (out_channel_block20)) 
            ('conv_down_block_21', nn.MaxPool2d(kernel_size=2))
        ]))   
        
        in_channel_block30, out_channel_block30 = out_channel_block20, out_channel_block20 * 2
        self.block_down_30 = nn.Sequential(OrderedDict([
             ('conv_down_block_30', ConvResBlock(in_channel_block30, out_channel_block30, drop_prob, norm_layer)),
             ('conv_down_block_31', nn.MaxPool2d(kernel_size=2))
        ]))  
        
        in_channel_block40, out_channel_block40 = out_channel_block30, out_channel_block30 * 2
        self.block_down_40 = nn.Sequential(OrderedDict([
            ('conv_down_block_40', ConvResBlock(in_channel_block40, out_channel_block40, drop_prob, norm_layer)),
            ('conv_down_block_41', nn.MaxPool2d(kernel_size=2))
        ]))

        in_channel_block50, out_channel_block50 = out_channel_block40, out_channel_block40
        self.block_down_50 = nn.Sequential(OrderedDict([
            ('conv_down_block_50', ConvResBlock(in_channel_block50, out_channel_block50, drop_prob, norm_layer)),
            ('conv_down_block_51', nn.MaxPool2d(kernel_size=2))
        ]))    
        
        in_channel_block60, out_channel_block60 = out_channel_block50, out_channel_block50
        self.block_down_60 = nn.Sequential(OrderedDict([
            ('conv_down_block_60', ConvDilationBlock(in_channel_block60, out_channel_block60, drop_prob, norm_layer)),
            #('conv_down_block_61', nn.MaxPool2d(kernel_size=2))
        ]))   

        # --------Squeeze layer------------------
        in_channel_block_mid, out_channel_block_mid = out_channel_block60, out_channel_block60
        self.block_middle = nn.Sequential(OrderedDict([
            ('conv_down_block_middle', ConvResBlock(in_channel_block_mid, out_channel_block_mid, drop_prob, norm_layer))           
        ]))  

        #-------------Upsampling path-------------- 
        in_channel_block61, out_channel_block61  = out_channel_block_mid + out_channel_block50 , (out_channel_block_mid + out_channel_block50) //2
        self.up_conv_dilation_6 = nn.Conv2d(out_channel_block_mid, out_channel_block_mid, 1, 1)
        self.block_up_merge_61 = nn.Sequential(OrderedDict([           
            ('conv_up_down_merge_block_61', ConvResBlock(in_channel_block61, out_channel_block61, drop_prob, norm_layer)) 
            
        ])) 

        #in_channel_block51, out_channel_block51  = out_channel_block_mid + out_channel_block40 , (out_channel_block_mid + out_channel_block40) //4
        in_channel_block51  = out_channel_block61 + out_channel_block40
        out_channel_block51 = (out_channel_block61 + out_channel_block40)//4

        self.block_up_merge_51 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_51', ConvResBlock(in_channel_block51, out_channel_block51, drop_prob, norm_layer))  
        ])) 

        in_channel_block41  = out_channel_block51 + out_channel_block30
        out_channel_block41 = (out_channel_block51 + out_channel_block30)//4
        self.block_up_merge_41 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_41', ConvResBlock(in_channel_block41, out_channel_block41, drop_prob, norm_layer))  
        ])) 

        in_channel_block31  = out_channel_block41 + out_channel_block20
        out_channel_block31 = (out_channel_block41 + out_channel_block20)//4
        self.block_up_merge_31 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_31', ConvResBlock(in_channel_block31, out_channel_block31, drop_prob, norm_layer))  
        ])) 

        in_channel_block21  = out_channel_block31 + out_channel_block10
        out_channel_block21 = (out_channel_block31 + out_channel_block10)//2
        self.block_up_merge_21 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_21', ConvResBlock(in_channel_block21, out_channel_block21, drop_prob, norm_layer))  
        ])) 

        # ----------Classifier--------------
        in_channel_block11, out_channel_block11 = out_channel_block21, out_chans
        self.block_classifier_11 = nn.Sequential(OrderedDict([             
            ('conv11', nn.Conv2d(in_channel_block11, in_channel_block11 //2 , kernel_size=1, stride=1, padding=0)),
            ('conv12', nn.Conv2d(in_channel_block11//2, out_channel_block11 , kernel_size=1, stride=1, padding=0)),
            ('conv13', nn.Conv2d(out_channel_block11, out_channel_block11, kernel_size=1, stride=1, padding=0)),
        ]))
               

    def forward(self, input):
        #------------Forward downsampling path-------------
        out_block_down_10 = self.block_down_10(input)
        out_block_down_20 = self.block_down_20(out_block_down_10)
        out_block_down_30 = self.block_down_30(out_block_down_20)
        out_block_down_40 = self.block_down_40(out_block_down_30)
        out_block_down_50 = self.block_down_50(out_block_down_40)
        out_block_down_60 = self.block_down_60(out_block_down_50)
         #------------Forward middle path-------------
        out_block_middle = self.block_middle(out_block_down_60)
        #print (out_block_middle.size())
        #------------Forward upsampling path-------------
        out_block_up_61 = self.up_conv_dilation_6(out_block_middle)
        #print (out_block_up_61.size())
        out_block_up_down_61 = torch.cat([out_block_up_61,out_block_down_50], 1)
        #print (out_block_up_down_61.size())
        out_block_up_down_61_merge = self.block_up_merge_61(out_block_up_down_61)
        #print (out_block_up_down_61_merge.size())
        out_block_up_51 = F.interpolate(out_block_up_down_61_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_51 = torch.cat([out_block_up_51,out_block_down_40], 1)
        out_block_up_down_51_merge = self.block_up_merge_51(out_block_up_down_51)

        out_block_up_41 = F.interpolate(out_block_up_down_51_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_41 = torch.cat([out_block_up_41,out_block_down_30], 1)
        out_block_up_down_41_merge = self.block_up_merge_41(out_block_up_down_41)
        
        out_block_up_31 = F.interpolate(out_block_up_down_41_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_31 = torch.cat([out_block_up_31, out_block_down_20], 1) 
        out_block_up_down_31_merge = self.block_up_merge_31(out_block_up_down_31) 
        
        out_block_up_21 = F.interpolate(out_block_up_down_31_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_21 = torch.cat([out_block_up_21,out_block_down_10], 1)
        out_block_up_down_21_merge = self.block_up_merge_21(out_block_up_down_21) 

        out_block_classifier_11= self.block_classifier_11(out_block_up_down_21_merge)      
        return out_block_classifier_11  


class UnetUpsamplingDilationResGenerator(nn.Module):

    def __init__(self, in_chans=1, out_chans = 1, num_init_features=32,  norm_layer=nn.InstanceNorm2d, drop_prob= 0.0):

        super(UnetUpsamplingDilationResGenerator, self).__init__()
        #--------------- Downsampling path ----------------------
        in_channel_block10, out_channel_block10 = in_chans, num_init_features
        self.block_down_10 = nn.Sequential(OrderedDict([
            ('conv_down_block_10', ConvResBlock(in_channel_block10, num_init_features, drop_prob,norm_layer )),
            #('conv_down_block_10_se', ChannelSpatialSELayer (num_init_features))    
        ]))   
        
        in_channel_block20, out_channel_block20 = out_channel_block10, out_channel_block10 * 2
        self.block_down_20 = nn.Sequential(OrderedDict([            
            ('conv_down_block_20', ConvResBlock(in_channel_block20, out_channel_block20, drop_prob, norm_layer)),
            #('conv_down_block_20_se', ChannelSpatialSELayer (out_channel_block20)) 
            ('conv_down_block_21', nn.MaxPool2d(kernel_size=2))
        ]))   
        
        in_channel_block30, out_channel_block30 = out_channel_block20, out_channel_block20 * 2
        self.block_down_30 = nn.Sequential(OrderedDict([
             ('conv_down_block_30', ConvResBlock(in_channel_block30, out_channel_block30, drop_prob, norm_layer)),
             ('conv_down_block_31', nn.MaxPool2d(kernel_size=2))
        ]))  
        
        in_channel_block40, out_channel_block40 = out_channel_block30, out_channel_block30 * 2
        self.block_down_40 = nn.Sequential(OrderedDict([
            ('conv_down_block_40', ConvResBlock(in_channel_block40, out_channel_block40, drop_prob, norm_layer)),
            ('conv_down_block_41', nn.MaxPool2d(kernel_size=2))
        ]))

        in_channel_block50, out_channel_block50 = out_channel_block40, out_channel_block40
        self.block_down_50 = nn.Sequential(OrderedDict([
            ('conv_down_block_50', ConvResBlock(in_channel_block50, out_channel_block50, drop_prob, norm_layer)),
            ('conv_down_block_51', nn.MaxPool2d(kernel_size=2))
        ]))    
        
        in_channel_block60, out_channel_block60 = out_channel_block50, out_channel_block50
        self.block_down_60 = nn.Sequential(OrderedDict([
            ('conv_down_block_60', ConvDilationBlock(in_channel_block60, out_channel_block60, drop_prob, norm_layer)),
            #('conv_down_block_61', nn.MaxPool2d(kernel_size=2))
        ]))   

        # --------Squeeze layer------------------
        in_channel_block_mid, out_channel_block_mid = out_channel_block60, out_channel_block60
        self.block_middle = nn.Sequential(OrderedDict([
            ('conv_down_block_middle', ConvResBlock(in_channel_block_mid, out_channel_block_mid, drop_prob, norm_layer))           
        ]))  

        #-------------Upsampling path-------------- 
        in_channel_block61, out_channel_block61  = out_channel_block_mid + out_channel_block50 , (out_channel_block_mid + out_channel_block50) //2
        self.up_conv_dilation_6 = nn.Conv2d(out_channel_block_mid, out_channel_block_mid, 1, 1)
        self.block_up_merge_61 = nn.Sequential(OrderedDict([           
            ('conv_up_down_merge_block_61', ConvResBlock(in_channel_block61, out_channel_block61, drop_prob, norm_layer)) 
            
        ])) 

        #in_channel_block51, out_channel_block51  = out_channel_block_mid + out_channel_block40 , (out_channel_block_mid + out_channel_block40) //4
        in_channel_block51  = out_channel_block61 + out_channel_block40
        out_channel_block51 = (out_channel_block61 + out_channel_block40)//4

        self.block_up_merge_51 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_51', ConvResBlock(in_channel_block51, out_channel_block51, drop_prob, norm_layer))  
        ])) 

        in_channel_block41  = out_channel_block51 + out_channel_block30
        out_channel_block41 = (out_channel_block51 + out_channel_block30)//4
        self.block_up_merge_41 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_41', ConvResBlock(in_channel_block41, out_channel_block41, drop_prob, norm_layer))  
        ])) 

        in_channel_block31  = out_channel_block41 + out_channel_block20
        out_channel_block31 = (out_channel_block41 + out_channel_block20)//4
        self.block_up_merge_31 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_31', ConvResBlock(in_channel_block31, out_channel_block31, drop_prob, norm_layer))  
        ])) 

        in_channel_block21  = out_channel_block31 + out_channel_block10
        out_channel_block21 = (out_channel_block31 + out_channel_block10)//2
        self.block_up_merge_21 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_21', ConvResBlock(in_channel_block21, out_channel_block21, drop_prob, norm_layer))  
        ])) 
        # Conv reduction for concat
        in_channel_block61_reduction, out_channel_block61_reduction = out_channel_block61, out_channel_block21
        self.block_up_merge_61_reduction =  nn.Sequential(OrderedDict([
            ('conv_up_down_merge_reduction_block_61', ConvResBlock(in_channel_block61_reduction, out_channel_block61_reduction, drop_prob, norm_layer))  
        ])) 
        in_channel_block51_reduction, out_channel_block51_reduction = out_channel_block51, out_channel_block21
        self.block_up_merge_51_reduction =  nn.Sequential(OrderedDict([
            ('conv_up_down_merge_reduction_block_51', ConvResBlock(in_channel_block51_reduction, out_channel_block51_reduction, drop_prob, norm_layer))  
        ])) 
        in_channel_block41_reduction, out_channel_block41_reduction = out_channel_block41, out_channel_block21
        self.block_up_merge_41_reduction =  nn.Sequential(OrderedDict([
            ('conv_up_down_merge_reduction_block_41', ConvResBlock(in_channel_block41_reduction, out_channel_block41_reduction, drop_prob, norm_layer))  
        ])) 
        in_channel_block31_reduction, out_channel_block31_reduction = out_channel_block31, out_channel_block21
        self.block_up_merge_31_reduction =  nn.Sequential(OrderedDict([
            ('conv_up_down_merge_reduction_block_31', ConvResBlock(in_channel_block31_reduction, out_channel_block31_reduction, drop_prob, norm_layer))  
        ])) 

        # ----------Classifier--------------
        in_channel_block11, out_channel_block11 = out_channel_block21*5, out_chans
        self.block_classifier_11 = nn.Sequential(OrderedDict([             
            ('conv11', nn.Conv2d(in_channel_block11, in_channel_block11 //2 , kernel_size=1, stride=1, padding=0)),
            ('conv12', nn.Conv2d(in_channel_block11//2, out_channel_block11 , kernel_size=1, stride=1, padding=0)),
            ('conv13', nn.Conv2d(out_channel_block11, out_channel_block11, kernel_size=1, stride=1, padding=0)),
        ]))
               

    def forward(self, input):
        #------------Forward downsampling path-------------
        out_block_down_10 = self.block_down_10(input)
        out_block_down_20 = self.block_down_20(out_block_down_10)
        out_block_down_30 = self.block_down_30(out_block_down_20)
        out_block_down_40 = self.block_down_40(out_block_down_30)
        out_block_down_50 = self.block_down_50(out_block_down_40)
        out_block_down_60 = self.block_down_60(out_block_down_50)
         #------------Forward middle path-------------
        out_block_middle = self.block_middle(out_block_down_60)
        #print (out_block_middle.size())
        #------------Forward upsampling path-------------
        out_block_up_61 = self.up_conv_dilation_6(out_block_middle)
        out_block_up_down_61 = torch.cat([out_block_up_61,out_block_down_50], 1)
        out_block_up_down_61_merge = self.block_up_merge_61(out_block_up_down_61)

        out_block_up_51 = F.interpolate(out_block_up_down_61_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_51 = torch.cat([out_block_up_51,out_block_down_40], 1)
        out_block_up_down_51_merge = self.block_up_merge_51(out_block_up_down_51)

        out_block_up_41 = F.interpolate(out_block_up_down_51_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_41 = torch.cat([out_block_up_41,out_block_down_30], 1)
        out_block_up_down_41_merge = self.block_up_merge_41(out_block_up_down_41)
        
        out_block_up_31 = F.interpolate(out_block_up_down_41_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_31 = torch.cat([out_block_up_31, out_block_down_20], 1) 
        out_block_up_down_31_merge = self.block_up_merge_31(out_block_up_down_31) 
        
        out_block_up_21 = F.interpolate(out_block_up_down_31_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_21 = torch.cat([out_block_up_21,out_block_down_10], 1)
        out_block_up_down_21_merge = self.block_up_merge_21(out_block_up_down_21) 

        out_block_up_down_61_merge_up = F.interpolate(self.block_up_merge_61_reduction(out_block_up_down_61_merge), scale_factor=16, mode='bilinear', align_corners=False)
        out_block_up_down_51_merge_up = F.interpolate(self.block_up_merge_51_reduction(out_block_up_down_51_merge), scale_factor=8, mode='bilinear', align_corners=False)
        out_block_up_down_41_merge_up = F.interpolate(self.block_up_merge_41_reduction(out_block_up_down_41_merge), scale_factor=4, mode='bilinear', align_corners=False)
        out_block_up_down_31_merge_up = F.interpolate(self.block_up_merge_31_reduction(out_block_up_down_31_merge), scale_factor=2, mode='bilinear', align_corners=False)

        out_block_up_down_merge_up_cat = torch.cat([out_block_up_down_61_merge_up, out_block_up_down_51_merge_up, out_block_up_down_41_merge_up, out_block_up_down_31_merge_up, out_block_up_down_21_merge] ,1)
        out_block_classifier_11= self.block_classifier_11(out_block_up_down_merge_up_cat)      
        return out_block_classifier_11  



class UnetUpsamplingscSEGenerator(nn.Module):

    def __init__(self, in_chans=1, out_chans = 1, num_init_features=32,  norm_layer=nn.InstanceNorm2d, drop_prob= 0.0):

        super(UnetUpsamplingscSEGenerator, self).__init__()
        #--------------- Downsampling path ----------------------
        in_channel_block10, out_channel_block10 = in_chans, num_init_features
        self.block_down_10 = nn.Sequential(OrderedDict([
            ('conv_down_block_10', ConvBlock(in_channel_block10, num_init_features, drop_prob,norm_layer )),
            ('conv_down_block_10_se', ChannelSpatialSELayer (num_init_features))    
        ]))   
        
        in_channel_block20, out_channel_block20 = out_channel_block10, out_channel_block10 * 2
        self.block_down_20 = nn.Sequential(OrderedDict([            
            ('conv_down_block_20', ConvBlock(in_channel_block20, out_channel_block20, drop_prob, norm_layer)),
            ('conv_down_block_20_se', ChannelSpatialSELayer (out_channel_block20)), 
            ('conv_down_block_21', nn.MaxPool2d(kernel_size=2))
        ]))   
        
        in_channel_block30, out_channel_block30 = out_channel_block20, out_channel_block20 * 2
        self.block_down_30 = nn.Sequential(OrderedDict([
             ('conv_down_block_30', ConvBlock(in_channel_block30, out_channel_block30, drop_prob, norm_layer)),
             ('conv_down_block_30_se', ChannelSpatialSELayer (out_channel_block30)),
             ('conv_down_block_31', nn.MaxPool2d(kernel_size=2))
        ]))  
        
        in_channel_block40, out_channel_block40 = out_channel_block30, out_channel_block30 * 2
        self.block_down_40 = nn.Sequential(OrderedDict([
            ('conv_down_block_40', ConvBlock(in_channel_block40, out_channel_block40, drop_prob, norm_layer)),
            ('conv_down_block_40_se', ChannelSpatialSELayer (out_channel_block40)),
            ('conv_down_block_41', nn.MaxPool2d(kernel_size=2))
        ]))

        in_channel_block50, out_channel_block50 = out_channel_block40, out_channel_block40
        self.block_down_50 = nn.Sequential(OrderedDict([
            ('conv_down_block_50', ConvBlock(in_channel_block50, out_channel_block50, drop_prob, norm_layer)),
            ('conv_down_block_50_se', ChannelSpatialSELayer (out_channel_block50)),
            ('conv_down_block_51', nn.MaxPool2d(kernel_size=2))
        ]))    
        
        # --------Squeeze layer------------------
        in_channel_block_mid, out_channel_block_mid = out_channel_block50, out_channel_block50
        self.block_middle = nn.Sequential(OrderedDict([
            ('conv_down_block_middle', ConvBlock(in_channel_block_mid, out_channel_block_mid, drop_prob, norm_layer)),
            ('conv_down_block_middle_se', ChannelSpatialSELayer (out_channel_block_mid)),          
        ]))  

        #-------------Upsampling path-------------- 
        in_channel_block51, out_channel_block51  = out_channel_block_mid + out_channel_block40 , (out_channel_block_mid + out_channel_block40) //4
        self.block_up_merge_51 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_51', ConvBlock(in_channel_block51, out_channel_block51, drop_prob, norm_layer)),
            ('conv_up_down_merge_block_51_se', ChannelSpatialSELayer (out_channel_block51)), 
        ])) 

        in_channel_block41  = out_channel_block51 + out_channel_block30
        out_channel_block41 = (out_channel_block51 + out_channel_block30)//4
        self.block_up_merge_41 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_41', ConvBlock(in_channel_block41, out_channel_block41, drop_prob, norm_layer)),
            ('conv_up_down_merge_block_41_se', ChannelSpatialSELayer (out_channel_block41)), 
        ])) 

        in_channel_block31  = out_channel_block41 + out_channel_block20
        out_channel_block31 = (out_channel_block41 + out_channel_block20)//4
        self.block_up_merge_31 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_31', ConvBlock(in_channel_block31, out_channel_block31, drop_prob, norm_layer)),
            ('conv_up_down_merge_block_31_se', ChannelSpatialSELayer (out_channel_block31)),   
        ])) 

        in_channel_block21  = out_channel_block31 + out_channel_block10
        out_channel_block21 = (out_channel_block31 + out_channel_block10)//2
        self.block_up_merge_21 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_21', ConvBlock(in_channel_block21, out_channel_block21, drop_prob, norm_layer)),
            ('conv_up_down_merge_block_21_se', ChannelSpatialSELayer (out_channel_block21)), 
        ])) 

        # ----------Classifier--------------
        in_channel_block11, out_channel_block11 = out_channel_block21, out_chans
        self.block_classifier_11 = nn.Sequential(OrderedDict([             
            ('conv11', nn.Conv2d(in_channel_block11, in_channel_block11 //2 , kernel_size=1, stride=1, padding=0)),
            ('conv12', nn.Conv2d(in_channel_block11//2, out_channel_block11 , kernel_size=1, stride=1, padding=0)),
            ('conv13', nn.Conv2d(out_channel_block11, out_channel_block11, kernel_size=1, stride=1, padding=0)),
        ]))
               

    def forward(self, input):
        #------------Forward downsampling path-------------
        out_block_down_10 = self.block_down_10(input)
        out_block_down_20 = self.block_down_20(out_block_down_10)
        out_block_down_30 = self.block_down_30(out_block_down_20)
        out_block_down_40 = self.block_down_40(out_block_down_30)
        out_block_down_50 = self.block_down_50(out_block_down_40)
       
        #------------Forward middle path-------------
        out_block_middle = self.block_middle(out_block_down_50)
        #------------Forward upsampling path-------------
        out_block_up_51 = F.interpolate(out_block_middle, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_51 = torch.cat([out_block_up_51,out_block_down_40], 1)
        out_block_up_down_51_merge = self.block_up_merge_51(out_block_up_down_51)

        out_block_up_41 = F.interpolate(out_block_up_down_51_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_41 = torch.cat([out_block_up_41,out_block_down_30], 1)
        out_block_up_down_41_merge = self.block_up_merge_41(out_block_up_down_41)
        
        out_block_up_31 = F.interpolate(out_block_up_down_41_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_31 = torch.cat([out_block_up_31, out_block_down_20], 1) 
        out_block_up_down_31_merge = self.block_up_merge_31(out_block_up_down_31) 
        
        out_block_up_21 = F.interpolate(out_block_up_down_31_merge, scale_factor=2, mode='bilinear', align_corners=False)
        out_block_up_down_21 = torch.cat([out_block_up_21,out_block_down_10], 1)
        out_block_up_down_21_merge = self.block_up_merge_21(out_block_up_down_21) 

        out_block_classifier_11= self.block_classifier_11(out_block_up_down_21_merge)      
        return out_block_classifier_11  


class UnetResGenerator(nn.Module):

    def __init__(self, in_chans=1, out_chans = 1, num_init_features=32,  norm_layer=nn.BatchNorm2d, drop_prob= 0.0):

        super(UnetResGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        #--------------- Downsampling path ----------------------
        in_channel_block10, out_channel_block10 = in_chans, num_init_features
        self.block_down_10 = nn.Sequential(OrderedDict([
            ('conv_down_block_10', ConvBlock(in_channel_block10, num_init_features, drop_prob))     
        ]))   
        
        in_channel_block20, out_channel_block20 = out_channel_block10, out_channel_block10 * 2
        self.block_down_20 = nn.Sequential(OrderedDict([            
            ('conv_down_block_20', ConvBlock(in_channel_block20, out_channel_block20, drop_prob)),
            #('conv_down_block_21', nn.MaxPool2d(kernel_size=2))
        ]))   
        self.block_down_20_pool =  nn.MaxPool2d(kernel_size=2)

        in_channel_block30, out_channel_block30 = out_channel_block20, out_channel_block20 * 2
        self.block_down_30 = nn.Sequential(OrderedDict([
             ('conv_down_block_30', ConvBlock(in_channel_block30, out_channel_block30, drop_prob)),
             #('conv_down_block_31', nn.MaxPool2d(kernel_size=2))
        ]))  
        self.block_down_30_pool =  nn.MaxPool2d(kernel_size=2)

        in_channel_block40, out_channel_block40 = out_channel_block30, out_channel_block30 * 2
        self.block_down_40 = nn.Sequential(OrderedDict([
            ('conv_down_block_40', ConvBlock(in_channel_block40, out_channel_block40, drop_prob)),
            #('conv_down_block_41', nn.MaxPool2d(kernel_size=2))
        ]))       
        self.block_down_40_pool =  nn.MaxPool2d(kernel_size=2)

        # --------Squeeze layer------------------
        in_channel_block_mid, out_channel_block_mid = out_channel_block40, out_channel_block40
        self.block_middle = nn.Sequential(OrderedDict([
            ('conv_down_block_middle', ConvBlock(in_channel_block_mid, out_channel_block_mid, drop_prob))           
        ]))   

        #-------------Upsampling path-------------- 
        in_channel_block41, out_channel_block41  = out_channel_block40, out_channel_block40
        self.block_up_41 = nn.Sequential(OrderedDict([
            ('conv41', nn.ConvTranspose2d(in_channel_block41, out_channel_block41, kernel_size=4, stride=2, padding=1, bias=use_bias)),
            ('norm41', norm_layer(out_channel_block41)),   
        ])) 
        in_channel_block_merge_41 = in_channel_block41 + out_channel_block30
        out_channel_block_merge_41 = in_channel_block_merge_41 // 2
        self.block_up_down_merge_41 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_41', ConvBlock(in_channel_block_merge_41, out_channel_block_merge_41, drop_prob))       
        ])) 
        
        in_channel_block31, out_channel_block31  = out_channel_block_merge_41, out_channel_block_merge_41
        self.block_up_31 = nn.Sequential(OrderedDict([
            ('conv31', nn.ConvTranspose2d(in_channel_block31, out_channel_block31, kernel_size=4, stride=2, padding=1, bias=use_bias)),
            ('norm31', norm_layer(out_channel_block31)),   
        ])) 
        in_channel_block_merge_31 = in_channel_block31 + out_channel_block20
        out_channel_block_merge_31 = in_channel_block_merge_31 // 2
        self.block_up_down_merge_31 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_31', ConvBlock(in_channel_block_merge_31, out_channel_block_merge_31, drop_prob))       
        ])) 

        in_channel_block21, out_channel_block21  = out_channel_block_merge_31, out_channel_block_merge_31
        self.block_up_21 = nn.Sequential(OrderedDict([
            ('conv21', nn.ConvTranspose2d(in_channel_block21, out_channel_block21, kernel_size=4, stride=2, padding=1, bias=use_bias)),
            ('norm21', norm_layer(out_channel_block21)),   
        ])) 
        in_channel_block_merge_21 = in_channel_block21 + out_channel_block10
        out_channel_block_merge_21 = in_channel_block_merge_31 // 2
        self.block_up_down_merge_21 = nn.Sequential(OrderedDict([
            ('conv_up_down_merge_block_21', ConvBlock(in_channel_block_merge_21, out_channel_block_merge_21, drop_prob))       
        ])) 
        # ----------Classifier--------------
        in_channel_block11, out_channel_block11 = out_channel_block_merge_21, out_chans
        self.block_classifier_11 = nn.Sequential(OrderedDict([             
            ('conv11', nn.Conv2d(in_channel_block11, in_channel_block11 //2 , kernel_size=1, stride=1, padding=0)),
            ('conv12', nn.Conv2d(in_channel_block11//2, out_channel_block11 , kernel_size=1, stride=1, padding=0)),
            ('conv13', nn.Conv2d(out_channel_block11, out_channel_block11, kernel_size=1, stride=1, padding=0)),
        ]))
               

    def forward(self, input):
        #------------Forward downsampling path-------------
        out_block_down_10 = self.block_down_10(input)        
        out_block_down_20 = self.block_down_20(out_block_down_10) + out_block_down_10
        out_block_down_20_pool = self.block_down_20_pool(out_block_down_20)

        out_block_down_30 = self.block_down_30(out_block_down_20_pool) + out_block_down_20_pool
        out_block_down_30_pool = self.block_down_30_pool(out_block_down_30)

        out_block_down_40 = self.block_down_40(out_block_down_30_pool) + out_block_down_30
        out_block_down_40_pool = self.block_down_40_pool(out_block_down_40) 
        #------------Forward middle path-------------
        out_block_middle = self.block_middle(out_block_down_40) + out_block_down_40

        #------------Forward upsampling path-------------
        out_block_up_41 = self.block_up_41(out_block_middle)
        out_block_up_down_41 = torch.cat([out_block_up_41,out_block_down_30], 1)
        out_block_up_down_41_merge = self.block_up_down_merge_41(out_block_up_down_41) + out_block_up_down_41
        
        out_block_up_31 = self.block_up_31(out_block_up_down_41_merge)
        out_block_up_down_31 = torch.cat([out_block_up_31, out_block_down_20], 1) 
        out_block_up_down_31_merge = self.block_up_down_merge_31(out_block_up_down_31) + out_block_up_down_31
        
        out_block_up_21 = self.block_up_21(out_block_up_down_31_merge)
        out_block_up_down_21 = torch.cat([out_block_up_21,out_block_down_10], 1)
        out_block_up_down_21_merge = self.block_up_down_merge_21(out_block_up_down_21) + out_block_up_down_21

        out_block_classifier_11= self.block_classifier_11(out_block_up_down_21_merge)      
        return out_block_classifier_11    

class mwcnnGen(nn.Module):
    def __init__(self,num_ins, num_chans, conv=common.default_conv):
        super(mwcnnGen, self).__init__()
        #n_resblocks = args.n_resblocks
        n_feats = num_chans
        kernel_size = 3
        self.scale_idx = 1
        nColor = num_ins

        act = nn.ReLU(True)

        self.DWT = common.DWT()
        self.IWT = common.IWT()

        n = 1
        m_head = [common.BBlock(conv, nColor, n_feats, kernel_size, act=act)]
        d_l0 = []
        d_l0.append(common.DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=False))


        d_l1 = [common.BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bn=False)]
        d_l1.append(common.DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False))

        d_l2 = []
        d_l2.append(common.BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bn=False))
        d_l2.append(common.DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False))
        pro_l3 = []
        pro_l3.append(common.BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=False))
        pro_l3.append(common.BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bn=False))

        i_l2 = [common.DBlock_inv1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=False)]
        i_l2.append(common.BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bn=False))

        i_l1 = [common.DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=False)]
        i_l1.append(common.BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bn=False))

        i_l0 = [common.DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=False)]

        m_tail = [conv(n_feats, nColor, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        x1 = self.d_l1(self.DWT(x0))
        x2 = self.d_l2(self.DWT(x1))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
        x = self.tail(self.i_l0(x_)) + x

        return x

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx

def define_Gen(input_nc, output_nc, ngf, netG, norm='batch', drop_prob=0.0):
    gen_net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'unet_transpose':
        gen_net = UnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, drop_prob= drop_prob) 
    elif netG == 'unet_upsampling':
        gen_net = UnetUpsamplingGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, drop_prob= drop_prob) 
    elif netG == 'unet_upsampling_dilation_res':
        gen_net = UnetUpsamplingDilationResGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, drop_prob= drop_prob)
    elif netG == 'unet_upsampling_dilation':
        gen_net = UnetUpsamplingDilationGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, drop_prob= drop_prob) 
    elif netG == 'unet_upsampling_scSE':
        gen_net = UnetUpsamplingscSEGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, drop_prob= drop_prob) 
    elif netG == 'unet_transpose_res':
        gen_net = UnetResGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, drop_prob= drop_prob) 
    elif netG == 'mwcnn':
        gen_net = mwcnnGen(input_nc, ngf) 
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_network(gen_net)
