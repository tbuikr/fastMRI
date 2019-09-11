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

    def __init__(self, in_chans, out_chans, drop_prob):
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

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_chans),
            nn.ReLU(),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1),
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

def define_Gen(input_nc, output_nc, ngf, netG, norm='batch', drop_prob=0.0):
    gen_net = None
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'unet_transpose':
        gen_net = UnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, drop_prob= drop_prob) 
    elif netG == 'unet_transpose_res':
        gen_net = UnetResGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, drop_prob= drop_prob) 
    
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)

    return init_network(gen_net)