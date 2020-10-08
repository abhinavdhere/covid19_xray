'''Custom ResNet with multi scale residual attention.'''
# import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    ''' Single residual block with variable number of conv layers.
    Attention can be turned on or off.
    '''
    def __init__(self, init_feats, feats, num_layers, attention=True,
                 downsample=False):
        '''
        Args:
            init_feats (int): number of input channels in first layer
            feats (int): number of feature channels
            num_layers (int): number of layers in the block
            attention (bool): To set attention on/off
            downsample (bool): To set downsampling on/off
        '''
        super(ResBlock, self).__init__()
        self.attention = attention
        self.downsample = downsample
        if self.downsample:
            stride = 2
            self.down_conv = nn.Conv2d(init_feats, feats, kernel_size=1,
                                       stride=2, bias=False)
        else:
            stride = 1
        layers = []
        for layer_num in range(num_layers):
            conv_layer = nn.Conv2d(init_feats, feats, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            stride = 1
            init_feats = feats
            norm_layer = nn.BatchNorm2d(feats)
            relu = nn.ReLU(inplace=True)
            layers.append(conv_layer)
            layers.append(norm_layer)
            if layer_num < (num_layers - 1):
                layers.append(relu)
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        if self.downsample:
            identity = self.down_conv(identity)
        if self.attention:
            x = torch.sigmoid(x)
            x = identity*x
        out = identity+out
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, num_blocks, num_layers, num_classes=2,
                 num_feats=64, downsample_freq=1):
        '''
        Args:
            in_channels (int): number of channels in input
            num_blocks (int): number of residual block
            num_layers (int): number of layers in each block
            num_classes (int): number of classes in classification
            num_feats (int): number of feature channels in first layer
            downsample_freq (int): number of blocks after which to downsample.
        '''
        super(ResNet, self).__init__()
        self.downsample_freq = downsample_freq
        self.conv1 = nn.Conv2d(in_channels, num_feats, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = []
        init_feats = num_feats
        downsample = False
        for block_num in range(num_blocks):
            block = ResBlock(init_feats, num_feats, num_layers,
                             downsample=downsample, attention=False)
            init_feats = num_feats
            layers.append(block)
            downsample = False
            cond = (block_num + 1 != num_blocks)
            if ((block_num+1) % self.downsample_freq == 0) and cond:
                num_feats *= 2
                downsample = True
        self.main_arch = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Linear(num_feats, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.main_arch(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.final(x)
        return out
