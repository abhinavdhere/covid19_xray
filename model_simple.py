import torch.nn as nn
import torch.nn.functional as F
from model import SelfAttentionModule


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.init_layer = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

        self.block1 = BasicConvBlock(2, 16)
        self.conv_pool1 = nn.conv2d(32, 32, kernel_size=2, stride=2, padding=0)
        self.attn1 = SelfAttentionModule(32)

        self.block2 = BasicConvBlock(2, 32)
        self.conv_pool2 = nn.conv2d(64, 64, kernel_size=2, stride=2, padding=0)

        self.block3 = BasicConvBlock(2, 64)
        self.conv_pool3 = nn.conv2d(128, 128, kernel_size=2, stride=2,
                                    padding=0)
        self.attn2 = SelfAttentionModule(128)

        self.semi_final = nn.Conv2d(128, 1, kernel_size=1)
        self.final = nn.Linear(64*64, 2)

    def forward(self, x):
        x = F.relu(self.init_layer(x))
        x = self.block1(x)
        x = self.conv_pool1(x)
        x = self.attn1(x)
        x = self.block2(x)
        x = self.conv_pool2(x)
        x = self.block3(x)
        x = self.conv_pool3(x)
        x = self.attn2(x)
        x = F.relu(self.semi_final(x))
        out = F.softmax(self.final(x), 1)
        return out


class BasicConvBlock(nn.Module):
    def __init__(self, num_layers, in_channels):
        super(BasicConvBlock, self).__init__()
        self.layers = []
        self.bn = nn.BatchNorm2d(in_channels*2, in_channels*2)
        for layernum in range(num_layers):
            convLayer = nn.Conv2d(in_channels, in_channels*2, kernel_size=3,
                                  stride=1, padding=1)
            self.layers.append(convLayer)

    def forward(self, x):
        for layernum in range(len(self.layers)):
            x = F.relu(self.layers[layernum](x))
        x = self.bn(x)
        return x
