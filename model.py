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
        self.attn_scaler = nn.Parameter(torch.Tensor([1]))
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
        if isinstance(x, tuple):
            x = x[0]
        identity = x
        out = self.conv_block(x)
        if self.downsample:
            identity = self.down_conv(identity)
        if self.attention:
            attn = torch.sigmoid(out)
            attn_out = self.attn_scaler*identity*attn
            # out = torch.sigmoid(out)
            # out = identity*out
            res = identity+out+attn_out
        else:
            res = identity+out
        res = F.relu(res)
        return res, attn


class AuxClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxClassifier, self).__init__()
        self.conv0 = nn.Conv2d(in_channels, in_channels*4, kernel_size=3,
                               stride=4, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Linear(in_channels*4, num_classes)

    def forward(self, x):
        x = F.relu(self.conv0(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.final(x)
        return out


class MARL(nn.Module):
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
        super(MARL, self).__init__()
        self.downsample_freq = downsample_freq
        self.conv1 = nn.Conv2d(in_channels, num_feats, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_feats)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = []
        self.num_blocks = num_blocks
        init_feats = num_feats
        downsample = False
        for block_num in range(num_blocks):
            block = ResBlock(init_feats, num_feats, num_layers,
                             downsample=downsample, attention=True)
            init_feats = num_feats
            self.layers.append(block.cuda())
            downsample = False
            cond = (block_num + 1 != num_blocks)
            if ((block_num+1) % self.downsample_freq == 0) and cond:
                num_feats *= 2
                downsample = True
        self.main_arch = nn.Sequential(*self.layers)
        self.attn_conv = nn.Conv2d(960, 512, kernel_size=3, stride=4)
        # self.attn_conv = nn.Conv2d(1920, 512, kernel_size=3, stride=4)
        self.semifinal = nn.Conv2d(512, 512, kernel_size=3, stride=1,
                                   padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Linear(num_feats, num_classes)
        self.aux_classifier = AuxClassifier(128, num_classes)
        # self.weights = nn.Parameter(torch.ones(num_blocks, 128, 128)).cuda()

    def forward(self, x):
        attn_map_list = []
        conicity_sum = 0
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for num, block in enumerate(self.main_arch, 1):
            x, attn_map = block(x)
            if self.training and num == (self.num_blocks // 2):
                aux = self.aux_classifier(x)
            conicity = self.get_conicity(attn_map)
            conicity_sum += conicity
            # attn_map = F.interpolate(attn_map, (52, 56),
            #                          align_corners=False, mode='bilinear')
            attn_map_list.append(attn_map)
        mid_size = attn_map_list[(len(attn_map_list)//2) - 1].shape
        for idx, attn_map in enumerate(attn_map_list):
            attn_map_list[idx] = F.interpolate(attn_map,
                                               (mid_size[2], mid_size[3]),
                                               align_corners=False,
                                               mode='bilinear')
        # import pdb
        # pdb.set_trace()
        stacked_attn_map = torch.cat(attn_map_list, 1)
        ms_attn = torch.sigmoid(self.attn_conv(stacked_attn_map))
        x = ms_attn*x
        x = self.semifinal(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.final(x)
        if self.training:
            return out, aux, conicity_sum
        else:
            return out #, conicity_sum

    def get_conicity(self, attn_map):
        atm = 0
        attn_map = torch.reshape(attn_map, (-1, attn_map.shape[1],
                                            attn_map.shape[2]
                                            * attn_map.shape[3]))
        # taking each channel as vector
        mean_vec = torch.mean(attn_map, 1).unsqueeze(1)
        # for i in range(attn_map.shape[1]):
        #     atm += F.cosine_similarity(attn_map[:, i], mean_vec)
        # conicity = atm.float()/(i+1)
        atm = F.cosine_similarity(attn_map, mean_vec, 2)
        conicity = torch.mean(atm, 1)
        return conicity


class ARL(MARL):
    def __init__(self, in_channels, num_blocks, num_layers, num_classes=2,
                 num_feats=64, downsample_freq=1):
        super(ARL, self).__init__(
            in_channels, num_blocks, num_layers, num_classes, num_feats=64,
            downsample_freq=1
        )

    def forward(self, x):
        conicity_sum = 0
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        for num, block in enumerate(self.main_arch, 1):
            x, attn_map = block(x)
            if self.training and num == (self.num_blocks // 2):
                aux = self.aux_classifier(x)
            conicity = self.get_conicity(attn_map)
            conicity_sum += conicity
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.final(x)
        if self.training:
            return out, aux, conicity_sum
        else:
            return out, conicity_sum
