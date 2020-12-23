""" Network utils for dense UNet """
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.normalization as norms


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            if in_size < 64:
                nGrps = 4
            else:
                nGrps = 16
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1),
                                       norms.GroupNorm(nGrps, out_size),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
                                       norms.GroupNorm(nGrps, out_size),
                                       nn.ReLU(),)
#            self.conv3 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1),
#                                       norms.GroupNorm(nGrps,out_size),
#                                       nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 2),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 2),
                                       nn.ReLU(),)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
#        outputs = self.conv3(outputs)
        return outputs


class Combiner(nn.Module):
    '''
    Combines outputs of two layers by padding (if needed) and concatenation.
    '''
    def __init__(self):
        super(Combiner, self).__init__()

    def getPadding(self, offset):
        if offset % 2 == 0:
            padding = 2*[np.sign(offset)*(np.abs(offset) // 2)]
        else:
            padding = [np.sign(offset)*(np.abs(offset) // 2),
                       np.sign(offset)*((np.abs(offset) // 2) + 1)]
        return padding

    def forward(self, input1, input2):
        '''
        input1 - from decoder ; input2 - from encoder.
        '''
        offset1 = input2.size()[2] - input1.size()[2]
        padding1 = self.getPadding(offset1)
        offset2 = input2.size()[3] - input1.size()[3]
        padding2 = self.getPadding(offset2)
        padding = padding2+padding1
        output1 = F.pad(input1, padding)
        return torch.cat([output1, input2], 1)


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        nGrps = 16
        # self.bn1 = nn.BatchNorm2d(nChannels)
        self.bn1 = norms.GroupNorm(nGrps, nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=True)
        self.bn2 = norms.GroupNorm(nGrps, interChannels)
        # self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=0, bias=True)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = F.pad(out, (1, 1, 1, 1), mode='replicate')
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        if nChannels < 64:
            nGrps = 4
        else:
            nGrps = 16
        self.bn1 = norms.GroupNorm(nGrps, nChannels)
#        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)
        self.dnConv1 = nn.Conv2d(nOutChannels, nOutChannels, 2, stride=2)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.dnConv1(out)
#        out = F.avg_pool2d(out, 2)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
#        self.bn1 = nn.BatchNorm2d(nChannels)
        if nChannels < 64:
            nGrps = 4
        else:
            nGrps = 8  # 16
        self.bn1 = norms.GroupNorm(nGrps, nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


def _make_dense(nChannels, growthRate, nDenseBlocks, bottleneck):
    layers = []
    for i in range(int(nDenseBlocks)):
        if bottleneck:
            layers.append(Bottleneck(nChannels, growthRate))
        else:
            layers.append(SingleLayer(nChannels, growthRate))
        nChannels += growthRate
    return nn.Sequential(*layers)
