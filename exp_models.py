import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from model import AuxClassifier

class RobustDenseNet(nn.Module):
    """ DenseNet with aux output """
    def __init__(self, pretrained, num_classes):
        super(RobustDenseNet, self).__init__()
        self.base_network = torchvision.models.densenet121(
            pretrained=pretrained)
        self.base_network.classifier = nn.Linear(
            in_features=1024, out_features=num_classes, bias=True)
        self.aux_classifier = AuxClassifier(512, num_classes)

    def setup_hook(self):
        self.mid_output = []
        def hook(module, input, output):
            self.mid_output.append(output)
        hook_obj = self.base_network.features.denseblock2.register_forward_hook(hook)
        return hook_obj

    def forward(self, x):
        hook_obj = self.setup_hook()
        pred = self.base_network.forward(x)
        if self.training:
            mid_out = self.mid_output
            aux_pred = self.aux_classifier(mid_out[0])
            hook_obj.remove()
            return pred, aux_pred
        else:
            return pred


class RobustEfficientNet(RobustDenseNet):
    def __init__(self, pretrained, num_classes):
        super(RobustEfficientNet, self).__init__()
        model = EfficientNet.from_pretrained('efficientnet-b4')
        self.base_network.classifier = nn.Linear(
            in_features=1792, out_features=num_classes, bias=True)
        self.aux_classifier = AuxClassifier(112, num_classes)

    def setup_hook(self):
        self.mid_output = []
        def hook(module, input, output):
            self.mid_output.append(output)
        hook_obj = self.base_network._blocks[16].register_forward_hook(hook)
        return hook_obj


class BasicNet2(nn.Module):
    def __init__(self):
        super(BasicNet2, self).__init__()
        initNum = 128
        self.init_layer = nn.Conv2d(3, initNum, kernel_size=3, stride=1,
                                    padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1_a = nn.Conv2d(initNum, initNum*2, kernel_size=3, stride=1,
                                 padding=1)
        self.conv1_b = nn.Conv2d(initNum*2, initNum*2, kernel_size=3, stride=1,
                                 padding=1)
        self.conv1_c = nn.Conv2d(initNum*2, initNum*2, kernel_size=3, stride=1,
                                 padding=1)
        self.bn1 = nn.BatchNorm2d(initNum*2, initNum*2)

        # self.conv_dn1 = nn.Conv2d(initNum*2, initNum*2, kernel_size=2,
        # stride=2, padding=0)

        initNum = initNum*2
        self.conv2_a = nn.Conv2d(initNum, initNum*2, kernel_size=3, stride=1,
                                 padding=1)
        self.conv2_b = nn.Conv2d(initNum*2, initNum*2, kernel_size=3, stride=1,
                                 padding=1)
        self.conv2_c = nn.Conv2d(initNum*2, initNum*2, kernel_size=3, stride=1,
                                 padding=1)
        self.bn2 = nn.BatchNorm2d(initNum*2, initNum*2)

        # self.conv_dn2 = nn.Conv2d(initNum*2, initNum*2, kernel_size=2,
        #                           stride=2, padding=0)

        initNum = initNum*2
        self.conv3_a = nn.Conv2d(initNum, initNum*2, kernel_size=3, stride=1,
                                 padding=1)
        self.conv3_b = nn.Conv2d(initNum*2, initNum*2, kernel_size=3, stride=1,
                                 padding=1)
        self.conv3_c = nn.Conv2d(initNum*2, initNum*2, kernel_size=3, stride=1,
                                 padding=1)
        self.bn3 = nn.BatchNorm2d(initNum*2, initNum*2)
        initNum = initNum*2

        # self.conv_dn3 = nn.Conv2d(initNum, initNum, kernel_size=2, stride=2,
        # padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Linear(initNum, 2)

    def forward(self, x):
        x = F.relu(self.init_layer(x))

        x = F.relu(self.bn1(self.conv1_a(x)))
        x = F.relu(self.bn1(self.conv1_b(x)))
        x1 = F.relu(self.bn1(self.conv1_c(x)))
        x = x1 + x
        x = self.maxpool(x)

        x = F.relu(self.bn2(self.conv2_a(x)))
        x = F.relu(self.bn2(self.conv2_b(x)))
        x1 = F.relu(self.bn2(self.conv2_c(x)))
        x = x1 + x
        x = self.maxpool(x)

        x = F.relu(self.bn3(self.conv3_a(x)))
        x = F.relu(self.bn3(self.conv3_b(x)))
        x1 = F.relu(self.bn3(self.conv3_c(x)))
        x = x1 + x
        x = self.maxpool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.final(x)
        return x


class BasicNet(nn.Module):
    def __init__(self):
        super(BasicNet, self).__init__()
        self.init_layer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.block1 = BasicConvBlock(64)
        self.conv_pool1 = nn.Conv2d(128, 128, kernel_size=2, stride=2,
                                    padding=0)
        # self.attn1 = SelfAttentionModule(32)

        self.block2 = BasicConvBlock(128)
        self.conv_pool2 = nn.Conv2d(256, 256, kernel_size=2, stride=2,
                                    padding=0)

        self.block3 = BasicConvBlock(256)
        self.conv_pool3 = nn.Conv2d(512, 512, kernel_size=2, stride=2,
                                    padding=0)
        self.attn2 = SelfAttentionModule(512)

        # self.semi_final = nn.Conv2d(128, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.final = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.init_layer(x))
        x = self.block1(x)
        x = self.conv_pool1(x)
        # x = self.attn1(x)
        x = self.block2(x)
        x = self.conv_pool2(x)
        x = self.block3(x)
        x = self.conv_pool3(x)
        # x, attn_map = self.attn2(x)
        # x = F.relu(self.semi_final(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.final(x)
        return out


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(BasicConvBlock, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels*2, in_channels*2)
        self.convLayer1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3,
                                    stride=1, padding=1)
        self.convLayer2 = nn.Conv2d(in_channels*2, in_channels*2,
                                    kernel_size=3,
                                    stride=1, padding=1)
        self.convLayer3 = nn.Conv2d(in_channels*2, in_channels*2,
                                    kernel_size=3,
                                    stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.bn(self.convLayer1(x)))
        out = F.relu(self.bn(self.convLayer2(x)))
        out = F.relu(self.bn(self.convLayer3(out)))
        out = out + x
        return x
