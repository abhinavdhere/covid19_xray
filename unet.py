"""
Standard U-Net with options for backbone in encoder.
Abhinav Dhere ;
"""
# import pdb
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# ------------- #
# U-Net Module
# ------------- #


class UNet(nn.Module):
    def __init__(self, n_classes=1, backbone='resnet34',
                 encoder_weights_init='ImageNet'):
        '''
        backbone : name of network to use as encoder.
        encoder_weights_init : how encoder weights are initialized -
        random or ImageNet.
        '''
        super(UNet, self).__init__()
        available_models = {'resnet34': torchvision.models.resnet34,
                            'resnet50': torchvision.models.resnet50,
                            'densenet121': torchvision.models.densenet121,
                            'resnext50_32x4d': (torchvision.models.
                                                resnext50_32x4d)}
        if encoder_weights_init == 'ImageNet':
            pretrained = True
        else:
            pretrained = False
        backbone_model = available_models[backbone](pretrained,
                                                    num_classes=n_classes)
        self.encoder = nn.Sequential(*list(backbone_model.children())[:-2])
        self.decoder = Decoder(n_classes)

    def setup_hooks(self):
        self.int_outputs = []

        def hook(module, input, output):
            self.int_outputs.append(output)

        encoder_layers = [self.encoder.layer3, self.encoder.layer2,
                          self.encoder.layer1]
        for layer in encoder_layers:
            layer.register_forward_hook(hook)

    def forward(self, x):
        self.setup_hooks()
        out_layer4 = self.encoder.forward(x)
        out_layer1, out_layer2, out_layer3 = self.int_outputs
        self.int_outputs = []
        out = self.decoder(out_layer4, out_layer3, out_layer2,
                           out_layer1)
        return out


class Decoder(nn.Module):
    def __init__(self, nClasses):
        self.up_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2)
        self.bn1 = nn.BatchNorm2d(512, 512)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.up_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(256, 256)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.up_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2)
        self.bn3 = nn.BatchNorm2d(128, 128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.up_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2)
        self.bn4 = nn.BatchNorm2d(64, 64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.up_conv5 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2)
        self.bn5 = nn.BatchNorm2d(32, 32)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(16, 16)

        self.final = nn.Conv2d(16, nClasses, kernel_size=3, padding=1)

    def forward(self, x, out_layer3, out_layer2, out_layer1):
        x = self.bn1(self.up_conv1(x))
        x = torch.cat((x, out_layer3), 1)
        x = F.relu(self.bn2(self.conv1(x)))

        x = self.bn2(self.up_conv2(x))
        x = torch.cat((x, out_layer2), 1)
        x = F.relu(self.bn3(self.conv2(x)))

        x = self.bn3(self.up_conv3(x))
        x = torch.cat((x, out_layer1), 1)
        x = F.relu(self.bn4(self.conv3(x)))

        x = self.bn4(self.up_conv4(x))
        x = F.relu(self.bn5(self.conv4(x)))

        x = self.bn5(self.up_conv5(x))
        x = F.relu(self.bn6(self.conv5(x)))

        out = self.final(x)
        return out
