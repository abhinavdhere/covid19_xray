import torch.nn as nn
import torch.nn.functional as F
from network_utils import unetConv2, Combiner, _make_dense, Transition


class dense_unet_encoder(nn.Module):
    def __init__(self):
        super(dense_unet_encoder, self).__init__()
        self.growthRate = 32
        nFeat = 1
        self.nLayers = [2, 8, 18, 24]
        isBottleneck = True
        self.inputLayer = nn.Conv2d(nFeat, 64, 7, stride=2, padding=3)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        nFeat = 64  # (3, 3, nSlices)
        self.conv1 = _make_dense(nFeat, self.growthRate, self.nLayers[0],
                                 isBottleneck)
        nFeat += self.growthRate*self.nLayers[0]
        # Transition layer applies 1x1 conv,  brings to given output ...
        # ... channels (16) and applies stride 2 conv to downsample
        self.trans1 = Transition(nFeat, nFeat//2)
        nFeat = nFeat // 2
        self.conv2 = _make_dense(
            nFeat, self.growthRate, self.nLayers[1], isBottleneck)
        nFeat += self.growthRate*self.nLayers[1]
        self.trans2 = Transition(nFeat, nFeat//2)
        nFeat = nFeat // 2
        self.conv3 = _make_dense(
            nFeat, self.growthRate, self.nLayers[2], isBottleneck)
        nFeat += self.growthRate*self.nLayers[2]
        self.trans3 = Transition(nFeat, nFeat//2)
        nFeat = nFeat // 2
        self.center = _make_dense(
            nFeat, self.growthRate, self.nLayers[3], isBottleneck)
        nFeat += self.growthRate*self.nLayers[3]
        self.final_num_feat = nFeat

    def forward(self, x):
        x = self.inputLayer(x)
        x = self.mp(x)
        c1_out = self.trans1(self.conv1(x))
        c2_out = self.trans2(self.conv2(c1_out))
        c3_out = self.trans3(self.conv3(c2_out))
        x = self.center(c3_out)
        return x, c1_out, c2_out, c3_out


class dense_unet_decoder(nn.Module):
    def __init__(self, nClasses, centerNumFeat, nLayers, growthRate):
        super(dense_unet_decoder, self).__init__()
        self.combiner = Combiner()
        # +16 due to added channels from encoder, ...
        # ... always 16 due to transition layer
        next_num_channels = centerNumFeat - growthRate*nLayers[-1]
        self.conv4 = unetConv2(
            centerNumFeat+next_num_channels, 256, True)
        self.upTrans2 = nn.ConvTranspose2d(256, 128, 2, 2)
        next_num_channels = next_num_channels*2 - growthRate*nLayers[-2]
        self.conv5 = unetConv2(128+next_num_channels, 64, True)
        self.upTrans3 = nn.ConvTranspose2d(64, 32, 4, 4)
        next_num_channels = next_num_channels*2 - growthRate*nLayers[-3]
        self.conv6 = unetConv2(32+next_num_channels, 16, True)
        self.upTrans4 = nn.ConvTranspose2d(16, 4, 4, 4)
        self.final = nn.Conv2d(4, nClasses, kernel_size=1)

    def forward(self, x, c1_out, c2_out, c3_out):
        x = self.conv4(self.combiner(c3_out, x))
        x = self.conv5(self.combiner(c2_out, self.upTrans2(x)))
        x = self.conv6(self.combiner(c1_out, self.upTrans3(x)))
        x = self.upTrans4(x)
        # # extra upsampling to compensate for input layer's stride 2
        # x = self.upTrans4(x)
        out = F.softmax(self.final(x), 1)
        return out


class dense_unet_autoencoder(nn.Module):
    def __init__(self, centerNumFeat):
        super(dense_unet_autoencoder, self).__init__()
        self.conv1 = unetConv2(centerNumFeat, 256, True)
        self.upTrans1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv2 = unetConv2(128, 64, True)
        self.upTrans2 = nn.ConvTranspose2d(64, 32, 4, 4)
        self.conv3 = unetConv2(32, 32, True)
        self.upTrans3 = nn.ConvTranspose2d(32, 16, 4, 4)
        self.recons = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(self.upTrans1(x))
        x = self.conv3(self.upTrans2(x))
        x = self.upTrans3(x)
        x = self.recons(x)
        return x


class DUN(nn.Module):
    '''
    Dense U-Net for segmentation.
    '''
    def __init__(self):
        super(DUN, self).__init__()
        self.encoder = dense_unet_encoder()
        self.decoder = dense_unet_decoder(
            2, self.encoder.final_num_feat, self.encoder.nLayers,
            self.encoder.growthRate
        )
        self.autoEncoderModel = dense_unet_autoencoder(
            self.encoder.final_num_feat)

    def forward(self, x):
        x, c1_out, c2_out, c3_out = self.encoder(x)
        out = self.decoder(x, c1_out, c2_out, c3_out)
        recons = self.autoEncoderModel(x)
        return out, recons
