import sys
import captum.attr
from captum.attr import visualization as viz
import torch
# import torch.nn as nn
import torch.nn.functional as F
from model import ResNet
from PIL import Image
import numpy as np
from aux import loadModel, get_nBatches
from learner import dataLoader
from config import path
# import matplotlib.pyplot as plt
import pdb


val_nBatches = get_nBatches(path, 'val', 1, 1)
valDataLoader = dataLoader(path, 'val', 1, val_nBatches)
model = ResNet(in_channels=1, num_blocks=4, num_layers=4,
               downsample_freq=1).cuda()
# model = nn.DataParallel(model)
model_name = sys.argv[1]
successFlag = loadModel('chkpt', model, model_name)
print(successFlag)
model.eval()
for i in range(val_nBatches):
    X, y, fName = valDataLoader.__next__()
    with torch.no_grad():
        pred = F.softmax(model.forward(X), 1)
    if y.item() == 1 and torch.argmax(pred).item() == 1:
        pdb.set_trace()
        gcObj = captum.attr.LayerGradCam(model.forward, model.main_arch[3]
                                         .conv_block[10])
        attr = gcObj.attribute(X, 1)
        attrRescaled = Image.fromarray(attr.detach().cpu()
                                       .numpy()[0, 0, :, :]).resize(
                                           (X.shape[3], X.shape[2]))
        pltObj = viz.visualize_image_attr(np.expand_dims(attrRescaled, -1),
                                          X.permute(0, 2, 3, 1)[0, :, :, :]
                                          .detach().cpu().numpy(),
                                          method="blended_heat_map",
                                          cmap='jet',
                                          sign="absolute_value",
                                          show_colorbar=True,
                                          title="Overlayed Attributions")
        pltObj[0].savefig('./gradcam_res/gradCam'+fName[0].split('.')[0]
                          + '.png')
        # plt.imshow(X.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy()
        # .astype('uint8'))
        # plt.show()
        # plt.imshow(np.array(attrRescaled),cmap='jet')
        # plt.show()
