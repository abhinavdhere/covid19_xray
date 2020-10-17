import sys
import captum.attr
from captum.attr import visualization as viz
import torch
# import torch.nn as nn
import torch.nn.functional as F
from model import ResNet
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from aux import loadModel, get_nBatches
from learner import dataLoader
from config import path
# import matplotlib.pyplot as plt
import pdb


def load_BB(fName, img):
    name_orig = fName[0].split('_')[-2].split('.')[0]
    bb_data = pd.read_csv('/home/abhinav/CXR_datasets/RSNA_dataset/'
                          'stage_2_train_labels.csv').values
    img_bb_all = bb_data[bb_data[:, 0] == name_orig, 1:-1]
    img_bb_all /= 2
    for i in range(img_bb_all.shape[0]):
        img_bb = list(map(int, img_bb_all[i].tolist()))
        img_bb_overlayed = cv2.rectangle(img, (img_bb[0], img_bb[1]),
                                         (img_bb[0]+img_bb[2],
                                          img_bb[1]+img_bb[3]),
                                         color=(1, 1, 1), thickness=2)
    return img_bb_overlayed


val_nBatches = get_nBatches(path, 'val', 1, 1)
valDataLoader = dataLoader(path, 'val', 1, val_nBatches)
model = ResNet(in_channels=1, num_blocks=4, num_layers=4,
               downsample_freq=1).cuda()
# model = nn.DataParallel(model)
model_name = sys.argv[1]
successFlag = loadModel('chkpt', model, model_name)
get_bb_flag = True
model.eval()
for i in range(val_nBatches):
    X, y, fName = valDataLoader.__next__()
    with torch.no_grad():
        pred = model.forward(X)
        pred = F.softmax(pred, 1)
    if y.item() == 1 and torch.argmax(pred).item() == 1:
        pdb.set_trace()
        # gcObj = captum.attr.LayerGradCam(model.forward, model.main_arch[3]
        #                                  .conv_block[10])
        gcObj = captum.attr.LayerGradCam(model.forward, model.semifinal)
        attr = gcObj.attribute(X, 1)
        attrRescaled = Image.fromarray(attr.detach().cpu()
                                       .numpy()[0, 0, :, :]).resize(
                                           (X.shape[3], X.shape[2]))
        img = X.permute(0, 2, 3, 1)[0, :, :, :].detach().cpu().numpy()
        if get_bb_flag:
            img = load_BB(fName, img)
        pltObj = viz.visualize_image_attr(np.expand_dims(attrRescaled, -1),
                                          img, method="blended_heat_map",
                                          cmap='jet', sign="absolute_value",
                                          show_colorbar=True,
                                          title="Overlayed Attributions")
        pltObj[0].savefig('./gradcam_res/gradCam'+fName[0].split('.')[0]
                          + '.png')
        # plt.imshow(X.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy()
        # .astype('uint8'))
        # plt.show()
        # plt.imshow(np.array(attrRescaled),cmap='jet')
        # plt.show()
