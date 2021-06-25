import sys
import captum.attr
from captum.attr import visualization as viz
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MARL
from resnet import resnet18
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from aux import loadModel
# from data_handler import DataLoader
import data_handler
import matplotlib.pyplot as plt
import config
# import pdb
# from SmoothGradCAMplusplus import cam


def load_BB_with_lung_seg(fName, lung_mask, img, draw=True):
    if draw:
        name_orig = fName[0].split('_')[-2].split('.')[0]
    else:
        name_orig = fName.split('_')[-1].split('.')[0]
    blank = np.zeros((512, 512))
    bb_data = pd.read_csv('/home/abhinav/CXR_datasets/RSNA_dataset'
                          '/stage_2_train_labels.csv').values
    img_bb_all = bb_data[bb_data[:, 0] == name_orig, 1:-1]
    img_bb_all /= 2
    # contours in blank need to be filled when extraction is needed
    # for computational purposes. For visualization, 2 is better.
    if not draw:
        bb_thickness = -1
    else:
        bb_thickness = 2
    for i in range(img_bb_all.shape[0]):
        img_bb = list(map(int, img_bb_all[i].tolist()))
        blank = cv2.rectangle(blank, (img_bb[0], img_bb[1]),
                              (img_bb[0]+img_bb[2],
                               img_bb[1]+img_bb[3]),
                              color=(1, 1, 1), thickness=bb_thickness)
    min_row, max_row = np.where(np.any(lung_mask, 0))[0][[0, -1]]
    min_col, max_col = np.where(np.any(lung_mask, 1))[0][[0, -1]]
    blank = blank[min_col:max_col, min_row:max_row]
    blank = cv2.resize(blank, (352, 384), cv2.INTER_AREA)
    contours, _ = cv2.findContours(blank.astype('uint8'), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if draw:
        for cnt in contours:
            img = cv2.drawContours(img, [cnt], 0,
                                   color=(1, 1, 1), thickness=2)
        return img
    else:
        return contours


def load_BB(fName, img):
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
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


if __name__ == '__main__':
    model_name = sys.argv[1]
    folder_name = sys.argv[2]
    foldNum = sys.argv[3]
    # val_data_handler = DataLoader('val', foldNum, 1, 'none')
    # trn_data_handler = data_handler.DataLoader('trn', foldNum, 1, None, 0)
    # val_data_handler = data_handler.DataLoader('val', foldNum, 1, None, 0)
    tst_data_handler = data_handler.DataLoader('tst', foldNum, 1, None, 0)
    # num_batches = trn_data_handler.num_batches
    # num_batches = val_data_handler.num_batches
    num_batches = tst_data_handler.num_batches
    model = MARL(in_channels=1, num_blocks=4, num_layers=4,
                 downsample_freq=1).cuda()
    # model = resnet18(num_classes=2).cuda()
    model = nn.DataParallel(model)
    successFlag = loadModel('chkpt', model, model_name)
    # successFlag = loadModel('main', model, model_name)
    print(successFlag)
    get_bb_flag = False
    model.eval()
    # pred_list = []
    for i in range(num_batches):
        X, y, fName = tst_data_handler.datagen.__next__()
        if fName[0].split('_')[1] == '2':
            continue
        # X, y, fName = val_data_handler.datagen.__next__()
        mask_name = '_'.join(fName[0].split('_')[:-1])
        lung_mask = np.load(config.PATH.rsplit('/', 1)[0]
                            # + '/bimcv_iitj_lungSeg/'+mask_name+'.npy')
                            + '/lung_seg_raw/'+mask_name+'.npy')
        with torch.no_grad():
            pred = model.forward(X)
            pred = F.softmax(pred, 1)
        if (y.item() == 0 and torch.argmax(pred).item() == 0):
            # and (pred[0, 1].item() > 0.8):
            # pred_list.append(pred[0, 1].item())
    # print(len(pred_list))
    # plt.hist(np.array(pred_list))
    # plt.savefig('hist_pred.png')
            # gcObj = captum.attr.LayerGradCam(
            #     model.forward, model.module.main_arch[3].conv_block[10])
            # import pdb
            # pdb.set_trace()
            # gcObj = cam.GradCAMpp(model, model.main_arch[3].conv_block[10])
            # attr, _ = gcObj.forward(X, 1)
            # ig = captum.attr.IntegratedGradients(model)
            # attr = ig.attribute(X, target=0, internal_batch_size=1, n_steps=60)
            # gcObj = captum.attr.LayerGradCam(model.forward,
            #                                  model.module.layer4)
            gcObj = captum.attr.LayerGradCam(model.forward, model.module.semifinal)
            attr = gcObj.attribute(X, 1)
            attr = torch.abs(attr)
            attrRescaled = Image.fromarray(attr.detach().cpu()
                                           .numpy()[0, 0, :, :]).resize(
                                               (X.shape[3], X.shape[2]))
            img = X[0, 0, :, :].detach().cpu().numpy()
            if get_bb_flag:
                # img = load_BB(fName, img)
                img = load_BB_with_lung_seg(fName, lung_mask, img)
            min_row, max_row = np.where(np.any(lung_mask, 0))[0][[0, -1]]
            min_col, max_col = np.where(np.any(lung_mask, 1))[0][[0, -1]]
            lung_mask = lung_mask[min_col:max_col, min_row:max_row]
            lung_mask = cv2.resize(lung_mask, (352, 384), cv2.INTER_AREA)
            # plt.imshow(img, cmap='gray')
            attr_map = np.array(attrRescaled)
            # thresh = 0.8*np.max(attr_map)
            # attr_map[attr_map < thresh] = 0
            # attr_map[attr_map > thresh] = 1
            # plt.imshow(lung_mask*attr_map, cmap='jet', alpha=0.3)
            # plt.imshow(attr_map, cmap='jet', alpha=0.3)
            # plt.title('Overlayed Attributions')
            # plt.axis('off')
            # plt.colorbar()
            # # plt.savefig('test_threshold_covid_bimcv.png')
            # # plt.savefig('./gradcam_misc/rsna_march_21/resnet1/' +
            # #             fName[0].split('.')[0]+'.png')
            # plt.savefig('./gradcam_misc/rsna_jun21/marl_viz_normal/' +
            #             fName[0].split('.')[0]+'.png')
            # plt.savefig('./gradcam_misc/bimcv_stage2/raw_unthresh_rerun1/viz_original/'
            #             # guided_thresholded/'
            #             'covid/'+model_name+'_' + fName[0].split('.')[0]
            #             + '.png')
            # plt.savefig('./gradcam_misc/pd_stage2_fold3/guided_thresholded/raw/pneumonia/'+model_name+'_'
            #             + fName[0].split('.')[0] + '.png')

            # np.save('./gradcam_misc/bimcv_stage2/raw_unthresh_val_rerun/pneumonia/'
            #         + model_name+'_' + fName[0].split('.')[0]
            #         + '.npy', attr_map)
            np.save('./gradcam_misc/rsna_jun21/marl_raw_normal/'
                    + fName[0].split('.')[0] + '.npy', attr_map)
            # np.save('./gradcam_misc/rsna_march_21/guided_thresholded/'
            #         'raw_thresh80/resnet1/' + fName[0].split('.')[0]
            #         + '.npy', attr_map)

            # pltObj = viz.visualize_image_attr(np.expand_dims(attrRescaled, -1),
            #                                   img, method="blended_heat_map",
            #                                   cmap='jet', sign="absolute_value",
            #                                   show_colorbar=True,
            #                                   title="Overlayed Attributions")
            # pltObj[0].savefig('./gradcam_misc/rsna_march_21/resnet_noLungSeg/'
            #                   + fName[0].split('.')[0] + '.png')
            # plt.close()
            # plt.imshow(X.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy()
            # .astype('uint8'))
            # plt.show()
            # plt.imshow(np.array(attrRescaled),cmap='jet')
            # plt.show()
