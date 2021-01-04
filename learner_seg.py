'''
Primary module. Includes dataloader,  trn/val/test functions. Reads
options from user and runs training.
'''
import os
from collections import namedtuple
from itertools import combinations
from enum import Enum
# import pdb

import torch
import torch.nn.functional as F
from tqdm import trange
# from pytorchcv.model_provider import get_model
# from fastai.vision.models.unet import DynamicUnet
import torch.nn as nn
# from torchvision.models import resnext50_32x4d
# import torchvision
# import pydicom as dcm
# from pytorch_model_summary import summary

import aux
# from aux import weightedBCE as lossBCE, dice_coeff as lossDice
from aux import dice_coeff as lossDice
from data_handler import SegDataLoader
from dense_unet import DUN
# from unet import UNet


def predict_compute_loss(X, model, y_OH, loss_wts, loss_list, gamma):
    """
    Run prediction and return losses
    Args:
        X (torch.Tensor): data batch from data loader
        model(torch.nn.Module): model being trained/predicted with
        y_OH (torch.Tensor): one-hot encoded labels
    Returns:
        pred (torch.Tensor): soft predictions for given batch
        loss (float): total loss for the batch
        loss_list (dict): break up of losses
    """
    pred, recons = model.forward(X)
    class_wts = torch.Tensor([1, 1]).view(1, -1, 1, 1).cuda()
    focal_loss_fn = aux.FocalLoss(class_wts, gamma=int(gamma),
                                  reduction='sum')
    dice_loss = 0
    focal_loss = focal_loss_fn(pred, y_OH) / (512*512)
    for i in range(2):
        # bce_loss = lossBCE(1, pred[:, i], y_OH[:, i]) / (512*512)
        dice_loss += lossDice(pred[:, i], y_OH[:, i])
    loss = (loss_wts[0]*focal_loss + loss_wts[1]*dice_loss)
    loss_list['focal_loss'] += focal_loss
    loss_list['dice'] += dice_loss.item()
    mse_loss = F.mse_loss(recons, X)
    loss += loss_wts[2]*mse_loss
    loss_list['mse'] += mse_loss
    return pred, loss, loss_list


def run_model(data_handler, model, optimizer, loss_wts, gamma, amp):
    '''
    process : 'trn', 'val' or 'tst'
    '''
    num_batches = data_handler.num_batches
    batch_size = data_handler.batch_size
    process = data_handler.data_type
    running_loss = 0
    running_dice = 0
    loss_list = {'focal_loss': 0, 'dice': 0, 'mse': 0}
    # pred_list = []
    label_list = []
    # softpred_list = []
    if amp:
        scaler = torch.cuda.amp.GradScaler()
    with trange(num_batches, desc=process, ncols=100) as t:
        for m in range(num_batches):
            X, y, fName = data_handler.datagen.__next__()
            y_onehot = aux.toCategorical(y, 'seg').cuda()
            if process == 'trn':
                optimizer.zero_grad()
                model.train()
                if amp:
                    with torch.cuda.amp.autocast():
                        pred, loss, loss_list = predict_compute_loss(
                            X, model, y_onehot, loss_wts, loss_list, gamma
                        )
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred, loss, loss_list = predict_compute_loss(
                        X, model, y_onehot, loss_wts, loss_list, gamma
                    )
                    loss.backward()
                    optimizer.step()
            elif process == 'val' or process == 'tst':
                model.eval()
                if amp:
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        pred, loss, loss_list = predict_compute_loss(
                            X, model, y_onehot, loss_wts, loss_list, gamma
                        )
                else:
                    with torch.no_grad():
                        pred, loss, loss_list = predict_compute_loss(
                            X, model, y_onehot, loss_wts, loss_list, gamma
                        )

            running_loss += loss
            hardPred = torch.argmax(pred, 1)
            # from PIL import Image
            import config
            import numpy as np
            for batch_idx in range(hardPred.shape[0]):
                np.save(
                    config.PATH.rsplit('/', 1)[0] + '/lung_seg_raw/'
                    + fName[batch_idx].rsplit('_', 1)[0],
                    hardPred[batch_idx].detach().cpu().numpy()
                )
                # img_to_save = Image.fromarray(
                #     hardPred[batch_idx].detach().cpu().numpy().astype('uint8'))
            #     img_to_save.save(
            #         config.PATH.rsplit('/', 1)[0] + '/lung_seg_noBCET/'
            #         + fName[batch_idx].rsplit('_', 1)[0])
            running_dice += aux.integral_dice(hardPred, y_onehot[:, 1], 1)
            # pred_list.append(hardPred.cpu())
            # softpred_list.append(pred.detach().cpu())
            label_list.append(y.cpu())
            t.set_postfix(loss=running_loss.item()/(float(m+1)*batch_size))
            t.update()
        final_loss = running_loss/(float(m+1)*batch_size)
        for loss_name in loss_list.keys():
            loss_list[loss_name] /= (float(m+1)*batch_size)
        # acc = aux.globalAcc(pred_list, label_list)
        # acc = acc.item() / (512*512)
        dice = running_dice/(m+1)
        metrics = Metrics(final_loss.item(), dice.item())
        # print(metrics.Acc, metrics.Dice)
        return metrics, loss_list


def main():
    # Take options and hyperparameters from user
    parser = aux.getOptions()
    args = parser.parse_args()
    if args.saveName is None:
        print("Warning! Savename unspecified. No logging will take place."
              "Model will not be saved.")
        bestValRecord = None
        logFile = None
    else:
        bestValRecord, logFile = aux.initLogging(args.saveName, 'Dice')
        with open(os.path.join('logs', bestValRecord), 'r') as statusFile:
            bestVal = float(statusFile.readline().strip('\n').split()[-1])
    aux.log_config(logFile, args)
    loss_wts = tuple(map(float, args.lossWeights.split(',')))
    amp = (args.amp == 'True')
    # Inits
    base_aug_names = ['normal', 'rotated', 'gaussNoise', 'mirror',
                      'blur', 'sharpen', 'translate']
    # taking pairs of aug. types + all individual aug.
    all_aug_names = [combo[0]+'+'+combo[1] for combo in combinations(
        base_aug_names[1:], 2)]
    all_aug_names += base_aug_names
    all_aug_names.remove('blur+sharpen')  # blur+sharpen is pointless
    # trn_data_handler = SegDataLoader('trn', args.foldNum, args.batchSize,
    #                                  'all',
    #                                  # 'random_class0_all_class1',
    #                                  undersample=False, sample_size=3000,
    #                                  aug_names=all_aug_names, in_channels=0)
    # val_data_handler = SegDataLoader('val', args.foldNum, args.batchSize,
    #                                  'none', in_channels=0)
    tst_data_handler = SegDataLoader('tst', args.foldNum, args.batchSize,
                                     'none', in_channels=0)
    # model = UNet(n_classes=2).cuda()
    # encoder = resnext50_32x4d(pretrained=True)
    # encoder = encoder[:-2]
    # encoder = nn.Sequential(*list(encoder.children())[:-2])
    # model = DynamicUnet(encoder, 2, (256, 256), norm_type=NormType.Batch,
    #                     self_attention=True, last_cross=False).cuda()
    # model.load_state_dict(torch.load('cbam_resnet34_unet_monty_Segment.pth'))
    model = DUN().cuda()
    model = nn.DataParallel(model)
    if args.loadModelFlag:
        successFlag = aux.loadModel(args.loadModelFlag, model, args.saveName)
        if successFlag == 0:
            return 0
        elif successFlag == 1:
            print("Model loaded successfully")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate,
                                 weight_decay=args.weightDecay)
    # # Learning
    if args.runMode == 'all':
        for epochNum in range(args.initEpochNum, args.initEpochNum
                              + args.nEpochs):
            trn_metrics, trn_loss_list = run_model(
                trn_data_handler, model, optimizer, loss_wts, args.gamma, amp
            )
            aux.logMetrics(epochNum, trn_metrics, trn_loss_list, 'trn',
                           logFile, 'segment')
            torch.save(model.state_dict(), 'savedModels/'+args.saveName+'.pt')
        # epochNum = 0
            val_metrics, val_loss_list = run_model(
                val_data_handler, model, optimizer, loss_wts, args.gamma, amp
            )
            aux.logMetrics(epochNum, val_metrics, val_loss_list, 'val',
                           logFile, 'segment')
            if bestValRecord and val_metrics.Dice > bestVal:
                bestVal = aux.save_chkpt(bestValRecord, bestVal,
                                         val_metrics.Dice, 'Dice',
                                         model, args.saveName)
        tst_metrics, tst_loss_list = run_model(
            tst_data_handler, model, optimizer, loss_wts, args.gamma, amp
        )
        aux.logMetrics(epochNum, tst_metrics, tst_loss_list, 'tst',
                       logFile, 'segment')
    elif args.runMode == 'val':
            val_metrics, val_loss_list = run_model(
                val_data_handler, model, optimizer, loss_wts, args.gamma, amp
            )
            aux.logMetrics(args.initEpochNum, val_metrics, val_loss_list,
                           'val', logFile, 'segment')
    elif args.runMode == 'tst':
        tst_metrics, tst_loss_list = run_model(
            tst_data_handler, model, optimizer, loss_wts, args.gamma, amp
        )
        aux.logMetrics(args.initEpochNum, tst_metrics, tst_loss_list,
                       'tst', logFile, 'segment')


if __name__ == '__main__':
    NormType = Enum('NormType', 'Batch BatchZero Weight Spectral Instance'
                    'InstanceZero')
    Metrics = namedtuple('Metrics', ['Loss', 'Dice'])
    main()

    # encoder = get_model('cbam_resnet34')
    # encoder = list(encoder.children())
    # encoder = encoder[-2][:-1]
    # model = DynamicUnet(encoder, 2, (512, 512), norm_type=None,
    #                     last_cross=False).cuda()
