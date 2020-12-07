'''
Primary module. Includes dataloader,  trn/val/test functions. Reads
options from user and runs training.
'''
import os
from collections import namedtuple
# import pdb

import torch
import torch.nn.functional as F
from tqdm import trange
# from pytorchcv.model_provider import get_model
from fastai.vision.models.unet import DynamicUnet
import torch.nn as nn
from torchvision.models import resnext50_32x4d
# import torchvision
# import pydicom as dcm
# from pytorch_model_summary import summary

import aux
from aux import weightedBCE as lossBCE, dice_coeff as lossDice
from data_handler import SegDataLoader
# from dense_unet import DUN
from unet import UNet


def runModel(data_handler, model, optimizer, lossWts):
    '''
    process : 'trn', 'val' or 'tst'
    '''
    num_batches = data_handler.num_batches
    batch_size = data_handler.batch_size
    process = data_handler.data_type
    runningLoss = 0
    runningDice = 0
    predList = []
    labelList = []
    softPredList = []
    with trange(num_batches, desc=process, ncols=100) as t:
        for m in range(num_batches):
            X, y, fName = data_handler.datagen.__next__()
            yOH = aux.toCategorical(y, 'seg').cuda()
            if process == 'trn':
                optimizer.zero_grad()
                model.train()
                pred = model.forward(X)
                pred = F.softmax(pred, 1)
                loss = 0
                for i in range(2):
                    loss += (lossWts[0]*lossBCE(1, pred[:, i], yOH[:, i])
                             + lossWts[1]*lossDice(pred[:, i], yOH[:, i]))
                # loss += lossWts[2]*F.mse_loss(recons, X)
                loss.backward()
                optimizer.step()
            elif process == 'val' or process == 'tst':
                model.eval()
                with torch.no_grad():
                    pred = model.forward(X)
                    pred = F.softmax(pred, 1)
                    loss = 0
                    for i in range(2):
                        loss += (lossWts[0]*lossBCE(1, pred[:, i], yOH[:, i])
                                 + lossWts[1]*lossDice(pred[:, i], yOH[:, i]))
                    # loss += lossWts[2]*F.mse_loss(recons, X)
            runningLoss += loss
            hardPred = torch.argmax(pred, 1)
            runningDice += aux.integral_dice(hardPred, yOH[:, 1], 1)
            predList.append(hardPred.cpu())
            softPredList.append(pred.detach().cpu())
            labelList.append(y.cpu())
            t.set_postfix(loss=runningLoss.item()/(float(m+1)*batch_size))
            t.update()
        finalLoss = runningLoss/(float(m+1)*batch_size)
        # acc = aux.globalAcc(predList, labelList)
        # acc = acc.item() / (512*512)
        dice = runningDice/(m+1)
        metrics = Metrics(finalLoss.item(), 0, dice.item())
        # print(metrics.Acc, metrics.Dice)
        return metrics


def logMetrics(epochNum, metrics, process, logFile, saveName):
    '''
    Print metrics to terminal and save to logfile in a proper format.
    '''
    line = (('Epoch num. {epochNum:d} \t {process} Loss : {lossVal:.7f};'
            '{process} Acc : {acc:.3f} ; {process} Dice : {dice:.3f}\n')
            .format(epochNum=epochNum, process=process, lossVal=metrics.Loss,
                    acc=metrics.Acc, dice=metrics.Dice))
    print(line.strip('\n'))
    if logFile:
        with open(os.path.join('logs', logFile), 'a') as f:
            f.write(line)


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
    lossWts = tuple(map(float, args.lossWeights.split(',')))
    # Inits
    aug_names = ['normal', 'rotated', 'gaussNoise', 'mirror',
                 'blur', 'sharpen', 'translate']
    trn_data_handler = SegDataLoader('trn', args.foldNum, args.batchSize,
                                     'random',
                                     # 'random_class0_all_class1',
                                     undersample=False, sample_size=3000,
                                     aug_names=aug_names, in_channels=3)
    val_data_handler = SegDataLoader('val', args.foldNum, args.batchSize,
                                     'none', in_channels=3)
    tst_data_handler = SegDataLoader('tst', args.foldNum, args.batchSize,
                                     'none', in_channels=3)
    # model = UNet(n_classes=2, backbone='resnext50_32x4d').cuda()
    encoder = resnext50_32x4d(pretrained=True)
    # encoder = encoder[:-2]
    encoder = nn.Sequential(*list(encoder.children())[:-2])
    model = DynamicUnet(encoder, 2, (512, 512), norm_type=None,
                        last_cross=False).cuda()
    # model.load_state_dict(torch.load('cbam_resnet34_unet_monty_Segment.pth'))
    # model = DUN().cuda()
    # model = nn.DataParallel(model)
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
            trn_metrics = runModel(trn_data_handler, model, optimizer,
                                   lossWts=lossWts)
            aux.logMetrics(epochNum, trn_metrics, 'trn', logFile,
                           args.saveName, 'segment')
            torch.save(model.state_dict(), 'savedModels/'+args.saveName+'.pt')
        # epochNum = 0
            val_metrics = runModel(val_data_handler, model, optimizer,
                                   lossWts)
            aux.logMetrics(epochNum, val_metrics, 'val', logFile,
                           args.saveName, 'segment')
            if bestValRecord and val_metrics.Dice > bestVal:
                bestVal = aux.save_chkpt(bestValRecord, bestVal,
                                         val_metrics.Dice, 'Dice',
                                         model, args.saveName)
        tst_metrics = runModel(tst_data_handler, model, optimizer,
                               lossWts)
        aux.logMetrics(epochNum, tst_metrics, 'tst', logFile,
                       args.saveName, 'segment')
    elif args.runMode == 'val':
        val_metrics = runModel(val_data_handler, model, optimizer,
                               lossWts)
        aux.logMetrics(1, val_metrics, 'val', logFile,
                       args.saveName, 'segment')
    elif args.runMode == 'tst':
        tst_metrics = runModel(tst_data_handler, model, optimizer,
                               lossWts)
        aux.logMetrics(1, tst_metrics, 'tst', logFile,
                       args.saveName, 'segment')


if __name__ == '__main__':
    Metrics = namedtuple('Metrics', ['Loss', 'Acc', 'Dice'])
    main()

    # encoder = get_model('cbam_resnet34')
    # encoder = list(encoder.children())
    # encoder = encoder[-2][:-1]
    # model = DynamicUnet(encoder, 2, (512, 512), norm_type=None,
    #                     last_cross=False).cuda()

