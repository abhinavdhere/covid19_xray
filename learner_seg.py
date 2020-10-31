'''
Primary module. Includes dataloader,  trn/val/test functions. Reads
options from user and runs training.
'''
import os
import pdb
from collections import namedtuple

import torch
import torch.nn.functional as F
# import torch.nn as nn
# import torchvision
from tqdm import trange
import numpy as np
import cv2
# import pydicom as dcm
# from pytorch_model_summary import summary

import aux
import config
from aux import weightedBCE as lossBCE, dice_coeff as lossDice
from unet import UNet
from augmentTools import korniaAffine,  augment_gaussian_noise


def augment(im, augType, dataType):
    if augType == 'normal':
        im = im
    elif augType == 'rotated':
        rotAng = np.random.choice([-10, 10])
        im = korniaAffine(im, rotAng, 'rotate', dataType)
    elif augType == 'gaussNoise':
        im = augment_gaussian_noise(im, (0, 0.5))
    elif augType == 'mirror':
        im = torch.flip(im, [-1])
    return im


def preprocess_data(full_name, file_type):
    '''
    Load image or label and preprocess it.
    file_type : 'data' or 'label'
    '''
    img = cv2.imread(full_name, cv2.IMREAD_ANYDEPTH)
    try:
        img = cv2.resize(img, (config.imgDims[0], config.imgDims[1]),
                         cv2.INTER_AREA)
    except cv2.error:
        pdb.set_trace()
    if file_type == 'data':
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = (img - np.mean(img)) / np.std(img)
    img = torch.Tensor(img)
    if file_type == 'label':
        img = img.unsqueeze(0)
    img = img.cuda()
    return img


def dataLoader(fPath, dataType, batchSize, nBatches):
    undersample = False
    sample_size = 2000
    fPath_img = os.path.join(fPath, 'images')
    fPath_lbl = os.path.join(fPath, 'labels')
    fList = aux.getFList(fPath, dataType)
    augNames = ['normal', 'rotated', 'gaussNoise', 'mirror']
    while True:
        augList = []
        if dataType == 'trn':
            for name in fList:
                augName = np.random.choice(augNames)
                augList.append(name+'_'+augName)
            if undersample:
                augList = np.random.choice(augList, (sample_size,),
                                           replace=False)
            augList = np.random.permutation(augList)
        else:
            augList += [name+'_normal' for name in fList]
        dataArr = []
        labelArr = []
        fNameArr = []
        count = 0
        batchCount = 0
        for fName_full in augList:
            fName = '_'.join(fName_full.split('_')[:-1])
            augName = fName_full.split('_')[-1]
            img_name_w_path = os.path.join(fPath_img,
                                           fName.split('.')[0]+'.jpeg')
            lbl_name_w_path = os.path.join(fPath_lbl, fName)
            img = preprocess_data(img_name_w_path, 'data')
            img = img.permute(2, 0, 1)
            img = augment(img, augName, 'data')
            lbl = preprocess_data(lbl_name_w_path, 'label')
            if augName in ['rotated', 'mirror']:
                lbl = augment(lbl, augName, 'label')
            lbl = lbl.cpu().long()
            if torch.std(img) == 0 or not torch.isfinite(img).all():
                pdb.set_trace()
            dataArr.append(img)
            labelArr.append(lbl)
            fNameArr.append(fName_full)
            count += 1
            if count == batchSize or ((nBatches-batchCount) == 2 and
                                      count == (len(fList) % batchSize)):
                yield torch.stack(dataArr),  torch.stack(labelArr), fNameArr
                batchCount += 1
                count, dataArr, labelArr, fNameArr = 0, [], [], []


def runModel(dataLoader, model, optimizer, process, batchSize,
             nBatches, lossWts):
    '''
    process : 'trn', 'val' or 'tst'
    '''
    runningLoss = 0
    runningDice = 0
    predList = []
    labelList = []
    softPredList = []
    with trange(nBatches, desc=process, ncols=100) as t:
        for m in range(nBatches):
            X, y, fName = dataLoader.__next__()
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
                loss.backward()
                optimizer.step()
            elif process == 'val' or process == 'tst':
                model.eval()
                with torch.no_grad():
                    pred = model.forward(X)
                    pred = F.softmax(pred, 1)
                    # pdb.set_trace()
                    loss = 0
                    for i in range(2):
                        loss += (lossWts[0]*lossBCE(1, pred[:, i], yOH[:, i])
                                 + lossWts[1]*lossDice(pred[:, i], yOH[:, i]))
            runningLoss += loss
            hardPred = torch.argmax(pred, 1)
            runningDice += aux.integral_dice(hardPred, yOH[:, 1], 1)
            predList.append(hardPred.cpu())
            softPredList.append(pred.detach().cpu())
            labelList.append(y.cpu())
            t.set_postfix(loss=runningLoss.item()/(float(m+1)*batchSize))
            t.update()
        finalLoss = runningLoss/(float(m+1)*batchSize)
        acc = aux.globalAcc(predList, labelList)
        acc = acc.item() / (512*512)
        dice = runningDice/(m+1)
        metrics = Metrics(finalLoss.item(), acc, dice.item())
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
        bestValRecord, logFile = aux.initLogging(args.saveName)
        with open(os.path.join('logs', bestValRecord), 'r') as statusFile:
            bestVal = float(statusFile.readline().strip('\n').split()[-1])
    lossWts = tuple(map(float, args.lossWeights.split(',')))
    # Inits
    trn_nBatches = aux.get_nBatches(config.path, 'trn', args.batchSize, 1)
    # trn_nBatches = 492  # 849
    trnDataLoader = dataLoader(config.path, 'trn', args.batchSize,
                               trn_nBatches)
    val_nBatches = aux.get_nBatches(config.path, 'val', args.batchSize, 1)
    valDataLoader = dataLoader(config.path, 'val', args.batchSize,
                               val_nBatches)
    tst_nBatches = aux.get_nBatches(config.path, 'tst', args.batchSize, 1)
    tstDataLoader = dataLoader(config.path, 'tst', args.batchSize,
                               tst_nBatches)
    model = UNet(n_classes=2).cuda()
    # model= nn.DataParallel(model)
    if args.loadModelFlag:
        successFlag = aux.loadModel(args.loadModelFlag, model, args.saveName)
        if successFlag == 0:
            return 0
        elif successFlag == 1:
            print("Model loaded successfully")
    # classWts = aux.getClassBalancedWt(0.9999, [4854, 485])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate,
                                 weight_decay=args.weightDecay)
    # # Learning
    if args.runMode == 'all':
        for epochNum in range(args.initEpochNum, args.initEpochNum
                              + args.nEpochs):
            trnMetrics = runModel(trnDataLoader, model, optimizer,
                                  'trn', args.batchSize, trn_nBatches,
                                  lossWts=lossWts)
            logMetrics(epochNum, trnMetrics, 'trn', logFile, args.saveName)
            torch.save(model.state_dict(), args.saveName+'.pt')
        # epochNum = 0
            valMetrics = runModel(valDataLoader, model, optimizer,
                                  'val', args.batchSize, val_nBatches, lossWts)
            logMetrics(epochNum, valMetrics, 'val', logFile, args.saveName)
            if bestValRecord and valMetrics.Dice > bestVal:
                bestVal = aux.saveChkpt(bestValRecord, bestVal, valMetrics,
                                        model, args.saveName)
        tstMetrics = runModel(tstDataLoader, model, optimizer,
                              'tst', args.batchSize, tst_nBatches, lossWts)
        logMetrics(epochNum, tstMetrics, 'tst', logFile, args.saveName)
    elif args.runMode == 'val':
        valMetrics = runModel(valDataLoader, model, optimizer,
                              'val', args.batchSize, val_nBatches, lossWts)
        logMetrics(1, valMetrics, 'val', logFile, args.saveName)
    elif args.runMode == 'tst':
        tstMetrics = runModel(tstDataLoader, model, optimizer,
                              'tst', args.batchSize, tst_nBatches, lossWts)
        logMetrics(1, tstMetrics, 'tst', logFile, args.saveName)


if __name__ == '__main__':
    Metrics = namedtuple('Metrics', ['Loss', 'Acc', 'Dice'])
    main()
