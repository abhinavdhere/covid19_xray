'''
Primary module. Includes dataloader,  trn/val/test functions. Reads
options from user and runs training.
'''
import os
from collections import namedtuple
# import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
# import torchvision
from tqdm import trange
# import pydicom as dcm
# from pytorch_model_summary import summary

import aux
from aux import weightedBCE as lossBCE, dice_coeff as lossDice
from data_handler import SegDataLoader
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
            runningLoss += loss
            hardPred = torch.argmax(pred, 1)
            runningDice += aux.integral_dice(hardPred, yOH[:, 1], 1)
            predList.append(hardPred.cpu())
            softPredList.append(pred.detach().cpu())
            labelList.append(y.cpu())
            t.set_postfix(loss=runningLoss.item()/(float(m+1)*batch_size))
            t.update()
        finalLoss = runningLoss/(float(m+1)*batch_size)
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
    aug_names = ['normal', 'rotated', 'gaussNoise', 'mirror',
                 'blur', 'sharpen', 'translate']
    trn_data_handler = SegDataLoader('trn', args.foldNum, args.batchSize,
                                     'all',
                                     # 'random_class0_all_class1',
                                     undersample=False, sample_size=3000,
                                     aug_names=aug_names, in_channels=3)
    val_data_handler = SegDataLoader('val', args.foldNum, args.batchSize,
                                     'none', in_channels=3)
    tst_data_handler = SegDataLoader('tst', args.foldNum, args.batchSize,
                                     'none', in_channels=3)
    model = UNet(n_classes=2).cuda()
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
            trnMetrics = runModel(trn_data_handler, model, optimizer,
                                  lossWts=lossWts)
            aux.logMetrics(epochNum, trnMetrics, 'trn', logFile, args.saveName)
            torch.save(model.state_dict(), 'savedModels/'+args.saveName+'.pt')
        # epochNum = 0
            valMetrics = runModel(val_data_handler, model, optimizer,
                                  lossWts)
            aux.logMetrics(epochNum, valMetrics, 'val', logFile, args.saveName)
            if bestValRecord and valMetrics.Dice > bestVal:
                bestVal = aux.saveChkpt(bestValRecord, bestVal, valMetrics,
                                        model, args.saveName)
        tstMetrics = runModel(tst_data_handler, model, optimizer,
                              lossWts)
        aux.logMetrics(epochNum, tstMetrics, 'tst', logFile, args.saveName)
    elif args.runMode == 'val':
        valMetrics = runModel(val_data_handler, model, optimizer,
                              lossWts)
        aux.logMetrics(1, valMetrics, 'val', logFile, args.saveName)
    elif args.runMode == 'tst':
        tstMetrics = runModel(tst_data_handler, model, optimizer,
                              lossWts)
        aux.logMetrics(1, tstMetrics, 'tst', logFile, args.saveName)


if __name__ == '__main__':
    Metrics = namedtuple('Metrics', ['Loss', 'Acc', 'Dice'])
    main()
