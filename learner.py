'''
Primary module. Includes dataloader,  trn/val/test functions. Reads
options from user and runs training.
'''
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
from tqdm import trange
import numpy as np
import sklearn.metrics
import cv2
import aux
from aux import weightedBCE as lossFun
from myGlobals import path, Metrics, imgDims
# from model import inception_v3
from resnet import resnet18
from augmentTools import korniaAffine,  augment_gaussian_noise


def augment(im, augType):
    if augType == 'normal':
        im = im
    elif augType == 'rotated':
        rotAng = np.random.choice([-10, 10])
        im = korniaAffine(im, rotAng, 'rotate')
    elif augType == 'gaussNoise':
        im = augment_gaussian_noise(im, (0, 0.5))
    elif augType == 'mirror':
        im = torch.flip(im, [-1])
    return im


def dataLoader(fPath, dataType, batchSize, nBatches):
    fList = aux.getFList(fPath, dataType)
    if dataType == 'trn':
        augNames = ['normal']  # , 'rotated', 'gaussNoise', 'mirror']
    else:
        augNames = ['normal']
    augList = []
    for augName in augNames:
        augList += [name+'_'+augName for name in fList]
    while True:
        augList = np.random.permutation(augList)
        dataArr = []
        labelArr = []
        fNameArr = []
        count = 0
        batchCount = 0
        # defective = []
        for fName_full in augList:
            fName = '_'.join(fName_full.split('_')[:-1])
            augName = fName_full.split('_')[-1]
            # import pdb ; pdb.set_trace()
            try:
                img = cv2.imread(os.path.join(fPath, dataType, fName),
                                 cv2.IMREAD_ANYDEPTH)
            except OSError:
                print(fName)
                continue
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img, (imgDims[0], imgDims[1]), cv2.INTER_AREA)
            nameParts = fName.split('_')
            lbl = int(nameParts[1])
            # if lbl > 1:
            #     lbl = 1
            img = (img - np.mean(img)) / np.std(img)
            img = torch.Tensor(img).cuda()
            img = img.permute(2, 0, 1)
            img = augment(img, augName)
            if torch.std(img) == 0 or not torch.isfinite(img).all():
                pdb.set_trace()
            lbl = torch.Tensor(np.array([lbl])).long()
            dataArr.append(img)
            labelArr.append(lbl)
            fNameArr.append(fName_full)
            count += 1
            if count == batchSize or ((nBatches-batchCount) == 2 and
                                      count == (len(fList) % batchSize)):
                yield torch.stack(dataArr),  torch.stack(labelArr), fNameArr
                batchCount += 1
                count = 0 ; dataArr = [] ; labelArr = []; fNameArr = []


def runModel(dataLoader, model, optimizer, classWts, process, batchSize,
             nBatches, lossWts):
    '''
    process : 'trn', 'val' or 'tst'
    '''
    runningLoss = 0
    predList = []
    labelList = []
    softPredList = []
    with trange(nBatches, desc=process, ncols=100) as t:
        for m in range(nBatches):
            X, y, fName = dataLoader.__next__()
            yOH = aux.toCategorical(y).cuda()
            # print(fName)
            if process == 'trn':
                optimizer.zero_grad()
                model.train()
                pdb.set_trace()
                # pred, auxPred = model.forward(X)
                pred, _ = model.forward(X)
                # import pdb; pdb.set_trace()
                pred = F.softmax(pred, 1)
                # auxPred = F.softmax(auxPred, 1)
                loss = 0
                for i in range(2):
                    loss += lossWts[0]*lossFun(classWts[i], pred[:, i],
                                               yOH[:, i])  # +\
                            # lossWts[1]*lossFun(classWts[i], auxPred[:, i],
                            #                    yOH[:, i])
                loss.backward()
                optimizer.step()
            elif process == 'val' or process == 'tst':
                model.eval()
                with torch.no_grad():
                    pred, attn_map = model.forward(X)
                    pred = F.softmax(pred, 1)
                    loss = (lossFun(classWts[0], pred[:, 0], yOH[:, 0])
                            + lossFun(classWts[1], pred[:,  1], yOH[:, 1]))
            runningLoss += loss
            hardPred = torch.argmax(pred, 1)
            predList.append(hardPred.cpu())
            softPredList.append(pred.detach().cpu())
            labelList.append(y.cpu())
            t.set_postfix(loss=runningLoss.item()/(float(m+1)*batchSize))
            t.update()
        finalLoss = runningLoss/(float(m+1)*batchSize)
        acc = aux.globalAcc(predList, labelList)
        f1 = sklearn.metrics.f1_score(torch.cat(labelList),
                                      torch.cat(predList),  labels=None)
        auroc, auprc, fpr_tpr_arr, precision_recall_arr = aux.AUC(softPredList,
                                                                  labelList)
        metrics = Metrics(finalLoss, acc, f1, auroc, auprc, fpr_tpr_arr,
                          precision_recall_arr)
        # toSave = torch.cat((torch.cat(predList).unsqueeze(-1),
        # torch.cat(labelList)), axis=1)
        # toSave1 = torch.cat((toSave.float(), torch.cat(softPredList)),
        # axis=1)
        # np.savetxt('tst_nih_wAug_preds.csv', toSave1.numpy(), delimiter=',')
        return metrics


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
    trn_nBatches = aux.get_nBatches(path, 'trn', args.batchSize, 1)
    trnDataLoader = dataLoader(path, 'trn', args.batchSize, trn_nBatches)
    val_nBatches = aux.get_nBatches(path, 'val', args.batchSize, 1)
    valDataLoader = dataLoader(path, 'val', args.batchSize, val_nBatches)
    tst_nBatches = aux.get_nBatches(path, 'tst', args.batchSize, 1)
    tstDataLoader = dataLoader(path, 'tst', args.batchSize, tst_nBatches)
    # model = inception_v3(pretrained=False, progress=True,  num_classes=2,
    #                      aux_logits=True, init_weights=True).cuda()
    model = resnet18(pretrained=False, progress=True, num_classes=2).cuda()
    # import pdb ; pdb.set_trace()
    # model = AttnModel().cuda()
    model = nn.DataParallel(model)
    # model = BasicNet2().cuda()
    if args.loadModelFlag:
        successFlag = aux.loadModel(args.loadModelFlag, model, args.saveName)
        if successFlag == 0:
            return 0
        elif successFlag == 1:
            print("Model loaded successfully")
    # lossFun = nn.BCELoss(reduction='sum')
    # classWts = aux.getClassBalancedWt(0.9999, [810, 754])
    classWts = aux.getClassBalancedWt(0.9999, [6039, 6039])
    # [1431, 1431])#[15608,19917])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate,
                                 weight_decay=args.weightDecay)
    # # Learning
    for epochNum in range(args.initEpochNum, args.initEpochNum+args.nEpochs):
        trnMetrics = runModel(trnDataLoader, model, optimizer, classWts,
                              'trn', args.batchSize, trn_nBatches,
                              lossWts=lossWts)
        aux.logMetrics(epochNum, trnMetrics, 'trn', logFile, args.saveName)
        torch.save(model.state_dict(), args.saveName+'.pt')
    # epochNum = 0
        valMetrics = runModel(valDataLoader, model, optimizer, classWts,
                              'val', args.batchSize, val_nBatches, None)
        aux.logMetrics(epochNum, valMetrics, 'val', logFile, args.saveName)
        if bestValRecord and valMetrics.AUROC > bestVal:
            bestVal = aux.saveChkpt(bestValRecord, bestVal, valMetrics, model,
                                    args.saveName)
    tstMetrics = runModel(tstDataLoader, model, optimizer, classWts,
                          'tst', args.batchSize, tst_nBatches, None)
    aux.logMetrics(epochNum, tstMetrics, 'tst', logFile, args.saveName)
    # # for cross dataset testing
    # tstMetrics = runModel(trnDataLoader, model, optimizer, classWts, 'tst',
    # args.batchSize, trn_nBatches, None)
    # logMetrics(0, tstMetrics, 'tst', logFile, args.saveName)


if __name__ == '__main__':
    main()
