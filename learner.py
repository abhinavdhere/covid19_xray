'''
Primary module. Includes dataloader,  trn/val/test functions. Reads
options from user and runs training.
'''
import os
import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
# import torchvision
from tqdm import trange
import numpy as np
import sklearn.metrics
import cv2
# import pydicom as dcm
# from pytorch_model_summary import summary
import aux
import config
from aux import weightedBCE as lossFun
from model import ResNet
# from unet import UNet
# from resnet import resnet18
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


def preprocess_data(full_name):
    img = cv2.imread(full_name, cv2.IMREAD_ANYDEPTH)
    # img = dcm.dcmread(full_name).pixel_array
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img = np.load(full_name)
    # img[img < config.window[0]] = config.window[0]
    # img[img > config.window[1]] = config.window[1]
    img = cv2.resize(img, (config.imgDims[0], config.imgDims[1]),
                     cv2.INTER_AREA)
    # crop1 = config.crop_params[0]
    # crop2 = config.crop_params[1]
    # center = (img.shape[0]//2, img.shape[1]//2)
    # img = img[center[0]-crop1:center[0]+crop1,
    # center[1]-crop2:center[1]+crop2]
    img = (img - np.mean(img)) / np.std(img)
    img = torch.Tensor(img).cuda()
    # img = img.permute(2, 0, 1)
    # img = img.unsqueeze(0)
    return img


def dataLoader(fPath, dataType, batchSize, nBatches, fold_num):
    undersample = False
    sample_size = 2000
    fList = aux.getFList(fPath, dataType, fold_num)
    augNames = ['normal', 'rotated', 'gaussNoise', 'mirror']
    # for augName in augNames:
    #     augList += [name+'_'+augName for name in fList]
    while True:
        # augList_classA = []
        augList_classB = []
        augList = []
        if dataType == 'trn':
            for name in fList:
                # augName = np.random.choice(augNames)
                # augList.append(name+'_'+augName)
                augList += [name + '_' + augName for augName in augNames]
                # if int(name.split('_')[1]) == 2:
                #     augList_classA += [name+'_'+augName for augName in
                #                        augNames]
                # else:
                #     augName = np.random.choice(augNames)
                #     augList_classB.append(name+'_'+augName)
            if undersample:
                augList_classB = np.random.choice(augList_classB,
                                                  (sample_size,),
                                                  replace=False)
            # augList = augList_classA + augList_classB.tolist()
            augList = np.random.permutation(augList)
        else:
            augList += [name+'_normal' for name in fList]
        # print(len(augList))
        dataArr = []
        labelArr = []
        fNameArr = []
        count = 0
        batchCount = 0
        for fName_full in augList:
            fName = '_'.join(fName_full.split('_')[:-1])
            augName = fName_full.split('_')[-1]
            nameParts = fName.split('_')
            lbl = int(nameParts[1])
            # if lbl == 0:
            #     # lbl = 1
            #     continue
            # lbl -= 1
            name_w_path = os.path.join(fPath, fName)
            img = preprocess_data(name_w_path)
            pdb.set_trace()
            lung_mask = cv2.imread(fPath.rsplit('/', 1)[0]
                                   + '/lungSeg/'+fName, cv2.IMREAD_GRAYSCALE)
            lung_mask[lung_mask == 255] = 1
            img = img*lung_mask
            img = img.unsqueeze(0)
            img = augment(img, augName)
            if lbl > 1:
                lbl = 1
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
                count, dataArr, labelArr, fNameArr = 0, [], [], []


def runModel(dataLoader, model, optimizer, classWts, process, batchSize,
             nBatches, lossWts):
    '''
    process : 'trn', 'val' or 'tst'
    '''
    runningLoss = 0
    predList = []
    labelList = []
    softPredList = []
    find_optimal_threshold = False
    # gc = GuidedGradCam(model, model.avgpool)
    with trange(nBatches, desc=process, ncols=100) as t:
        for m in range(nBatches):
            X, y, fName = dataLoader.__next__()
            yOH = aux.toCategorical(y).cuda()
            # print(fName)
            # pdb.set_trace()
            # attribution = gc.attribute(X, 1)
            if process == 'trn':
                optimizer.zero_grad()
                model.train()
                # pred, auxPred = model.forward(X)
                pred, conicity = model.forward(X)
                pred = F.softmax(pred, 1)
                # auxPred = F.softmax(auxPred, 1)
                loss = 0
                for i in range(2):
                    loss += lossFun(classWts[i], pred[:, i], yOH[:, i])
                    # lossWts[1]*lossFun(classWts[i], auxPred[:, i],
                    #                    yOH[:, i])
                loss = lossWts[0]*loss + lossWts[1]*torch.sum(conicity)
                loss.backward()
                optimizer.step()
            elif process == 'val' or process == 'tst':
                model.eval()
                with torch.no_grad():
                    pred, conicity = model.forward(X)
                    pred = F.softmax(pred, 1)
                    loss = 0
                    for i in range(2):
                        loss += lossFun(classWts[i], pred[:, i], yOH[:, i])
                    # loss = (lossFun(classWts[0], pred[:, 0], yOH[:, 0])
                    #         + lossFun(classWts[1], pred[:,  1], yOH[:, 1]))
                    loss = lossWts[0]*loss + lossWts[1]*torch.sum(conicity)
            runningLoss += loss
            hardPred = torch.argmax(pred, 1)
            # hardPred = (pred[:, 1] > 0.38699).int()
            # hardPred = (pred[:, 1] > 0.00415).int()
            predList.append(hardPred.cpu())
            softPredList.append(pred.detach().cpu())
            labelList.append(y.cpu())
            t.set_postfix(loss=runningLoss.item()/(float(m+1)*batchSize))
            t.update()
        finalLoss = runningLoss/(float(m+1)*batchSize)
        acc = aux.globalAcc(predList, labelList)
        f1 = sklearn.metrics.f1_score(torch.cat(labelList),
                                      torch.cat(predList),  labels=None,
                                      average='binary')
        auroc, auprc, fpr_tpr_arr, precision_recall_arr = aux.AUC(softPredList,
                                                                  labelList)
        if find_optimal_threshold and process == 'val':
            softPredList = np.concatenate(softPredList, 0)
            labelList = np.concatenate(labelList, 0)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(labelList,
                                                             softPredList[:,
                                                                          1],
                                                             pos_label=1)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            print("Threshold value is:", optimal_threshold)
        metrics = config.Metrics(finalLoss, acc, f1, auroc, auprc, fpr_tpr_arr,
                                 precision_recall_arr)
        print(metrics.Acc, metrics.F1)
        # metrics = config.Metrics(finalLoss, acc, f1, 0, 0, None, None)
        return metrics


def two_stage_inference(dataLoader, model1, model2, nBatches):
    predList = []
    labelList = []
    softPredList = []
    for m in range(nBatches):
        X, y, fName = dataLoader.__next__()
        # yOH = aux.toCategorical(y).cuda()
        pred, conicity = model1.forward(X)
        pred = F.softmax(pred, 1)
        hardPred = torch.argmax(pred, 1)
        if hardPred[0]:
            pred, _ = model2.forward(X)
            pred = F.softmax(pred, 1)
            hardPred = torch.argmax(pred, 1)
            hardPred += 1
        predList.append(hardPred.cpu())
        softPredList.append(pred.detach().cpu())
        labelList.append(y.cpu())
    acc = aux.globalAcc(predList, labelList)
    f1 = sklearn.metrics.f1_score(torch.cat(labelList),
                                  torch.cat(predList),  labels=None,
                                  average='macro')
    print(acc, f1)
    # auroc, auprc, fpr_tpr_arr, precision_recall_arr = aux.AUC(softPredList,
    #                                                           labelList)
    # metrics = config.Metrics(0, acc, f1, auroc, auprc, fpr_tpr_arr,
    #                          precision_recall_arr)
    # return metrics


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
    trn_nBatches = aux.get_nBatches(config.path, 'trn', args.batchSize,
                                    4, args.foldNum)
    # trn_nBatches = 492  # 849
    trnDataLoader = dataLoader(config.path, 'trn', args.batchSize,
                               trn_nBatches, args.foldNum)
    val_nBatches = aux.get_nBatches(config.path, 'val', args.batchSize,
                                    1, args.foldNum)
    valDataLoader = dataLoader(config.path, 'val', args.batchSize,
                               val_nBatches, args.foldNum)
    tst_nBatches = aux.get_nBatches(config.path, 'tst', args.batchSize,
                                    1, args.foldNum)
    tstDataLoader = dataLoader(config.path, 'tst', args.batchSize,
                               tst_nBatches, args.foldNum)
    model = ResNet(in_channels=1, num_blocks=4, num_layers=4,
                   num_classes=2, downsample_freq=1).cuda()
    model = nn.DataParallel(model)
    if args.loadModelFlag:
        print(args.saveName)
        successFlag = aux.loadModel(args.loadModelFlag, model, args.saveName)
        if successFlag == 0:
            return 0
        elif successFlag == 1:
            print("Model loaded successfully")
    # lossFun = nn.BCELoss(reduction='sum')
    # classWts = aux.getClassBalancedWt(0.9999, [810, 754])
    # classWts = aux.getClassBalancedWt(0.9999, [2720, 2703])
    classWts = aux.getClassBalancedWt(0.9999, [6726, 4610+461])
    # classWts = aux.getClassBalancedWt(0.9999, [4854, 485])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate,
                                 weight_decay=args.weightDecay)
    # # Learning
    if args.runMode == 'all':
        for epochNum in range(args.initEpochNum, args.initEpochNum
                              + args.nEpochs):
            trnMetrics = runModel(trnDataLoader, model, optimizer, classWts,
                                  'trn', args.batchSize, trn_nBatches,
                                  lossWts=lossWts)
            aux.logMetrics(epochNum, trnMetrics, 'trn', logFile, args.saveName)
            torch.save(model.state_dict(), 'savedModels/'+args.saveName+'.pt')
        # epochNum = 0
            valMetrics = runModel(valDataLoader, model, optimizer, classWts,
                                  'val', args.batchSize, val_nBatches, lossWts)
            aux.logMetrics(epochNum, valMetrics, 'val', logFile, args.saveName)
            if bestValRecord and valMetrics.F1 > bestVal:
                bestVal = aux.saveChkpt(bestValRecord, bestVal, valMetrics,
                                        model, args.saveName)
        tstMetrics = runModel(tstDataLoader, model, optimizer, classWts,
                              'tst', args.batchSize, tst_nBatches, lossWts)
        aux.logMetrics(epochNum, tstMetrics, 'tst', logFile, args.saveName)
    elif args.runMode == 'val':
        valMetrics = runModel(valDataLoader, model, optimizer, classWts,
                              'val', args.batchSize, val_nBatches, lossWts)
        # aux.logMetrics(1, valMetrics, 'val', logFile, args.saveName)
    elif args.runMode == 'tst':
        tstMetrics = runModel(tstDataLoader, model, optimizer, classWts,
                              'tst', args.batchSize, tst_nBatches, lossWts)
        aux.logMetrics(1, tstMetrics, 'tst', logFile, args.saveName)
    elif args.runMode == 'two_stage_inference':
        model_stage1 = ResNet(in_channels=1, num_blocks=4, num_layers=4,
                              num_classes=2, downsample_freq=1).cuda()
        model_stage1 = nn.DataParallel(model_stage1)
        model_stage2 = ResNet(in_channels=1, num_blocks=4, num_layers=4,
                              num_classes=2, downsample_freq=1).cuda()
        model_stage2 = nn.DataParallel(model_stage2)
        aux.loadModel('chkpt', model_stage1,
                      ('attn_multiScale_channelsStacked_'
                       'conicity_resnet_covidx_wAug_stage1_rerun'))
        aux.loadModel('chkpt', model_stage2,
                      ('attn_multiScale_channelsStacked_'
                       'conicity_resnet_covidx_wAug_'
                       'undersample_stage2'))
        two_stage_inference(valDataLoader, model_stage1,
                            model_stage2, val_nBatches)
        two_stage_inference(tstDataLoader, model_stage1,
                            model_stage2, tst_nBatches)


if __name__ == '__main__':
    main()

# Graveyard
    # toSave = torch.cat((torch.cat(predList).unsqueeze(-1),
    # torch.cat(labelList)), axis=1)
    # toSave1 = torch.cat((toSave.float(), torch.cat(softPredList)),
    # axis=1)
    # np.savetxt('tst_nih_wAug_preds.csv', toSave1.numpy(), delimiter=',')
    # model = inception_v3(pretrained=False, progress=True,  num_classes=2,
    #                      aux_logits=True, init_weights=True).cuda()
    # model = resnet18(pretrained=False, progress=True, num_classes=2).cuda()
    # # for cross dataset testing
    # tstMetrics = runModel(trnDataLoader, model, optimizer, classWts, 'tst',
    # args.batchSize, trn_nBatches, None)
    # logMetrics(0, tstMetrics, 'tst', logFile, args.saveName)
    # lung_mask_soft = seg_model.forward(img)
    # lung_mask = torch.argmax(lung_mask_soft[0], 0)
    # img = img[0, 0, :, :]*lung_mask
    # seg_model = UNet(n_classes=2).cuda()
    # aux.loadModel('chkpt', seg_model, 'lung_seg')
