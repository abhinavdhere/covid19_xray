'''
Primary module. Includes dataloader,  trn/val/test functions. Reads
options from user and runs training.
'''
import os
# import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
# import torchvision
from tqdm import trange
import numpy as np
import sklearn.metrics
# import pydicom as dcm
# from pytorch_model_summary import summary
# from torchvision.models import resnet18
import aux
import config
from data_handler import DataLoader
from aux import weightedBCE as lossFun
from model import ResNet
# from unet import UNet
# from resnet import resnet18


def runModel(data_handler, model, optimizer, classWts, lossWts):
    '''
    process : 'trn', 'val' or 'tst'
    '''
    num_batches = data_handler.num_batches
    batch_size = data_handler.batch_size
    process = data_handler.data_type
    runningLoss = 0
    predList = []
    labelList = []
    softPredList = []
    find_optimal_threshold = False
    # gc = GuidedGradCam(model, model.avgpool)
    with trange(num_batches, desc=process, ncols=100) as t:
        for m in range(num_batches):
            X, y, file_name = data_handler.datagen.__next__()
            yOH = aux.toCategorical(y).cuda()
            # print(file_name)
            # pdb.set_trace()
            if process == 'trn':
                optimizer.zero_grad()
                model.train()
                # pred = model.forward(X)
                pred, auxPred, conicity = model.forward(X)
                conicity = torch.abs(conicity)
                pred = F.softmax(pred, 1)
                auxPred = F.softmax(auxPred, 1)
                loss = 0
                for i in range(2):
                    loss += (lossWts[0]*lossFun(classWts[i], pred[:, i],
                                                yOH[:, i])
                             + lossWts[1]*lossFun(classWts[i], auxPred[:, i],
                                                  yOH[:, i]))
                loss = loss + lossWts[2]*torch.sum(conicity)
                loss.backward()
                optimizer.step()
            elif process == 'val' or process == 'tst':
                model.eval()
                with torch.no_grad():
                    # pred = model.forward(X)
                    pred, conicity = model.forward(X)
                    conicity = torch.abs(conicity)
                    pred = F.softmax(pred, 1)
                    loss = 0
                    for i in range(2):
                        loss += lossFun(classWts[i], pred[:, i], yOH[:, i])
                    # loss = (lossFun(classWts[0], pred[:, 0], yOH[:, 0])
                    #         + lossFun(classWts[1], pred[:,  1], yOH[:, 1]))
                    loss = lossWts[0]*loss + lossWts[1]*torch.sum(conicity)
            runningLoss += loss
            hardPred = torch.argmax(pred, 1)
            # hardPred = (pred[:, 1] > 0.983).int()
            # hardPred = (pred[:, 1] > 0.2181).int()
            predList.append(hardPred.cpu())
            softPredList.append(pred.detach().cpu())
            labelList.append(y.cpu())
            t.set_postfix(loss=runningLoss.item()/(float(m+1)*batch_size))
            t.update()
        finalLoss = runningLoss/(float(m+1)*batch_size)
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
            # precision, recall, thresholds = \
            #     sklearn.metrics.precision_recall_curve(labelList,
            #                                            softPredList[:, 1],
            #                                            pos_label=1)
            optimal_idx = np.argmax(tpr - fpr)
            # fscore = (2 * precision * recall) / (precision + recall)
            # optimal_idx = np.argmax(fscore)
            optimal_threshold = thresholds[optimal_idx]
            print("Threshold value is:", optimal_threshold)
        metrics = config.Metrics(finalLoss, acc, f1, auroc, auprc, fpr_tpr_arr,
                                 precision_recall_arr)
        # print(metrics.Acc, metrics.F1)
        # metrics = config.Metrics(finalLoss, acc, f1, 0, 0, None, None)
        return metrics


def two_stage_inference(data_handler, model1, model2):
    predList = []
    labelList = []
    softPredList = []
    num_batches = data_handler.num_batches
    for m in range(num_batches):
        X, y, file_name = data_handler.datagen.__next__()
        # yOH = aux.toCategorical(y).cuda()
        model1.eval()
        model2.eval()
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
    trn_data_handler = DataLoader('trn', args.foldNum, args.batchSize,
                                  'all',
                                  # 'random_class0_all_class1',
                                  undersample=False, sample_size=3000,
                                  aug_names=aug_names)
    val_data_handler = DataLoader('val', args.foldNum, args.batchSize, 'none')
    tst_data_handler = DataLoader('tst', args.foldNum, args.batchSize, 'none')
    model = ResNet(in_channels=1, num_blocks=4, num_layers=4,
                   num_classes=2, downsample_freq=1).cuda()
# print(summary(model, torch.zeros((1, 1, 512, 512)).cuda(), show_input=True))
    # model = resnet18(num_classes=2).cuda()
    model = nn.DataParallel(model)
    if args.loadModelFlag:
        print(args.saveName)
        successFlag = aux.loadModel(args.loadModelFlag, model, args.saveName)
        if successFlag == 0:
            return 0
        elif successFlag == 1:
            print("Model loaded successfully")
#    classWts = aux.getClassBalancedWt(0.9999, [1203, 1176+390])
    # classWts = aux.getClassBalancedWt(0.9999, [4610, 461])
    classWts = aux.getClassBalancedWt(0.9999, [6726, 4610+461])
    # classWts = aux.getClassBalancedWt(0.9999, [4810, 4810])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate,
                                 weight_decay=args.weightDecay)
    # # Learning
    if args.runMode == 'all':
        for epochNum in range(args.initEpochNum, args.initEpochNum
                              + args.nEpochs):
            trnMetrics = runModel(trn_data_handler, model, optimizer, classWts,
                                  lossWts=lossWts)
            aux.logMetrics(epochNum, trnMetrics, 'trn', logFile, args.saveName)
            torch.save(model.state_dict(), 'savedModels/'+args.saveName+'.pt')
        # epochNum = 0
            valMetrics = runModel(val_data_handler, model, optimizer, classWts,
                                  lossWts)
            aux.logMetrics(epochNum, valMetrics, 'val', logFile, args.saveName)
            if bestValRecord and valMetrics.F1 > bestVal:
                bestVal = aux.saveChkpt(bestValRecord, bestVal, valMetrics,
                                        model, args.saveName)
        tstMetrics = runModel(tst_data_handler, model, optimizer, classWts,
                              lossWts)
        aux.logMetrics(epochNum, tstMetrics, 'tst', logFile, args.saveName)
    elif args.runMode == 'val':
        valMetrics = runModel(val_data_handler, model, optimizer, classWts,
                              lossWts)
        aux.logMetrics(1, valMetrics, 'val', logFile, args.saveName)
    elif args.runMode == 'tst':
        tstMetrics = runModel(tst_data_handler, model, optimizer, classWts,
                              lossWts)
        aux.logMetrics(1, tstMetrics, 'tst', logFile, args.saveName)
    elif args.runMode == 'two_stage_inference':
        model_stage1 = ResNet(in_channels=1, num_blocks=4, num_layers=4,
                              num_classes=2, downsample_freq=1).cuda()
        model_stage1 = nn.DataParallel(model_stage1)
        model_stage2 = ResNet(in_channels=1, num_blocks=4, num_layers=4,
                              num_classes=2, downsample_freq=1).cuda()
        model_stage2 = nn.DataParallel(model_stage2)
        flg1 = aux.loadModel('chkpt', model_stage1,
                             'stage1_noLungSeg_allAug_wAux_absConicity_fold'
                             + str(args.foldNum))
        flg2 = aux.loadModel('main', model_stage2,
                             'stage2_noLungSeg_allAug_wAux_absConicity_fold'
                             + str(args.foldNum))
        print(flg1, flg2)
        two_stage_inference(val_data_handler, model_stage1, model_stage2)
        two_stage_inference(tst_data_handler, model_stage1, model_stage2)


if __name__ == '__main__':
    main()

# Graveyard
    # trn_nBatches = 492  # 849
    # img = dcm.dcmread(full_name).pixel_array
    # img = np.load(full_name)
    # img[img < config.window[0]] = config.window[0]
    # img[img > config.window[1]] = config.window[1]
    # crop1 = config.crop_params[0]
    # crop2 = config.crop_params[1]
    # center = (img.shape[0]//2, img.shape[1]//2)
    # img = img[center[0]-crop1:center[0]+crop1,
    # center[1]-crop2:center[1]+crop2]
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
    # auroc, auprc, fpr_tpr_arr, precision_recall_arr = aux.AUC(softPredList,
    #                                                           labelList)
    # metrics = config.Metrics(0, acc, f1, auroc, auprc, fpr_tpr_arr,
    #                          precision_recall_arr)
    # return metrics
