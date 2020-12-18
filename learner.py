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
# from aux import weightedBCE as lossFun
from model import ResNet
from exp_models import RobustDenseNet
# from unet import UNet
# from resnet import resnet18


def predict_compute_loss(X, model, y_OH, class_wts, loss_wts, loss_list,
                         process, gamma, amp):
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
    # if amp:
    #     with torch.cuda.amp.autocast():
    if process == 'trn':
        # pred, aux_pred, conicity = model.forward(X)
        pred, aux_pred = model.forward(X)
        aux_pred = F.softmax(aux_pred, 1)
    else:
        # pred, conicity = model.forward(X)
        pred = model.forward(X)
    # else:
    #         if process == 'trn':
    #             pred, aux_pred, conicity = model.forward(X)
    #             aux_pred = F.softmax(aux_pred, 1)
    #         else:
    #             pred, conicity = model.forward(X)
    # conicity = torch.abs(conicity)
    pred = F.softmax(pred, 1)
    focal_loss_fn = aux.FocalLoss(class_wts, gamma=gamma, reduction='sum')
    # loss = 0
    # for i in range(2):
    # main_bce_loss = lossFun(class_wts[i], pred[:, i], y_OH[:, i])
    main_focal_loss = focal_loss_fn(pred, y_OH)
    if process == 'trn':
        main_aux_loss = focal_loss_fn(aux_pred, y_OH)
    # main_aux_loss = lossFun(class_wts[i], aux_pred[:, i], y_OH[:, i])
        loss = (loss_wts[0]*main_focal_loss + loss_wts[1]*main_aux_loss)
        loss_list['aux_focal_loss'] += main_aux_loss
    else:
        loss = loss_wts[0]*main_focal_loss
    loss_list['main_focal_loss'] += main_focal_loss
    # loss = loss + loss_wts[2]*torch.sum(conicity)
    # loss_list['conicity'] += torch.sum(conicity)
    return pred, loss, loss_list


def run_model(data_handler, model, optimizer, class_wts, loss_wts, gamma, amp):
    '''
    Loads data from given data_handler object, runs model prediction,
    collects losses/metrics and computes gradient & updates weights
    if process is trn.
    Args:
        data_handler (DataLoader): data loader object
        model (torch.nn.Module): model for training/inference
        optimizer (torch.optim module): optimizer
        class_wts (List[float]): class weights for weighted loss
        loss_wts (List[float]): weightage for loss functions
        gamma (int): focusing factor gamma for focal loss
        amp (bool): Whether to use mixed precision
    Returns:
        metrics (NamedTuple[Metrics]): Containing selected metrics for epoch
        loss_list (dict): Dictionary containing breakup of loss over the epoch
    '''
    num_batches = data_handler.num_batches
    batch_size = data_handler.batch_size
    process = data_handler.data_type
    running_loss = 0
    # loss_list = {'main_bce': 0, 'aux_bce': 0, 'conicity': 0}
    loss_list = {'main_focal_loss': 0, 'aux_focal_loss': 0}
    pred_list = []
    label_list = []
    softpred_list = []
    find_optimal_threshold = False
    if amp:
        scaler = torch.cuda.amp.GradScaler()
    with trange(num_batches, desc=process, ncols=100) as t:
        for m in range(num_batches):
            X, y, file_name = data_handler.datagen.__next__()
            y_onehot = aux.toCategorical(y).cuda()
            # print(file_name)
            # pdb.set_trace()
            if process == 'trn':
                optimizer.zero_grad()
                model.train()
                # pred = model.forward(X)
                pred, loss, loss_list = predict_compute_loss(
                    X, model, y_onehot, class_wts, loss_wts, loss_list,
                    process, gamma, amp
                )
                if amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            elif process == 'val' or process == 'tst':
                model.eval()
                with torch.no_grad():
                    # pred = model.forward(X)
                    pred, loss, loss_list = predict_compute_loss(
                        X, model, y_onehot, class_wts, loss_wts, loss_list,
                        process, gamma, amp
                    )
            running_loss += loss
            hardPred = torch.argmax(pred, 1)
            # hardPred = (pred[:, 1] > 0.983).int()
            # hardPred = (pred[:, 1] > 0.2181).int()
            pred_list.append(hardPred.cpu())
            softpred_list.append(pred.detach().cpu())
            label_list.append(y.cpu())
            t.set_postfix(loss=running_loss.item()/(float(m+1)*batch_size))
            t.update()
        finalLoss = running_loss/(float(m+1)*batch_size)
        for loss_name in loss_list.keys():
            loss_list[loss_name] /= (float(m+1)*batch_size)
        acc = aux.globalAcc(pred_list, label_list)
        f1 = sklearn.metrics.f1_score(torch.cat(label_list),
                                      torch.cat(pred_list),  labels=None,
                                      average='binary')
        auroc, auprc, fpr_tpr_arr, precision_recall_arr = aux.AUC(
            softpred_list, label_list
        )
        if find_optimal_threshold and process == 'val':
            softpred_list = np.concatenate(softpred_list, 0)
            label_list = np.concatenate(label_list, 0)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                label_list, softpred_list[:, 1], pos_label=1
            )
            # precision, recall, thresholds = \
            #     sklearn.metrics.precision_recall_curve(label_list,
            #                                            softpred_list[:, 1],
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
        return metrics, loss_list


def two_stage_inference(data_handler, model1, model2):
    pred_list = []
    label_list = []
    softpred_list = []
    num_batches = data_handler.num_batches
    for m in range(num_batches):
        X, y, file_name = data_handler.datagen.__next__()
        # y_onehot = aux.toCategorical(y).cuda()
        model1.eval()
        model2.eval()
        # pred, conicity = model1.forward(X)
        pred = model1.forward(X)
        pred = F.softmax(pred, 1)
        hardPred = torch.argmax(pred, 1)
        if hardPred[0]:
            X = X[:, 0, :, :]
            X = X.unsqueeze(1)
            pred, _ = model2.forward(X)
            pred = F.softmax(pred, 1)
            hardPred = torch.argmax(pred, 1)
            hardPred += 1
        pred_list.append(hardPred.cpu())
        softpred_list.append(pred.detach().cpu())
        label_list.append(y.cpu())
    acc = aux.globalAcc(pred_list, label_list)
    f1 = sklearn.metrics.f1_score(torch.cat(label_list),
                                  torch.cat(pred_list),  labels=None,
                                  average='macro')
    print(acc, f1)


def main():
    # Take options and hyperparameters from user
    torch.autograd.set_detect_anomaly(True)
    parser = aux.getOptions()
    args = parser.parse_args()
    if args.saveName is None:
        print("Warning! Savename unspecified. No logging will take place."
              "Model will not be saved.")
        bestValRecord = None
        logFile = None
    else:
        bestValRecord, logFile = aux.initLogging(args.saveName, 'F1')
        with open(os.path.join('logs', bestValRecord), 'r') as statusFile:
            bestVal = float(statusFile.readline().strip('\n').split()[-1])
    loss_wts = tuple(map(float, args.lossWeights.split(',')))
    amp = (args.amp == 'True')
    # Inits
    aug_names = ['normal', 'rotated', 'gaussNoise', 'mirror',
                 'blur', 'sharpen', 'translate']
    trn_data_handler = DataLoader('trn', args.foldNum, args.batchSize,
                                  'random',
                                  # 'random_class0_all_class1',
                                  undersample=False, sample_size=2000,
                                  aug_names=aug_names, in_channels=3)
    val_data_handler = DataLoader('val', args.foldNum, args.batchSize,
                                  'none', in_channels=3)
    tst_data_handler = DataLoader('tst', args.foldNum, args.batchSize,
                                  'none', in_channels=3)
    # model = ResNet(in_channels=1, num_blocks=4, num_layers=4,
    #                num_classes=2, downsample_freq=1).cuda()
    model = RobustDenseNet(pretrained=True, num_classes=2).cuda()
# print(summary(model, torch.zeros((1, 1, 512, 512)).cuda(), show_input=True))
    # model = resnet18(num_classes=2).cuda()
    # model = nn.DataParallel(model)
    if args.loadModelFlag:
        print(args.saveName)
        successFlag = aux.loadModel(args.loadModelFlag, model, args.saveName)
        if successFlag == 0:
            return 0
        elif successFlag == 1:
            print("Model loaded successfully")
#    class_wts = aux.getClassBalancedWt(0.9999, [1203, 1176+390])
    class_wts = aux.getClassBalancedWt(0.9999, [8308, 5676+258])
    # class_wts = aux.getClassBalancedWt(0.9999, [5676, 258])
    # class_wts = aux.getClassBalancedWt(0.9999, [4610, 461])
    # class_wts = aux.getClassBalancedWt(0.9999, [6726, 4610+461])
    # class_wts = aux.getClassBalancedWt(0.9999, [4810, 4810])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate,
                                 weight_decay=args.weightDecay)
    # # Learning
    if args.runMode == 'all':
        for epochNum in range(args.initEpochNum, args.initEpochNum
                              + args.nEpochs):
            trnMetrics, trn_loss_list = run_model(
                trn_data_handler, model, optimizer, class_wts,
                loss_wts=loss_wts, gamma=args.gamma, amp=amp
            )
            aux.logMetrics(epochNum, trnMetrics, trn_loss_list, 'trn', logFile,
                           'classify')
            torch.save(model.state_dict(), 'savedModels/'+args.saveName+'.pt')
        # epochNum = 0
            valMetrics, val_loss_list = run_model(
                val_data_handler, model, optimizer, class_wts, loss_wts,
                args.gamma, amp
            )
            aux.logMetrics(epochNum, valMetrics, val_loss_list, 'val', logFile,
                           'classify')
            if bestValRecord and valMetrics.F1 > bestVal:
                bestVal = aux.save_chkpt(bestValRecord, bestVal, valMetrics.F1,
                                         'F1', model, args.saveName)
        tstMetrics, tst_loss_list = run_model(
            tst_data_handler, model, optimizer, class_wts, loss_wts,
            args.gamma, amp
        )
        aux.logMetrics(epochNum, tstMetrics, tst_loss_list, 'tst', logFile,
                       'classify')
    elif args.runMode == 'val':
        valMetrics, val_loss_list = run_model(
            val_data_handler, model, optimizer, class_wts, loss_wts,
            args.gamma, amp
        )
        aux.logMetrics(1, valMetrics, val_loss_list, 'val', logFile,
                       'classify')
    elif args.runMode == 'tst':
        tstMetrics, tst_loss_list = run_model(
            tst_data_handler, model, optimizer, class_wts, loss_wts,
            args.gamma, amp
        )
        aux.logMetrics(1, tstMetrics, tst_loss_list, 'tst', logFile,
                       'classify')
    elif args.runMode == 'two_stage_inference':
        model_stage1 = RobustDenseNet(pretrained=True, num_classes=2).cuda()
        # model_stage1 = ResNet(in_channels=1, num_blocks=4, num_layers=4,
        #                       num_classes=2, downsample_freq=1).cuda()
        # model_stage1 = nn.DataParallel(model_stage1)
        model_stage2 = ResNet(in_channels=1, num_blocks=4, num_layers=4,
                              num_classes=2, downsample_freq=1).cuda()
        model_stage2 = nn.DataParallel(model_stage2)
        flg1 = aux.loadModel('chkpt', model_stage1,
                             'stage1_covidx_split1_densenet121_wAux_FL')
# 'stage1_covidx_split1')
        flg2 = aux.loadModel('chkpt', model_stage2,
                             'stage2_covidx_split1')
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
    # toSave = torch.cat((torch.cat(pred_list).unsqueeze(-1),
    # torch.cat(label_list)), axis=1)
    # toSave1 = torch.cat((toSave.float(), torch.cat(softpred_list)),
    # axis=1)
    # np.savetxt('tst_nih_wAug_preds.csv', toSave1.numpy(), delimiter=',')
    # model = inception_v3(pretrained=False, progress=True,  num_classes=2,
    #                      aux_logits=True, init_weights=True).cuda()
    # model = resnet18(pretrained=False, progress=True, num_classes=2).cuda()
    # # for cross dataset testing
    # tstMetrics = run_model(trnDataLoader, model, optimizer, class_wts, 'tst',
    # args.batchSize, trn_nBatches, None)
    # logMetrics(0, tstMetrics, 'tst', logFile, args.saveName)
    # lung_mask_soft = seg_model.forward(img)
    # lung_mask = torch.argmax(lung_mask_soft[0], 0)
    # img = img[0, 0, :, :]*lung_mask
    # seg_model = UNet(n_classes=2).cuda()
    # aux.loadModel('chkpt', seg_model, 'lung_seg')
    # auroc, auprc, fpr_tpr_arr, precision_recall_arr = aux.AUC(softpred_list,
    #                                                           label_list)
    # metrics = config.Metrics(0, acc, f1, auroc, auprc, fpr_tpr_arr,
    #                          precision_recall_arr)
    # return metrics
