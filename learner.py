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
import sklearn.metrics
# import pydicom as dcm
# from pytorch_model_summary import summary
# from torchvision.models import resnet18
import aux
import config
from data_handler import DataLoader
# from aux import weightedBCE as lossFun
from model import MARL
from exp_models import RobustDenseNet
from analyze_performance import PredAnalyzer
# from unet import UNet
from resnet import resnet18


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
    focal_loss_fn = aux.FocalLoss(class_wts, gamma=gamma, reduction='sum')
    if amp:
        with torch.cuda.amp.autocast(enabled=False):
            if process == 'trn':
                pred, aux_pred, conicity = model.forward(X)
                # pred, aux_pred = model.forward(X)
                # pred = model.forward(X)
                aux_pred = F.softmax(aux_pred, 1)
            else:
                pred, conicity = model.forward(X)
                # pred = model.forward(X)
            pred = F.softmax(pred.float(), 1)
            main_focal_loss = focal_loss_fn(pred, y_OH)
            if process == 'trn':
                main_aux_loss = focal_loss_fn(aux_pred, y_OH)
                loss = (loss_wts[0]*main_focal_loss +
                        loss_wts[1]*main_aux_loss)
                loss_list['aux_focal_loss'] += main_aux_loss
                # loss = (loss_wts[0]*main_focal_loss)
            else:
                loss = loss_wts[0]*main_focal_loss
            loss_list['main_focal_loss'] += main_focal_loss
            loss = loss + loss_wts[2]*torch.sum(conicity)
            loss_list['conicity'] += torch.sum(conicity).item()
        # pred = model.forward(X)
    # else:
    #         if process == 'trn':
    #             pred, aux_pred, conicity = model.forward(X)
    #             aux_pred = F.softmax(aux_pred, 1)
    #         else:
    #             pred, conicity = model.forward(X)
    # conicity = torch.abs(conicity)
    # main_aux_loss = lossFun(class_wts[i], aux_pred[:, i], y_OH[:, i])
    # loss = 0
    # for i in range(2):
    # main_bce_loss = lossFun(class_wts[i], pred[:, i], y_OH[:, i])
    return pred, loss, loss_list


def run_model(data_handler, model, optimizer, class_wts, loss_wts, gamma, amp,
              save_name):
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
        save_name (str): name of model (for saving predictions)
    Returns:
        metrics (NamedTuple[Metrics]): Containing selected metrics for epoch
        loss_list (dict): Dictionary containing breakup of loss over the epoch
    '''
    num_batches = data_handler.num_batches
    batch_size = data_handler.batch_size
    process = data_handler.data_type
    running_loss = 0
    # loss_list = {'main_bce': 0, 'aux_bce': 0, 'conicity': 0}
    loss_list = {'main_focal_loss': 0, 'aux_focal_loss': 0, 'conicity': 0}
    # loss_list = {'main_focal_loss': 0, 'aux_focal_loss': 0}
    # loss_list = {'main_focal_loss': 0}
    pred_list = []
    label_list = []
    softpred_list = []
    filename_list = []
    if amp:
        scaler = torch.cuda.amp.GradScaler()
    with trange(num_batches, desc=process, ncols=100) as t:
        for m in range(num_batches):
            X, y, file_names = data_handler.datagen.__next__()
            y_onehot = aux.toCategorical(y).cuda()
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
            pred_list.append(hardPred.cpu())
            softpred_list.append(pred.detach().cpu())
            label_list.append(y.cpu())
            filename_list += file_names
            t.set_postfix(loss=running_loss.item()/(float(m+1)*batch_size))
            t.update()
        final_loss = running_loss/(float(m+1)*batch_size)
        for loss_name in loss_list.keys():
            loss_list[loss_name] /= (float(m+1)*batch_size)
        metrics = compute_metrics(pred_list, label_list, softpred_list,
                                  filename_list, final_loss, process,
                                  save_name)
        # print(metrics.Acc, metrics.F1)
        # metrics = config.Metrics(finalLoss, acc, f1, 0, 0, None, None)
        return metrics, loss_list


def test_time_aug(process, model, aug_names, class_wts, loss_wts, gamma,
                  fold_num, save_name):
    pred_list = []
    label_list = []
    softpred_list = []
    filename_list = []
    running_loss = 0
    loss_list = {'main_focal_loss': 0, 'aux_focal_loss': 0, 'conicity': 0}
    data_handler = DataLoader(process, fold_num, len(aug_names),
                              'all', in_channels=0, aug_names=aug_names)
    num_batches = data_handler.num_batches
    with trange(num_batches, desc=process, ncols=100) as t:
        for m in range(num_batches):
            X, y, file_names = data_handler.datagen.__next__()
            y_onehot = aux.toCategorical(y).cuda()
            model.eval()
            with torch.no_grad():
                pred, loss, loss_list = predict_compute_loss(
                    X, model, y_onehot, class_wts, loss_wts, loss_list,
                    process, gamma, amp=True
                )
                pred = pred.mean(axis=0).unsqueeze(0)
                running_loss += loss
                hardPred = torch.argmax(pred, 1)
                pred_list.append(hardPred.cpu())
                softpred_list.append(pred.detach().cpu())
                label_list.append(y[0].cpu().unsqueeze(0))
                filename_list.append(file_names[0])
                t.set_postfix(loss=running_loss.item()/(float(m+1)*7))
                t.update()
    final_loss = running_loss/(float(m+1))
    metrics = compute_metrics(pred_list, label_list, softpred_list,
                              filename_list, final_loss, process,
                              save_name+'_tta')
    return metrics, loss_list


def compute_metrics(pred_list, label_list, softpred_list, filename_list,
                    final_loss, process, save_name, plot=None):
    acc = aux.globalAcc(pred_list, label_list)
    f1 = sklearn.metrics.f1_score(torch.cat(label_list),
                                  torch.cat(pred_list),  labels=None,
                                  average='binary')
    auroc, auprc, fpr_tpr_arr, precision_recall_arr = aux.AUC(
        softpred_list, label_list
    )
    if plot == 'AUROC':
        display = sklearn.metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auroc)
        display.plot()
        plt.show()
    aux.save_predictions(save_name, process, filename_list, softpred_list,
                         pred_list, label_list)
    metrics = config.Metrics(final_loss, acc, f1, auroc, auprc, fpr_tpr_arr,
                             precision_recall_arr)
    return metrics


def two_stage_inference(data_handler, model1, model2):
    pred_list = []
    label_list = []
    softpred_list = []
    num_batches = data_handler.num_batches
    for m in range(num_batches):
        X, y, file_names = data_handler.datagen.__next__()
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


def two_stage_inference_offline(analyzer_stg1, analyzer_stg2, model_stg2,
                                tst_data_handler):
    """
    Use saved predictions from the two stages to obtain overall performance
    """
    pred_list_stg1 = analyzer_stg1.get_analysis(['Normal', 'Pneumonia'],
                                                silent=True)
    pred_list_stg2 = analyzer_stg2.get_analysis(['Pneumonia', 'COVID'],
                                                silent=True)
    name_list_stg1 = analyzer_stg1.name_list
    name_list_stg2 = analyzer_stg2.name_list
    stg2_map = {}
    final_pred_list = []
    for idx, name in enumerate(name_list_stg2):
        stg2_map[name] = pred_list_stg2[idx]

    ct = 0
    for idx, name in enumerate(name_list_stg1):
        pred_stg1 = pred_list_stg1[idx]
        if pred_stg1 == 1:
            if name in name_list_stg2:
                pred_stg2 = stg2_map[name]
            else:
                # for case when a sample was misclassified as normal in stage1
                ct += 1
                img = tst_data_handler.preprocess_data(
                    config.PATH+'/'+name.rsplit('_', 1)[0], 'normal', False)
                model_stg2.eval()
                pred_soft, _ = model_stg2.forward(img.unsqueeze(0))
                pred_soft = F.softmax(pred_soft, 1)
                pred_stg2 = torch.argmax(pred_soft).item()
            pred = pred_stg2 + 1
        else:
            pred = pred_stg1
        final_pred_list.append(pred)
    label_list = [int(name.split('_')[1]) for name in name_list_stg1]
    report = sklearn.metrics.classification_report(
        label_list, final_pred_list,
        target_names=['Normal', 'Pneumonia', 'COVID'], digits=4)
    print(report)
    print(ct)


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
    all_aug_names = ['normal', 'rotated', 'gaussNoise', 'mirror',
                     'blur', 'sharpen', 'translate']
    trn_data_handler = DataLoader('trn', args.foldNum, args.batchSize,
                                  # 'unequal_all',
                                  'random',
                                  # None,
                                  # 'random_class0_all_class1',
                                  undersample=False, sample_size=3000,
                                  # in_channels=0)
                                  aug_names=all_aug_names, in_channels=0)
    val_data_handler = DataLoader('val', args.foldNum, args.batchSize,
                                  None, in_channels=0)
    tst_data_handler = DataLoader('tst', args.foldNum, args.batchSize,
                                  None, in_channels=0)
    model = MARL(in_channels=1, num_blocks=4, num_layers=4,
                 num_classes=2, downsample_freq=1).cuda()
    # model = RobustDenseNet(pretrained=True, num_classes=2).cuda()
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
    # class_wts = aux.getClassBalancedWt(0.9999, [1203, 1190+394])
    # class_wts = aux.getClassBalancedWt(0.9999, [1190, 394])
    # class_wts = aux.getClassBalancedWt(0.9999, [8308, 5676+258])
    # class_wts = aux.getClassBalancedWt(0.9999, [5676, 258])
    # class_wts = aux.getClassBalancedWt(0.9999, [7081+442, 4854+302])

    # class_wts = aux.getClassBalancedWt(0.9999, [4610, 461])
    # class_wts = aux.getClassBalancedWt(0.9999, [6726, 4610+461])
    # class_wts = aux.getClassBalancedWt(0.9999, [4810, 4810])

    # class_wts = aux.getClassBalancedWt(0.9999, [2040, 2007])
    # class_wts = aux.getClassBalancedWt(0.9999, [7081, 8790])
    class_wts = aux.getClassBalancedWt(0.9999, [4853, 3937])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learningRate,
                                 weight_decay=args.weightDecay)
    # # Learning
    if args.runMode == 'all':
        for epochNum in range(args.initEpochNum, args.initEpochNum
                              + args.nEpochs):
            trnMetrics, trn_loss_list = run_model(
                trn_data_handler, model, optimizer, class_wts,
                loss_wts=loss_wts, gamma=args.gamma, amp=amp,
                save_name=args.saveName
            )
            aux.logMetrics(epochNum, trnMetrics, trn_loss_list, 'trn', logFile,
                           'classify')
            torch.save(model.state_dict(), 'savedModels/'+args.saveName+'.pt')
        # epochNum = 0
            valMetrics, val_loss_list = run_model(
                val_data_handler, model, optimizer, class_wts, loss_wts,
                args.gamma, amp, save_name=args.saveName
            )
            aux.logMetrics(epochNum, valMetrics, val_loss_list, 'val', logFile,
                           'classify')
            if bestValRecord and valMetrics.F1 > bestVal:
                bestVal = aux.save_chkpt(bestValRecord, bestVal, valMetrics.F1,
                                         'F1', model, args.saveName)
        tstMetrics, tst_loss_list = run_model(
            tst_data_handler, model, optimizer, class_wts, loss_wts,
            args.gamma, amp, args.saveName
        )
        aux.logMetrics(epochNum, tstMetrics, tst_loss_list, 'tst', logFile,
                       'classify')
    elif args.runMode == 'two_stage_inference':
        model_stage1 = RobustDenseNet(pretrained=False, num_classes=2).cuda()
        # model_stage1 = MARL(in_channels=1, num_blocks=4, num_layers=4,
        #                     num_classes=2, downsample_freq=1).cuda()
        # model_stage1 = nn.DataParallel(model_stage1)
        model_stage2 = MARL(in_channels=1, num_blocks=4, num_layers=4,
                            num_classes=2, downsample_freq=1).cuda()
        model_stage2 = nn.DataParallel(model_stage2)
        flg1 = aux.loadModel('chkpt', model_stage1,
                             'stage1_covidx_split1_densenet121_wAux_FL')
                             # 'covidx_stage1_noSeg_segTest')
# 'stage1_covidx_split1')
        flg2 = aux.loadModel('chkpt', model_stage2,
                             'covidx_stage2_wSeg_segTest')
                             # 'stage2_covidx_split1')
                             # 'covidx_stage2_noSeg_FL_pairAug_attn_fold0')
        print(flg1, flg2)
        two_stage_inference(val_data_handler, model_stage1, model_stage2)
        two_stage_inference(tst_data_handler, model_stage1, model_stage2)
    elif args.runMode == 'two_stage_inference_offline':
        filename1 = (
            'predictions/bimcv_stage1_wSeg_FL_pairAug_tst_preds.csv')
        filename2 = (
            'predictions/bimcv_stage2_wSeg_FL_pairAug_attn_tst_preds.csv')
        analyzer_stg1 = PredAnalyzer(filename1, True, None)
        # analyzer_stg2 = PredAnalyzer(filename2, True, 'AUROC')
        analyzer_stg2 = PredAnalyzer(filename2, True, None)
        model_stage2 = MARL(in_channels=1, num_blocks=4, num_layers=4,
                            num_classes=2, downsample_freq=1).cuda()
        model_stage2 = nn.DataParallel(model_stage2)
        flg2 = aux.loadModel('chkpt', model_stage2,
                             'bimcv_stage2_wSeg_FL_pairAug_attn')
        two_stage_inference_offline(analyzer_stg1, analyzer_stg2, model_stage2,
                                    tst_data_handler)
    else:
        if args.runMode == 'val':
            data_handler = val_data_handler
        elif args.runMode == 'tst':
            data_handler = tst_data_handler
        if args.tta == 'True':
            metrics, loss_list = test_time_aug(
                args.runMode, model, all_aug_names, class_wts, loss_wts,
                args.gamma, args.foldNum, args.saveName
            )
        else:
            metrics, loss_list = run_model(
                data_handler, model, optimizer, class_wts, loss_wts,
                args.gamma, amp, args.saveName
            )
        aux.logMetrics(1, metrics, loss_list, args.runMode, logFile,
                       'classify')


if __name__ == '__main__':
    main()
