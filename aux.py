import numpy as np
import torch
import sklearn.metrics
import argparse
import os
from myGlobals import dataSetList


''' Auxiliary functions for learner to use '''


def getFList(path, process):
    '''
    Generate list of files as per process (dataType) and dataset.
    '''
    allFileList = os.listdir(os.path.join(path, process))
    fList = []
    for fName in allFileList:
        if fName.split('_')[0] in dataSetList:
            fList.append(fName)
    return fList


def get_nBatches(path, process, batchSize, augFactor):
    '''
    Compute number of batches.
    '''
    fList = getFList(path, process)
    nSamples = augFactor*len(fList)  # os.listdir(os.path.join(path, process)))
    if nSamples % batchSize == 0:
        nBatches = nSamples // batchSize
    else:
        nBatches = (nSamples // batchSize) + 1
    return nBatches


def logMetrics(epochNum, metrics, process, logFile, saveName):
    '''
    Print metrics to terminal and save to logfile in a proper format.
    '''
    line = ('Epoch num. {epochNum:d} \t {process} Loss : {lossVal:.7f};'
            '{process} Acc : {acc:.3f} ; {process} F1 : {f1:.3f} ; '
            '{process} AUROC : {auroc:.3f} ; {process} AUPRC :'
            '{auprc:.3f}\n').format(epochNum=epochNum, process=process,
                                    lossVal=metrics.Loss, acc=metrics.Acc,
                                    f1=metrics.F1, auroc=metrics.AUROC,
                                    auprc=metrics.AUPRC)
    print(line.strip('\n'))
    if logFile:
        with open(os.path.join('logs', logFile), 'a') as f:
            f.write(line)
    np.savetxt('logs/FprTpr_'+saveName.split('.')[0] + '.csv',
               metrics.fpr_tpr_arr, delimiter=', ')
    np.savetxt('logs/PrecisionRecall_'+saveName.split('.')[0] + '.csv',
               metrics.precision_recall_arr, delimiter=', ')


def loadModel(loadModelFlag, model, saveName):
    '''
    Load saved weights. loadModelFlag: main, chkpt or None.
    Sends abort signal if saved model does not exist.
    '''
    try:
        if loadModelFlag == 'main':
            model.load_state_dict(torch.load(saveName+'.pt'))
        elif loadModelFlag == 'chkpt':
            model.load_state_dict(torch.load('chkpt_'+saveName+'.pt'))
        successFlag = 1
    except FileNotFoundError:
        print('Model does not exist! Aborting...')
        successFlag = 0
    return successFlag


def saveChkpt(bestValRecord, bestVal, metrics, model, saveName):
    '''
    Save checkpoint model
    '''
    diff = metrics.AUROC - bestVal
    bestVal = metrics.AUROC
    with open(os.path.join('logs', bestValRecord), 'w') as statusFile:
        statusFile.write('Best AUROC so far: '+str(bestVal))
    torch.save(model.state_dict(), 'chkpt_'+saveName+'.pt')
    print('Model checkpoint saved since AUROC has improved by '+str(diff))
    return bestVal


def initLogging(saveName):
    '''
    Create files for storing best metric value and logs if not existing
    already. Returns names of the files.
    '''
    bestValRecord = 'bestVal_'+saveName+'.txt'
    logFile = 'log_'+saveName+'.txt'
    if not os.path.exists(os.path.join('logs', bestValRecord)):
        os.system('echo "Best F1 so far: 0.0" > '+os.path.join('logs',
                                                               bestValRecord))
    if not os.path.exists(os.path.join('logs', logFile)):
        os.system('touch '+os.path.join('logs', logFile))
    return bestValRecord, logFile


def getOptions():
    '''
    Set options for argument parser to take hyperparameters.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--saveName", help="Name by which model will be saved"
                        "Names of logging files depend on this",
                        type=str)
    parser.add_argument("--initEpochNum", help="Serial number of starting"
                        "epoch, for display.", type=int, default=1)
    parser.add_argument("--nEpochs", help="Number of epochs", type=int,
                        default=10)
    parser.add_argument("--batchSize", help="Batch Size", type=int, default=12)
    parser.add_argument("-wd", "--weightDecay", help="Weight decay for"
                        "optimizer", type=float, default='1e-5')
    parser.add_argument("-lr", "--learningRate", help="Learning rate",
                        type=float, default='1e-4')
    parser.add_argument("-lwts", "--lossWeights", help="Weights for main"
                        "and auxiliary loss. Pass as a string in format wt1,"
                        "wt2 such that wt1+wt2=1", type=str,
                        default='0.8, 0.2')
    parser.add_argument("-loadflg", "--loadModelFlag", help="Whether and"
                        "which model to load. main, chkpt or None"
                        "(not passed)", type=str)
    return parser


def getClassBalancedWt(beta, samplesPerCls, nClasses=2):
    '''
    As per https://towardsdatascience.com/handling-class-imbalanced-data-
    using-a-loss-specifically-made-for-it-6e58fd65ffab
    '''
    effectiveNum = 1.0 - np.power(beta, samplesPerCls)
    weights = (1.0 - beta) / np.array(effectiveNum)
    weights = weights / np.sum(weights) * nClasses
    return torch.Tensor(weights).cuda()


def weightedBCE(weight, pred, target):
    normVal = 1e-24
    weights = 1 + (weight-1)*target
    loss = -((weights*target)*pred.clamp(min=normVal).log()
             + (1-target)*(1-pred).clamp(min=normVal).log()).sum()
    return loss


def toCategorical(yArr):
    '''
    One Hot encoding for softmax
    '''
    y_OH = torch.FloatTensor(yArr.shape[0], 2)
    y_OH.zero_()
    y_OH.scatter_(1, yArr, 1)
    return y_OH


def AUC(soft_predList, labelList):
    """
    Use the probabilities to get AUROC and AUPRC values.
    """
    # pdb.set_trace()
    if isinstance(soft_predList, list):
        soft_predList = np.concatenate(soft_predList, 0)
    if isinstance(labelList, list):
        labelList = np.concatenate(labelList, 0)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(labelList,
                                                    soft_predList[:, 1],
                                                    pos_label=1)
    auc_roc = sklearn.metrics.auc(fpr, tpr)
    precision, recall,\
        threshold = sklearn.metrics.precision_recall_curve(labelList,
                                                           soft_predList[:, 1],
                                                           pos_label=1)
    auc_prc = sklearn.metrics.auc(recall, precision)
    # save fpr & tpr for plotting
    fpr_tpr_arr = np.array([fpr, tpr])
    precision_recall_arr = np.array([precision, recall])
    return auc_roc, auc_prc, fpr_tpr_arr, precision_recall_arr


def globalAcc(predList, labelList):
    '''
    Compute accuracy based on all predictions and labels at the end of an
    epoch.
    '''
    if not isinstance(predList, torch.Tensor):
        predList = torch.cat(predList)
    if not isinstance(labelList, torch.Tensor):
        labelList = torch.cat(labelList)
    acc = torch.sum(predList == labelList[:, 0]).float()/(predList.shape[0])
    return acc
