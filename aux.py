import numpy as np
import torch
import sklearn.metrics
import argparse
import os
### Auxiliary functions for learner to use
def initLogging(saveName):
    '''
    Create files for storing best metric value and logs if not existing already.
    Returns names of the files.
    '''
    bestValRecord = 'bestVal_'+saveName+'.txt'
    logFile = 'log_'+saveName+'.txt'
    if not os.path.exists(os.path.join('logs',bestValRecord)):
       os.system('echo "Best F1 so far: 0.0" > '+os.path.join('logs',bestValRecord))
    if not os.path.exists(os.path.join('logs',logFile)):
        os.system('touch '+os.path.join('logs',logFile))
    return bestValRecord,logFile

def getOptions():
    '''
    Set options for argument parser to take hyperparameters.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--saveName", help="Name by which model will be saved. Names of logging files depend on this",
                        type=str)
    parser.add_argument("--initEpochNum", help="Serial number of starting epoch, for display.", type=int, default=1)
    parser.add_argument("--nEpochs", help="Number of epochs", type=int, default=10)
    parser.add_argument("--batchSize", help="Batch Size", type=int, default=12)
    parser.add_argument("-wd", "--weightDecay", help="Weight decay for optimizer", type=float, default='1e-5')
    parser.add_argument("-lr", "--learningRate", help="Learning rate", type=float, default='1e-4')
    parser.add_argument("-lwts", "--lossWeights", help="Weights for main and auxiliary loss. Pass as a string in format wt1,wt2 such that wt1+wt2=1", type=str, default='0.8,0.2')
    return parser

def toCategorical(yArr):
# One Hot encoding for softmax
    y_OH = torch.FloatTensor(yArr.shape[0],2)
    y_OH.zero_()
    y_OH.scatter_(1,yArr,1)
    return y_OH

def AUC(soft_predList,labelList):
    """
    Use the probabilities to get AUROC and AUPRC values.
    """
    # pdb.set_trace()
    if isinstance(soft_predList,list):
        soft_predList = np.concatenate(soft_predList,0)
    if isinstance(labelList,list):
        labelList = np.concatenate(labelList,0)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(labelList, soft_predList[:,1], pos_label = 1)
    auc_roc = sklearn.metrics.auc(fpr, tpr)
    precision, recall, threshold = sklearn.metrics.precision_recall_curve(labelList, soft_predList[:,1], pos_label = 1)
    auc_prc = sklearn.metrics.auc(recall,precision)
    #save fpr & tpr for plotting
    np.savetxt('logs/FprTpr_'+saveName.split('.')[0]+ '.csv',np.array([fpr,tpr]), delimiter = ',')
    np.savetxt('logs/PrecisionRecall_'+saveName.split('.')[0]+ '.csv',np.array([precision,recall]), delimiter = ',')
    return auc_roc, auc_prc

def globalAcc(predList,labelList):
    '''
    Compute accuracy based on all predictions and labels at the end of an epoch.
    '''
    if not isinstance(predList,torch.Tensor):
        predList = torch.cat(predList)
    if not isinstance(labelList,torch.Tensor):
        labelList = torch.cat(labelList)
    #pdb.set_trace()
    acc = torch.sum(predList==labelList).float()/( predList.shape[0] )
    return acc
