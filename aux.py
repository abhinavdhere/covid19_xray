''' Supplmentary functions for learner to use '''
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import sklearn.metrics
import argparse
import os


# # --------- Argparse options ------------
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
    parser.add_argument("--gamma", help="Gamma for focal loss", type=float,
                        default=2)
    parser.add_argument("--foldNum", help="Fold number for k fold"
                        "cross-validation", type=int, default='1')
    parser.add_argument("-loadflg", "--loadModelFlag", help="Whether and"
                        "which model to load. main, chkpt or None"
                        "(not passed)", type=str)
    parser.add_argument("--runMode", help="all : trn, val, tst \n trn: train"
                        "only \n val: val only \n tst: test only", type=str,
                        default="all")
    parser.add_argument("--amp", help="Whether mixed precision will be used"
                        ". Valid values are True or False", default='False')
    parser.add_argument("--tta", help="Whether to use test time augmentation",
                        default='False')
    return parser


# #--------- Logging and model loading/saving ---------
def logMetrics(epochNum, metrics, loss_list, process, logFile, task):
    '''
    Print metrics to terminal and save to logfile in a proper format.
    '''
    if task == 'classify':
        line = (
            f'Epoch num. {epochNum} - {process}'
            f' Main_FL : {loss_list["main_focal_loss"]:.6f} ;'
            f' Aux_loss : {loss_list["aux_focal_loss"]:.6f} ;'
            # f' Conicity : {loss_list["conicity"]:.6f} ;'
            f' Acc : {metrics.Acc:.3f} ; F1 : {metrics.F1:.3f} ;'
            f' AUROC : {metrics.AUROC:.3f} ;  AUPRC : {metrics.AUPRC}\n'
        )
    elif task == 'segment':
        line = (
            f'Epoch num. {epochNum} - {process}'
            f' Focal_loss : {loss_list["focal_loss"]:.6f} ;'
            f' Dice_loss : {loss_list["dice"]:.6f} ;'
            f' MSE : {loss_list["mse"]:.6f} ;'
            f' Dice_score : {metrics.Dice:.4f}\n'
        )
    print(line.strip('\n'))
    if logFile:
        with open(os.path.join('logs', logFile), 'a') as f:
            f.write(line)
    # np.savetxt('logs/FprTpr_'+save_name.split('.')[0] + '.csv',
    #            metrics.fpr_tpr_arr, delimiter=', ')
    # np.savetxt('logs/PrecisionRecall_'+save_name.split('.')[0] + '.csv',
    #            metrics.precision_recall_arr, delimiter=', ')


def loadModel(loadModelFlag, model, save_name):
    '''
    Load saved weights. loadModelFlag: main, chkpt or None.
    Sends abort signal if saved model does not exist.
    '''
    try:
        if loadModelFlag == 'main':
            model.load_state_dict(torch.load(os.path.join('savedModels',
                                                          save_name+'.pt')))
        elif loadModelFlag == 'chkpt':
            model.load_state_dict(torch.load('savedModels/chkpt_'
                                             + save_name+'.pt'))
        successFlag = 1
    except FileNotFoundError:
        print('Model does not exist! Aborting...')
        successFlag = 0
    return successFlag


def save_chkpt(best_val_record, best_val, metric_val, metric_name, model,
               savename):
    '''
    Save checkpoint model if performance exceeds previous best.

    Args:
        best_val_record (str): name of text file storing best_val so far
        best_val (int): best performance so far by the model in
                        selected metric as read from best_val_record
                        text file
        metric_val (float): curent value of selected metric
        metric_name: name of metric to compare
    '''
    diff = metric_val - best_val
    best_val = metric_val
    with open(os.path.join('logs', best_val_record), 'w') as statusFile:
        statusFile.write('Best ' + metric_name + ' so far: '+str(best_val))
    torch.save(model.state_dict(), 'savedModels/chkpt_'+savename+'.pt')
    print('Model checkpoint saved since ' + metric_name + ' has improved by '
          + str(diff))
    return best_val


def initLogging(save_name, metric_name):
    '''
    Create files for storing best metric value and logs if not existing
    already. Returns names of the files.
    '''
    bestValRecord = 'bestVal_'+save_name+'.txt'
    logFile = 'log_'+save_name+'.txt'
    if not os.path.exists(os.path.join('logs', bestValRecord)):
        os.system('echo "Best '+metric_name+' so far: 0.0" > '
                  + os.path.join('logs', bestValRecord))
    if not os.path.exists(os.path.join('logs', logFile)):
        os.system('touch '+os.path.join('logs', logFile))
    return bestValRecord, logFile


def log_config(log_file_name, args):
    '''
    Information regarding selected args to be logged in file along
    with time and date of run.
    Args:
        log_file_name (str): name of log file
        args : args obtained from argparser
    '''
    from datetime import datetime
    run_time = datetime.now()
    run_time_str = run_time.strftime('%d-%m-%Y %H:%M:%S')
    config_desc = ''
    for arg_name, arg_val in vars(args).items():
        config_desc += f'{arg_name} : {arg_val} \t'
    with open(os.path.join('logs', log_file_name), 'a') as f:
        f.write(('-'*20)+'\n')
        f.write(f'Run initiated at {run_time_str} \n')
        f.write(config_desc)
        f.write('\n')


def save_predictions(save_name, process, filename_list, softpred_list,
                     pred_list, label_list):
    """
    Save soft & thresholded predictions to a csv for later analysis
    Args:
        save_name (str): name of model
        process (str): 'trn', 'val', or 'tst'
        filename_list (List[str]): names of images
        softpred_list (List[torch.Tensor]): output of softmax from model -
                                            shape(n, 2)
        pred_list (List[torch.Tensor]): hard thresholded predictions -
                                        shape(n,)
        label_list (List[torch.Tensor]): labels - shape(n, 1)
    """
    softpred_list, pred_list, label_list = list(
        map(np.concatenate, [softpred_list, pred_list, label_list], [0]*3))
    pred_list = np.expand_dims(pred_list, -1)
    filename_list = np.expand_dims(np.array(filename_list), -1)
    all_data = np.concatenate((filename_list, softpred_list, pred_list,
                               label_list), 1)
    header = np.array([['Image_name', 'Probability_0', 'Probability_1',
                        'Predictions', 'Labels']])
    all_data = np.concatenate((header, all_data), 0)
    np.savetxt(f'predictions/{save_name.split(".")[0]}_{process}_preds.csv',
               all_data, delimiter=',', fmt='%s')


# #--------- Loss functions ------------
def getClassBalancedWt(beta, samples_per_cls, num_classes=2):
    '''
    As per https://towardsdatascience.com/handling-class-imbalanced-data-
    using-a-loss-specifically-made-for-it-6e58fd65ffab
    '''
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_classes
    return torch.Tensor(weights).cuda()


class FocalLoss(nn.Module):
    """ Simple focal loss implementation """
    def __init__(self, alpha, gamma, reduction):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-12

    def forward(self, pred, label_one_hot):
        pred = pred + self.eps
        focus_weight = torch.pow(torch.tensor(1.) - pred, self.gamma)
# self.gamma.to(pred.dtype))
        loss = -(torch.sum(self.alpha*focus_weight*label_one_hot*pred.log(),
                           dim=1))
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


def weightedBCE(weight, pred, target):
    norm_val = 1e-24
    weights = 1 + (weight-1)*target
    loss = -((weights*target)*pred.clamp(min=norm_val).log()
             + (1-target)*(1-pred).clamp(min=norm_val).log()).sum()
    return loss


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


# #--------- Metrics --------------
def integral_dice(pred, gt, k):
    '''
    Dice coefficient for multiclass hard thresholded prediction consisting
    of integers instead of binary values.
    k = integer for class for which Dice is being calculated.
    '''
    return ((torch.sum(pred[gt == k] == k)*2.0 /
            (torch.sum(pred[pred == k] == k)
             + torch.sum(gt[gt == k] == k)).float()))


def AUC(softpred_list, label_list):
    """
    Use the probabilities to get AUROC and AUPRC values.
    """
    # pdb.set_trace()
    if isinstance(softpred_list, list):
        softpred_list = np.concatenate(softpred_list, 0)
    if isinstance(label_list, list):
        label_list = np.concatenate(label_list, 0)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label_list,
                                                    softpred_list[:, 1],
                                                    pos_label=1)
    auc_roc = sklearn.metrics.auc(fpr, tpr)
    precision, recall,\
        threshold = sklearn.metrics.precision_recall_curve(label_list,
                                                           softpred_list[:, 1],
                                                           pos_label=1)
    auc_prc = sklearn.metrics.auc(recall, precision)
    # save fpr & tpr for plotting
    fpr_tpr_arr = np.array([fpr, tpr])
    precision_recall_arr = np.array([precision, recall])
    return auc_roc, auc_prc, fpr_tpr_arr, precision_recall_arr


def globalAcc(pred_list, label_list):
    '''
    Compute accuracy based on all predictions and labels at the end of an
    epoch.
    '''
    if not isinstance(pred_list, torch.Tensor):
        pred_list = torch.cat(pred_list)
    if not isinstance(label_list, torch.Tensor):
        label_list = torch.cat(label_list)
    acc = torch.sum(pred_list == label_list[:, 0]).float()/(pred_list.shape[0])
    return acc


# #-------- Misc. --------
def toCategorical(yArr, *args):
    '''
    One Hot encoding for softmax
    '''
    if args and args[0] == 'seg':
            y_OH = torch.FloatTensor(yArr.shape[0], 2, yArr.shape[2],
                                     yArr.shape[3])
    else:
        y_OH = torch.FloatTensor(yArr.shape[0], 2)
    y_OH.zero_()
    y_OH.scatter_(1, yArr, 1)
    return y_OH


def BCET(min_out_img, max_out_img, mean_out_img, in_img):
    """
    Obtain and apply BCET function for given input image and target
    output params.
    Translated from MATLAB code at
    (https://www.imageeprocessing.com/2017/11/balance-contrast
    -enhancement-technique.html)
    Args:
       min_out_img (float): min value of target image
       max_out_img (float): max value of target image
       mean_out_img (float): mean value of target image
       in_img (np.array): input image to be transformed
    Returns:
       out_img (np.array): transformed output image
    """
    in_img = in_img.astype('float32')  # INPUT IMAGE
    Lmin = np.min(in_img)  # MINIMUM OF INPUT IMAGE
    Lmax = np.max(in_img)  # MAXIMUM OF INPUT IMAGE
    Lmean = np.mean(in_img)  # MEAN OF INPUT IMAGE
    LMssum = np.mean(in_img**2)  # MEAN SQUARE SUM OF INPUT IMAGE

    bnum = ((Lmax**2)*(mean_out_img - min_out_img)
            - LMssum*(max_out_img - min_out_img)
            + (Lmin**2)*(max_out_img - mean_out_img))
    bden = (2*(Lmax * (mean_out_img - min_out_img)
               - Lmean*(max_out_img - min_out_img)
               + Lmin * (max_out_img - mean_out_img)))

    b = bnum/bden

    a = (max_out_img-min_out_img)/((Lmax-Lmin)*(Lmax+Lmin-2*b))

    c = min_out_img - a * (Lmin-b)**2

    out_img = a * ((in_img-b)**2) + c  # PARABOLIC FUNCTION

    return out_img
