### Primary module. Includes dataloader, trn/val/test functions. Reads
### options from user and runs training.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm,trange

import os
from collections import namedtuple

import numpy as np
import sklearn.metrics
from PIL import Image

from aux import *

import pdb
def dataLoader(fPath,dataType,batchSize,nBatches):
    fList = os.listdir(os.path.join(fPath,dataType))
    while True:
        fListShuffled = np.random.permutation(fList)
        dataArr = []
        labelArr = []
        count = 0
        batchCount = 0
        for fName in fListShuffled:
            img = Image.open(os.path.join(fPath,dataType,fName)).convert('RGB')
            img = img.resize((360,328),Image.BICUBIC)
            nameParts = fName.split('_')
            lbl = int(nameParts[1])
            img = torch.Tensor(np.array(img)).cuda()
            img = img.permute(2,0,1)
            lbl = torch.Tensor(np.array([lbl])).long()
            dataArr.append(img)
            labelArr.append(lbl)
            count+=1
            if count==batchSize or ((nBatches-batchCount)==2 and count==(len(fList)%batchSize)):
                yield torch.stack(dataArr), torch.stack(labelArr)
                batchCount += 1
                count = 0 ; dataArr = [] ; labelArr = []

def runModel(dataLoader,model,optimizer,lossFun,process,batchSize,nBatches,lossWts):
    '''
    process : 'trn', 'val' or 'tst'
    '''
    runningLoss = 0
    predList = []
    labelList = []
    softPredList = []
    with trange(nBatches,desc=process,ncols=100) as t:
        for m in range(nBatches):
            X,y = dataLoader.__next__()
            yOH = toCategorical(y).cuda()
            if process=='trn':
                optimizer.zero_grad()
                model.train()
                pred, auxPred = model.forward(X)
                pred = F.softmax(pred,1)
                auxPred = F.softmax(auxPred,1)
                loss = 0
                for i in range(2):
                    loss += lossWts[0]*lossFun(pred[i],yOH[i]) + lossWts[1]*lossFun(auxPred[i],yOH[i])
                loss.backward()
                optimizer.step()
            elif process=='val' or process=='tst':
                model.eval()
                with torch.no_grad():
                    pred = F.softmax(model.forward(X),1)
                    loss = lossFun(pred[0],yOH[0]) + lossFun(pred[1],yOH[1])
            runningLoss += loss
            hardPred = torch.argmax(pred,1)
            predList.append(hardPred.cpu())
            softPredList.append(pred.cpu())
            labelList.append(y.cpu())
            t.set_postfix(loss=runningLoss.item()/(float(m+1)*batchSize)) #for tqdm thingy
            t.update()
        finalLoss = runningLoss/(float(m+1)*batchSize)
        acc = globalAcc(predList,labelList)
        f1 = sklearn.metrics.f1_score(torch.cat(labelList), torch.cat(predList), labels=None) #needs list as np array
        auroc, auprc, fpr_tpr_arr, precision_recall_arr = AUC(softPredList, labelList)
        metrics = Metrics(finalLoss,acc,f1,auroc,auprc,fpr_tpr_arr,precision_recall_arr)
        return metrics

def main():
    ## Take options and hyperparameters from user
    parser = getOptions()
    args = parser.parse_args()
    if args.saveName==None:
        print("Warning! Savename unspecified. No logging will take place. Model will not be saved.")
        bestValRecord = None ; logFile = None
    else:
        bestValRecord, logFile = initLogging(args.saveName)
        with open(os.path.join('logs',bestValRecord),'r') as statusFile:
            bestVal = float(statusFile.readline().strip('\n').split()[-1])
    lossWts = tuple(map(float,args.lossWeights.split(',')))
    ## Inits
    trn_nBatches = get_nBatches(path,'trn',args.batchSize)
    trnDataLoader = dataLoader(path,'trn',args.batchSize,trn_nBatches)
    val_nBatches = get_nBatches(path,'val',args.batchSize)
    valDataLoader = dataLoader(path,'val',args.batchSize,val_nBatches)
    tst_nBatches = get_nBatches(path,'tst',args.batchSize)
    tstDataLoader = dataLoader(path,'tst',args.batchSize,tst_nBatches)
    model = torchvision.models.inception_v3(pretrained=False,progress=True, num_classes=2, aux_logits=True, init_weights=True).cuda()
    model = nn.DataParallel(model)
    lossFun = nn.BCELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learningRate, weight_decay=args.weightDecay)
    ## Learning
    for epochNum in range(args.initEpochNum,args.nEpochs+1):
        trnMetrics = runModel(trnDataLoader,model,optimizer,lossFun,'trn',args.batchSize,lossWts=lossWts)
        logMetrics(epochNum,trnMetrics,'trn',logFile)
        torch.save(model.state_dict(),saveName+'.pt')
        valMetrics = runModel(valDataLoader,model,optimizer,lossFun,'val',args.batchSize,val_nBatches,None)
        logMetrics(epochNum,valMetrics,'val',logFile,args.saveName)
        if bestValRecord and valMetrics.F1>bestVal:
            bestVal = saveChkpt(bestValRecord,bestVal,valMetrics,model,args.saveName)
    tstMetrics = runModel(tstDataLoader,model,optimizer,lossFun,'tst',args.batchSize,tst_nBatches,None)
    logMetrics(epochNum,tstMetrics,'tst',logFile)

if __name__=='__main__':
    path = '/home/abhinav/covid19_data_xray'    # dataset path
    Metrics = namedtuple('Metrics',['Loss','Acc','F1','AUROC','AUPRC','fpr_tpr_arr','precision_recall_arr'])
    main()
