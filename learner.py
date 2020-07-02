### Primary module. Includes dataloader, trn/val/test functions. Reads
### options from user and runs training.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm,trange

import os

import numpy as np
import sklearn.metrics
from PIL import Image
import cv2
# from skimage import io
# from skimage import transform
# from skimage import color
from aux import *
from aux import weightedBCE as lossFun
from myGlobals import *
from augmentTools import korniaAffine, augment_gaussian_noise

import pdb
def augment(im,augType):
    if augType=='normal':
        im = im
    elif augType=='rotated':
        rotAng = np.random.choice([-10,10])
        im = korniaAffine(im,rotAng,'rotate')
    elif augType=='gaussNoise':
        im = augment_gaussian_noise(im,(0,0.5))
    elif augType=='mirror':
        im = torch.flip(im, [-1])
    return im

def dataLoader(fPath,dataType,batchSize,nBatches):
    fList = getFList(fPath,dataType)
    if dataType=='trn':
        augNames = ['normal','rotated','gaussNoise','mirror']
    else:
        augNames = ['normal']
    augList = []
    for augName in augNames:
        augList+=[name+'_'+augName for name in fList]
    while True:
        augList = np.random.permutation(augList)
        dataArr = []
        labelArr = []
        fNameArr = []
        count = 0
        batchCount = 0
        defective = []
        for fName_full in augList:
            fName = '_'.join(fName_full.split('_')[:-1])
            augName = fName_full.split('_')[-1]
            try:
                img = cv2.imread(os.path.join(fPath,dataType,fName),cv2.IMREAD_ANYDEPTH)
                # img = io.imread(os.path.join(fPath,dataType,fName))
                # img = Image.open(os.path.join(fPath,dataType,fName)).convert('RGB')
            except OSError:
                print(fName)
                continue
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img,(imgDims[0],imgDims[1]),cv2.INTER_AREA)
            # img = color.gray2rgb(img)
            # img = transform.resize(img,(imgDims[0],imgDims[1]),preserve_range=True)
            # img = img.resize((imgDims[1],imgDims[0]),Image.BICUBIC)
            # img = np.array(img)
            nameParts = fName.split('_')
            lbl = int(nameParts[1])
            img = torch.Tensor(img).cuda()
            img = img.permute(2,0,1)
            img = augment(img,augName)
            img = ((img - torch.mean(img))/torch.std(img))
            if torch.std(img)==0 or not torch.isfinite(img).all():
                pdb.set_trace()
            lbl = torch.Tensor(np.array([lbl])).long()
            dataArr.append(img)
            labelArr.append(lbl)
            fNameArr.append(fName_full)
            count+=1
            if count==batchSize or ((nBatches-batchCount)==2 and count==(len(fList)%batchSize)):
                yield torch.stack(dataArr), torch.stack(labelArr), fNameArr
                batchCount += 1
                count = 0 ; dataArr = [] ; labelArr = []; fNameArr = []

def runModel(dataLoader,model,optimizer,classWts,process,batchSize,nBatches,lossWts):
    '''
    process : 'trn', 'val' or 'tst'
    '''
    runningLoss = 0
    predList = []
    labelList = []
    softPredList = []
    with trange(nBatches,desc=process,ncols=100) as t:
        for m in range(nBatches):
            X,y,fName = dataLoader.__next__()
            yOH = toCategorical(y).cuda()
            if process=='trn':
                optimizer.zero_grad()
                model.train()
                pred, auxPred = model.forward(X)
                pred = F.softmax(pred,1)
                auxPred = F.softmax(auxPred,1)
                loss = 0
                for i in range(2):
                    loss += lossWts[0]*lossFun(classWts[i],pred[i],yOH[i]) + lossWts[1]*lossFun(classWts[i],auxPred[i],yOH[i])
                loss.backward()
                optimizer.step()
            elif process=='val' or process=='tst':
                model.eval()
                with torch.no_grad():
                    pred = F.softmax(model.forward(X),1)
                    loss = lossFun(classWts[0],pred[0],yOH[0]) + lossFun(classWts[1],pred[1],yOH[1])
            runningLoss += loss
            hardPred = torch.argmax(pred,1)
            predList.append(hardPred.cpu())
            softPredList.append(pred.detach().cpu())
            labelList.append(y.cpu())
            t.set_postfix(loss=runningLoss.item()/(float(m+1)*batchSize)) #for tqdm thingy
            t.update()
        finalLoss = runningLoss/(float(m+1)*batchSize)
        acc = globalAcc(predList,labelList)
        f1 = sklearn.metrics.f1_score(torch.cat(labelList), torch.cat(predList), labels=None) #needs list as np array
        auroc, auprc, fpr_tpr_arr, precision_recall_arr = AUC(softPredList, labelList)
        metrics = Metrics(finalLoss,acc,f1,auroc,auprc,fpr_tpr_arr,precision_recall_arr)
        # toSave = torch.cat(torch.cat(predList),torch.cat(labelList),axis=1)
        # np.savetxt('tst_combinedData_wNorm_bce_preds.csv',toSave.numpy(),delimiter=',')
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
    trn_nBatches = get_nBatches(path,'trn',args.batchSize,4)
    trnDataLoader = dataLoader(path,'trn',args.batchSize,trn_nBatches)
    val_nBatches = get_nBatches(path,'val',args.batchSize,4)
    valDataLoader = dataLoader(path,'val',args.batchSize,val_nBatches)
    tst_nBatches = get_nBatches(path,'tst',args.batchSize,4)
    tstDataLoader = dataLoader(path,'tst',args.batchSize,tst_nBatches)
    model = torchvision.models.inception_v3(pretrained=False,progress=True, num_classes=2, aux_logits=True, init_weights=True).cuda()
    model = nn.DataParallel(model)
    if args.loadModelFlag:
        successFlag = loadModel(args.loadModelFlag,model,args.saveName)
        if successFlag==0:
            return 0
        elif successFlag==1:
            print("Model loaded successfully")
    # lossFun = nn.BCELoss(reduction='sum')
    classWts = getClassBalancedWt(0.9999,[1431,1431])#[15608,19917])
    optimizer = torch.optim.Adam(model.parameters(),lr=args.learningRate, weight_decay=args.weightDecay)
    ## Learning
    for epochNum in range(args.initEpochNum,args.initEpochNum+args.nEpochs):
        trnMetrics = runModel(trnDataLoader,model,optimizer,classWts,'trn',args.batchSize,trn_nBatches,lossWts=lossWts)
        logMetrics(epochNum,trnMetrics,'trn',logFile,args.saveName)
        torch.save(model.state_dict(),args.saveName+'.pt')
        valMetrics = runModel(valDataLoader,model,optimizer,classWts,'val',args.batchSize,val_nBatches,None)
        logMetrics(epochNum,valMetrics,'val',logFile,args.saveName)
        if bestValRecord and valMetrics.F1>bestVal:
            bestVal = saveChkpt(bestValRecord,bestVal,valMetrics,model,args.saveName)
    tstMetrics = runModel(tstDataLoader,model,optimizer,classWts,'tst',args.batchSize,tst_nBatches,None)
    logMetrics(args.initEpochNum,tstMetrics,'tst',logFile,args.saveName)
    ## for cross dataset testing
    # tstMetrics = runModel(trnDataLoader,model,optimizer,classWts,'tst',args.batchSize,trn_nBatches,None)
    # logMetrics(0,tstMetrics,'tst',logFile,args.saveName)

if __name__=='__main__':
    main()
