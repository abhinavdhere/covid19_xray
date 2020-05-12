### Primary module. Includes dataloader, trn/val/test functions. Reads
### options from user and runs training.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
from PIL import Image

import pdb
def dataLoader(fPath,dataType,batchSize):
    fList = os.listdir(os.path.join(fPath,dataType))
    while True:
        fListShuffled = np.random.permutation(fList)
        dataArr = []
        labelArr = []
        count = 0
        for fName in fListShuffled:
            img = Image.open(os.path.join(fPath,dataType,fName))
            img = img.resize((360,328),Image.BICUBIC)
            nameParts = fName.split('_')
            lbl = int(nameParts[1])
            img = torch.Tensor(np.array(img)).cuda()
            lbl = torch.Tensor(np.array([lbl])).cuda()
            dataArr.append(img)
            labelArr.append(lbl)
            count+=1
            if count==batchSize:
                yield torch.stack(dataArr).unsqueeze(1), torch.stack(labelArr)
                count = 0 ; dataArr = [] ; labelArr = []

def runModel(dataLoader,model,optimizer,process,lossWts=(0.6,0.4)):
    '''
    process : 'trn', 'val' or 'tst'
    '''
    with trange(nBatches,desc='Epoch '+str(epoch+1),ncols=100) as t:
        for m in range(nBatches):
            X,y = dataLoader.__next__()
            if process=='trn':
                optimizer.zero_grad()
                model.train()
                pred, auxPred = model.forward(X)
                pred = F.softmax(pred,1)
                auxPred = F.softmax(auxPred,1)
                loss = lossWts[0]*lossFun(pred,y) + lossWts[1]*lossFun(auxPred,y)
                loss.backward()
                optimizer.step()
            elif process=='val' or process=='tst':
                model.eval()
                pred = F.softmax(model.forward(X),1)
                loss = lossFun(pred,y)
            t.set_postfix(loss=runningLoss/(float(m+1)*batchSize)) #for tqdm thingy
            t.update()

path = '/home/abhinav/covid19_data_xray'
## Hyperparameters
nEpochs = 10
batchSize = 12
lrVal = 1e-3
wtDec = 1e-5
lossWts = (0.8,0.2)
## Inits
trnDataLoader = dataLoader(path,'trn',batchSize)
valDataLoader = dataLoader(path,'val',batchSize)
tstDataLoader = dataLoader(path,'tst',batchSize)
model = torchvision.models.inception_v3(pretrained=False).cuda()
lossFun = nn.BCELoss(reduction='sum')
optimizer = torch.optim.Adam(lr=lrVal, weight_decay=wtDec)
## Learning
for epochNum in range(initEpochNum,nEpochs+1):
    runModel(trnDataLoader,model,optimizer,lossFun,'trn',lossWts)

