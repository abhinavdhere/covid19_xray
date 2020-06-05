import captum.attr
import torch
import torch.nn as nn
import torchvision
from aux import loadModel,get_nBatches
from learner import dataLoader
from myGlobals import path

import pdb

pdb.set_trace()

val_nBatches = get_nBatches(path,'val',1)
valDataLoader = dataLoader(path,'val',1,val_nBatches)
model = torchvision.models.inception_v3(pretrained=True,progress=True, num_classes=2, aux_logits=True, init_weights=False).cuda()
model = nn.DataParallel(model)
successFlag = loadModel('chkpt',model,'kaggle_inceptionv3')
X,y,fName = valDataLoader.__next__()
gcObj = captum.attr.LayerGradCam(model.forward,model.module.Mixed_7c)
attr = gcObj.attribute(X,1)

