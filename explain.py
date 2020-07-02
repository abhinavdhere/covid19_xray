import captum.attr
#import captum.attr.visualization as viz
from captum.attr import visualization as viz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
from aux import loadModel,get_nBatches
from learner import dataLoader
from myGlobals import path
import matplotlib.pyplot as plt
import pdb

#pdb.set_trace()

val_nBatches = get_nBatches(path,'val',1)
valDataLoader = dataLoader(path,'val',1,val_nBatches)
model = torchvision.models.inception_v3(pretrained=True,progress=True, num_classes=2, aux_logits=True, init_weights=False).cuda()
model = nn.DataParallel(model)
successFlag = loadModel('chkpt',model,'nih_inceptionv3')
print(successFlag)
model.eval()
for i in range(val_nBatches):
    X,y,fName = valDataLoader.__next__()
    with torch.no_grad():
        pred = F.softmax(model.forward(X),1)
    if y.item()==1 and torch.argmax(pred).item()==1:
        pdb.set_trace()
        gcObj = captum.attr.LayerGradCam(model.forward,model.module.Mixed_7c)
        attr = gcObj.attribute(X,1)
        attrRescaled = Image.fromarray(attr.detach().cpu().numpy()[0,0,:,:]).resize((X.shape[3],X.shape[2]))
    # viz.visualize_image_attr(np.array(attrRescaled),X[0,0,:,:].detach().cpu().numpy(),method='blended_heat_map')
        pltObj = viz.visualize_image_attr(np.expand_dims(attrRescaled,-1), X.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy(), method="blended_heat_map",cmap='jet', sign="absolute_value",show_colorbar=True, title="Overlayed Attributions")
        pltObj[0].savefig('gradCam'+fName.split('.')[0]+'.png')
        # plt.imshow(X.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy().astype('uint8'))
        # plt.show()
        # plt.imshow(np.array(attrRescaled),cmap='jet')
        # plt.show()
