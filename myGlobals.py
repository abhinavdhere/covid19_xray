from collections import namedtuple

path = '/scratch/amd/cats_and_dogs'    # dataset path
Metrics = namedtuple('Metrics',['Loss','Acc','F1','AUROC','AUPRC','fpr_tpr_arr','precision_recall_arr'])
dataSetList = ['catdog']#['chxs','kg','padch']
imgDims = (400, 400)#(328,360)(485,664)#(328,360)
