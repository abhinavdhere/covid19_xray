from collections import namedtuple

path = '/home/abhinavdhere/covid19_data_xray'    # dataset path
Metrics = namedtuple('Metrics',['Loss','Acc','F1','AUROC','AUPRC','fpr_tpr_arr','precision_recall_arr'])
dataSetList = ['kg']
imgDims = (485,664) #328,360
