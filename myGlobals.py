from collections import namedtuple

path = '/home/abhinav/covid19_data_xray'    # dataset path
Metrics = namedtuple('Metrics',['Loss','Acc','F1','AUROC','AUPRC','fpr_tpr_arr','precision_recall_arr'])
dataSetList = ['nih']
imgDims = (512,512)#(485,664) #328,360
