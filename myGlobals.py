from collections import namedtuple

# path = '/home/abhinav/cats_and_dogs'    # dataset path
path = '/home/abhinav/covid19_data_xray/'
Metrics = namedtuple('Metrics', ['Loss', 'Acc', 'F1', 'AUROC', 'AUPRC',
                                 'fpr_tpr_arr', 'precision_recall_arr'])
# dataSetList = ['chxs', 'kg', 'padch']
dataSetList = ['chxs']
imgDims = (328, 360)  # (400, 400) (328,360)(485,664)#(328,360)
