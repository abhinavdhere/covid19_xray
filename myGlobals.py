from collections import namedtuple

# path = '/home/abhinav/cats_and_dogs'    # dataset path
# path = '/home/abhinav/covid19_data_xray/'
# path = '/home/abhinav/CXR_datasets/COVIDx_data'
path = '/home/abhinav/CXR_datasets/RSNA_dataset/my_splits'
Metrics = namedtuple('Metrics', ['Loss', 'Acc', 'F1', 'AUROC', 'AUPRC',
                                 'fpr_tpr_arr', 'precision_recall_arr'])
# dataSetList = ['chxs', 'kg', 'padch']
dataSetList = ['rsna']
# dataSetList = ['actmed', 'fig1', 'sirm', 'rsna']
imgDims = (512, 512)  # (400, 400) (328,360)(485,664)#(328,360)
