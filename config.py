from collections import namedtuple

path = '/home/abhinav/CXR_datasets/COVIDx_data'
Metrics = namedtuple('Metrics', ['Loss', 'Acc', 'F1', 'AUROC', 'AUPRC',
                                 'fpr_tpr_arr', 'precision_recall_arr'])
imgDims = (512, 512)
dataSetList = ['actmed', 'fig1', 'sirm', 'rsna']

window = (-79, 200)
# crop_params = (210, 215)
crop_params = (208, 224)

# # Used variables
# window = (-79, 304)
# path = '/home/abhinav/CXR_datasets/RSNA_dataset/my_splits'
# path = '/home/abhinav/kits_2d'
# path = '/home/abhinav/covid19_data_xray/'
# path = '/home/abhinav/cats_and_dogs'    # dataset path

# dataset_list = ['rsna']
# dataSetList = ['chxs', 'kg', 'padch']
# dataset_list = ['kits']
# dataSetList = ['catdog']
