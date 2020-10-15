from collections import namedtuple

path = '/home/abhinav/kits_2d'
Metrics = namedtuple('Metrics', ['Loss', 'Acc', 'F1', 'AUROC', 'AUPRC',
                                 'fpr_tpr_arr', 'precision_recall_arr'])
dataset_list = ['kits']
imgDims = (512, 512)

window = (-79, 200)
crop_params = (210, 215)

# # Used variables
# window = (-79, 304)
# path = '/home/abhinav/covid19_data_xray/'
# path = '/home/abhinav/CXR_datasets/COVIDx_data'
# path = '/home/abhinav/CXR_datasets/RSNA_dataset/my_splits'
# path = '/home/abhinav/cats_and_dogs'    # dataset path

# dataSetList = ['chxs', 'kg', 'padch']
# dataSetList = ['rsna']
# dataSetList = ['catdog']
# dataSetList = ['actmed', 'fig1', 'sirm', 'rsna']
