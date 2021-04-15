""" Constants to be shared across modules

Attributes:
    PATH (str): disk path for all images.
    PATH_FLIST (str): disk path for file lists folder.
    IMG_DIMS (Tuple[int]): dimensions to which images will be resized.
    DATASET_LIST (List[str]): datasets which will be allowed
    TASK (str): can be either 'normal_vs_pneumonia', 'pneumonia_vs_covid'
        or 'lung_seg'
"""
from collections import namedtuple

# PATH = '/home/abhinav/CXR_datasets/PDCOVIDNet_data/all_images'
# PATH_FLIST = '/home/abhinav/CXR_datasets/PDCOVIDNet_data/file_lists'
PATH = '/home/abhinav/CXR_datasets/RSNA_dataset/all_images'
PATH_FLIST = '/home/abhinav/CXR_datasets/RSNA_dataset/file_lists'
# PATH = '/home/abhinav/CXR_datasets/COVIDx_data/all_data'
# PATH = '/home/abhinav/CXR_datasets/lung_seg_dataset_v7labs/all_data'
# PATH_FLIST = '/home/abhinav/CXR_datasets/lung_seg_dataset_v7labs/file_lists'
# PATH_FLIST = '/home/abhinav/CXR_datasets/COVIDx_data/file_lists'
# PATH_FLIST = '/home/abhinav/CXR_datasets/COVIDx_data/small_set_flists'
# PATH_FLIST = '/home/abhinav/CXR_datasets/COVIDx_data/small_set_5fold_flists'

# PATH = '/home/abhinav/CXR_datasets/bimcv_pos/bimcv_iitj'
# PATH_FLIST = '/home/abhinav/CXR_datasets/bimcv_pos/bimcv_iitj_flists'
Metrics = namedtuple('Metrics', ['Loss', 'Acc', 'F1', 'AUROC', 'AUPRC',
                                 'fpr_tpr_arr', 'precision_recall_arr'])
IMG_DIMS = (512, 512)
# IMG_DIMS = (256, 256)
DATASET_LIST = ['rsna']
# DATASET_LIST = ['sirm', 'cohen', 'kg']
# DATASET_LIST = ['actmed', 'fig1', 'sirm', 'cohen', 'rsna']
# DATASET_LIST = ['bimcv', 'chxs']
TASK = 'normal_vs_pneumonia'
# TASK = 'pneumonia_vs_covid'
# TASK = 'two_stage'

# # Used variables
# crop_params = (210, 215)
# crop_params = (208, 224)
# window = (-79, 304)
# path = '/home/abhinav/CXR_datasets/RSNA_dataset/my_splits'
# path = '/home/abhinav/kits_2d'
# path = '/home/abhinav/covid19_data_xray/'
# path = '/home/abhinav/cats_and_dogs'    # dataset path

# dataset_list = ['rsna']
# dataSetList = ['chxs', 'kg', 'padch']
# dataset_list = ['kits']
# dataSetList = ['catdog']
