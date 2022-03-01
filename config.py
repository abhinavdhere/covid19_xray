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


PATH = '/home/abhinav/CXR_datasets/bimcv_pos/bimcv_iitj'
PATH_FLIST = '/home/abhinav/CXR_datasets/bimcv_pos/5fold_flists'

Metrics = namedtuple('Metrics', ['Loss', 'Acc', 'F1', 'AUROC', 'AUPRC',
                                 'fpr_tpr_arr', 'precision_recall_arr'])
IMG_DIMS = (512, 512)

DATASET_LIST = ['bimcv', 'chxs']
# TASK = 'normal_vs_pneumonia'
# TASK = 'pneumonia_vs_covid'
TASK = 'two_stage'

