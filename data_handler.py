""" All functions for data handling """
import os

import numpy as np
import torch
import cv2
# import pydicom as dcm

import config
from augment_tools import augment


class DataLoader:
    def __init__(self, data_type, fold_num, batch_size, aug_setup,
                 in_channels=0, undersample=False, sample_size=None,
                 aug_names=['normal']):
        """
        Args:
            data_type (str): type of data to be loaded ('trn', 'val' or 'tst')
            fold_num (int): split number for k fold validation
            batch_size (int): batch size
            aug_setup (str): fashion in which augmentations to be applied.
                Allowed values are
                'random_class0_all_class1' - random augmentations in class0
                and all augmentations in class1
                'random' - random augmentations for all classes
                'all' - all augmentations for all classes
            in_channels (int): number of input channels. 3 for RGB and 0
                for grayscale (U-Net requires RGB)
            undersample (bool): whether to use undersampling for class0
                Defaults to False
            sample_size (int, optional) - samples to be selected for
                undersampling. Defaults to None. Compulsory if undersample
                is True.
            aug_names (List[str]): list of augmentations to be applied.
                Defaults to ['normal'] i.e. no augmentations.
        """
        self.path = config.PATH
        self.data_type = data_type
        self.in_channels = in_channels
        self.fold_num = fold_num
        self.batch_size = batch_size

        self._undersample = undersample
        self._sample_size = sample_size
        self._aug_names = aug_names
        self._aug_setup = aug_setup
        self._file_list = self.get_file_list()

        self.num_batches = self.get_num_batches()
        self.datagen = self.dataloader()

    @property
    def aug_list(self):
        return self.set_augmentations()

    def get_file_list(self):
        """Generate list of files as per data_type, dataset and task.

        Returns:
            file_list (List[str]): original list of file names from directory
        """
        # allFileList = os.listdir(os.path.join(path, data_type))
        if self.data_type == 'val':
            flist_name = (self.path.rsplit('/', 1)[0]+'/file_lists/'
                          '5fold_split_val.txt')
        # 'val.txt')
        else:
            flist_name = (self.path.rsplit('/', 1)[0]+'/file_lists/'
                          '5fold_split_' + str(self.fold_num)
                          + '_' + self.data_type + '.txt')
        # + self.data_type + '.txt')
        all_filelist = np.loadtxt(flist_name, delimiter='\n', dtype=str)
        file_list = []
        for file_name in all_filelist:
            if file_name.split('_')[0] in config.DATASET_LIST:
                lbl = file_name.split('_')[1]
                if config.TASK == 'pneumonia_vs_covid' and lbl == '0':
                    continue
                else:
                    file_list.append(file_name)
        return file_list

    def set_random_aug(self, name):
        '''Set random augmentation flag while maintaining 50% chance of having
        normal (i.e. no aug) and 50% of any other augmentation.

        Args:
            name (str): name of image
        Returns:
            name_w_code (str): name with augmentation code appended
        '''
        augment_flag = np.random.choice([0, 1])
        if augment_flag:
            aug_name = np.random.choice(self._tmp_aug_names)
        else:
            aug_name = 'normal'
        name_w_code = name+'_'+aug_name
        return name_w_code

    def set_augmentations(self):
        """Set appropriate augmentations by appending codes to file names.

        Returns:
            aug_list (List[str]): list of file names with augmentation code
                appended.
        """
        aug_list = []
        if self.data_type == 'trn':
            if self._aug_setup == 'random':
                self._tmp_aug_names = self._aug_names.copy()
                self._tmp_aug_names.remove('normal')
                for name in self._file_list:
                    name_w_code = self.set_random_aug(name)
                    aug_list.append(name_w_code)
            elif self._aug_setup == 'all':
                for name in self._file_list:
                    aug_list += [name + '_' + aug_name for aug_name in
                                 self._aug_names]
            elif self._aug_setup == 'random_class0_all_class1':
                aug_list_classA = []
                aug_list_classB = []
                self._tmp_aug_names = self._aug_names.copy()
                self._tmp_aug_names.remove('normal')
                for name in self._file_list:
                    if int(name.split('_')[1]) == 2:
                        aug_list_classA += [name+'_'+aug_name for aug_name in
                                            self._aug_names]
                    else:
                        name_w_code = self.set_random_aug(name)
                        aug_list_classB.append(name_w_code)
                if self._undersample:
                    if not self._sample_size:
                        raise ValueError('Sample size not passed for'
                                         'undersampling')
                    aug_list_classB = np.random.choice(aug_list_classB,
                                                       (self._sample_size,),
                                                       replace=False)
                    aug_list = aug_list_classA + aug_list_classB.tolist()
                else:
                    aug_list = aug_list_classA + aug_list_classB
            aug_list = np.random.permutation(aug_list)
        else:
            aug_list += [name+'_normal' for name in self._file_list]
        return aug_list

    def get_num_batches(self):
        """ Compute number of batches based on batch_size. """
        num_samples = len(self.aug_list)
        if num_samples % self.batch_size == 0:
            num_batches = num_samples // self.batch_size
        else:
            num_batches = (num_samples // self.batch_size) + 1
        return num_batches

    def preprocess_data(self, full_name, aug_name, segment_lung):
        """ Load images and do preprocessing as required

        Args:
            full_name (str): name of image with path and without
                augmentation code
            aug_name (str): augmentation code (see self.set_augmentations)
            segment_lung (bool): whether to apply lung segmentation

        Returns:
            img (torch.Tensor): CUDA tensor of size (in_channels, size0, size1)
                with required preprocessing.
        """
        img = cv2.imread(full_name, cv2.IMREAD_ANYDEPTH)
        # img = dcm.dcmread(full_name)
        # img = img.pixel_array
        if self.in_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.permute(2, 0, 1)
        img = cv2.resize(img, (config.IMG_DIMS[0], config.IMG_DIMS[1]),
                         cv2.INTER_AREA)
        img = (img - np.mean(img)) / np.std(img)
        if segment_lung:
            img = self.apply_seg_mask(img, full_name.split('/')[-1],
                                      crop=True)
        img = torch.Tensor(img).cuda()
        if self.in_channels == 0:
            img = img.unsqueeze(0)
        img = augment(img, aug_name)
        return img

    def apply_seg_mask(self, img, file_name, crop):
        """ Load lung segmentation mask from disk and mask the image with it.

        Args:
            img (torch.Tensor): Image tensor of size (in_channels,
                size0, size1)
            file_name (str): name of image file without path
            crop (bool): whether to crop to segmented region

        Returns:
            img (torch.Tensor): CUDA Image tensor with lungs region only
        """
        lung_mask = cv2.imread(config.PATH.rsplit('/', 1)[0]
                               + '/lungSeg/'+file_name, cv2.IMREAD_GRAYSCALE)
        lung_mask[lung_mask == 255] = 1
        img = img*lung_mask
        if crop:
            min_row, max_row = np.where(np.any(lung_mask, 0))[0][[0, -1]]
            min_col, max_col = np.where(np.any(lung_mask, 1))[0][[0, -1]]
            img = img[min_col:max_col, min_row:max_row]
            img = cv2.resize(img, (352, 384), cv2.INTER_AREA)
        return img

    def dataloader(self):
        """ Generator for yielding batches of data & corresponding labels

        Args:
            batch_size (int): size of each batch

        Yields:
            data_arr (torch.Tensor): CUDA tensor of shape (batch_size,
                in_channels, size0, size1)
            label_arr (torch.Tensor): tensor of shape(batch_size, 1)
            file_name_arr (List[str]): list of images names for images
                in the batch
        """
        while True:
            aug_list = self.aug_list
            # print(len(aug_list))
            count, batch_count, data_arr, label_arr,\
                file_name_arr = 0, 0, [], [], []
            for file_name_full in aug_list:
                file_name = '_'.join(file_name_full.split('_')[:-1])
                aug_name = file_name_full.split('_')[-1]
                nameParts = file_name.split('_')
                lbl = int(nameParts[1])
                if config.TASK == 'pneumonia_vs_covid':
                    lbl -= 1
                elif config.TASK == 'normal_vs_pneumonia' and lbl > 1:
                    lbl = 1
                name_w_path = os.path.join(config.PATH, file_name)
                img = self.preprocess_data(name_w_path, aug_name,
                                           segment_lung=False)
                # pdb.set_trace()
                if torch.std(img) == 0 or not torch.isfinite(img).all():
                    raise ValueError('Image intensity inappropriate'
                                     '(std is 0 or image has infinity')
                lbl = torch.Tensor(np.array([lbl])).long()
                data_arr.append(img)
                label_arr.append(lbl)
                file_name_arr.append(file_name_full)
                count += 1
                last_batch_flag = ((self.num_batches-batch_count) == 2 and
                                   count == (len(aug_list) % self.batch_size))
                if (count == self.batch_size) or last_batch_flag:
                    yield torch.stack(data_arr),  torch.stack(label_arr),\
                            file_name_arr
                    batch_count += 1
                    count, data_arr, label_arr, file_name_arr = 0, [], [], []
