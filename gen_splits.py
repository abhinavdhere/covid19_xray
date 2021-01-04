""" Generate random splits """
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split


def get_options():
    """ Return argparser with options specified here """
    parser = argparse.ArgumentParser()
    parser.add_argument("split_ratios", help=('Ratio or number of samples for'
                        'required splits'), type=str, default=0.8)
    parser.add_argument("split_type", help='List of splits, ex. trn,val,tst',
                        default='trn,val,tst', type=str)
    parser.add_argument("flist_path", help='Path where file lists stored',
                        required=True, type=str)
    parser.add_argument("image_path", help='Path where images stored',
                        required=True, type=str)
    parser.add_argument("random_state", help='leave blank for None',
                        default=None, type=int)
    return parser


def process_split(split_list, split_labels, split_type):
    split_arr = np.expand_dims(np.array(split_list), -1)
    class_dist = [np.sum(split_labels == lbl_id) for lbl_id in range(3)]
    print(split_type+'-Class {class0:s}: {dist0:d} \n Class {class1:s}:'
          '{dist1:d}Class {class2:s}: {dist2:d}'.format(
              class0=LABEL_NAMES[0], dist0=class_dist[0],
              class1=LABEL_NAMES[1], dist1=class_dist[1],
              class2=LABEL_NAMES[2], dist2=class_dist[2]))
    np.savetxt(os.path.join(args.flist_path, split_type), split_arr,
               delimiter='\n', fmt='%s')


if __name__ == '__main__':
    parser = get_options()
    args = parser.parse_args()
    splits = args.split_type.split(',')
    random_state = args.random_state
    LABEL_NAMES = {0: 'Normal', 1: 'Pneumonia', 2: 'COVID'}
    available_files = os.listdir(args.image_path)
    available_labels = [int(fname.split('_')[1]) for fname in available_files]
    for idx, split_type in enumerate(splits[:-1]):
        split_list, available_files,\
             split_labels, available_labels = train_test_split(
                available_files,
                train_ratio=args.split_ratios[idx],
                stratify=available_labels,
                random_state=random_state)
        process_split(split_list, split_labels, split_type)
    process_split(available_files, available_labels, splits[-1])
