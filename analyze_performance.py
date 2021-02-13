""" Reads from saved predictions and provides detailed analysis """
import argparse
from tabulate import tabulate
import numpy as np
import sklearn
from sklearn.metrics import classification_report


class PredAnalyzer:
    def __init__(self, filename):
        all_data = np.loadtxt(filename, delimiter=',', dtype=str)
        self.softpred_list = all_data[1:, 1:3].astype('float32')
        self.pred_list = all_data[1:, 3].astype('uint8')
        self.label_list = all_data[1:, 4].astype('uint8')
        self.name_list = all_data[1:, 0]

    def optimize_threshold(self, measure):
        if measure == 'AUROC':
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(
                self.label_list, self.softpred_list[:, 1], pos_label=1
            )
            optimal_idx = np.argmax(tpr - fpr)
        elif measure == 'AUPRC':
            precision, recall, thresholds = \
                sklearn.metrics.precision_recall_curve(
                    self.label_list, self.softpred_list[:, 1], pos_label=1
                )
            fscore = (2 * precision * recall) / (precision + recall)
            optimal_idx = np.argmax(fscore)
        optimal_threshold = thresholds[optimal_idx]
        print("Threshold value is:", optimal_threshold)
        return optimal_threshold

    def get_analysis(self, target_names, tta=False, thresh_optim=None):
        if thresh_optim:
            threshold = self.optimize_threshold(thresh_optim)
            pred_list = (self.softpred_list[:, 1] > threshold).astype('uint8')
        else:
            pred_list = self.pred_list
        print('Confusion matrix is:')
        conf_mat = sklearn.metrics.confusion_matrix(self.label_list.tolist(),
                                                    pred_list.tolist())
        print(tabulate([[target_names[0], conf_mat[0, 0], conf_mat[0, 1]],
                        [target_names[1], conf_mat[1, 0], conf_mat[1, 1]]],
                       headers=[target_names[0], target_names[1]]))
        print('Metrics are:')
        report = classification_report(self.label_list.tolist(),
                                       pred_list.tolist(),
                                       target_names=target_names)
        print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", help="Name by which model is saved",
                        type=str)
    parser.add_argument("--data_category", help="trn, val or tst",
                        type=str)
    parser.add_argument("--test_time_aug", "-tta", help="Whether test-time"
                        "augmentation is needed")
    parser.add_argument("--optimize_threshold", "-thresh", help="Method for "
                        "optimizing threshold : using AUROC or AUPRC")
    parser.add_argument("--classes", help="comma separated names of classes")
    args = parser.parse_args()
    tta = (args.test_time_aug == 'True')
    target_names = args.classes.split(',')
    if tta:
        filename = (f'predictions/{args.modelname}_tta_'
                    f'{args.data_category}_preds.csv')
    else:
        filename = (f'predictions/{args.modelname}_'
                    f'{args.data_category}_preds.csv')
    print(f'{args.data_category} data for the model {args.modelname}')
    print(f'Test time augmentation is {tta} and threshold '
          f'optimization is {args.optimize_threshold}.')
    print(f'Classes are {target_names}')
    analyzer = PredAnalyzer(filename)
    analyzer.get_analysis(target_names, tta, args.optimize_threshold)
