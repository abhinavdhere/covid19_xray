"""
Two experiments for quantitative testing of GradCams from my models -
1. Computing IOU for normal vs pnemomnia with RSNA GT
2. Computing portion of attr. in vertical portions of lung
(for covid vs pneumonia, to quantify geographical diversity of opacity)
"""
import os
import csv
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

import explain


def load_BB(fName):
    name_orig = fName[0].split('_')[-2].split('.')[0]
    bb_data = pd.read_csv('/home/abhinav/CXR_datasets/RSNA_dataset/'
                          'stage_2_train_labels.csv').values
    img_bb_all = bb_data[bb_data[:, 0] == name_orig, 1:-1]
    return img_bb_all


def compute_iou(bb_pred, bb_gt):
    """
    Follows https://stackoverflow.com/a/57247833/4670262
    """
    polygon_pred = Polygon(bb_pred)
    polygon_gt = Polygon(bb_gt)
    iou = (polygon_pred.intersection(polygon_gt).area /
           polygon_pred.union(polygon_gt).area)
    return iou


def measure_attr_iou(attr, bb_gt_list, lung_mask):
    attr_bin = attr.copy()
    attr_bin[attr_bin != 0] = 1
    contours, _ = cv2.findContours(attr_bin.astype('uint8'), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                      reverse=True)[:3]
    attr_bb_list = []
    for cnt in contours:
        _, bb = LungSections.fit_lung_bb(cnt)
        # bb = cv2.boundingRect(cnt)
        attr_bb_list.append(bb)

    iou_list = []
    for bb_gt in bb_gt_list:
        _iou_list = []
        cv2.drawContours(lung_mask, [bb_gt], 0, (0, 255, 0), 2)
        for bb in attr_bb_list:
            cv2.drawContours(lung_mask, [bb], 0, (255, 0, 0), 2)
            iou_val = compute_iou(bb, bb_gt)
            _iou_list.append(iou_val)
        if len(_iou_list) == 0:
            iou_list.append(0)
        else:
            iou_list.append(np.max(_iou_list))
    return iou_list, lung_mask


def measure_attr_areas_bb(attr, bb_gt_list, lung_mask):
    total_attr_area = np.sum(attr > 0)
    epsilon = 1e-15
    area_list = []
    for bb in bb_gt_list:
        bb_mask, section_area \
            = get_lung_section_mask(lung_mask.copy(), bb)
        overlap = attr*bb_mask
        overlap_area = np.sum(overlap)
        area_ratio = overlap_area / (total_attr_area + epsilon)
        area_list.append(area_ratio*100)
    return area_list


def compare_attr_iou(attr_path, lung_mask_path):
    # with open('attr_area_marl_individual_BB_wLungSeg.csv', 'a') as f:
    with open('attr_area_marl_noConicity.csv', 'a') as f:
        writer = csv.writer(f)
        for fname in os.listdir(attr_path):
            attr = np.load(os.path.join(attr_path, fname))
            thresh = 0.8*np.max(attr)
            attr[attr < thresh] = 0
            attr[attr > thresh] = 1
            mask_name = 'rsna_'+'_'.join(fname.split('_')[1:]).split('.')[
                0]+'.dcm.npy'
            lung_mask = np.load(os.path.join(lung_mask_path,
                                             mask_name))
                                             # fname.rsplit('.', 1)[0]
            contours = explain.load_BB_with_lung_seg(fname, lung_mask, None,
                                                     draw=False)
            min_row, max_row = np.where(np.any(lung_mask, 0))[0][[0, -1]]
            min_col, max_col = np.where(np.any(lung_mask, 1))[0][[0, -1]]
            lung_mask = lung_mask[min_col:max_col, min_row:max_row]
            lung_mask = cv2.resize(lung_mask, (352, 384), cv2.INTER_AREA)
            attr = attr[min_col:max_col, min_row:max_row]
            attr = cv2.resize(attr, (352, 384), cv2.INTER_AREA)
            # attr = attr[:, :, 0]*lung_mask
            attr = attr*lung_mask
            bb_gt_list = []
            for cnt in contours:
                _, bb_gt = LungSections.fit_lung_bb(cnt)
                bb_gt_list.append(bb_gt)
            lung_mask_rgb = np.stack((lung_mask*255, lung_mask*255,
                                      lung_mask*255), -1)
            iou_list, overlayed_mask = measure_attr_iou(attr, bb_gt_list,
                                                        lung_mask_rgb)
            # area_list = measure_attr_areas_bb(attr, bb_gt_list, lung_mask)
            # cv2.imwrite('attr_iou_plots/covidnet_jun21' +
            #             fname.rsplit('.', 1)[0]+'.png', overlayed_mask)
            # for val in iou_list:
            #     writer.writerow([fname, val])
            writer.writerow([fname]+[np.mean(iou_list)])
            # writer.writerow([fname]+[np.mean(area_list)])


def get_lung_section_mask(lung_mask, bb_section):
    bb_mask = np.zeros((384, 352))
    # bb_mask = np.zeros((512, 512))
    cv2.drawContours(bb_mask, [bb_section], 0, [255, 255, 255], -1)
    bb_mask[bb_mask == 255] = 1
    lung_section_mask = lung_mask*bb_mask
    cnt, hiers = cv2.findContours(lung_section_mask.astype('uint8'),
                                  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cnt = sorted(cnt, key=lambda x: cv2.contourArea(x),
                 reverse=True)[0]
    area = cv2.contourArea(cnt)
    return lung_section_mask, area


def measure_attr_areas(attr, bb_gt_list, lung_mask):
    contours, _ = cv2.findContours(attr.astype('uint8'), cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                      reverse=True)  # [:3]
    total_cnt_area = sum([cv2.contourArea(cnt) for cnt in contours])
    area_list = []
    # num = 0
    for bb_section in bb_gt_list:
        _area_list = []
        lung_section_mask, section_area \
            = get_lung_section_mask(lung_mask.copy(), bb_section)
        # cv2.imwrite('temp_lung_'+str(num)+'.png', lung_section_mask*255)
        # num += 1
        for cnt in contours:
            # cnt_mask = np.zeros((384, 352))
            cnt_mask = np.zeros((512, 512))
            cv2.drawContours(cnt_mask, [cnt], 0, [255, 255, 255], -1)
            cnt_mask[cnt_mask == 255] = 1
            overlap = cnt_mask*lung_section_mask
            overlap_cnt, _ = cv2.findContours(
                overlap.astype('uint8'), cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE)
            if len(overlap_cnt) == 0:
                overlap_area = 0
            else:
                overlap_area = cv2.contourArea(overlap_cnt[0])
            # area_ratio = overlap_area / section_area
            area_ratio = overlap_area / total_cnt_area
            _area_list.append(area_ratio*100)
        area_list.append(np.sum(_area_list))
    return area_list


def measure_geo_dist(path, lung_mask_path, class_name, measure):
    path = os.path.join(path)#, class_name)
    with open('geo_dist_attr_cntr_area_lungwise_'+class_name+'.csv', 'a') as f:
        writer = csv.writer(f)
        for item in os.listdir(path):
            if (os.path.isfile(os.path.join(path, item))):
                attr = np.load(os.path.join(path, item))
                thresh = 0.8*np.max(attr)
                attr[attr < thresh] = 0
                attr[attr > thresh] = 1
                if class_name == 'covid':
                    lung_mask_name = (
                        '_'.join(item.split('_')[6:]).rsplit(
                            '.', 1)[0]+'.png.npy')
                elif class_name == 'pneumonia':
                    lung_mask_name = (
                        '_'.join(item.split('_')[6:]).rsplit(
                            '.', 1)[0]+'.jpg.npy')
                else:
                    lung_mask_name = 'rsna_'+'_'.join(
                        item.split('_')[1:]).split('.')[
                        0]+'.dcm.npy'
                try:
                    lung_mask = np.load(os.path.join(lung_mask_path,
                                                     lung_mask_name))
                except FileNotFoundError:
                    continue
                    # print(item)
                min_row, max_row = np.where(np.any(lung_mask, 0))[0][[0, -1]]
                min_col, max_col = np.where(np.any(lung_mask, 1))[0][[0, -1]]
                lung_mask = lung_mask[min_col:max_col, min_row:max_row]
                lung_mask = cv2.resize(lung_mask, (352, 384), cv2.INTER_AREA)
                try:
                    lung_sections = LungSections(lung_mask)
                except IndexError:
                    print(item)
                    continue
                if measure == 'iou':
                    lung_iou_list = []
                    lung_mask_rgb = np.stack(
                        (lung_mask*255, lung_mask*255, lung_mask*255), -1)
                    for i in range(2):
                        sections = lung_sections.divide_sections(i)
                        iou_list, plot = measure_attr_iou(attr, sections,
                                                          lung_mask_rgb)
                        lung_iou_list.append(iou_list)
                        cv2.imwrite(
                            'geo_dist_attr_plots/'+item.rsplit('.', 1)[0]
                            + '.png', plot
                        )
                    writer.writerow(lung_iou_list[0]+lung_iou_list[1])
                elif measure == 'area':
                    lung_area_list = []
                    for i in range(2):
                        sections = lung_sections.divide_sections(i)
                        # single_lung_mask = np.zeros((384, 352))
                        # if i == 0:
                        #     lung_box = lung_sections.lung_box1
                        # else:
                        #     lung_box = lung_sections.lung_box2
                        # single_lung_mask = cv2.drawContours(
                        #     single_lung_mask, [lung_box], 0, 255, -2)
                        # single_lung_mask[single_lung_mask == 255] = 1
                        # attr_lung = attr*single_lung_mask*lung_mask
                        attr_lung = attr*lung_mask
                        try:
                            # area_list = measure_attr_areas(attr, sections,
                            #                                lung_mask)
                            area_list = measure_attr_areas_bb(
                                attr_lung, sections, lung_mask)
                            lung_area_list.append(area_list)
                        except ZeroDivisionError:
                            print(item)
                            lung_area_list.append([0, 0, 0])
                    record = [
                        lung_area_list[0][0] + lung_area_list[1][0],
                        lung_area_list[0][1] + lung_area_list[1][1],
                        lung_area_list[0][2] + lung_area_list[1][2]
                    ]
                    # writer.writerow(record)
                    writer.writerow(lung_area_list[0]+lung_area_list[1])


class LungSections:
    """
    Fit a rotated rectangle on lung contour and divide the rectangle in
    three horizontal sections.
    """
    def __init__(self, lung_mask):
        # Extract contours and get two largest contours (for two lungs)
        contours, _ = cv2.findContours(lung_mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours = sorted(contours, key=lambda x: cv2.contourArea(x),
                          reverse=True)
        # import pdb
        # pdb.set_trace()
        lungs = contours[:2]
        # Fit rotated boxes for both lungs
        rect1, lung_box1 = self.fit_lung_bb(lungs[0])
        rect2, lung_box2 = self.fit_lung_bb(lungs[1])
        # Sorting to get left lung first
        lungs, rects = zip(*sorted(zip([lung_box1, lung_box2], [rect1, rect2]),
                           key=lambda x: x[1][0][0], reverse=False))
        self.lung_box1 = lungs[0]
        self.lung_box2 = lungs[1]

    @staticmethod
    def fit_lung_bb(cnt):
        """ Fit a rotated rectangle over contour """
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return rect, box

    @staticmethod
    def find_first_point_position(box):
        """
        Find whether first point is left bottom or right bottom
        See https://stackoverflow.com/a/58185432/4670262
        """
        # Find second bottom most point
        min_y = 512
        min_idx = 0
        for idx in range(1, box.shape[0]):
            if box[idx, 1] < min_y:
                min_y = box[idx, 1]
                min_idx = idx
        # Check position of first point relative to second bottom most point
        if box[min_idx, 0] > box[0, 0]:
            return "left"
        else:
            return "right"

    @staticmethod
    def divide_line(start_pt, end_pt, a, b):
        """ Divide a line into 3 parts given start and end point """
        # mid_x = int(np.mean([start_pt[0], end_pt[0]]))
        # mid_y = int(np.mean([start_pt[1], end_pt[1]]))
        left_quarter_x = (a*end_pt[0] + b*start_pt[0]) / (a + b)
        left_quarter_x = int(np.round(left_quarter_x))
        left_quarter_y = (a*end_pt[1] + b*start_pt[1]) / (a + b)
        left_quarter_y = int(np.round(left_quarter_y))
        # left_quarter_x = int(np.mean([start_pt[0], mid_x]))
        # left_quarter_y = int(np.mean([start_pt[1], mid_y]))
        a, b = b, a
        right_quarter_x = (a*end_pt[0] + b*start_pt[0]) / (a + b)
        right_quarter_x = int(np.round(right_quarter_x))
        right_quarter_y = (a*end_pt[1] + b*start_pt[1]) / (a + b)
        right_quarter_y = int(np.round(right_quarter_y))
        # right_quarter_x = int(np.mean([mid_x, end_pt[0]]))
        # right_quarter_y = int(np.mean([mid_y, end_pt[1]]))
        return ((left_quarter_x, left_quarter_y),
                (right_quarter_x, right_quarter_y))

    def get_section_boxes(self, bottom_pts, top_pts):
        """ Obtain boxes for lung sections using given points """
        bottom_left_quarter, bottom_right_quarter \
            = self.divide_line(bottom_pts[0], bottom_pts[1], 1, 3)
        top_left_quarter, top_right_quarter \
            = self.divide_line(top_pts[0], top_pts[1], 1, 3)
        left_section = np.array((bottom_pts[0], top_pts[0],
                                 top_left_quarter, bottom_left_quarter))
        mid_section = np.array((bottom_left_quarter, top_left_quarter,
                                top_right_quarter, bottom_right_quarter))
        right_section = np.array((bottom_right_quarter, top_right_quarter,
                                  top_pts[1], bottom_pts[1]))
        return (left_section, mid_section, right_section)

    def divide_sections(self, box_idx):
        """ Interface method for dividing sections in lung """
        if box_idx == 0:
            box = self.lung_box1
        elif box_idx == 1:
            box = self.lung_box2
        else:
            raise ValueError('Invalid box index specified')
        first_pt_pos = self.find_first_point_position(box)
        if first_pt_pos == 'left':
            bottom_pts = (box[0], box[3])
            top_pts = (box[1], box[2])
        else:
            bottom_pts = (box[0], box[1])
            # top_pts = (box[2], box[3])
            top_pts = (box[3], box[2])
        sections = self.get_section_boxes(bottom_pts, top_pts)
        return sections


if __name__ == '__main__':
    # attr_path = ('/home/abhinav/covid19_xray/gradcam_misc/bimcv_stage2/'
    #              'raw_unthresh_rerun1/covid')
    # lung_mask_path = '/home/abhinav/CXR_datasets/bimcv_pos/lung_seg_raw/'
    attr_path = ('gradcam_misc/rsna_jun21/marl_raw_normal')
    # attr_path = ('gradcam_misc/rsna_march_21/guided_thresholded/'
    #              'raw_thresh80/resnet1')
    lung_mask_path = '/home/abhinav/CXR_datasets/RSNA_dataset/lung_seg_raw'
    # compare_attr_iou(attr_path, lung_mask_path)
    measure_geo_dist(attr_path, lung_mask_path, 'normal', 'area')
    # fname = 'bimcv_2_sub-S03047_ses-E07985_run-1_bp-chest_vp-ap_cr.png.npy'
    # fname = 'bimcv_2_sub-S03072_ses-E06166_run-1_bp-chest_vp-ap_cr.png.npy'
    # fname = 'bimcv_2_sub-S03215_ses-E06438_run-1_bp-chest_vp-ap_cr.png.npy'
    # # attr = np.load('gradcam_misc/bimcv_stage2/raw_unthresh_rerun1/covid/bimcv_stage2_wSeg_FL_pairAug_attn_bimcv_2_sub-S03072_ses-E06166_run-1_bp-chest_vp-ap_cr.npy')
    # attr = np.load('gradcam_misc/bimcv_stage2/raw_unthresh_rerun1/covid/bimcv_stage2_wSeg_FL_pairAug_attn_bimcv_2_sub-S03215_ses-E06438_run-1_bp-chest_vp-ap_cr.png')
    # thresh = 0.8*np.max(attr)
    # attr[attr > thresh] = 1
    # attr[attr < thresh] = 0
    # lung_mask = np.load(lung_mask_path+fname)
    # min_row, max_row = np.where(np.any(lung_mask, 0))[0][[0, -1]]
    # min_col, max_col = np.where(np.any(lung_mask, 1))[0][[0, -1]]
    # lung_mask = lung_mask[min_col:max_col, min_row:max_row]
    # lung_mask = cv2.resize(lung_mask, (352, 384), cv2.INTER_AREA)
    # attr = attr*lung_mask
    # lung_sections = LungSections(lung_mask)
    # sections = lung_sections.divide_sections(0)
    # lung_mask *= 255
    # lung_mask = np.stack((lung_mask, lung_mask, lung_mask), -1)
    # # # cv2.drawContours(lung_mask, [lung_sections.lung_box2], 0, (255, 0, 0), 2)
    # cv2.drawContours(lung_mask, list(sections), 0, (0, 140, 255), 2)
    # cv2.drawContours(lung_mask, list(sections), 2, (0, 140, 255), 2)
    # cv2.drawContours(lung_mask, list(sections), 1, (212, 0, 145), 2)
    # # # cv2.drawContours(lung_mask, list(sections), 1, (0, 0, 0), cv2.FILLED)
    # # cv2.drawContours(lung_mask, list(sections), 0, (0, 0, 255), 2)
    # # # cv2.drawContours(lung_mask, list(sections), 1, (0, 0, 0), cv2.FILLED)
    # sections = lung_sections.divide_sections(1)
    # # # cv2.drawContours(lung_mask, [lung_sections.lung_box1], 0, (255, 0, 0), 2)
    # cv2.drawContours(lung_mask, list(sections), 0, (0, 140, 255), 2)
    # cv2.drawContours(lung_mask, list(sections), 2, (0, 140, 255), 2)
    # cv2.drawContours(lung_mask, list(sections), 1, (212, 0, 145), 2)
    # # # cv2.drawContours(lung_mask, list(sections), 1, (0, 0, 0), cv2.FILLED)
    # import matplotlib.pyplot as plt
    # plt.imshow(lung_mask)
    # plt.imshow(attr, cmap='gray', alpha=0.3)
    # plt.colorbar()
    # plt.axis('off')
    # plt.savefig('lung_sections_overlay_attr.png')
    # # cv2.imwrite('lung_mask_sections_final.png', lung_mask)
