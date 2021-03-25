"""
Two experiments for quantitative testing of GradCams from my models -
1. Computing IOU for normal vs pnemomnia with RSNA GT
2. Computing portion of attr. in vertical portions of lung
(for covid vs pneumonia, to quantify geographical diversity of opacity)
"""
import os
import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


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


def compare_gradcam(path, fname):
    attr = np.load(os.path.join(path, fname))
    bb_gt_list = load_BB(fname)
    _, contours, _ = cv2.findContours(attr, cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_SIMPLE)

    attr_bb_list = []
    for cnt in contours:
        bb = cv2.boundingRect(cnt)
        # attr_bb_list(bb)

    iou_list = []
    for bb in attr_bb_list:
        _iou_list = []
        for bb_gt in bb_gt_list:
            iou_val = compute_iou(bb, bb_gt)
            _iou_list.append(iou_val)
        iou_list.append(max(_iou_list))
    return np.mean(iou_list)


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
        lungs = contours[:2]
        # Fit rotated boxes for both lungs
        self.lung_box1 = self.fit_lung_bb(lungs[0])
        self.lung_box2 = self.fit_lung_bb(lungs[1])

    @staticmethod
    def fit_lung_bb(cnt):
        """ Fit a rotated rectangle over contour """
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box

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
    def divide_line(start_pt, end_pt):
        """ Divide a line into 3 parts given start and end point """
        mid_x = int(np.mean([start_pt[0], end_pt[0]]))
        mid_y = int(np.mean([start_pt[1], end_pt[1]]))
        left_quarter_x = int(np.mean([start_pt[0], mid_x]))
        left_quarter_y = int(np.mean([start_pt[1], mid_y]))
        right_quarter_x = int(np.mean([mid_x, end_pt[0]]))
        right_quarter_y = int(np.mean([mid_y, end_pt[1]]))
        return ((left_quarter_x, left_quarter_y),
                (right_quarter_x, right_quarter_y))

    def get_section_boxes(self, bottom_pts, top_pts):
        """ Obtain boxes for lung sections using given points """
        bottom_left_quarter, bottom_right_quarter \
            = self.divide_line(bottom_pts[0], bottom_pts[1])
        top_left_quarter, top_right_quarter \
            = self.divide_line(top_pts[0], top_pts[1])
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
    fname = 'bimcv_2_sub-S03047_ses-E07985_run-1_bp-chest_vp-ap_cr.png.npy'
    lung_mask = np.load(fname)
    lung_sections = LungSections(lung_mask)
    sections = lung_sections.divide_sections(1)
    lung_mask *= 255
    lung_mask = np.stack((lung_mask, lung_mask, lung_mask), -1)
    cv2.drawContours(lung_mask, [lung_sections.lung_box2], 0, (255, 0, 0), 2)
    cv2.drawContours(lung_mask, list(sections), 0, (0, 0, 255), 2)
    cv2.drawContours(lung_mask, list(sections), 2, (0, 255, 0), 2)
    sections = lung_sections.divide_sections(0)
    cv2.drawContours(lung_mask, [lung_sections.lung_box1], 0, (255, 0, 0), 2)
    cv2.drawContours(lung_mask, list(sections), 0, (0, 0, 255), 2)
    cv2.drawContours(lung_mask, list(sections), 2, (0, 255, 0), 2)
    cv2.imwrite('lung_mask_sections_test1.png', lung_mask)
