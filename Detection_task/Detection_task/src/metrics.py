# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 01:44:49 2020

@author: 33787
"""

import numpy as np
from src import utils 



    
def evaluate_mean_distance(model, dataset):
    """Calculates the average euclidean distance for OD or Fovea"""
    dist_OD = 0
    dist_Fovea = 0
    for i in range(dataset.__len__()):
        #§ get ground truth and predicted boxes
        _, OD_true_box, Fovea_true_box, OD_predicted_box, Fovea_predicted_box = utils.get_boxes(model, dataset, threshold = 0.008, img_idx = i)
        # compute euclidean distance
        dist_OD += utils.get_center_distance(OD_true_box, OD_predicted_box)
        dist_Fovea += utils.get_center_distance(Fovea_true_box, Fovea_predicted_box)
        
    return dist_OD/dataset.__len__() , dist_Fovea/dataset.__len__()


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou
    
    
def non_max_suppression(boxes, scores, threshold):
    """Performs non-maximum suppression and returns indices of kept boxes.
    boxes: [N, (y1, x1, y2, x2)]. Notice that (y2, x2) lays outside the box.
    scores: 1-D array of box scores.
    threshold: Float. IoU threshold to use for filtering.
    """
    assert boxes.shape[0] > 0
    if boxes.dtype.kind != "f":
        boxes = boxes.astype(np.float32)
    
    """ FILL HERE """
    # Compute box areas
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = (y2 - y1) * (x2 - x1)
    
    # Get indicies of boxes sorted by scores (highest first)
    ixs = scores.argsort()[::-1]
    
    pick = []
    while len(ixs) > 0:
        # Pick top box and add its index to the list
        i = ixs[0]
        pick.append(i)
        # Compute IoU of the picked box with the rest
        iou = compute_iou(boxes[i], boxes[ixs[1:]], area[i], area[ixs[1:]])
        # Identify boxes with IoU over the threshold. This
        # returns indices into ixs[1:], so add 1 to get
        # indices into ixs.
        remove_ixs = np.where(iou > threshold)[0] + 1
        # Remove indices of the picked and overlapped boxes.
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)


def accuracy(gt_labels, pred_labels):
    return len(np.where(gt_labels == pred_labels)[0])/len(gt_labels)

def f1_score(gt_labels, pred_labels):
    tp = len(gt_labels == pred_labels)
    fp = len(np.where(pred_labels[np.where(gt_labels == 2)] == 1)[0])
    fn = len(np.where(pred_labels[np.where(gt_labels == 1)] == 2)[0])
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('Precision: {:.3f}'.format(precision))
    print('Recall: {:.3f}'.format(recall))
    return 2 * precision * recall / (precision + recall)


def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA1, yA1, xA2, yA2 = boxA 
    xB1, yB1, xB2, yB2 = boxB
    x1 = max(xA1, xB1)
    y1 = max(yA1, yB1)
    x2 = min(xA2, xB2)
    y2 = min(yA2, yB2)
    # compute the area of intersection rectangle
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (xA2 - xA1 + 1) * (yA2 - yA1 + 1)
    boxBArea = (xB2 - xB1 + 1) * (yB2 - yB1 + 1)
    union = boxAArea + boxBArea - intersection
    return intersection / union