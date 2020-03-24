# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 22:22:39 2020

@author: 33787
"""
import math
import sys
from src import utils
import numpy as np
from tqdm import tqdm
from src.metrics import IoU


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print("\n------------------------Training---------------------------------------\n")
    for i, values in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images, targets,_ = values
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Feed the training samples to the model and compute the losses
        loss_dict = model(images, targets)
        
        losses = sum(loss for loss in loss_dict.values())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Pytorch function to initialize optimizer
        optimizer.zero_grad()
        # Compute gradients or the backpropagation
        losses.backward()
        # Update current gradient
        optimizer.step()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Record losses to plot learning curves
        if i == 0: 
            history = {key: val.cpu().detach() for key, val in loss_dict_reduced.items()}
            history['loss'] = losses_reduced.cpu().detach()
        else:
            for key, val in loss_dict_reduced.items():history[key] += val.cpu().detach()
            history['loss'] += losses_reduced.cpu().detach()
    return history


def evaluate(model, dataset, device):
    model.eval()
    dist_OD = 0
    dist_Fovea = 0
    gt_boxes, pred_boxes = np.empty(4), np.empty(4)
    print("\n------------------------Validation---------------------------------------\n")
    for idx in tqdm(range(dataset.__len__())):
        img, target, factor = dataset[idx]
        
        _, OD_true_box, Fovea_true_box, OD_predicted_box, Fovea_predicted_box = utils.get_boxes(model, dataset, threshold = 0.008, img_idx = idx)
        gt_boxes = np.vstack((gt_boxes, OD_true_box ))
        gt_boxes = np.vstack((gt_boxes, Fovea_true_box ))
        pred_boxes = np.vstack((pred_boxes, OD_predicted_box ))
        pred_boxes = np.vstack((pred_boxes, Fovea_predicted_box ))
        
        dist_OD += ( 1/factor[0] )  * utils.get_center_distance(OD_true_box, OD_predicted_box)
        dist_Fovea +=  ( 1/factor[1] )  * utils.get_center_distance(Fovea_true_box, Fovea_predicted_box)
    
    average_iou = np.mean([IoU(gt_boxes[i], pred_boxes[i]) for i in range(len(gt_boxes))])
    print("\n Average IoU over validation set: {:.2f}".format(average_iou))
    mean_dist_OD = dist_OD/dataset.__len__()
    print('Mean distance between OD centers over validation set: {:.3f}'.format(mean_dist_OD))   
    mean_dist_Fov = dist_Fovea/dataset.__len__()
    print('Mean distance between Fovea centers over validation set: {:.3f} \n'.format(mean_dist_Fov))
    
    return average_iou
  
