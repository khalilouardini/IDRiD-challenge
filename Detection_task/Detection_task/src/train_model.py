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
import torch


def train_one_epoch_FasterRCNN(model, optimizer, data_loader, device, epoch, print_freq):
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
    return history['loss'].numpy()

def train_one_epoch_RetinaNet(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    #model.module.freeze_bn()
    epoch_loss = []

    print("\n------------------------Training---------------------------------------\n")
    for iter_num, (img,target,factor) in enumerate(data_loader):
        annotations = utils.get_annotations_retinanet(target[0])
        img = img[0].unsqueeze(0)
        optimizer.zero_grad()
        classification_loss, regression_loss = model([ img.to(device), annotations.to(device)])
        try:

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            if bool(loss == 0):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            epoch_loss.append(float(loss))                                
            if iter_num % print_freq == 0 :
                print(
                        'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                                epoch, iter_num, float(classification_loss), float(regression_loss), np.mean(epoch_loss)))
            del classification_loss
            del regression_loss
        except Exception as e:
            print(e)
            continue
    return epoch_loss

def evaluate(model, dataset, device, model_type):
    model.eval()
    dist_OD = 0
    dist_Fovea = 0
    gt_boxes_OD, pred_boxes_OD = np.empty(4), np.empty(4)
    gt_boxes_Fovea, pred_boxes_Fovea = np.empty(4), np.empty(4)
    print("\n------------------------Validation---------------------------------------\n")
    for idx in tqdm(range(dataset.__len__())):
        img, target, factor = dataset[idx]
        
        _, OD_true_box, Fovea_true_box, OD_predicted_box, Fovea_predicted_box = utils.get_boxes(model, dataset, threshold = 0.008, img_idx = idx, model_type= model_type)
        gt_boxes_OD = np.vstack((gt_boxes_OD, OD_true_box ))
        gt_boxes_Fovea = np.vstack((gt_boxes_Fovea, Fovea_true_box ))
        pred_boxes_OD = np.vstack((pred_boxes_OD, OD_predicted_box ))
        pred_boxes_Fovea = np.vstack((pred_boxes_Fovea, Fovea_predicted_box ))
        
        dist_OD +=  utils.get_center_distance(OD_true_box, OD_predicted_box,factor)
        dist_Fovea +=   utils.get_center_distance(Fovea_true_box, Fovea_predicted_box, factor)
    
    average_iou_OD = np.mean([IoU(gt_boxes_OD[i], pred_boxes_OD[i]) for i in range(len(gt_boxes_OD))])
    print("\nAverage IoU for OD detection over validation set: {:.3f}".format(average_iou_OD))
    average_iou_Fovea = np.mean([IoU(gt_boxes_Fovea[i], pred_boxes_Fovea[i]) for i in range(len(gt_boxes_Fovea))])
    print("Average IoU for Fovea detection over validation set: {:.3f}".format(average_iou_Fovea))
    mean_dist_OD = dist_OD/dataset.__len__()
    print('Mean distance between OD centers over validation set: {:.3f}'.format(mean_dist_OD))   
    mean_dist_Fov = dist_Fovea/dataset.__len__()
    print('Mean distance between Fovea centers over validation set: {:.3f} \n'.format(mean_dist_Fov))
    
    return (average_iou_OD + average_iou_Fovea) /2
  
