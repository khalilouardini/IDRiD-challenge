# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 13:54:47 2020

@author: 33787
"""
import matplotlib.pyplot as plt
from src.utils import get_center
import numpy as np


def plot_random_batch(dataset):
    """Plots 4 images selected randomly from dataset
    with the OD and Fovea boxes """
    idx = np.random.randint(0, dataset.__len__() , 4) 
    ax = []
    f = plt.figure(figsize=(14,14))
    for n,i in enumerate(idx,0):
        img, target,_ = dataset[i]
        ax.append(f.add_subplot(1,4,n+1))
        plt.title('Sample ' + str(i+1))
        show_image(img, target["boxes"][0], target["boxes"][1], ax=ax[n], box = True)
        
        
def show_image(image, OD, Fovea, ax, box = False):
    """Show image with OD and Fovea"""
    ax.imshow(image.permute(1, 2, 0))
    _OD = get_center(OD)
    _Fovea = get_center(Fovea)

    ax.scatter(_OD[0], _OD[1], c ='#1f77b4', marker = 'o', label = 'OD')
    ax.scatter(_Fovea[0], _Fovea[1], c ='#2ca02c', marker = 'o', label = 'Fovea' )
    
    if box :
        _x1,_y1,_x2,_y2 = OD
        x1,y1,x2,y2 = Fovea

        ax.plot([_x1, _x1, _x2, _x2, _x1], [_y1, _y2, _y2, _y1, _y1], c ='#1f77b4',linewidth=2)
        ax.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1],c ='#2ca02c',linewidth=2)

    ax.legend()
    #plt.show()
    
    
def plot_prediction(img, OD_true_box, Fovea_true_box, OD_predicted_box, Fovea_predicted_box, idx):
    
    # Get bounding box coordinates
    xx1,yy1,xx2,yy2 = OD_true_box
    OD_true_center = get_center(OD_true_box)
    _xx1,_yy1,_xx2,_yy2 = Fovea_true_box
    Fovea_true_center = get_center(Fovea_true_box)
        
    # Retrieve predicted bounding boxes and scores 
    _x1,_y1,_x2,_y2 = OD_predicted_box
    OD_predicted_center = get_center(OD_predicted_box)
    x1,y1,x2,y2 = Fovea_predicted_box
    Fovea_predicted_center = get_center(Fovea_predicted_box)
        

        
    plt.figure(figsize=(10,10))
    plt.imshow(img.mul(255).permute(1, 2, 0).byte().numpy(), cmap="gray")
    plt.plot([xx1, xx1, xx2, xx2, xx1], [yy1, yy2, yy2, yy1, yy1], 'c-', label = "ground truth OD")
    plt.plot([_x1, _x1, _x2, _x2, _x1], [_y1, _y2, _y2, _y1, _y1], 'r-', label = "predicted OD")
    plt.plot([_xx1, _xx1, _xx2, _xx2, _xx1], [_yy1, _yy2, _yy2, _yy1, _yy1], 'b-', label = "ground truth Fovea")
    plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], 'w-', label = "predicted Fovea")
    plt.scatter(OD_true_center[0], OD_true_center[1], c ='c', marker = 'o')
    plt.scatter(Fovea_true_center[0], Fovea_true_center[1], c ='b', marker = 'o')
    plt.scatter(OD_predicted_center[0], OD_predicted_center[1], c ='r', marker = 'o')
    plt.scatter(Fovea_predicted_center[0], Fovea_predicted_center[1], c ='w', marker = 'o')
    
    
    plt.legend()
    plt.savefig('./figures/{}.png'.format(idx))
    
    plt.axis('off')
    plt.show()