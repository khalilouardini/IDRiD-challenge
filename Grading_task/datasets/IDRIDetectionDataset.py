# -*- coding: utf-8 -*-
"""
Created on Thur Feb 27 16:47:27 2020

@author: Ramzi Charradi
"""
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class IDRID_Detection_Dataset(Dataset):
    """Detection of OD ans Fovea dataset."""

    def __init__(self, csv_od, csv_fovea, root_dir, transform=None, box_width_OD = (120,120), box_width_Fovea = (120,120), image_size = (800,800)):
        
        """
        Parameters
        ----------
        csv_od :
              Path to csv containing optical disc coordinates.
        csv_fovea : 
              Path to csv containing Fovea coordinates.
        root_dir : 
              Path of data folder containing train and test images in separate folders.
        transform : 
              transformations to be applied to the image (expect resizing).
        box_width :
              widh of the boxes to be detected
        image_size:
              all images will be resized to this
        """
        
        self.od = pd.read_csv(csv_od)
        self.fovea = pd.read_csv(csv_fovea)
        self.root_dir = root_dir
        self.transform = transform
        self.box_width_OD = box_width_OD
        self.box_width_Fovea = box_width_Fovea
        self.image_size = image_size
        
       

    def __len__(self):
        
        """return the length of the dataset"""
        for i,id in enumerate(self.od['Image No']):
            if not isinstance(id, str):
                break
        return i
        
    def __reshape__(self, sample):
        """reshape the image to a given size and update coordinates
        NB : the coordinates depend on the size of the image
             we use self.scale_factor to track th changes  
        """
        # resize + to tensor
        scale_factor = np.ones(2)
        image = sample['image']
        init_shape = np.array(list(image.size))
        scale = transforms.Resize(self.image_size)
        to_tensor = transforms.ToTensor()
        composed = transforms.Compose([scale, to_tensor])
        sample['image'] = composed(sample['image'])
        final_shape = np.array([sample['image'].shape[1], sample['image'].shape[2]])
        # update coordinates
        if not set(final_shape) == set(init_shape):
            scale_factor *= (final_shape/init_shape)
            sample['OD'] *=  scale_factor
            sample['Fovea'] *= scale_factor  
            
        return sample, scale_factor
            
    def __get_boxes__(self,sample, tpe ='OD'):
        """return the bounding boxes for a given type [OD, Fovea]"""
        
        # create image boxes
        if tpe =='OD':
            width = self.box_width_OD
        else:
            width = self.box_width_Fovea
            
        bbox = []
        bbox.append(sample[tpe][0]-width[0]/2)
        bbox.append(sample[tpe][1]-width[1]/2)
        bbox.append(sample[tpe][0]+width[0]/2)
        bbox.append(sample[tpe][1]+width[1]/2)
        
        return bbox

    def __getitem__(self, idx):
        """return image, target dict and scale factor"""
        
        #format index
        if torch.is_tensor(idx):
            idx = idx.tolist()
               
        # get coordinates
        OD = np.array([self.od.iloc[idx, 1], self.od.iloc[idx, 2] ]).astype('float')
        Fovea = np.array([self.fovea.iloc[idx, 1], self.fovea.iloc[idx, 2] ]).astype('float')
        img_name = self.od.iloc[idx, 0]
            
        # read image
        img_path = os.path.join(self.root_dir,img_name+'.jpg')
        image = Image.open(img_path)
        
        # create the sample dictionary
        sample = {'image': image, 'OD': OD, 'Fovea': Fovea }
        # reshape the image and update coordinates
        sample, scale_factor = self.__reshape__(sample)
  
        # create bounding boxes
        boxes = []
        boxes.append(self.__get_boxes__(sample, tpe ='OD'))
        boxes.append(self.__get_boxes__(sample, tpe ='Fovea'))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        #create labels
        labels = torch.tensor([1,2], dtype=torch.int64)

        #image_id
        image_id = torch.tensor([idx])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((2), dtype=torch.int64)

        # create target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        #target["area"] = torch.tensor([60*60],dtype=torch.int64)
        target["area"] = torch.tensor([self.box_width_OD[0]*self.box_width_OD[1], \
                                       self.box_width_Fovea[0]*self.box_width_Fovea[1] ],\
                                        dtype=torch.int64)
        target["iscrowd"] = iscrowd
 
        # apply transformations
        img = sample['image']
        if self.transform is not None:
            img , target = self.transform(img, target)

            
        return img, target, scale_factor
