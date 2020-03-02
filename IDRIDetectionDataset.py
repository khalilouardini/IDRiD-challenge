# -*- coding: utf-8 -*-
"""
Created on Thur Feb 27 16:47:27 2020

@author: Ramzi Charradi
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from bengraham_preprocessing import preprocess

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class IDRiDDetectionDataset(Dataset):
    """Detection of OD ans Fovea dataset."""

    def __init__(self, csv_od, csv_fovea, root_dir, transform=None, set_type = "train"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.od = pd.read_csv(csv_od)
        self.fovea = pd.read_csv(csv_fovea)
        self.root_dir = root_dir
        self.transform = transform
        self.set_type = set_type

    def __len__(self):
        if self.set_type=="train":
            return 413
        else :
            return 103
        

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
            
        # labels
        OD = np.array([self.od.iloc[idx, 1], self.od.iloc[idx, 2] ]).astype('float')
        Fovea = np.array([self.fovea.iloc[idx, 1], self.fovea.iloc[idx, 2] ]).astype('float')
        img_name = self.od.iloc[idx, 0]
            
        # image
        # get img path
        img_path = os.path.join(self.root_dir,img_name+'.jpg')
        image = Image.open(img_path)
        sample = {'image': image, 'OD': OD, 'Fovea': Fovea }
        
        # shape
        sc_factor = np.ones(2)
        init_shape = np.array(list(image.size))

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            # modify coordinates after reshape
            final_shape = np.array([sample['image'].shape[1], sample['image'].shape[2]])
            if not set(final_shape) == set(init_shape):
                sc_factor *= (final_shape/init_shape)
                sample['OD'] *=  sc_factor
                sample['Fovea'] *= sc_factor
            
        return sample