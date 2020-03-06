import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.tensorboard
import torchvision
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageEnhance
import random
import time
import numpy as np
from skimage import transform
import matplotlib.pyplot as plt
import SimpleITK as sitk
import Transforms
from UNet import UNet

import pandas as pd
import sklearn.model_selection as model_selection
from tqdm import tqdm


def get_files_info(data_path):
    # The data is store in the folder 'Segmentation.nosync/'
    files = os.listdir(data_path)  # All the files in the folder 'Segmentation.nosync/'
    print('\t Files info')
    print('Content of the folder : {}'.format(files))
    print('Lenght of the folder : {}'.format(len(files)))

    # In the masks folder we have 1 folder per task we want to segment
    tasks = ['MA', 'HE', 'EX', 'SE', 'OD']
    for task in tasks:
        #     task = 'MA'
        task_path = os.path.join(data_path, 'train_masks/' + task + '/')
        task_files = os.listdir(task_path)
        print('Number of train masks for ' + task + ' task :', len(task_files))

    train_images_path = os.path.join(data_path, 'train_images')
    print('Number of train images : %s' % len(os.listdir(train_images_path)))

    test_images_path = os.path.join(data_path, 'test_images')
    print('Number of test images : %s' % len(os.listdir(test_images_path)))
    print('-' * 20)
    print('\n')


def load_sitk(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


class IDRiDDataset(Dataset):
    def __init__(self, mode='train', root_dir='Segmentation.nosync/',
                 transform=None, tasks=['MA', 'HE', 'EX', 'SE', 'OD'], data_augmentation=True):

        super(IDRiDDataset, self).__init__()
        # After resize image
        IMG_SIZE = 640
        if mode == 'train':
            mask_file, image_file = 'train_masks/', 'train_images/'

        elif mode == 'val':
            mask_file, image_file = 'test_masks/', 'test_images/'

        else:
            raise EnvironmentError('You should put a valid mode to generate the dataset')

        self.transform = transform
        self.mask_file = mask_file
        self.image_file = image_file
        self.root_dir = root_dir
        self.tasks = tasks
        self.data_augmentation = data_augmentation
        self.process_image = False

    def __len__(self):
        task = self.tasks[0]  # Assuming all the masks folders have the same length
        mask_path = os.path.join(self.root_dir, self.mask_file + task)
        return len(os.listdir(mask_path))

    def _crop_resize_image(self, sample):
        image, mask = sample['image'], sample['mask']
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        rate = [0.248, 0.25, 0.252]
        index = random.randint(0, 2)
        old_size = image.size
        new_size = tuple([int(x * rate[index]) for x in old_size])
        resize_image = image.resize(size=new_size, resample=3)
        resize_mask = mask.resize(size=new_size, resample=3)
        crop_image = resize_image.crop((66, 0, 930, 712))
        crop_mask = resize_mask.crop((66, 0, 930, 712))
        padding = (0, 76, 0, 76)
        image = ImageOps.expand(crop_image, padding)
        mask = ImageOps.expand(crop_mask, padding)
        if image.size != mask.size:
            raise ValueError("Image and mask size not equal !!!")

        # random crop image to size (640, 640)
        width, height = image.size
        resize = int(self.image_options["resize_size"])
        x = random.randint(0, width - resize - 1)
        y = random.randint(0, height - resize - 1)
        image = image.crop((x, y, x + resize, y + resize))
        mask = mask.crop((x, y, x + resize, y + resize))

        if self.data_augmentation is True:
            # light
            enh_bri = ImageEnhance.Brightness(image)
            brightness = round(random.uniform(0.8, 1.2), 2)
            image = enh_bri.enhance(brightness)

            # color
            enh_col = ImageEnhance.Color(image)
            color = round(random.uniform(0.8, 1.2), 2)
            image = enh_col.enhance(color)

            # contrast
            enh_con = ImageEnhance.Contrast(image)
            contrast = round(random.uniform(0.8, 1.2), 2)
            image = enh_con.enhance(contrast)

            method = random.randint(0, 7)
            # print(method)
            if method < 7:
                image = image.transpose(method)
                mask = mask.transpose(method)
            degree = random.randint(-5, 5)
            image = image.rotate(degree)
            mask = mask.rotate(degree)

        image_array = np.array(image)
        # standardization image
        if self.process_image is True:
            image_array = image_array * (1.0 / 255)
            # image_array = per_image_standardization(image_array)

        mask_array = np.array(mask)
        return np.array(image_array), mask_array

    def __getitem__(self, idx):
        'Generate one batch of data'
        sample = self.load(idx)
        return sample

    def load(self, idx):
        # Get masks from a particular idx
        masks = []

        for task in self.tasks:
            suffix = '.tif'
            mask_name = 'IDRiD_{:02d}_'.format(idx + 1) + task + suffix  # if idx = 0. we look for the image 1
            mask_path = os.path.join(self.root_dir, self.mask_file + task + '/' + mask_name)
            mask = load_sitk(mask_path)
            mask = mask[:, :, 0] / 255
            masks.append(mask)

        masks = np.stack(masks, axis=0)

        # Get original images
        image_name = 'IDRiD_{:02d}'.format(idx + 1) + '.jpg'
        image_path = os.path.join(self.root_dir, self.image_file + image_name)
        image = load_sitk(image_path)

        # Define output sample
        sample = {'image': image, 'masks': masks}

        # If transform apply transformation
        if self.transform:
            sample = self.transform(sample)

        #         # One hot encoding
        #         label = 4
        #         mask = mask.astype(np.int16)
        #         mask = np.rollaxis(np.eye(label, dtype=np.uint8)[mask], -1, 0)

        return sample


def load_train_val_data(tasks=['EX', 'MA'], data_path='Segmentation.nosync/'):
    # The data is store in the folder 'Segmentation.nosync/'
    batch_size = 5

    get_files_info(data_path)

    # Create train dataset
    n_classes = len(tasks)  # One mask per task
    n_channels = 3  # RGB image as input

    transforms_list = [Transforms.Rescale(650),
                       Transforms.RandomCrop(640),
                       Transforms.RandomRotate90(),
                       Transforms.ImageEnhencer(),
                       Transforms.ToTensor()]

    ## Image is now (512, 512, 3)
    transformation = torchvision.transforms.Compose(transforms_list)

    print('\t Loading Train and Validation Datasets... \n')
    train_data = IDRiDDataset(tasks=tasks, transform=transformation)
    val_data = IDRiDDataset(mode='val', tasks=tasks, transform=transformation)

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(val_data,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True)

    print('Length of train dataset: ', len(train_loader.dataset))
    print('Shape of image :', train_loader.dataset[10]['image'].shape)
    print('Shape of mask : ', train_loader.dataset[10]['masks'][0].shape)

    return train_loader, val_loader

#
# if __name__ == '__main__':
#     train_loader, val_loader = load_train_val_data(tasks=['EX', 'MA'], data_path='Segmentation.nosync/')
