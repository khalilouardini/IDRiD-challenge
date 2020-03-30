import torchvision
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from Transforms import Resize, RandomCrop, RandomRotate90, ToTensor, ImageEnhencer, ApplyCLAHE



def get_files_info(data_path):
    '''
        Gets information about files and folders in data_path.
    '''
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
                 transform=None, tasks=['MA', 'HE', 'EX', 'SE', 'OD'], data_augmentation=True, green=False):

        super(IDRiDDataset, self).__init__()
        # After resize image
        IMG_SIZE = 640
        if mode == 'train':
            mask_file, image_file = 'train_masks/', 'train_images/'

        elif mode == 'val':
            mask_file, image_file = 'test_masks/', 'test_images/'

        else:
            raise EnvironmentError('You should put a valid mode to generate the dataset')

        self.mode = mode
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

    def __getitem__(self, idx):
        'Generate one batch of data'
        sample = self.load(idx)
        return sample

    def load(self, idx):
        # Get masks from a particular idx
        masks = []

        if self.mode == 'val':
            idx += 54

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
        masks = masks.astype(np.int16)
        orig_img = np.copy(image)

        # new_masks = [] # list of masks each mask composed of a background mask and a foreground mask [2, 512, 512]
        # for mask in masks:
        #     bg = np.zeros(masks[0].shape).astype(np.int16)
        #     fg = np.zeros(masks[0].shape).astype(np.int16)
        #     bg[masks[0] == 0] = 1 #background
        #     fg[masks[0] != 0] = 1 #foreground
        #     new_mask = np.stack([bg, fg])
        #     new_masks.append(new_mask)

        # Define output sample
        sample = {'image': image, 'masks': masks}
        # sample = {'image': image, 'masks': new_masks}

        # If transform apply transformation
        if self.transform:
            sample = self.transform(sample)

        trans_image = sample['image']
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.title('Original image')
        plt.axis('off')
        plt.imshow(orig_img)

        plt.subplot(122)
        plt.title('Transformed image')
        plt.axis('off')
        plt.imshow(np.array(trans_image[0]))
        plt.show()

        return sample


def load_train_val_data(tasks=['EX', 'MA'], data_path='Segmentation.nosync/', batch_size=8, green=False):
    # The data is store in the folder 'Segmentation.nosync/'
    get_files_info(data_path)

    # Create train dataset
    n_classes = len(tasks)  # One mask per task
    n_channels = 3  # RGB image as input

    transforms_list = [
        Resize(520),  # resize to 520x782
        RandomCrop(512),
        RandomRotate90(),
        ImageEnhencer(color_jitter=False, green=green),
        # ApplyCLAHE(green=green),
        ToTensor(green=green)]

    ## Image is now (512, 512, 3)
    transformation = torchvision.transforms.Compose(transforms_list)

    print('\t Loading Train and Validation Datasets... \n')
    train_data = IDRiDDataset(tasks=tasks, transform=transformation, root_dir=data_path)
    val_data = IDRiDDataset(mode='val', tasks=tasks, transform=transformation, root_dir=data_path)

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
    print('Shape of mask : ', train_loader.dataset[10]['masks'].shape)
    print('-' * 20)
    print('\n')

    return train_loader, val_loader


if __name__ == '__main__':
    train_loader, val_loader = load_train_val_data(tasks=['EX'], data_path='Segmentation.nosync/', green=True)
