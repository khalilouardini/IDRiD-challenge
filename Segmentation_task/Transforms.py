from PIL import Image, ImageEnhance
import numpy as np
import torch
from torchvision import transforms
import time
from albumentations.augmentations.transforms import CLAHE


IMG_SIZE = 640
image_options = {'resize': True, 'resize_size': IMG_SIZE}


class Resize(object):
    '''
        Resize an image.

    Args:
        output_size: (int, tuple) if is an int the image will keep its aspect ratio and have
                    its lowest dimension resized to the given int.
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int))
        self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        image = Image.fromarray(image)
        image = transforms.Resize(self.output_size, interpolation=2)(image)
        image = np.array(image)

        new_masks = []
        for mask in masks:
            # new_mask = []
            # for i in range(mask.shape[0]): # each label (foreground, task1, task2 ...)
            #     img = Image.fromarray(mask[i])
            #     img = transforms.Resize(self.output_size, interpolation=2)(img)
            #     new_mask.append(np.array(img))

            # new_mask = np.stack(new_mask)
            # new_masks.append(new_mask)
            img = Image.fromarray(mask)
            img = transforms.Resize(self.output_size, interpolation=2)(img)
            new_masks.append(np.array(img))

        new_masks = np.array(new_masks)
        new_sample = {'image': image, 'masks': new_masks}
        return new_sample


class RandomRotate90:
    '''
        Randomly rotates an image
    '''

    def __init__(self, num_rot=(1, 2, 3, 4)):
        self.num_rot = num_rot
        self.axes = (0, 1)  # axes of rotation

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        n = np.random.choice(self.num_rot)
        image_rotate = np.ascontiguousarray(np.rot90(image, n, self.axes))
        new_masks = []
        for (i, mask) in enumerate(masks):
            # new_mask = []
            # for j in range(mask.shape[0]):
            #     rot_mask = np.rot90(mask[j], n, self.axes)
            #     new_mask.append(rot_mask)

            # new_mask = np.stack(new_mask)
            # new_masks.append(new_mask)
            new_masks.append(np.rot90(mask, n, self.axes))

        new_sample = {'image': image_rotate, 'masks': new_masks}
        # print('Rotate90 done')
        return new_sample


class ApplyCLAHE(object):
    '''
        Applies CLAHE (Contrast Limited Adaptative Histogram Equalization)
        transformation to the dataset image.

    Args:
        green: bool, if True only returns the green channel of the image
    '''

    def __init__(self, green=False):
        self.green = green

    def __call__(self, sample):
        light = CLAHE(p=1)  # transformation
        image, masks = sample['image'], sample['masks']
        image = np.uint8(image)

        image = light(image=image)['image']
        if self.green:
            image = [image[:, :, 1]]  # only green channel

        new_sample = {'image': image, 'masks': masks}
        return new_sample


class ImageEnhencer(object):
    '''
        Enhances the brightness/color and contrast of a dataset image

    Args:
        green: bool, if True only returns the green channel of the enhanced image
        color_jitter: bool, if True applies a random enhancement of the brightness/contrast/hue
    '''

    def __init__(self, color_jitter=False, green=False):
        self.color_jitter = color_jitter
        self.green = green

    def __call__(self, sample, color_jitter=False):
        t1 = time.time()
        image, masks = sample['image'], sample['masks']
        image = np.uint8(image)
        image = Image.fromarray(image)

        if self.color_jitter:
            # apply random enlightment/hue/contrast
            image = transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0, hue=0.05)(image)
            image = np.array(image)

        else:
            # light
            enh_bri = ImageEnhance.Brightness(image)
            # brightness = round(random.uniform(0.8, 1.2), 2)
            brightness = 1.3
            image = enh_bri.enhance(brightness)

            # color
            enh_col = ImageEnhance.Color(image)
            # color = round(random.uniform(0.8, 1.2), 2)
            color = 1.0
            image = enh_col.enhance(color)

            # contrast
            enh_con = ImageEnhance.Contrast(image)
            # contrast = round(random.uniform(0.8, 1.2), 2)
            contrast = 1.2
            image = enh_con.enhance(contrast)
            image = np.array(image)

        if self.green:
            image = [image[:, :, 1]]  # only green channel

        masks = masks  # if only one task
        new_sample = {'image': image, 'masks': masks}

        return new_sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        new_masks = []
        for mask in masks:
            # new_mask = []
            # for i in range(mask.shape[0]):
            #     mask_crop = mask[i][top: top + new_h, left: left + new_w]
            #     new_mask.append(mask_crop)
            # new_mask = np.stack(new_mask)
            # new_masks.append(new_mask)
            mask_crop = mask[top: top + new_h, left: left + new_w]
            new_masks.append(mask_crop)

        new_masks = np.array(new_masks)

        return {'image': image, 'masks': new_masks}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.

    Args:
        green: bool, if True does not perform swap of color axis (HxWxC -> CxHxW)

    """

    def __init__(self, green=False):
        self.green = green

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

        image = np.array(image)
        masks = torch.from_numpy(np.array(masks))

        if not self.green:
            # if RGB channel
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = np.rollaxis(image, 2, 0)

        image = torch.from_numpy(image)

        return {'image': image,
                'masks': masks}

