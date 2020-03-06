from PIL import Image, ImageOps, ImageEnhance
import random
import numpy as np
import torch
from torchvision import transforms
from skimage import transform

IMG_SIZE = 640
image_options = {'resize': True, 'resize_size': IMG_SIZE}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        new_masks = []
        for mask in masks:
            mask = transform.resize(mask, (new_h, new_w))
            new_masks.append(mask)

        return {'image': img, 'masks': new_masks}


class RandomRotate90:
    def __init__(self, num_rot=(1, 2, 3, 4)):
        self.num_rot = num_rot
        self.axes = (0, 1)  # axes of rotation

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        n = np.random.choice(self.num_rot)
        image_rotate = np.ascontiguousarray(np.rot90(image, n, self.axes))
        new_masks = []
        for (i, mask) in enumerate(masks):
            new_masks.append(np.rot90(mask, n, self.axes))

        new_sample = {'image': image, 'masks': new_masks}
        # print('Rotate90 done')
        return new_sample


class ImageEnhencer:
    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        image = np.uint8(image)
        image = Image.fromarray(image)

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
        image = np.array(image)

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
            mask = mask[top: top + new_h, left: left + new_w]
            new_masks.append(mask)

        new_masks = np.array(new_masks)

        return {'image': image, 'masks': new_masks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(image)
        masks = np.array(masks)
        image = np.rollaxis(image, 2, 0)

        return {'image': torch.from_numpy(image),
                'masks': torch.from_numpy(masks)}

# def RandomRotate(theta_max=20):
