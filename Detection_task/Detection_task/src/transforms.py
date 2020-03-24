import random
from PIL import ImageEnhance
from torchvision.transforms import functional as F
from torchvision import transforms



        
class Compose(object):
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, image, target):
        for t in self.transformations:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox

        return image, target

class ImageEnhencer:
    def __call__(self, image, target):
        #image, masks = sample['image'], sample['masks']
        #image = np.uint8(image)
        image = transforms.ToPILImage()(image).convert("RGB")

        # light
        #♠enh_bri = ImageEnhance.Brightness(image)
        #brightness = round(random.uniform(0.8, 1.2), 2)
        #image = enh_bri.enhance(brightness)

        # color
        #enh_col = ImageEnhance.Color(image)
        #color = round(random.uniform(0.8, 1.2), 2)
        #image = enh_col.enhance(color)

        # contrast
        enh_con = ImageEnhance.Contrast(image)
        contrast = round(random.uniform(0.8, 1.2), 2)
        image = enh_con.enhance(contrast)
        #image = np.array(image)

        return transforms.ToTensor()(image), target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    

    
    
def get_transform(train):
    
    transform = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transform.append(ImageEnhencer())

    if train:
        #during training, randomly flip the training images
        #and ground-truth for data augmentation
        transform.append(RandomHorizontalFlip(0.2))

    return Compose(transform)
