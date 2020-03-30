import matplotlib.pyplot as plt
from Grading_task.datasets.IDRiDGradingDataset import IDRiDGradingDataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch

def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, labels = sample_batched['image'], sample_batched['labels']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')

if __name__ == "__main__":

    #init
    root_dir = 'B. Disease Grading/1. Original Images/merged_dataset/'
    csv_file = 'B. Disease Grading/2. Groundtruths/merged_labels.csv'
    bengraham = True

    # transform (preprocessing)
    scale = transforms.Resize((256, 256))
    to_tensor = transforms.ToTensor()
    horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
    vertical_flip =transforms.RandomVerticalFlip(p=0.3)
    color_jitter = transforms.ColorJitter(brightness=0.05*torch.abs(torch.randn(1)).item(),
                                            contrast=0.05*torch.abs(torch.randn(1)).item(),
                                            saturation=torch.abs(0.05*torch.randn(1)).item(),
                                            hue=torch.abs(0.05*torch.randn(1)).item()
                                          )
    #random_perspective = transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)
    random_rotation = transforms.RandomRotation(30)
    #center_crop = transforms.CenterCrop(256)
    composed = transforms.Compose([scale,
                                    to_tensor]
                                  )


    # init class
    idrid_dataset = IDRiDGradingDataset(csv_file, root_dir, composed, bengraham)

    #test: 1 sample
    idx = 45
    print(len(idrid_dataset))
    sample = idrid_dataset[idx]
    img, label = sample['image'], sample['labels']

    iterate = False
    # Iterate through dataset
    if iterate:
        for i in range(len(idrid_dataset)):
            sample = idrid_dataset[i]
            print(f"item:{i} | size:{sample['image'].size()} | label:{sample['labels']}")

    do_show_batch = True
    dataloader = DataLoader(idrid_dataset, batch_size=4,
                            shuffle=True, num_workers=0)
    # dataloader + plot
    if do_show_batch:
        for i_batch, sample_batched in enumerate(dataloader):
            print(f"item:{i_batch} | size:{sample_batched['image'].size()}, label: {label}")
            # observe 4th batch and stop.
            if i_batch == 4:
                plt.figure()
                show_batch(sample_batched)
                plt.axis('off')
                plt.ioff()
                plt.plot()
                plt.show()
                break
            plt.show()

    print("Done")