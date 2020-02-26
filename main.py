from PIL import Image
import os
import matplotlib.pyplot as plt
from IDRiDGradingDataset import IDRiDGradingDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

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
    root_dir = 'B. Disease Grading/1. Original Images/a. Training Set/'
    csv_file = 'B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'
    bengraham = False

    # transform (preprocessing)
    scale = transforms.Resize((512, 512))
    to_tensor = transforms.ToTensor()
    composed = transforms.Compose([scale,
                                   to_tensor])


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
            if i_batch == 3:
                plt.figure()
                show_batch(sample_batched)
                plt.axis('off')
                plt.ioff()
                plt.show()
                break

    print("Done")