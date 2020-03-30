import torch
from Grading_task.models.IDRiDClassifier import IDRiDClassifier
from Grading_task.datasets.IDRiDGradingDataset import IDRiDGradingDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from Grading_task.utils.classifier_utils import train, test
from torch.utils.data.sampler import SubsetRandomSampler

if __name__ == "__main__":
    # Prepare data (CHANGE PATH HERE)
    root_dir_train = '/Users/khalilouardini/Desktop/projects/dlmi/B. Disease Grading/1. Original Images/a. Training Set'
    csv_file_train = '/Users/khalilouardini/Desktop/projects/dlmi/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'

    root_dir_test = '/Users/khalilouardini/Desktop/projects/dlmi/B. Disease Grading/1. Original Images/b. Testing Set'
    csv_file_test = '/Users/khalilouardini/Desktop/projects/dlmi/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv'
    bengraham = False

    # transform (preprocessing)
    scale = transforms.Resize((256, 256))
    to_tensor = transforms.ToTensor()
    horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
    vertical_flip = transforms.RandomVerticalFlip(p=0.3)
    color_jitter = transforms.ColorJitter(brightness=0.01 * torch.abs(torch.randn(1)).item(),
                                          contrast=0.01 * torch.abs(torch.randn(1)).item(),
                                          saturation=torch.abs(0.01 * torch.randn(1)).item(),
                                          hue=torch.abs(0.01 * torch.randn(1)).item()
                                          )
    # random_perspective = transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3)
    random_rotation = transforms.RandomRotation(30)
    center_crop = transforms.CenterCrop(256)
    composed = transforms.Compose([scale,
                                   horizontal_flip,
                                   random_rotation,
                                   vertical_flip,
                                   color_jitter,
                                   center_crop,
                                   to_tensor]
                                  )

    #composed = transforms.Compose([scale,
                                   #to_tensor])
    # train and test set
    train_set = IDRiDGradingDataset(csv_file_train, root_dir_train, composed, bengraham)
    test_set = IDRiDGradingDataset(csv_file_test, root_dir_test, composed, bengraham)

    # dataloaders
    batch_size = 4
    n = len(train_set)
    train_indices = list(range(int(0.8 * n)))
    val_indices = list(range(int(0.8 * n) + 1, n))

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_indices), num_workers=0)
    val_loader = DataLoader(train_set, batch_size=batch_size,
                            sampler=SubsetRandomSampler(val_indices), num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=True, num_workers=0)

    print(
        f"# batches in train set {len(train_loader)} | # batches in val set {len(val_loader)}  # batches in test set {len(test_loader)}")

    # model
    use_cuda = torch.cuda.is_available()
    model = IDRiDClassifier()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        model = torch.nn.DataParallel(model)
    model.to(device)

    # intialize variables
    n_epochs = 1
    lr = 1e-5
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_model = train(model, train_loader, val_loader, criterion, optimizer, n_epochs, device)
    results_df, results_metrics = test(model, test_loader, criterion, device)
    print("done")






