import torch
from Grading_task.models.IDRiDClassifier import IDRiDClassifier
from Grading_task.datasets.IDRiDGradingDataset import IDRiDGradingDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from Grading_task.utils.classifier_utils import test
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import numpy as np
import itertools


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
    #composed = transforms.Compose([scale,
                                   #horizontal_flip,
                                   #random_rotation,
                                   #vertical_flip,
                                   #to_tensor]
                                 # )

    composed = transforms.Compose([scale,
                                   to_tensor])
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
    state_dict = torch.load('Grading_task/model_dr.pth', map_location=torch.device('cpu'))
    model = torch.nn.DataParallel(IDRiDClassifier())
    model.load_state_dict(state_dict)
    model.eval()

    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    results_df, results_metrics = test(model, test_loader, criterion, 'cpu')


    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting normalize=True.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    classes = ['Sane', 'Mild', 'Moderate', 'Severe', 'Severe+']
    plot_confusion_matrix(results_metrics['confusion_matrix'], classes)

    print("done")






