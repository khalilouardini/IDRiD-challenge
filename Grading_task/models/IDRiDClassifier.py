import torch.nn as nn
from torchvision import models
import torch


class IDRiDClassifier(nn.Module):

    def __init__(self, n_classes=5):
        super(IDRiDClassifier, self).__init__()

        self.classes = n_classes
        self.Avg = nn.AvgPool2d(4)
        self.ReLU = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(1000)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.linear1 = nn.Linear(1000, 1000)
        self.linear2 = nn.Linear(1000, 512)
        self.linear3 = nn.Linear(512, self.classes)

        ## ResNet101 features extractor
        self.res50 = models.resnet50(pretrained=True)
        # Freezing all layers
        for child in list(self.res50.children()):
            for param in child.parameters():
                param.requires_grad = False
        # Removing the softmax layer
        self.res101 = nn.Sequential(*list(self.res50.children())[:-1])

    def forward(self, x):
        x = self.res50(x)
        x = x.view(-1, 1000)
        x = nn.Dropout(0.5)(self.ReLU(self.batchnorm1(self.linear1(x))))
        x = nn.Dropout(0.7)(self.ReLU(self.batchnorm2(self.linear2(x))))
        return torch.nn.functional.softmax(self.linear3(x))
