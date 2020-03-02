import torch.nn as nn
from torchvision import models
import torch

class IDRiDClassifier(nn.Module):

    def __init__(self, n_classes=5):
        super(IDRiDClassifier, self).__init__()

        self.classes = n_classes
        self.Avg = nn.AvgPool2d(4)
        self.ReLU = nn.ReLU()
        self.Dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(1000, self.classes)

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
        return self.Dropout(self.linear(x))
