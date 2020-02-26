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
        self.linear = nn.Linear(2048, self.classes)

        ## ResNet101 features extractor
        self.res101 = models.resnet101(pretrained=True)
        # Freezing all layers
        for child in list(self.res101.children()):
            for param in child.parameters():
                param.requires_grad = False
        # Removing the softmax layer
        self.res101 = nn.Sequential(*list(self.res101.children())[:-1])

    def forward(self, x):
        x1 = self.res101(x)
        x1 = x1.view(-1, 2048)
        return self.Dropout(self.linear(x))
