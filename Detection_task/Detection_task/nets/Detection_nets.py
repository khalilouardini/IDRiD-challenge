import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn.functional as F

    
class Regression(nn.Module):
    
    def __init__(self):
        super(Regression, self).__init__()
        
        self.l1 = nn.Linear(800*800*3, 500)
        self.l2 = nn.Linear(500, 100)
        self.l3 = nn.Linear(100, 52)
        self.l4 = nn.Linear(52, 2)
        
    def forward(self, inputs):
        x = inputs.view(800*800*3)
        x = self.l1(x)
        x = F.tanh (x)
        x = self.l2(x)
        x = F.tanh (x)
        x = self.l3(x)
        x = F.tanh (x)
        x = self.l4(x)
        return x


def FasterRCNN(image_mean, image_std, num_classes=3):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                             image_mean=image_mean,
                                                             image_std=image_std)
    # get number of input channels for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model
