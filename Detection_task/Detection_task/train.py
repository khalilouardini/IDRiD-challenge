import argparse
import torch
import torch.optim as optim
import os
import numpy as np


from datasets.IDRIDetectionDataset import IDRID_Detection_Dataset
from nets.Detection_nets import FasterRCNN
from src.train_model import train_one_epoch_RetinaNet, train_one_epoch_FasterRCNN, evaluate
from src import utils
import src.transforms as T
from nets import models


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["FasterRCNN", "RetinaNet"],
                    help = "choose model FasterRCNN or RetinaNet.", required=True)
parser.add_argument("--epochs", type=int, default = 5)
parser.add_argument("--depth", type=int, choices=[18,34,50,101,152], default = 101,
                    help = "depth of resnet modele for RetinaNet.")
parser.add_argument("--print_freq", type=int, default = 50)



device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    
    args = parser.parse_args()
    torch.manual_seed(41)
    
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    print("Training {}".format(args.model))
    print("Using device : %s" % device)
    device = torch.device(device)
    num_epochs = args.epochs
    print("Num_epochs : ",num_epochs)
    depth = args.depth
    print_freq = args.print_freq
    num_classes = 3
    
    
    
    # Train data paths
    root_dir_train = 'C. Localization/1. Original Images/a. Training Set/'
    csv_od_train = 'C. Localization/2. Groundtruths/1. Optic Disc Center Location/a. IDRiD_OD_Center_Training Set_Markups.csv'
    csv_fovea_train = 'C. Localization/2. Groundtruths/2. Fovea Center Location/IDRiD_Fovea_Center_Training Set_Markups.csv'

    # validation data Paths
    root_dir_test = 'C. Localization/1. Original Images/b. Testing Set/'
    csv_od_test = 'C. Localization/2. Groundtruths/1. Optic Disc Center Location/b. IDRiD_OD_Center_Testing Set_Markups.csv'
    csv_fovea_test = 'C. Localization/2. Groundtruths/2. Fovea Center Location/IDRiD_Fovea_Center_Testing Set_Markups.csv'
    

    # train and test set
    train_set = IDRID_Detection_Dataset(csv_od_train, csv_fovea_train, root_dir_train, T.get_transform(train=True),box_width_OD = (108,148), box_width_Fovea = (120,120),image_size = (800,800))
    test_set = IDRID_Detection_Dataset(csv_od_test, csv_fovea_test, root_dir_test,  T.get_transform(train=False),box_width_OD = (108,148), box_width_Fovea = (120,120),image_size = (800,800))
    

    data_loader = torch.utils.data.DataLoader(
                train_set, batch_size=1, shuffle=True, num_workers=0,
                collate_fn= utils.collate_fn)

    test_data_loader = torch.utils.data.DataLoader(
                test_set, batch_size=1, shuffle=True, num_workers=0,
                collate_fn= utils.collate_fn)
    
    

    
    if args.model == "FasterRCNN" :
        print("Computing images statistics :")
        image_mean = (0.46737722, 0.24098666, 0.10314517)
        image_std = (0.04019115, 0.024475794, 0.02510888)
        #image_mean = utils.compute_means(train_set)
        #image_std = utils.compute_stds(test_set)
        print("Means: {}".format(image_mean))
        print("Stds: {}".format(image_std))
        model = FasterRCNN(image_mean, image_std, num_classes).to(device)
        train_one_epoch = train_one_epoch_FasterRCNN

            
    else :
        # Create the model
        if depth == 18:
            model = models.resnet18(num_classes=3, pretrained=True)
        elif depth == 34:
            model = models.resnet34(num_classes=3, pretrained=True)
        elif depth == 50:
            model = models.resnet50(num_classes=3, pretrained=True)
        elif depth == 101:
            model = models.resnet101(num_classes=3, pretrained=True)
        elif depth == 152:
            model = models.resnet152(num_classes=3, pretrained=True)
            
        model.to(device)
        model.training = True
        model.train()
        #model.module.freeze_bn()
        train_one_epoch = train_one_epoch_RetinaNet
        
        
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)


    max_iou = 0.0
    for epoch in range(num_epochs):
        # Train for one epoch, printing every 10 iterations
        epoch_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq)
        iou = evaluate(model, test_set, device, args.model)
        scheduler.step(np.mean(epoch_loss))
        if iou > max_iou :
            max_iou = iou
            save_path = "./models/{}.pth".format(args.model)
            print("Saving checkpoint {:s} at epoch {:d}".format(save_path, epoch))
            torch.save(model.state_dict(), save_path)
        #torch.cuda.empty_cache()
                
                

    