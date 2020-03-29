import argparse
import torch
import os


from datasets.IDRIDetectionDataset import IDRID_Detection_Dataset
from nets.Detection_nets import FasterRCNN
from src.train_model import evaluate
from src import utils
import src.transforms as T
from nets import models
from src.plots import plot_prediction


parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, choices=["evaluate", "infer"], required=True)
parser.add_argument("--model", type=str, choices=["FasterRCNN", "RetinaNet"],
                    help = "choose model FasterRCNN or RetinaNet.", required=True)
parser.add_argument("--depth", type=int, choices=[18,34,50,101,152], default = 101,
                    help = "depth of resnet modele for RetinaNet.")
parser.add_argument("--weights", "-w", required=True, metavar="WEIGHTS_FILE",
                    type=str,
                    help="Model weights.")
parser.add_argument("--img_idx", type=int, default = None)
parser.add_argument("--dataset", type=str,choices = ["train","test"],
                    help="dataset to be evaluated", default = "test")




device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    
    args = parser.parse_args()
    torch.manual_seed(41)
    
    
    if not os.path.exists("models"):
        os.makedirs("models")
        
    
    print("Using device : %s" % device)
    device = torch.device(device)
    depth = args.depth
    num_classes = 3
    
    CHECKPOINT_FILE = args.weights
    print("Checkpoint file: {:s}".format(CHECKPOINT_FILE))
    model_state_dict = torch.load(CHECKPOINT_FILE)
    
    
    
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
    
    

    
    if args.model == "FasterRCNN" :
        print("Computing images statistics :")
        image_mean = (0.46737722, 0.24098666, 0.10314517)
        image_std = (0.04019115, 0.024475794, 0.02510888)
        #image_mean = utils.compute_means(train_set)
        #image_std = utils.compute_stds(test_set)
        print("Means: {}".format(image_mean))
        print("Stds: {}".format(image_std))
        model = FasterRCNN(image_mean, image_std, num_classes)
        model.load_state_dict(model_state_dict).to(device)

            
    else :
        # Create the model
        if depth == 50:
            model = models.resnet50(num_classes=3, pretrained=True)
        elif depth == 101:
            model = models.resnet101(num_classes=3, pretrained=True)
        elif depth == 152:
            model = models.resnet152(num_classes=3, pretrained=True)
            
        
        model.load_state_dict(model_state_dict)
        model.to(device)
    
        
    if args.dataset == "train":
        dataset = train_set
    else :
        dataset = test_set
     
    if args.task == "evaluate":
        print("Evaluating {}".format(args.model))
        evaluate(model, dataset, device, args.model)
    else :
        if args.img_idx is not None:
            if not os.path.exists("./figures"):
                os.makedirs("./figures")
            img, OD_true_box, Fovea_true_box, OD_predicted_box, Fovea_predicted_box = utils.get_boxes(model, dataset, threshold = 0.008, img_idx = args.img_idx, model_type = args.model)    
            plot_prediction(img, OD_true_box, Fovea_true_box, OD_predicted_box, Fovea_predicted_box, args.img_idx) 

                
                

    