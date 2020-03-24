from datasets.IDRIDetectionDataset import IDRID_Detection_Dataset
from nets.Detection_nets import FasterRCNN
from src.train_fasterrcnn import train_one_epoch, evaluate
import src.transforms as T
from src.metrics import evaluate_mean_distance
import torch
from src import utils
from src.plots import plot_prediction, plot_random_batch





if __name__ == "__main__":
    torch.manual_seed(0)
    # Prepare data
    root_dir_train = 'C. Localization/1. Original Images/a. Training Set/'
    csv_od_train = 'C. Localization/2. Groundtruths/1. Optic Disc Center Location/a. IDRiD_OD_Center_Training Set_Markups.csv'
    csv_fovea_train = 'C. Localization/2. Groundtruths/2. Fovea Center Location/IDRiD_Fovea_Center_Training Set_Markups.csv'


    root_dir_test = 'C. Localization/1. Original Images/b. Testing Set/'
    csv_od_test = 'C. Localization/2. Groundtruths/1. Optic Disc Center Location/b. IDRiD_OD_Center_Testing Set_Markups.csv'
    csv_fovea_test = 'C. Localization/2. Groundtruths/2. Fovea Center Location/IDRiD_Fovea_Center_Testing Set_Markups.csv'
    

    # train and test set
    train_set = IDRID_Detection_Dataset(csv_od_train, csv_fovea_train, root_dir_train, T.get_transform(train=True),box_width_OD = (108,148), box_width_Fovea = (120,120),image_size = (800,800))
    valid_set = IDRID_Detection_Dataset(csv_od_test, csv_fovea_test, root_dir_test,  T.get_transform(train=False),box_width_OD = (108,148), box_width_Fovea = (120,120),image_size = (800,800))
    

    # Data loaders
    torch.manual_seed(1)

    data_loader = torch.utils.data.DataLoader(
                train_set, batch_size=2, shuffle=True, num_workers=0,
                collate_fn= utils.collate_fn)

    val_data_loader = torch.utils.data.DataLoader(
                valid_set, batch_size=2, shuffle=True, num_workers=0,
                collate_fn= utils.collate_fn)
    
    #image_mean = utils.compute_means(train_set)
    #image_std = utils.compute_stds(valid_set)
        
    image_mean = (0.43456945, 0.21095228, 0.07047101)
    image_std = (0.040260073, 0.028208755, 0.02999968)


    print("Means: {}".format(image_mean))
    print("Stds: {}".format(image_std))

#%% # plot 4 images randomly
    plot_random_batch(train_set)
    
#%% 
    # load a model pre-trained on COCO
    model = FasterRCNN(image_mean, image_std, num_classes = 3)

#%%
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # our dataset has 3 classes only - background, tumour and no tumour
    num_classes = 3
    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    
    """ FILL HERE"""
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate
    # Change the scheduler type if you wish
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=1,
                                                   gamma=0.5)

#%%
    num_epochs = 3
    max_iou = 0.7
    for epoch in range(num_epochs):

        # Train for one epoch, printing every 10 iterations
        train_his_ = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        iou = evaluate(model, valid_set, device)
        lr_scheduler.step()
        if iou > max_iou :
            max_iou = iou
            torch.save(model, 'model_best_iou4.pt')
            print("Model saved")
        
        
        #print("Validation : Mean distance ",evaluate_mean_distance(model, valid_set))

    
        torch.cuda.empty_cache()
        
#%%
    #torch.save(model, 'model2.pt')
    #load model
    model = torch.load('model_best_iou3.pt')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%%
    img, OD_true_box, Fovea_true_box, OD_predicted_box, Fovea_predicted_box = utils.get_boxes(model, train_set, threshold = 0.008, img_idx = 1)
        
    plot_prediction(img, OD_true_box, Fovea_true_box, OD_predicted_box, Fovea_predicted_box)     
#%%
    print(evaluate_mean_distance(model, valid_set))
