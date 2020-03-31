from UNet import UNet
from main import train_loop
import torch.nn as nn
from DataSet import load_train_val_data
from torch.utils.tensorboard import SummaryWriter
import os
import time
import torch

if __name__ == '__main__':
    ## PARAMS
    data_path = 'Segmentation.nosync/' #path where dataset is stored
    save_path = 'best_model.MSELoss--UNet.pth.tar' #path where weights of our model was downloaded
    tensorboard_folder = save_path + 'tensorboard_logs/'
    session_name = 'Test_session' + '_' + time.strftime('%m.%d %Hh%M')
    tasks = ['EX']
    lr = 1e-4

    # Tensorboard logs
    log_dir = tensorboard_folder + session_name + '/'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    # Use GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Loading a model
    checkpoint = torch.load(save_path, map_location=device)
    model = UNet(1, 1) #green only and 1 task
    if torch.cuda.is_available():
        model = model.to(device)

    # load weights into model
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')

    # Load Datasets
    train_loader, val_loader = load_train_val_data(tasks=tasks, data_path=data_path, batch_size=8, green=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Choose optimize
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=7)

    with torch.no_grad():
        model.eval()
        print('Running Evaluation...')
        val_loss, val_aupr = train_loop(val_loader, model, criterion,
                                        optimizer, writer, 1)
        print('AUPR evaluated on validation set:', val_aupr)
