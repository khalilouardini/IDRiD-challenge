import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from torch.autograd import Variable
from utils import auc_on_batch, aupr_on_batch, plot
from DataSet import load_train_val_data
from UNet import UNet
from GCN import GCN
import torch.nn as nn

## PARAMETERS OF THE MODEL
use_cuda = torch.cuda.is_available()
learning_rate = 1e-4
image_size = (512, 512)
n_labels = 2
epochs = 20
batch_size = 8
print_frequency = 1
save_frequency = 10
save_model = True
tumor_percentage = 0.5
tensorboard = True

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def to_var(x, device):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    x = x.to(device)
    return x


def to_numpy(x):
    if not (isinstance(x, np.ndarray) or x is None):
        if x.is_cuda:
            x = x.data.cpu()
        x = x.numpy()
    return x


def print_summary(epoch, i, nb_batch, loss, loss_name, batch_time,
                  average_loss, average_time, auc, aupr, mode):
    '''
        mode = Train or Test
    '''
    summary = '[' + str(mode) + '] Epoch: [{0}][{1}/{2}]\t'.format(
        epoch, i, nb_batch)

    string = ''
    string += '{} : {:.4f} '.format(loss_name, loss)
    string += '(Average {:.4f}) '.format(average_loss)
    string += 'AUC {:.4f} '.format(auc)
    string += 'AUPR {:.4f} \t'.format(aupr)
    string += 'Batch Time {:.4f} '.format(batch_time)
    string += '(Average {:.4f}) \t'.format(average_time)

    summary += string

    print(summary)


def save_checkpoint(state, save_path):
    '''
        Save the current model.
        If the model is the best model since beginning of the training
        it will be copy
    '''

    print('Saving to ', save_path)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']  # epoch no
    best_model = state['best_model']  # bool
    val_loss = state['val_loss']  # loss value
    model = state['model']  # model type
    loss = state['loss']  # loss name

    if best_model:
        filename = save_path + '/' + \
                   'best_model.{}--{}.pth.tar'.format(loss, model)
    else:
        filename = save_path + '/' + \
                   'model.{}--{}--{:02d}.pth.tar'.format(loss, model, epoch)

    torch.save(state, filename)


def weighted_BCELoss(output, target, weights=[5, 1]):
    output = output.clamp(min=1e-5, max=1 - 1e-5)
    if weights is not None:
        assert len(weights) == 2

        loss = -weights[0] * (target * torch.log(output)) - weights[1] * ((1 - target) * torch.log(1 - output))
    else:
        loss = -target * torch.log(output) - (1 - target) * torch.log(1 - output)

    return torch.mean(loss)


def dice_loss(input, target):
    smooth = 1.
    target = target.float()
    input = input.float()
    input_flat = input.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (input_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) /
                (input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth))


def mean_dice_loss(input, target):
    channels = list(range(target.shape[1]))
    loss = 0
    for channel in channels:
        dice = dice_loss(input[:, channel, ...],
                         target[:, channel, ...])
        loss += dice

    return loss / len(channels)


def train_loop(loader, model, criterion, optimizer, writer, epoch, lr_scheduler=None, model_type='UNet'):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0
    train_aupr, train_auc = 0.0, 0.0  # train_auc is the train area under the ROC curve
    aupr_sum, auc_sum = 0.0, 0.0

    auprs = []
    for (i, sample) in enumerate(loader, 1):
        images, masks = sample['image'], sample['masks']
        try:
            loss_name = criterion._get_name()
        except AttributeError:
            loss_name = criterion.__name__

        # Take variable and put them to GPU
        model.to(device)
        images, masks = to_var(images.float(), device), to_var(masks.float(), device)

        # compute output
        output = model(images)

        # Compute loss
        if model_type == 'UNet':
            if loss_name == 'BCELoss':
                # flattening if using BCELoss
                loss = criterion(output.view(images.shape[0], -1), masks.view(images.shape[0], -1))
            else:
                loss = criterion(output, masks)  # Loss

        elif model_type == 'GCN':
            weights = [5, 1]
            if use_cuda:
                weights = torch.FloatTensor(weights).cuda()

            output = torch.sigmoid(output)  # apply last sigmoid activation for GCN model
            loss = criterion(output, masks, weights=weights)  # weighted BCE Loss

        # compute gradient and do SGD step
        if model.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Compute AUC scores
        # pred_masks = torch.argmax(output, dim=1, keepdim=True)
        if loss_name == 'BCELoss':
            thresh = Variable(torch.Tensor([0.1]))  # threshold
            pred_masks = (output > thresh).float()
        else:
            pred_masks = output

        train_auc = auc_on_batch(masks, pred_masks)
        train_aupr = aupr_on_batch(masks, pred_masks)
        auprs.append(train_aupr)

        # measure elapsed time
        batch_time = time.time() - end

        time_sum += batch_size * batch_time
        loss_sum += batch_size * loss
        average_loss = loss_sum / (i * batch_size)
        average_time = time_sum / (i * batch_size)

        end = time.time()
        torch.cuda.empty_cache()

        if i % print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), loss, loss_name, batch_time,
                          average_loss, average_time, train_auc, train_aupr, logging_mode)
        if tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_' + loss_name, loss.item(), step)

            # plot metrics in tensorboard
            writer.add_scalar(logging_mode + '_auc', train_auc, step)
            writer.add_scalar(logging_mode + '_aupr', train_aupr, step)

        torch.cuda.empty_cache()

    if tensorboard:
        # plots the last results images, predicted mask and ground truth
        n = images.shape[0]
        # images = to_numpy(images)
        masks = to_numpy(masks)

        for batch in range(n):
            fig = plot(images[batch],
                       masks[batch], pred_masks[batch])

            writer.add_figure(logging_mode + str(batch),
                              fig, epoch)

    if lr_scheduler is not None:
        lr_scheduler.step(loss)

    mean_aupr = np.mean(auprs)
    return loss, mean_aupr


def main_loop(data_path, batch_size=batch_size, model_type='UNet', green=False, tensorboard=True):
    # Load train and val data
    tasks = ['EX']
    data_path = data_path
    n_labels = len(tasks)
    n_channels = 1 if green else 3  # green or RGB
    train_loader, val_loader = load_train_val_data(tasks=tasks, data_path=data_path, batch_size=batch_size, green=green)

    if model_type == 'UNet':
        lr = learning_rate
        model = UNet(n_channels, n_labels)
        # Choose loss function
        criterion = nn.MSELoss()
        # criterion = dice_loss
        # criterion = mean_dice_loss
        # criterion = nn.BCELoss()


    elif model_type == 'GCN':
        lr = 1e-4
        model = GCN(n_labels, image_size[0])
        criterion = weighted_BCELoss
        # criterion = nn.BCELoss()

    else:
        raise TypeError('Please enter a valid name for the model type')

    try:
        loss_name = criterion._get_name()
    except AttributeError:
        loss_name = criterion.__name__

    if loss_name == 'BCEWithLogitsLoss':
        lr = 1e-4
        print('learning rate: ', lr)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Choose optimize
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=7)

    if tensorboard:
        log_dir = tensorboard_folder + session_name + '/'
        print('log dir: ', log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
    else:
        writer = None

    max_aupr = 0.0
    for epoch in range(epochs):  # loop over the dataset multiple times
        print('******** Epoch [{}/{}]  ********'.format(epoch + 1, epochs + 1))
        print(session_name)

        # train for one epoch
        model.train(True)
        print('Training with batch size : ', batch_size)
        train_loop(train_loader, model, criterion, optimizer, writer, epoch,
                   lr_scheduler=lr_scheduler,
                   model_type=model_type)

        # evaluate on validation set
        print('Validation')
        with torch.no_grad():
            model.eval()
            val_loss, val_aupr = train_loop(val_loader, model, criterion,
                                            optimizer, writer, epoch)

        # Save best model
        if val_aupr > max_aupr and epoch > 3:
            print('\t Saving best model, mean aupr on validation set: {:.4f}'.format(val_aupr))
            max_aupr = val_aupr
            save_checkpoint({'epoch': epoch,
                             'best_model': True,
                             'model': model_type,
                             'state_dict': model.state_dict(),
                             'val_loss': val_loss,
                             'loss': loss_name,
                             'optimizer': optimizer.state_dict()}, model_path)

        elif save_model and (epoch + 1) % save_frequency == 0:
            save_checkpoint({'epoch': epoch,
                             'best_model': False,
                             'model': model_type,
                             'loss': loss_name,
                             'state_dict': model.state_dict(),
                             'val_loss': val_loss,
                             'optimizer': optimizer.state_dict()}, model_path)

    return model


if __name__ == '__main__':
    ## PARAMS
    ## You should create the following save folder and tensorboard folder if the mkdirs command fails to create a folder

    main_path = 'Segmentation.nosync/' # folder with test_images folder and test_masks folder (containing 'EX' folder ...)
    sets_path = os.path.join(main_path, 'datasets/')
    csv_path = os.path.join(main_path, 'data/tumor_count.csv')
    data_folder = os.path.join(main_path, 'data/')
    save_path = 'save/'
    loss_function = 'dice_loss'
    session_name = 'Test_session' + '_' + time.strftime('%m.%d %Hh%M')
    model_path = save_path + 'models/' + session_name + '/'
    tensorboard_folder = save_path + 'tensorboard_logs/'

    model = main_loop(data_path=main_path, green=True, tensorboard=True)

