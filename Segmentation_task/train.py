import DataSet
from UNet import UNet
import time
import numpy as np
import torch
import os

## PARAMS
main_path = 'Segmentation.nosync/'
sets_path = os.path.join(main_path, 'datasets/')
csv_path = os.path.join(main_path, 'data/tumor_count.csv')
data_folder = os.path.join(main_path, 'data/')
save_path = '/content/save/'
loss_function = 'dice_loss'
session_name = 'Test_session' + '_' + time.strftime('%m.%d %Hh%M')
model_path = save_path + 'models/' + session_name + '/'
tensorboard_folder = save_path + 'tensorboard_logs/'

## PARAMETERS OF THE MODEL
learning_rate = 1e-4
image_size = (640, 640)
n_labels = 2
epochs = 10
batch_size = 5
print_frequency = 5
save_frequency = 10
save_model = True
tumor_percentage = 0.5
tensorboard = False


def print_summary(epoch, i, nb_batch, loss, batch_time,
                  average_loss, average_time, mode):
    '''
        mode = Train or Test
    '''
    summary = '[' + str(mode) + '] Epoch: [{0}][{1}/{2}]\t'.format(
        epoch, i, nb_batch)

    string = ''
    string += 'Dice Loss {:.4f} '.format(loss)
    string += '(Average {:.4f}) \t'.format(average_loss)
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

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    epoch = state['epoch']
    val_loss = state['val_loss']
    filename = save_path + '/' + \
               'model.{:02d}--{:.3f}.pth.tar'.format(epoch, val_loss)
    torch.save(state, filename)


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


def train_loop(loader, model, criterion, optimizer, writer, epoch):
    logging_mode = 'Train' if model.training else 'Val'

    end = time.time()
    time_sum, loss_sum = 0, 0

    for i, sample in enumerate(loader, 1):
        # Take variable and put them to GPU
        images, masks = sample['image'], sample['masks']
        # print(irms.shape) # Batch * Modality * Width * Height

        # compute output
        pred_masks = model(images)

        # compute loss
        dice_loss = criterion(pred_masks, masks)

        # compute gradient and do SGD step
        if model.training:
            optimizer.zero_grad()
            dice_loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time = time.time() - end

        time_sum += batch_size * batch_time
        loss_sum += batch_size * dice_loss
        average_loss = loss_sum / (i * batch_size)
        average_time = time_sum / (i * batch_size)

        end = time.time()

        if i % print_frequency == 0:
            print_summary(epoch + 1, i, len(loader), dice_loss, batch_time,
                          average_loss, average_time, logging_mode)
        if tensorboard:
            step = epoch * len(loader) + i
            writer.add_scalar(logging_mode + '_dice', dice_loss.item(), step)

    if tensorboard:
        n = images.shape[0]
        irms = np.array(images)
        masks = np.array(masks)
        pred_masks = np.array(pred_masks)

        for batch in range(n):
            fig = plot(images[batch, ...],
                       masks[batch, ...], pred_masks[batch, ...])

            writer.add_figure(logging_mode + str(batch),
                              fig, epoch)

    return dice_loss


def main_loop():
    # Load train and val data
    tasks = ['EX', 'MA']
    data_path = 'Segmentation.nosync/'
    n_labels = len(tasks)
    n_channels = 3
    train_loader, val_loader = DataSet.load_train_val_data(tasks=tasks, data_path=data_path)

    model = UNet(n_channels, n_labels)
    criterion = mean_dice_loss  # Choose loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Choose optimize

    if tensorboard:
        log_dir = tensorboard_folder + session_name + '/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    else:
        writer = None

    for epoch in range(epochs):  # loop over the dataset multiple times
        print('******** Epoch [{}/{}]  ********'.format(epoch + 1, epochs + 1))
        print(session_name)

        # train for one epoch
        model.train()
        print('Training')
        train_loop(train_loader, model, criterion, optimizer, writer, epoch)

        # evaluate on validation set
        print('Validation')
        with torch.no_grad():
            model.eval()
            val_loss = train_loop(val_loader, model, criterion,
                                  optimizer, writer, epoch)

        if save_model and epoch % save_frequency == 0:
            save_checkpoint({'epoch': epoch,
                             'state_dict': model.state_dict(),
                             'val_loss': val_loss,
                             'optimizer': optimizer.state_dict()}, model_path)


if __name__ == '__main__':
    main_loop()
    # train_loader, val_loader = DataSet.load_train_val_data()
    #
    # # Load train and val data
    # tasks = ['EX', 'MA']
    # data_path = 'Segmentation.nosync/'
    # n_labels = len(tasks)
    # n_channels = 3
    # train_loader, val_loader = DataSet.load_train_val_data(tasks=tasks, data_path=data_path)
    #
    # model = UNet(n_channels, n_labels)
    # criterion = mean_dice_loss  # Choose loss function
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Choose optimize
