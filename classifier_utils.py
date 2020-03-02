import numpy as np
import torch
import pandas as pd
from copy import deepcopy


def train(model, train_loader, criterion, optimizer, n_epochs, device):
    """
    Method used to train our classifier

    Args:
        model: (nn.Module) the neural network
        train_loader: (DataLoader) a DataLoader wrapping a MRIDataset
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images
        optimizer: (torch.optim) an optimization algorithm
        n_epochs: (int) number of epochs performed during training

    Returns:
        best_model: (nn.Module) the trained neural network
    """
    best_model = deepcopy(model)
    train_best_loss = np.inf

    for epoch in range(n_epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            # load batch
            images, labels = data['image'], data['labels']
            images = images.to(device)
            # - forward pass
            output = model.forward(images)
            # - loss computation
            loss = criterion(output, labels)
            # - backward pass (gradients computation)
            loss.backward()
            # - weights update
            optimizer.step()
            # - gradients set to 0
            optimizer.zero_grad()

        _, train_metrics = test(model, train_loader, criterion, test, device)

        print('Epoch %i: loss = %f, balanced accuracy = %f'
              % (epoch, train_metrics['mean_loss'],
                 train_metrics['balanced_accuracy']))

        if train_metrics['mean_loss'] < train_best_loss:
            best_model = deepcopy(model)
            train_best_loss = train_metrics['mean_loss']

    return best_model


def test(model, data_loader, criterion, device):
    """
    Method used to test a CNN

    Args:
        model: (nn.Module) the model
        data_loader: (DataLoader) a DataLoader
        criterion: (nn.Module) a method to compute the loss of a mini-batch of images

    Returns:
        results_df: (DataFrame) the label predicted for every subject
        results_metrics: (dict) a set of metrics
    """
    model.eval()
    columns = ["img_idx", "proba",
               "true_label", "predicted_label"]
    results_df = pd.DataFrame(columns=columns)
    total_loss = 0

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            images, labels = data['image'].to(device), data['labels'].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            proba = torch.nn.Softmax(dim=1)(outputs)
            predictions, predicted_labels = torch.max(outputs.data, dim=1).values, torch.max(outputs.data, dim=1).indices
            for idx, _ in enumerate(labels):
                row = [idx, predictions[idx], labels[idx], predicted_labels[idx]]
                row_df = pd.DataFrame([row], columns=columns)
                results_df = pd.concat([results_df, row_df])

    results_metrics = compute_metrics(results_df.true_label.values, results_df.predicted_label.values)
    results_df.reset_index(inplace=True, drop=True)
    results_metrics['mean_loss'] = total_loss / len(data_loader.dataset)

    return results_df, results_metrics



def compute_metrics(ground_truth, prediction):
    """Computes the accuracy, sensitivity, specificity and balanced accuracy"""
    tp = np.sum((prediction == 1) & (ground_truth == 1))
    tn = np.sum((prediction == 0) & (ground_truth == 0))
    fp = np.sum((prediction == 1) & (ground_truth == 0))
    fn = np.sum((prediction == 0) & (ground_truth == 1))

    metrics_dict = dict()
    metrics_dict['accuracy'] = (tp + tn) / (tp + tn + fp + fn)

    # Sensitivity
    if tp + fn != 0:
        metrics_dict['sensitivity'] = tp / (tp + fn)
    else:
        metrics_dict['sensitivity'] = 0.0

    # Specificity
    if fp + tn != 0:
        metrics_dict['specificity'] = tn / (fp + tn)
    else:
        metrics_dict['specificity'] = 0.0

    metrics_dict['balanced_accuracy'] = (metrics_dict['sensitivity'] + metrics_dict['specificity']) / 2

    return metrics_dict