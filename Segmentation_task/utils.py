import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

def tensor_to_image(x):
    '''Returns an array of shape CxHxW from a given tensor with shape HxWxC'''

    x = np.rollaxis(x.int().detach().cpu().numpy(), 0, 3)
    return x


def plot(image, masks=None, pred_masks=None):
    '''plots for a given image the ground truth mask and the corresponding predicted mask
      masks: tensor of shape (n_tasks, 512, 512)
    '''
    fig, ax = plt.subplots(1, 3, gridspec_kw={'wspace': 0.15, 'hspace': 0.2,
                                              'top': 0.85, 'bottom': 0.1,
                                              'left': 0.05, 'right': 0.95})

    ax[0].imshow(image.int().detach().cpu().numpy()[0])
    # ax[0].imshow(tensor_to_image(image))
    ax[0].axis('off')

    if masks is not None:
        # masks = np.argmax(masks, axis=0)
        ax[1].imshow(masks[0], cmap='gray')
        ax[1].axis('off')

    if pred_masks is not None:
        # pred_masks = np.argmax(pred_masks, axis=0)
        # Thresholding mask
        thresh = 0.1
        prediction = pred_masks[0].detach().cpu().numpy()
        max_prob = np.max(prediction)
        img_pred = np.zeros(prediction.shape)
        img_pred[prediction >= thresh * max_prob] = 1
        ax[2].imshow(img_pred, cmap='gray')
        ax[2].axis('off')

    ax[0].set_title('Original Image')
    ax[1].set_title('Ground Truth Seg EX')
    ax[2].set_title('Predicted Seg EX')

    fig.canvas.draw()

    return fig


def AUPR(mask, prediction):
    '''Computes the Area under Precision-Recall Curve for a given ground-truth mask and predicted mask'''
    threshold_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # list of thresholds
    precisions = []
    recalls = []

    for thresh in threshold_list:
        # thresholding the predicted mask
        thresh_pred = np.zeros(prediction.shape)
        thresh_pred[prediction >= thresh] = 1

        # Computing the True and False Positives
        P = np.count_nonzero(mask)
        TP = np.count_nonzero(mask * thresh_pred)
        FP = np.count_nonzero(thresh_pred - (mask * thresh_pred))

        if (P > 0) and (TP + FP > 0):  # avoid division by 0
            precision = TP * 1.0 / (TP + FP)
            recall = TP * 1.0 / P
        else:
            precision = 1
            recall = 0

        precisions.append(precision)
        recalls.append(recall)

    return auc(recalls, precisions)


def aupr_on_batch(masks, pred):
    '''Computes the mean AUPR over a batch during training'''
    auprs = []
    for i in range(pred.shape[0]):
        prediction = pred[i][0].cpu().detach().numpy()
        mask = masks[i].cpu().detach().numpy()
        auprs.append(AUPR(mask, prediction))

    return np.mean(auprs)


def auc_on_batch(masks, pred):
    '''Computes the mean Area Under ROC Curve over a batch during training'''
    aucs = []
    for i in range(pred.shape[0]):
        prediction = pred[i][0].cpu().detach().numpy()
        mask = masks[i].cpu().detach().numpy()
        aucs.append(roc_auc_score(mask.reshape(-1), prediction.reshape(-1)))
    return np.mean(aucs)
