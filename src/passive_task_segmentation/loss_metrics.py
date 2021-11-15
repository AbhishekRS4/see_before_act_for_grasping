import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# compute dice loss
def compute_dice_loss(pred, label, smooth=1e-5):
    pred = pred.contiguous()
    label = label.contiguous()

    pred_probs = torch.sigmoid(pred)
    pred_probs = pred_probs.contiguous()

    intersection = (pred_probs * label).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred_probs.sum(dim=2).sum(dim=2) + label.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

# compute binary cross entropy loss
def get_bce_loss():
    bce_loss = torch.nn.BCEWithLogitsLoss()

    return bce_loss

# compute mean pixel accuracy
def compute_mean_pixel_acc(pred, label):
    if label.shape != pred.shape:
        print("label has dimension", label.shape, ", pred values have shape", pred.shape)
        return

    if label.dim() != 4:
        print("label has dim", label.dim(), ", Must be 4.")
        return

    acc_sum = 0
    for i in range(label.shape[0]):
        label_arr = label[i, :, :, :].clone().detach().cpu().numpy()
        pred_arr = pred[i, :, :, :].clone().detach().cpu().numpy()
        pred_arr = pred_arr.astype(np.int32)

        same = (label_arr == pred_arr).sum()

        _, a, b = label_arr.shape
        total = a*b

        acc_sum += same / total

    mean_pixel_accuracy = acc_sum / label.shape[0]
    return mean_pixel_accuracy

# compute mean iou (intersection over union)
def compute_mean_iou(pred, label, smooth=1e-5):
    if label.shape != pred.shape:
        print("label has dimension", label.shape, ", pred values have shape", pred.shape)
        return

    if label.dim() != 4:
        print("label has dim", label.dim(), ", Must be 4.")
        return

    iou_sum = 0
    for i in range(label.shape[0]):
        label_arr = label[i, :, :, :].clone().detach().cpu().numpy()
        pred_arr = pred[i, :, :, :].clone().detach().cpu().numpy()
        pred_arr = pred_arr.astype(np.int32)

        intersection = np.logical_and(label_arr, pred_arr).sum() + smooth
        union = np.logical_or(label_arr, pred_arr).sum() + smooth
        iou_score = intersection / union
        iou_sum += iou_score
        print(f"{i}/{label.shape[0]}, pred 1 count : {pred_arr.sum()}, label 1 count : {label_arr.sum()}, intersection : {intersection:.3f}, union : {union:.3f}, iou_score : {iou_score:.3f}")

    mean_iou = iou_sum / label.shape[0]
    return mean_iou
