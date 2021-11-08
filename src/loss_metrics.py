import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_dice_loss(pred, label):
    pred = pred.contiguous()
    label = label.contiguous()

    intersection = (pred * label).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + label.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def compute_bce_loss(pred, label):
    bce_loss = F.binary_cross_entropy_with_logits(pred, label)

    return bce_loss

def compute_mean_pixel_acc(pred, label):
    if label.shape != pred.shape:
        print("label has dimension", label.shape, ", pred values have shape", pred.shape)
        return

    if label.dim() != 4:
        print("label has dim", label.dim(), ", Must be 4.")
        return

    acc_sum = 0
    for i in range(label.shape[0]):
        label_arr = label[i, :, :, :].clone().detach().cpu().numpy()#.argmax(0)
        pred_arr = pred[i, :, :, :].clone().detach().cpu().numpy()#.argmax(0)
        pred_arr = pred_arr.astype(np.int32)

        #print(label_arr)
        #print(pred_arr)
        #print(np.unique(pred_arr), np.sum(pred_arr))
        #print(np.unique(label_arr), np.sum(label_arr))
        #print(label_arr.shape, label_arr.dtype)
        #print(pred_arr.shape, pred_arr.dtype)
        same = (label_arr == pred_arr).sum()
        #print(same)
        _, a, b = label_arr.shape
        total = a*b
        #print(total)
        acc_sum += same / total

    mean_pixel_accuracy = acc_sum / label.shape[0]
    return mean_pixel_accuracy

def compute_mean_iou(pred, label, smooth=1e-5):
    if label.shape != pred.shape:
        print("label has dimension", label.shape, ", pred values have shape", pred.shape)
        return

    if label.dim() != 4:
        print("label has dim", label.dim(), ", Must be 4.")
        return

    iou_sum = 0
    for i in range(label.shape[0]):
        label_arr = label[i, :, :, :].clone().detach().cpu().numpy()#.argmax(0)
        pred_arr = pred[i, :, :, :].clone().detach().cpu().numpy()#.argmax(0)
        pred_arr = pred_arr.astype(np.int32)

        intersection = np.logical_and(label_arr, pred_arr).sum()
        union = np.logical_or(label_arr, pred_arr).sum()
        iou_score = (intersection + smooth) / (union + smooth)
        iou_sum += iou_score

    mean_iou = iou_sum / label.shape[0]
    return mean_iou
