import os
import time
import math
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from loss_metrics import *
from utils import CSVWriter
from model import GraspAffordanceNet, ResNetUNet
from dataset import RGBDGraspAffordanceDataset

def validation_loop(dataset_loader, model, device):
    bce_loss = get_bce_loss()
    size = len(dataset_loader.dataset)
    num_batches = len(dataset_loader)
    valid_loss, valid_acc, valid_iou = 0, 0, 0
    model.eval()

    with torch.no_grad():
        for input_color, label in dataset_loader:
            #input_color, input_depth, label = input_color.cuda(), input_depth.cuda(), label.cuda()
            input_color, label = input_color.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
            pred_logits = model(input_color)#, input_depth.float())

            pred_probs = torch.sigmoid(pred_logits)
            pred_label = pred_probs > 0.5

            #valid_loss += (bce_loss(pred, label.float()) + compute_dice_loss(pred, label.float()))
            valid_loss += bce_loss(pred_logits, label)#, label.float())
            valid_acc += compute_mean_pixel_acc(pred_label, label)
            valid_iou += compute_mean_iou(pred_label, label)

    valid_loss /= num_batches
    valid_acc /= num_batches
    valid_iou /= num_batches
    return valid_loss, valid_acc, valid_iou

def train_loop(dataset_loader, model, optimizer, device):
    bce_loss = get_bce_loss()
    size = len(dataset_loader.dataset)
    num_batches = len(dataset_loader)
    train_loss = 0
    model.train()

    for input_color, label in dataset_loader:
        # Compute prediction and loss
        #(input_color, input_depth), label = data
        input_color, label = input_color.to(device, dtype=torch.float), label.to(device, dtype=torch.float)
        optimizer.zero_grad()
        #print(input_color.shape, input_color.dtype)
        #print(input_depth.shape, input_depth.dtype)
        #print(label.shape, label.dtype)
        pred = model(input_color)#.float())#, input_depth.float())
        loss_1 = bce_loss(pred, label)#.float())
        #loss_1 = 0
        #loss_2 = compute_dice_loss(pred, label.float())
        loss = loss_1# + loss_2

        # Backpropagation
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= num_batches
    return train_loss

def batch_train(FLAGS):
    os.makedirs(FLAGS.dir_model)
    csv_writer = CSVWriter(file_name=os.path.join(FLAGS.dir_model, "train_metrics.csv"),
        column_names=["epoch", "train_loss", "valid_loss", "valid_acc", "valid_iou"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RGBDGraspAffordanceDataset(FLAGS.dir_train_dataset,
        transforms.Compose([transforms.ToTensor()]))
    train_dataset_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    valid_dataset = RGBDGraspAffordanceDataset(FLAGS.dir_valid_dataset,
        transforms.Compose([transforms.ToTensor()]))
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    #grasp_aff_model = GraspAffordanceNet(pretrained=bool(FLAGS.pretrained))
    grasp_aff_model = ResNetUNet(1)
    grasp_aff_model.to(device)
    for l in grasp_aff_model.base_layers:
        for param in l.parameters():
            param.requires_grad = False

    #"""
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, grasp_aff_model.parameters()), lr=FLAGS.learning_rate,
        momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    """
    optimizer = torch.optim.Adam(grasp_aff_model.parameters(), lr=FLAGS.learning_rate,
        weight_decay=0)
    """
    #print(grasp_aff_model.parameters())
    print("Training for Grasp part affordance prediction")
    for epoch in range(FLAGS.num_epochs):
        t_1 = time.time()
        train_loss = train_loop(train_dataset_loader, grasp_aff_model, optimizer, device)
        t_2 = time.time()
        print("-"*100)
        print(f"Epoch : {epoch+1}/{FLAGS.num_epochs}, train loss: {train_loss:.4f}, time: {(t_2-t_1):.2f} sec.")
        valid_loss, valid_acc, valid_iou = validation_loop(valid_dataset_loader, grasp_aff_model, device)
        print(f"validation loss: {valid_loss:.4f}, validation accuracy: {valid_acc:.4f}, validation iou: {valid_iou:.4f}")
        csv_writer.write_row([epoch+1, train_loss.cpu().detach().numpy(), valid_loss.cpu().detach().numpy(), valid_acc, valid_iou])
        torch.save(grasp_aff_model.state_dict(), os.path.join(FLAGS.dir_model, f"{FLAGS.file_model}_{epoch+1}.pt"))
        exp_lr_scheduler.step()
    print("Training complete!!!!")
    csv_writer.close()

def main():
    dir_train_dataset = "/home/abhishek/Desktop/cognitive_robotics_lab/part_affordance_subset/clutter/clutter_train/"
    dir_valid_dataset = "/home/abhishek/Desktop/cognitive_robotics_lab/part_affordance_subset/clutter/clutter_valid/"
    learning_rate = 3e-4
    num_epochs = 80
    batch_size = 4
    dir_model = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_model = "affordance_net"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dir_train_dataset", default=dir_train_dataset,
        type=str, help="directory with train dataset")
    parser.add_argument("--dir_valid_dataset", default=dir_valid_dataset,
        type=str, help="directory with validation dataset")

    parser.add_argument("--pretrained", default=1,
        type=int, choices=[0, 1], help="use pretrained encoder (1:True, 0:False)")
    parser.add_argument("--use_cuda", default=1, type=int,
        choices=[0, 1], help="use gpu for training (1:True, 0:False)")

    parser.add_argument("--learning_rate", default=learning_rate,
        type=float, help="learning rate")
    parser.add_argument("--num_epochs", default=num_epochs,
        type=int, help="number of epochs to train")
    parser.add_argument("--batch_size", default=batch_size,
        type=int, help="number of samples in a batch")

    parser.add_argument("--dir_model", default=dir_model,
        type=str, help="directory to save the model")
    parser.add_argument("--file_model", default=file_model,
        type=str, help="file name to save the model")

    FLAGS, unparsed = parser.parse_known_args()
    batch_train(FLAGS)

if __name__ == "__main__":
    main()
