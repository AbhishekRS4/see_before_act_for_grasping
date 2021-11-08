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
from torch.utils.data import DataLoader

from loss_metrics import *
from utils import CSVWriter
from model import GraspAffordanceNet
from dataset import RGBDGraspAffordanceDataset

def validation_loop(dataset_loader, model):
    size = len(dataset_loader.dataset)
    num_batches = len(dataset_loader)
    valid_loss, valid_acc, valid_iou = 0, 0, 0

    with torch.no_grad():
        for x, label in dataset_loader:
            input_color, input_depth = x
            pred = model(input_color.float(), input_depth.float())

            pred_probs = torch.sigmoid(pred)
            pred_label = pred_probs > 0.5

            valid_loss += compute_bce_loss(pred, label.float())
            valid_acc += compute_mean_pixel_acc(pred_label, label)
            valid_iou += compute_mean_iou(pred_label, label)

    valid_loss /= num_batches
    valid_acc /= num_batches
    valid_iou /= num_batches
    return valid_loss, valid_acc, valid_iou

def train_loop(dataset_loader, model, optimizer):
    size = len(dataset_loader.dataset)
    num_batches = len(dataset_loader)
    train_loss = 0

    for batch, data in enumerate(dataset_loader):
        # Compute prediction and loss
        (input_color, input_depth), label = data
        #print(input_color.shape, input_color.dtype)
        #print(input_depth.shape, input_depth.dtype)
        #print(label.shape, label.dtype)
        pred = model(input_color.float(), input_depth.float())
        loss = compute_bce_loss(pred, label.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss /= num_batches
    return train_loss#.detach().cpu().numpy()

def batch_train(FLAGS):
    os.makedirs(FLAGS.dir_model)
    csv_writer = CSVWriter(file_name=os.path.join(FLAGS.dir_model, "train_metrics.csv"),
        column_names=["epoch", "train_loss", "valid_loss", "valid_acc", "valid_iou"])

    train_dataset = RGBDGraspAffordanceDataset(FLAGS.dir_train_dataset,
        transforms.Compose([transforms.ToTensor()]))
    train_dataset_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    valid_dataset = RGBDGraspAffordanceDataset(FLAGS.dir_valid_dataset,
        transforms.Compose([transforms.ToTensor()]))
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=FLAGS.batch_size, shuffle=False)

    grasp_aff_model = GraspAffordanceNet()
    optimizer = torch.optim.SGD(grasp_aff_model.parameters(), lr=FLAGS.learning_rate,
        momentum=0.9, weight_decay=2e-5)

    print("Training for Grasp part affordance prediction")
    for epoch in range(FLAGS.num_epochs):
        t_1 = time.time()
        train_loss = train_loop(train_dataset_loader, grasp_aff_model, optimizer)
        t_2 = time.time()
        print("-"*100)
        print(f"Epoch : {epoch+1}/{FLAGS.num_epochs}, train loss: {train_loss:.4f}, time: {(t_2-t_1):.2f} sec.")
        valid_loss, valid_acc, valid_iou = validation_loop(valid_dataset_loader, grasp_aff_model)
        print(f"validation loss: {valid_loss:.4f}, validation accuracy: {valid_acc:.4f}, validation iou: {valid_iou:.4f}")
        csv_writer.write_row([epoch+1, train_loss, valid_loss, valid_acc, valid_iou])
        torch.save(grasp_aff_model.state_dict(), os.path.join(FLAGS.dir_model, f"{FLAGS.file_model}_{epoch+1}.pt"))
    print("Training complete!!!!")
    csv_writer.close()

def main():
    dir_train_dataset = "/home/abhishek/Desktop/cognitive_robotics_lab/part-affordance-clutter/clutter/scene_cumulative_train/"
    dir_valid_dataset = "/home/abhishek/Desktop/cognitive_robotics_lab/part-affordance-clutter/clutter/scene_cumulative_valid/"
    pretrained = True
    learning_rate = 1e-4
    num_epochs = 25
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
    parser.add_argument("--use_cuda", default=0, type=int,
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
