import os
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from model import GraspAffordanceNet
from dataset import RGBDGraspAffordanceDataset

def compute_bce_loss(pred, label):
    bce_loss = F.binary_cross_entropy_with_logits(pred, label)

    return bce_loss

def train_loop(dataset_loader, model, optimizer):
    size = len(dataset_loader.dataset)
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

        #f batch % 100 == 0:
        loss, current = loss.item(), batch * len(input_color)
        print(f"loss: {loss:.4f}, batch: [{current}/{size}]")

def batch_train(FLAGS):
    os.makedirs(FLAGS.dir_model)

    train_dataset = RGBDGraspAffordanceDataset(FLAGS.dir_dataset,
        transforms.Compose([transforms.ToTensor()]))
    train_dataset_loader = DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=False)
    grasp_aff_model = GraspAffordanceNet()
    optimizer = torch.optim.SGD(grasp_aff_model.parameters(), lr=FLAGS.learning_rate,
        momentum=0.9, weight_decay=2e-5)

    for epoch in range(FLAGS.num_epochs):
        print(f"Epoch : {epoch+1}/{FLAGS.num_epochs}")
        train_loop(train_dataset_loader, grasp_aff_model, optimizer)
        torch.save(grasp_aff_model.state_dict(), os.path.join(FLAGS.dir_model, f"{FLAGS.file_model}_{epoch+1}.pt"))
        print("-"*20)

def main():
    dir_dataset = "/home/abhishek/Desktop/cognitive_robotics_lab/part-affordance-clutter/clutter/scene_cumulative/"
    pretrained = True
    learning_rate = 1e-4
    num_epochs = 25
    batch_size = 4
    dir_model = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_model = "affordance_net"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dir_dataset", default=dir_dataset,
        type=str, help="directory with validation labels")

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
