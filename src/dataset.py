import os
import numpy as np
from scipy.io import loadmat
from skimage.io import imread

import torch
from torch.utils.data import Dataset

class RGBDGraspAffordanceDataset(Dataset):
    def __init__(self, dir_dataset, transform=None):
        self.dir_dataset = dir_dataset

        self.depth_images_list = [f for f in os.listdir(self.dir_dataset) if f.endswith(".png")]
        self.color_images_list = [f for f in os.listdir(self.dir_dataset) if f.endswith(".jpg")]
        self.labels_list = [f for f in os.listdir(self.dir_dataset) if f.endswith("label.mat")]

        self.color_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3)
        self.color_std = np.array([0.229, 0.224, 0.225]).reshape(1, 3)

        self.depth_mean = np.array([0.01, 0.01, 0.01]).reshape(1, 3)
        self.depth_std = np.array([0.03, 0.03, 0.03]).reshape(1, 3)

        self.transform = transform

    def __len__(self):
        return len(self.depth_images_list)

    def __getitem__(self, idx):
        label = loadmat(os.path.join(self.dir_dataset, self.labels_list[idx]))["gt_label"]
        label = (label == 1).astype(np.int32)
        label = np.expand_dims(label, 0)

        color_image = imread(os.path.join(self.dir_dataset, self.color_images_list[idx]))
        color_image = color_image.astype(np.float32) / 255.
        color_image = (color_image - self.color_mean) / self.color_std

        depth_image = imread(os.path.join(self.dir_dataset, self.depth_images_list[idx]))
        depth_image = np.repeat(np.expand_dims(depth_image, -1), 3, -1).astype(np.float32)
        depth_image = (depth_image - self.depth_mean) / self.depth_std

        if self.transform:
            color_image = self.transform(color_image)
            depth_image = self.transform(depth_image)

        return (color_image, depth_image), label
