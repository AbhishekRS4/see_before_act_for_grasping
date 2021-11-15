import os
import sys
import time
import argparse
import numpy as np
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GraspAffordanceNet

def argument_parser():
    dir_test_dataset = "/home/abhishek/Desktop/cognitive_robotics_lab/part_affordance_subset/clutter/clutter_test/"
    model_checkpoint = "/home/abhishek/Desktop/cognitive_robotics_lab/object_grasp_affordance/2021-11-08_23-43-38/affordance_net_50.pt"
    dir_predictions = "../dir_predictions"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dir_test_dataset", default=dir_test_dataset,
        type=str, help="directory with test dataset to be used for inference")

    parser.add_argument("--pretrained", default=1,
        type=int, choices=[0, 1], help="use pretrained encoder (1:True, 0:False)")
    parser.add_argument("--use_cuda", default=1, type=int,
        choices=[0, 1], help="use gpu for training (1:True, 0:False)")

    parser.add_argument("--dir_predictions", default=dir_predictions,
        type=str, help="full directory path to save the predictions")
    parser.add_argument("--model_checkpoint", default=model_checkpoint,
        type=str, help="full path to load checkpoint model")

    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS

def run_inference(model, input_color_data, input_depth_data):
    t_1 = time.time()
    pred_logits = model(input_color_data, input_depth_data)
    pred_probs = torch.sigmoid(pred_logits)
    pred_label = pred_probs > 0.5
    t_2 = time.time()

    print(f"inference time : {((t_2-t_1)*1000):.3f} milli sec.")

    return pred_probs.detach().cpu().numpy(), pred_label.detach().cpu().numpy()

def batch_inference(FLAGS):
    if not os.path.isdir(FLAGS.dir_predictions):
        os.makedirs(FLAGS.dir_predictions)
        print(f"Created directory : {FLAGS.dir_predictions}")

    # mean and std for color images
    color_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3)
    color_std = np.array([0.229, 0.224, 0.225]).reshape(1, 3)

    # mean and std for depth images
    depth_mean = np.array([0.01, 0.01, 0.01]).reshape(1, 3)
    depth_std = np.array([0.03, 0.03, 0.03]).reshape(1, 3)

    # init model and set it in evaluation mode
    grasp_aff_model = GraspAffordanceNet(pretrained=True)
    grasp_aff_model.eval()
    grasp_aff_model.load_state_dict(torch.load(FLAGS.model_checkpoint))

    if FLAGS.use_cuda:
        if torch.cuda.device_count() > 0:
            grasp_aff_model.cuda()

    list_color_images = sorted([c for c in os.listdir(FLAGS.dir_test_dataset) if c.endswith("rgb.jpg")])
    list_depth_images = sorted([d for d in os.listdir(FLAGS.dir_test_dataset) if d.endswith("depth.png")])

    num_test_images = len(list_color_images)

    for i in range(num_test_images):
        file_name = list_color_images[i]
        color_image = imread(os.path.join(FLAGS.dir_test_dataset, list_color_images[i]))
        color_image = color_image.astype(np.float32) / 255.
        color_image = (color_image - color_mean) / color_std
        color_image = np.expand_dims(color_image, 0)
        color_image = np.transpose(color_image, axes=[0, 3, 1, 2])
        color_image = torch.tensor(color_image).float()

        depth_image = imread(os.path.join(FLAGS.dir_test_dataset, list_depth_images[i]))
        depth_image = np.repeat(np.expand_dims(depth_image, -1), 3, -1).astype(np.float32)
        depth_max = np.max(depth_image)
        depth_image = depth_image / depth_max
        depth_image = (depth_image - depth_mean) / depth_std
        depth_image = np.expand_dims(depth_image, 0)
        depth_image = np.transpose(depth_image, axes=[0, 3, 1, 2])
        depth_image = torch.tensor(depth_image).float()

        if FLAGS.use_cuda:
            if torch.cuda.device_count() > 0:
                color_image, depth_image = color_image.cuda(), depth_image.cuda()

        pred_probs, pred_label = run_inference(grasp_aff_model, color_image, depth_image)
        np.save(os.path.join(FLAGS.dir_predictions, file_name.replace("rgb.jpg", "probs.npy")), np.squeeze(pred_probs))
        np.save(os.path.join(FLAGS.dir_predictions, file_name.replace("rgb.jpg", "label.npy")), np.squeeze(pred_label))

        print(f"Processed {i+1}/{num_test_images}")

    print("Inference completed")
    print(f"Predictions saved in {FLAGS.dir_predictions}")

def main():
    FLAGS = argument_parser()

    if not os.path.isfile(FLAGS.model_checkpoint):
        print("Error, model file {FLAGS.model_checkpoint} does not exist")
        sys.exit(0)
    elif not os.path.isdir(FLAGS.dir_test_dataset):
        print("Error, dir {FLAGS.dir_test_dataset} does not exist")
        sys.exit(0)
    else:
        batch_inference(FLAGS)

main()
