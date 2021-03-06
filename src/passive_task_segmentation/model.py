import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18
from torchvision.models.densenet import densenet121

class GraspAffordanceSegNet(nn.Module):
    # GraspAffordanceSegNet - GASNet
    # To train grasp affordance segmentation - passive vision task

    def __init__(self, pretrained=True):
        super(GraspAffordanceSegNet, self).__init__()
        self.grasp_color_encoder = densenet121(pretrained=pretrained)
        self.grasp_depth_encoder = densenet121(pretrained=pretrained)

        self.graspnet = nn.Sequential(OrderedDict(
            [
                ("grasp-relu0", nn.ReLU(inplace=True)),
                ("grasp-conv0", nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding="same")),
                ("grasp-relu1_1", nn.ReLU(inplace=True)),
                ("grasp-conv1_1", nn.Conv2d(512, 512, kernel_size=3, stride=1, padding="same")),
                ("grasp-relu1_2", nn.ReLU(inplace=True)),
                ("grasp-conv-up1_2", nn.ConvTranspose2d(512, 128, kernel_size=4, stride=4, bias=False, output_padding=0)),
                ("grasp-relu2_1", nn.ReLU(inplace=True)),
                ("grasp-conv2_1", nn.Conv2d(128, 128, kernel_size=3, stride=1, padding="same")),
                ("grasp-relu2_2", nn.ReLU(inplace=True)),
                ("grasp-conv-up2_2", nn.ConvTranspose2d(128, 32, kernel_size=4, stride=4, bias=False, output_padding=0)),
                ("grasp-relu3_1", nn.ReLU(inplace=True)),
                ("grasp-conv3_1", nn.Conv2d(32, 32, kernel_size=3, stride=1, padding="same")),
                ("grasp-relu3_2", nn.ReLU(inplace=True)),
                ("grasp-conv-up3_2", nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, bias=False, output_padding=0)),
                ("grasp-relu4", nn.ReLU(inplace=True)),
            ]
        ))

        self.final_conv = nn.Sequential(OrderedDict(
            [
                ("grasp-final-conv", nn.Conv2d(16, 1, kernel_size=1, stride=1, bias=False))
            ]
        ))

        for m in self.named_modules():
            if "grasp-" in m[0]:
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.ConvTranspose2d):
                    nn.init.xavier_normal_(m[1].weight.data)

    def forward(self, input_color_data, input_depth_data):
        interm_grasp_color_feat = self.grasp_color_encoder.features(input_color_data)
        interm_grasp_depth_feat = self.grasp_depth_encoder.features(input_depth_data)
        #print(interm_grasp_color_feat.shape)
        #print(interm_grasp_depth_feat.shape)

        interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
        #print(interm_grasp_feat.shape)

        output_g_net = self.graspnet(interm_grasp_feat)
        output_logits = self.final_conv(output_g_net)

        return output_logits

"""
def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        self.base_model = resnet18(pretrained=True)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
"""
