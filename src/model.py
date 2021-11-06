import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.densenet import densenet121

class GraspAffordanceNet(nn.Module):
    def __init__(self, use_cuda=False, pretrained=True):
        super(GraspAffordanceNet, self).__init__()
        self.grasp_color_encoder = densenet121(pretrained=pretrained)
        self.grasp_depth_encoder = densenet121(pretrained=pretrained)

        self.graspnet = nn.Sequential(OrderedDict(
            [
                ("grasp-norm0", nn.BatchNorm2d(2048)),
                ("grasp-relu0", nn.ReLU(inplace=True)),
                ("grasp-conv0", nn.Conv2d(2048, 64, kernel_size=1, stride=1, bias=False)),
                ("grasp-norm1", nn.BatchNorm2d(64)),
                ("grasp-relu1", nn.ReLU(inplace=True)),
                ("grasp-conv1", nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
            ]
        ))

        for m in self.named_modules():
            if "grasp-" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

    def forward(self, input_color_data, input_depth_data):
        interm_grasp_color_feat = self.grasp_color_encoder.features(input_color_data)
        interm_grasp_depth_feat = self.grasp_depth_encoder.features(input_depth_data)
        #print(interm_grasp_color_feat.shape)
        #print(interm_grasp_depth_feat.shape)

        interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)
        #print(interm_grasp_feat.shape)

        output_prob = nn.Upsample(scale_factor=32, mode="bilinear").forward(self.graspnet(interm_grasp_feat))
        #print(output_prob.shape)

        return output_prob
