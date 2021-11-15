import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models.densenet import densenet121

class GraspAffordanceSegNet(nn.Module):
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

    def forward(self, input_color_data, input_depth_data):
        interm_grasp_color_feat = self.grasp_color_encoder.features(input_color_data)
        interm_grasp_depth_feat = self.grasp_depth_encoder.features(input_depth_data)

        interm_grasp_feat = torch.cat((interm_grasp_color_feat, interm_grasp_depth_feat), dim=1)

        output_g_a_p_net = self.graspnet(interm_grasp_feat)

        return output_g_a_p_net

class GAPANet(nn.Module):
    """
    GAPANet - Grasp Affordance Passive to Active Net
    """
    def __init__(self, passive_model_checkpoint_path):
        super().__init__()
        self.passive_model = GraspAffordanceSegNet()
        self.passive_model.load_state_dict(torch.load(passive_model_checkpoint_path))

        self.pos_output = nn.Sequential(OrderedDict(
            [
                ("active-grasp-pos", nn.Conv2d(16, 1, kernel_size=2, padding="same"))
            ]
        ))

        self.cos_output =  nn.Sequential(OrderedDict(
            [
                ("active-grasp-cos", nn.Conv2d(16, 1, kernel_size=2, padding="same"))
            ]
        ))

        self.sin_output =  nn.Sequential(OrderedDict(
            [
                ("active-grasp-sin", nn.Conv2d(16, 1, kernel_size=2, padding="same"))
            ]
        ))

        self.width_output =  nn.Sequential(OrderedDict(
            [
                ("active-grasp-wid", nn.Conv2d(16, 1, kernel_size=2, padding="same"))
            ]
        ))

        for m in self.named_modules():
            if "active-grasp-" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.xavier_uniform_(m[1].weight, gain=1)

    def forward(self, input_color_data, input_depth_data):
        x = self.passive_model(input_color_data, input_depth_data)

        pos_output = self.pos_output(x)
        cos_output = self.cos_output(x)
        sin_output = self.sin_output(x)
        width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, input_color_data, input_depth_data, yc):
        y_pos, y_cos, y_sin, y_width = yc
        pos_pred, cos_pred, sin_pred, width_pred = self(input_color_data, input_depth_data)

        p_loss = F.mse_loss(pos_pred, y_pos)
        cos_loss = F.mse_loss(cos_pred, y_cos)
        sin_loss = F.mse_loss(sin_pred, y_sin)
        width_loss = F.mse_loss(width_pred, y_width)

        return {
            "loss": p_loss + cos_loss + sin_loss + width_loss,
            "losses": {
                "p_loss": p_loss,
                "cos_loss": cos_loss,
                "sin_loss": sin_loss,
                "width_loss": width_loss
            },
            "pred": {
                "pos": pos_pred,
                "cos": cos_pred,
                "sin": sin_pred,
                "width": width_pred
            }
        }
