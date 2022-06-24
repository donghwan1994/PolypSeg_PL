import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.loss import *
from lib.module import *
from lib.res2net import res2net50_v1b_26w_4s

class Protytpe_model(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channels=64, pretrained=True):
        super(Protytpe_model, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=pretrained)

        self.context2 = DW_RFB(512, channels)
        self.context3 = DW_RFB(1024, channels)
        self.context4 = DW_RFB(2048, channels)

        # edge decoder for low-level feature
        self.edge_decoder = RFB_edge(256, 32)

        self.decoder = PPD(channels)

        self.attention2 = Fore_Attention(channels, channels, 1, 3)
        self.attention3 = Fore_Attention(channels, channels, 1, 3)
        self.attention4 = Fore_Attention(channels, channels, 1, 3)

        self.loss_fn = bce_iou_loss

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        base_size = x.shape[-2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        edge = self.edge_decoder(x1)
        x2 = self.resnet.layer2(x1+edge)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        
        x2_context = self.context2(x2)
        x3_context = self.context3(x3)
        x4_context = self.context4(x4)

        f5, a5 = self.decoder(x4_context, x3_context, x2_context)
        out5 = F.interpolate(a5, size=base_size, mode='bilinear', align_corners=False)

        f4, a4 = self.attention4(x4_context, f5, a5)
        out4 = F.interpolate(a4, size=base_size, mode='bilinear', align_corners=False)

        f3, a3 = self.attention3(x3_context, f4, a4)
        out3 = F.interpolate(a3, size=base_size, mode='bilinear', align_corners=False)

        _, a2 = self.attention2(x2_context, f3, a3)
        out2 = F.interpolate(a2, size=base_size, mode='bilinear', align_corners=False)

        return edge, out2, out3, out4, out5