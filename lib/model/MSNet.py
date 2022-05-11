import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from lib.loss import *
from lib.module import *
from lib.res2net import res2net50_v1b_26w_4s

from typing import *


class MSNet(nn.Module):
    def __init__(self, channels) -> None:
        super(MSNet, self).__init__()
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)

        self.multi_sub1 = Multiscale_Subtraction(64, channels, 4, kernel_size=3, return_feat=False)
        self.multi_sub2 = Multiscale_Subtraction(256, channels, 3, kernel_size=3)
        self.multi_sub3 = Multiscale_Subtraction(512, channels, 2, kernel_size=3)
        self.multi_sub4 = Multiscale_Subtraction(1024, channels, 1, kernel_size=3)
        self.multi_sub5 = Multiscale_Subtraction(2048, channels, 0, kernel_size=3)

        self.decoder4 = Conv(channels, channels, 3, relu=True)
        self.decoder3 = Conv(channels, channels, 3, relu=True)
        self.decoder2 = Conv(channels, channels, 3, relu=True)
        self.decoder1 = Conv(channels, 1, 3, bn=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x1 = self.res2net.maxpool(x)

        x2 = self.res2net.layer1(x1)
        x3 = self.res2net.layer2(x2)
        x4 = self.res2net.layer3(x3)
        x5 = self.res2net.layer4(x4)

        x5, ms5 = self.multi_sub5(x5)
        x4, ms4 = self.multi_sub4(x4, ms5)
        x3, ms3 = self.multi_sub3(x3, ms4)
        x2, ms2 = self.multi_sub2(x2, ms3)
        x1 = self.multi_sub1(x1, ms2)

        x5 = F.interpolate(x5, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = self.decoder4(x4 + x5)

        x4 = F.interpolate(x4, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        x3 = self.decoder3(x3 + x4)

        x3 = F.interpolate(x3, size=x2.shape[-2:], mode='bilinear', align_corners=False)        
        x2 = self.decoder2(x2 + x3)
        
        x2 = F.interpolate(x2, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        pred = self.decoder1(x1 + x2)

        return pred


if __name__ == '__main__':
    x = torch.randn((1, 3, 352, 352)).cuda()
    m = MSNet(64).cuda()
    y = m(x)
    print(y.shape)