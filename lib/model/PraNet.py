import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from lib.loss import *
from lib.module import *
from lib.res2net import res2net50_v1b_26w_4s

from typing import *


class PraNet(nn.Module):
    def __init__(self, channels) -> None:
        super(PraNet, self).__init__()
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)
        
        self.rfb2 = RFB_modified(512, channels)
        self.rfb3 = RFB_modified(1024, channels)
        self.rfb4 = RFB_modified(2048, channels)
        self.pd = aggregation(channels)

        self.rev_attention2 = Reverse_Attention(512, 64, 2, 3)
        self.rev_attention3 = Reverse_Attention(1024, 64, 2, 3)
        self.rev_attention4 = Reverse_Attention(2048, 256, 3, 5)

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)

        x1 = self.res2net.layer1(x)
        x2 = self.res2net.layer2(x1)
        x3 = self.res2net.layer3(x2)
        x4 = self.res2net.layer4(x3)

        x2_rfb = self.rfb2(x2)
        x3_rfb = self.rfb3(x3)
        x4_rfb = self.rfb4(x4)

        pred5 = self.pd(x4_rfb, x3_rfb, x2_rfb)
        pred4 = self.rev_attention4(x4, pred5)
        pred3 = self.rev_attention3(x3, pred4)
        pred2 = self.rev_attention2(x2, pred3)

        return pred2, pred3, pred4, pred5


if __name__ == '__main__':
    x = torch.randn((1, 3, 352, 352)).cuda()
    m = PraNet(64).cuda()
    y = m(x)
    print(y.shape)