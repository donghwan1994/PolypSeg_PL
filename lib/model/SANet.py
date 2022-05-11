import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from lib.loss import *
from lib.module import *
from lib.res2net import res2net50_v1b_26w_4s

from typing import *


class SANet(nn.Module):
    def __init__(self, channels) -> None:
        super(SANet, self).__init__()
        self.res2net = res2net50_v1b_26w_4s(pretrained=True)

        self.shal_attention = Shallow_Attention(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.res2net.conv1(x)
        x = self.res2net.bn1(x)
        x = self.res2net.relu(x)
        x = self.res2net.maxpool(x)

        x1 = self.res2net.layer1(x)
        x2 = self.res2net.layer2(x1)
        x3 = self.res2net.layer3(x2)
        x4 = self.res2net.layer4(x3)

        pred = self.shal_attention(x2, x3, x4)

        return pred


if __name__ == '__main__':
    x = torch.randn((1, 3, 352, 352)).cuda()
    m = SANet(64).cuda()
    y = m(x)
    print(y.shape)