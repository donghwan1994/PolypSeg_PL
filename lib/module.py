import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn.common_types import _size_2_t
from typing import *


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        dilation: _size_2_t = 1,
        groups:int = 1,
        padding: Union[str, _size_2_t] = 'same',
        bias:bool = False,
        bn:bool = True,
        relu:bool = False
    ) -> None:
        super().__init__()

        self._body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, dilation, groups, bias=bias)
        )        
        if bn:
            self._body.append(
                nn.BatchNorm2d(out_channels)
            )
        if relu:
            self._body.append(
                nn.ReLU(inplace=True)
            )

    def forward(self, x: Tensor) -> Tensor:
        return self._body(x)


class RFB_modified(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.branch0 = Conv(in_channels, out_channels, 1)
        self.branch1 = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            Conv(out_channels, out_channels, kernel_size=(1, 3),
                 padding=(0, 1)),
            Conv(out_channels, out_channels, kernel_size=(3, 1),
                 padding=(1, 0)),
            Conv(out_channels, out_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            Conv(out_channels, out_channels, kernel_size=(1, 5),
                 padding=(0, 2)),
            Conv(out_channels, out_channels, kernel_size=(5, 1),
                 padding=(2, 0)),
            Conv(out_channels, out_channels, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            Conv(out_channels, out_channels, kernel_size=(1, 7),
                 padding=(0, 3)),
            Conv(out_channels, out_channels, kernel_size=(7, 1),
                 padding=(3, 0)),
            Conv(out_channels, out_channels, 3, padding=7, dilation=7)
        )

        self.conv_cat = Conv(4 * out_channels, out_channels, 3)
        self.conv_res = Conv(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        res = self.conv_res(x)
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.relu(res + self.conv_cat(x))

        return x


class aggregation(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = Conv(channel, channel, 3)
        self.conv_upsample2 = Conv(channel, channel, 3)
        self.conv_upsample3 = Conv(channel, channel, 3)
        self.conv_upsample4 = Conv(channel, channel, 3)
        self.conv_upsample5 = Conv(2 * channel, 2 * channel, 3)

        self.conv_cat2 = Conv(2 * channel, 2 * channel, 3)
        self.conv_cat3 = Conv(3 * channel, 3 * channel, 3)
        self.conv4 = Conv(3 * channel, 3 * channel, 3)
        self.conv5 = Conv(3 * channel, 1, 1)

    def forward(self, x4: Tensor, x3: Tensor, x2: Tensor) -> Tensor:
        x4_1 = x4
        x3_1 = self.conv_upsample1(self.upsample(x4)) * x3
        x2_1 = self.conv_upsample2(self.upsample(self.upsample(x4))) \
                * self.conv_upsample3(self.upsample(x3)) * x2

        x3_2 = torch.cat([x3_1, self.conv_upsample4(self.upsample(x4_1))], dim=1)
        x3_2 = self.conv_cat2(x3_2)

        x2_2 = torch.cat([x2_1, self.conv_upsample5(self.upsample(x3_2))], dim=1)
        x2_2 = self.conv_cat3(x2_2)

        x = self.conv4(x2_2)
        x = self.conv5(x)

        return x


class Reverse_Attention(nn.Module):
    def __init__(
        self,
        in_channels: int, 
        channels: int, 
        depth: int, 
        kernel_size: _size_2_t
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            Conv(in_channels, channels, 1)
        )
        for i in range(depth):
            self.conv.append(
                Conv(channels, channels, kernel_size, relu=True)
            )
        self.conv_out = Conv(channels, 1, 1)

    def forward(self, x: Tensor, map: Tensor) -> Tensor:
        map = F.interpolate(map, size=x.shape[-2:], mode='bilinear', align_corners=False)
        rmap = -1 * torch.sigmoid(map) + 1
        x = rmap.expand(-1, x.shape[1], -1, -1).mul(x)
        x = self.conv(x)
        x = self.conv_out(x)
        x += map

        return x


class Shallow_Attention(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()

        self.shallow2 = Conv(512, channels, 1, relu=True)
        self.shallow3 = Conv(1024, channels, 1, relu=True)
        self.shallow4 = Conv(2048, channels, 1, relu=True)

        self.conv_out = Conv(3 * channels, 1, 1, bn=False)

    def forward(self, x2: Tensor, x3: Tensor, x4: Tensor) -> Tensor:
        x2 = self.shallow2(x2)
        x3 = self.shallow3(x3)
        x4 = self.shallow4(x4)

        x4 = F.interpolate(x4, size=x2.shape[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x2.shape[-2:], mode='bilinear', align_corners=True)

        x = torch.cat([x4, x4 * x3, x4 * x3 * x2], dim=1)
        x = self.conv_out(x)

        return x


class Multiscale_Subtraction(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int, 
        depth: int, 
        kernel_size: _size_2_t,
        return_feat: Optional[bool] = True
    ) -> None:
        super().__init__()
        self.return_feat = return_feat
        self.conv_in = Conv(in_channels, out_channels, kernel_size, relu=True)
        if depth > 0:
            self.convs = nn.ModuleList()
            for i in range(depth):
                self.convs.append(
                    Conv(out_channels, out_channels, kernel_size, relu=True)
                )
        else:
            self.convs = None
        self.conv_ce = Conv(out_channels, out_channels, kernel_size, relu=True)

    def forward(self, x: Tensor, f_list: Optional[Union[List[Tensor], Any]] = None) -> Tuple[Tensor, List[Tensor]]:
        ms = []
        x = self.conv_in(x)
        ms.append(x)
        if f_list is not None:
            assert len(self.convs) == len(f_list)
            su = x
            for conv, f in zip(self.convs, f_list):
                f = F.interpolate(f, size=su.shape[-2:], mode='bilinear', align_corners=False)
                su = conv(torch.abs(su - f))
                x += su
                ms.append(su)   
        x = self.conv_ce(x)

        if self.return_feat:
            return x, ms
        return x
