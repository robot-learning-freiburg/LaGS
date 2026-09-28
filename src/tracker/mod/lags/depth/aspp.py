# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * BEVDepth (https://github.com/Megvii-BaseDetection/BEVDepth), Copyright (c) Megvii, licensed under MIT,
# * BEVDet (https://github.com/HuangJunJie2017/BEVDet), Copyright (c) Phigent Robotics, licensed under Apache-2.0.
# See the LICENSES/ directory for full license texts.

from typing import Callable, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t


class BasicConvBlock(nn.Module):
    """
    Basic convolutional block for the Atrous Spatial Pyramid Pooling (ASPP)
    module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t | str = 0,
        dilation: _size_2_t = 1,
        bias: bool = False,
        norm: Callable[[int], nn.Module] = nn.BatchNorm2d,
        activation: Callable[[], nn.Module] = nn.ReLU,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int | tuple[int, int]): Size of the convolution kernel.
            stride (int | tuple[int, int]): Stride of the convolution.
            padding (int | tuple[int, int] | str): Padding to be applied to the input.
            dilation (int | tuple[int, int]): Dilation to be applied to the convolution.
            bias (bool): Whether to use bias in the convolution.
            norm (Callable[[int], nn.Module]): Normalization layer to be used.
            activation (Callable[[], nn.Module]): Activation function to be used.
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.norm = norm(out_channels)
        self.act = activation()

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class Aspp(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        hidden_channels: int = 256,
        dilation: Sequence[_size_2_t] = (1, 6, 12, 18),
        kernel_size: Sequence[_size_2_t] = (1, 3, 3, 3),
        norm: Callable[[int], nn.Module] = nn.BatchNorm2d,
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.5,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int | None): Number of output channels. If None, defaults to in_channels.
            hidden_channels (int): Number of hidden channels.
            dilation (Sequence[int | tuple[int, int]]): Dilation rates for the ASPP blocks.
            kernel_size (Sequence[int | tuple[int, int]]): Kernel sizes for the ASPP blocks.
            norm (Callable[[int], nn.Module]): Normalization layer to be used.
            activation (Callable[[], nn.Module]): Activation function to be used.
            dropout (float): Dropout probability.
        """
        super().__init__()

        out_channels = out_channels or in_channels

        # validate kernel_size and dilation
        assert len(kernel_size) == len(dilation)
        assert all(k % 2 == 1 for k in kernel_size)

        # compute padding
        padding = [d * (k - 1) // 2 for d, k in zip(dilation, kernel_size)]
        num_blocks = len(kernel_size)

        # create ASPP convolutional blocks
        aspp_blocks = [
            BasicConvBlock(
                in_channels,
                hidden_channels,
                kernel_size=k,
                stride=1,
                padding=p,
                dilation=d,
                norm=norm,
                activation=activation,
            )
            for k, d, p in zip(kernel_size, dilation, padding)
        ]
        self.aspp_blocks = nn.ModuleList(aspp_blocks)

        # global pooling block for ASPP
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            norm(hidden_channels),
            activation(),
        )

        # combinator for ASPP blocks + global pooling
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_channels * (num_blocks + 1), out_channels, 1, bias=False),
            norm(out_channels),
            activation(),
        )

        # final dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # apply ASPP blocks
        xa = [block(x) for block in self.aspp_blocks]

        # apply global pooling
        size = xa[0].size()[2:]
        xg = self.global_avg_pool(x)
        xg = F.interpolate(xg, size=size, mode="bilinear", align_corners=True)

        # combine
        x = torch.cat((*xa, xg), dim=1)
        x = self.out_conv(x)

        return self.dropout(x)
