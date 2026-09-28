# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * COTR (https://github.com/NotACracker/COTR), licensed under Apache-2.0,
# * MMDetection (https://github.com/open-mmlab/mmdetection), Copyright (c) OpenMMLab, licensed under Apache-2.0.
# See the LICENSES/ directory for full license texts.

from typing import Collection, Sequence

import torch
from torch import nn
from torch.utils import checkpoint as ckpt


class BasicBlock3d(nn.Module):
    # pylint: disable=too-many-instance-attributes
    """
    Basic 3D ResNet block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x = self.act2(x + identity)

        return x


class ResNet3dLayer(nn.Sequential):
    """
    A 3D ResNet layer that consists of multiple BasicBlock3D blocks.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        blocks = [
            BasicBlock3d(
                in_channels,
                out_channels,
                stride=stride,
                dilation=dilation,
                downsample=nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm3d(out_channels),
                ),
            )
        ]

        blocks += [
            BasicBlock3d(out_channels, out_channels) for _ in range(1, num_blocks)
        ]

        super().__init__(*blocks)


class ResNet3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_blocks: Sequence[int],
        num_channels: Sequence[int] | None = None,
        strides: Sequence[int] | None = None,
        dilations: Sequence[int] | None = None,
        output_layers: Collection[int] | None = None,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        if num_channels is None:
            num_channels = [in_channels * 2**i for i in range(1, len(num_blocks) + 1)]

        if strides is None:
            strides = [2] * len(num_blocks)

        if dilations is None:
            dilations = [1] * len(num_blocks)

        if output_layers is None:
            output_layers = range(len(num_blocks))

        assert len(num_blocks) == len(num_channels)
        assert len(num_blocks) == len(strides)

        self.num_channels = num_channels
        self.output_layers = set(output_layers)
        self.use_checkpointing = use_checkpointing

        assert len(self.output_layers) > 0, "No output layers specified"

        num_channels = [in_channels] + list(num_channels)

        layers = [
            ResNet3dLayer(
                in_channels=num_channels[i],
                out_channels=num_channels[i + 1],
                num_blocks=blocks_per_layer,
                stride=strides[i],
                dilation=dilations[i],
            )
            for i, blocks_per_layer in enumerate(num_blocks)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []

        for i, layer in enumerate(self.layers):
            if self.use_checkpointing:
                x = ckpt.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

            if i in self.output_layers:
                outputs.append(x)

        return outputs
