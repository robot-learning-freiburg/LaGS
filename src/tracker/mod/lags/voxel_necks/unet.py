# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Sequence

import torch
from torch import nn


class UNetUpsamplingBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.conv1 = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.ConvTranspose3d(
            out_channels,
            out_channels,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2),
            padding=(0, 0, 0),
        )
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        return x


class UNetAggregate3d(nn.Module):
    def __init__(
        self,
        in_channels: tuple[int],
        out_channels: int,
        channels: tuple[int] | None = None,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if channels is None:
            channels = in_channels[:-1]

        assert len(channels) == len(in_channels) - 1

        in_channels = list(reversed(in_channels))
        up_channels = [0] + list(reversed(channels))
        channels = reversed(channels)

        blocks = []
        for ch_in, ch_up, ch_out in zip(in_channels[:-1], up_channels[:-1], channels):
            blocks.append(UNetUpsamplingBlock3d(ch_in + ch_up, ch_out))

        blocks.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels[-1] + up_channels[-1], out_channels, kernel_size=1
                ),
                nn.ReLU(inplace=True),
            )
        )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        b, _, d, h, w = feats[-1].shape

        up = torch.zeros(b, 0, d, h, w, device=feats[0].device)
        for block, x in zip(self.blocks, reversed(feats)):
            up = block(torch.cat([up, x], dim=1))

        return up
