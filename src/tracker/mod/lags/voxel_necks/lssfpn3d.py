# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * COTR (https://github.com/NotACracker/COTR), licensed under Apache-2.0.
# See the LICENSES/ directory for full license texts.

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint as ckpt


class LssFpn3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_size: tuple[int, int, int],
        reverse: bool = False,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        self.reverse = reverse
        self.spatial_size = spatial_size
        self.use_checkpointing = use_checkpointing

        if not reverse:
            self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.up2 = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        x1, x2, x4 = features

        # down/up-sample to common spatial size
        if not self.reverse:
            x2 = self.up1(x2)
            x4 = self.up2(x4)
        else:
            interp_kwargs = {
                "size": self.spatial_size,
                "mode": "trilinear",
                "align_corners": True,
            }

            x1 = F.interpolate(x1, **interp_kwargs)
            x2 = F.interpolate(x2, **interp_kwargs)

            if x4.shape[-3:] != self.spatial_size:
                x4 = F.interpolate(x4, **interp_kwargs)

        # combine
        x = torch.cat([x1, x2, x4], dim=1)

        # apply conv
        if self.use_checkpointing:
            x = ckpt.checkpoint(self.conv, x)
        else:
            x = self.conv(x)

        return x
