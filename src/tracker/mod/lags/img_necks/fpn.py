# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class FPN(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        out_levels: List[int],
        upsample_mode: str = "nearest",
    ) -> None:
        super().__init__()

        assert all(0 <= lvl < len(in_channels) for lvl in out_levels)
        assert out_levels == sorted(out_levels)

        self.min_level = min(out_levels)
        self.out_levels = [lvl - self.min_level for lvl in out_levels]
        self.upsample_mode = upsample_mode

        # build lateral layers
        lateral_layers = []
        for in_chs in in_channels[self.min_level :]:
            layer = nn.Conv2d(in_chs, out_channels, kernel_size=1)
            lateral_layers.append(layer)

        # build output layers
        out_layers = []
        for _ in out_levels:
            layer = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            out_layers.append(layer)

        self.lateral_layers = nn.ModuleList(lateral_layers)
        self.out_layers = nn.ModuleList(out_layers)

        # initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # drop bottom-up features below the minimum output level
        features = features[self.min_level :]

        # run lateral layers
        laterals = [layer(features[i]) for i, layer in enumerate(self.lateral_layers)]

        # top-down connections: sum from top to bottom
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], size=laterals[i - 1].shape[2:], mode=self.upsample_mode
            )

        # run output layers
        out = [layer(laterals[i]) for i, layer in zip(self.out_levels, self.out_layers)]

        return out
