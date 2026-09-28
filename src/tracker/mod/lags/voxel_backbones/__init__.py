# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

from omegaconf import OmegaConf
from torch import nn

from ....config.registry import Registry
from . import resnet3d
from .resnet3d import ResNet3d

registry = Registry("lags.voxel_backbones")
registry.register(ResNet3d)


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> nn.Module:
    return registry.from_config(conf)
