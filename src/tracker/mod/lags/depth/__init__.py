# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

from omegaconf import OmegaConf
from torch import nn

from ....config.registry import Registry
from . import aspp, depthnet, lift, loss, stereo
from .depthnet import DepthNet
from .lift import LiftAndPool
from .loss import DepthLoss
from .stereo import StereoDepthNet

registry = Registry("lags.depth")
registry.register(DepthNet)
registry.register(StereoDepthNet)
registry.register(DepthLoss)
registry.register(LiftAndPool)


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> nn.Module:
    return registry.from_config(conf)
