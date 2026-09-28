# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

from omegaconf import OmegaConf
from torch import nn

from ....config.registry import Registry
from . import lssfpn3d
from .lssfpn3d import LssFpn3d
from .unet import UNetAggregate3d

registry = Registry("lags.voxel_necks")
registry.register(LssFpn3d)
registry.register(UNetAggregate3d)


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> nn.Module:
    return registry.from_config_resolve(conf)
