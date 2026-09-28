# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import lightning
import lightning.pytorch.plugins
import torch
import torch.amp
from lightning.pytorch.plugins import (
    CheckpointIO,
    ClusterEnvironment,
    LayerSync,
    MixedPrecision,
    Precision,
)
from omegaconf import OmegaConf

from ..registry import Registry

registry = Registry("lighting.callback")
registry.register_from_module(lightning.pytorch.plugins, Precision)
registry.register_from_module(lightning.pytorch.plugins, ClusterEnvironment)
registry.register_from_module(lightning.pytorch.plugins, CheckpointIO)
registry.register_from_module(lightning.pytorch.plugins, LayerSync)


@registry.register(key="MixedPrecision", force=True)
def _mixed_precision(*args, **kwargs):
    scaler = kwargs.pop("scaler", None)
    device = kwargs.pop("device", "cuda")

    if scaler is not None:
        scaler = OmegaConf.to_container(scaler)
        device = scaler.pop("device", device)

        scaler = torch.amp.GradScaler(**scaler, device=device)

    return MixedPrecision(*args, device=device, scaler=scaler, **kwargs)


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> Any:
    return registry.from_config(conf)
