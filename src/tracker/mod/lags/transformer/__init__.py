# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

from omegaconf import OmegaConf
from torch import nn

from ....config.registry import Registry
from . import base, flex, msda2d, msda3d, sca, stateful
from .base import TransformerLayer, TransformerLayerSequence
from .occupancy import HierarchicalOccupancyTransformer, OccupancyTransformer
from .semantic import SemanticTransformer
from .temporal import TemporalTransformer
from .tracking import (
    DisjointTrackingTransformer,
    DualStreamTrackingTransformer,
    UnifiedTrackingTransformer,
)

registry = Registry("lags.transformer")
registry.register(TransformerLayerSequence)
registry.register(OccupancyTransformer)
registry.register(HierarchicalOccupancyTransformer)
registry.register(SemanticTransformer)
registry.register(TemporalTransformer)
registry.register(DisjointTrackingTransformer)
registry.register(DualStreamTrackingTransformer)
registry.register(UnifiedTrackingTransformer)


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> nn.Module:
    return registry.from_config(conf)
