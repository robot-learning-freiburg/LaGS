# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Sequence

from omegaconf import OmegaConf

from . import nuscenes, occupancy
from .registry import registry
from .writer import PredictionWriter


def get(key: str) -> Any:
    return registry.get(key)


def build_callbacks(conf: OmegaConf | None) -> list[PredictionWriter]:
    if conf is None:
        return []

    if not isinstance(conf, Sequence):
        conf = [conf]

    return [registry.from_config(cfg) for cfg in conf]
