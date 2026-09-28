# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

from omegaconf import OmegaConf

# pylint: disable-next=redefined-builtin
from . import (
    depth,
    filter,
    image_augment,
    image_base,
    lidar_augment,
    lidar_base,
    lidar_object_sampler,
    occupancy,
    pack,
)
from .collection import Collection
from .conditional import Conditional, Random
from .identity import Identity
from .registry import registry
from .transform import Transform


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> Transform:
    return registry.from_config(conf)
