# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

from omegaconf import OmegaConf

from ... import config
from .. import transform
from . import utils
from .dataset import Dataset, IterableDataset
from .nuscenes import NuScenes, NuScenesSequential
from .registry import registry
from .waymo_to import WaymoTO
from .wrapper import wrap


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> Dataset | IterableDataset:
    source = registry.from_config(conf.source)
    transforms = [transform.build(c) for c in conf.transforms]

    kwargs = config.utils.copy(conf, readonly=False)
    del kwargs.source
    del kwargs.transforms

    return wrap(source, transforms, **kwargs)
