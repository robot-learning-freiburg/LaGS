# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import lightning
from lightning.pytorch.profilers import (
    AdvancedProfiler,
    Profiler,
    PyTorchProfiler,
    SimpleProfiler,
    XLAProfiler,
)
from omegaconf import OmegaConf

from ..registry import Registry

registry = Registry("lighting.profiler")
registry.register_from_module(lightning.pytorch.profilers, Profiler)
registry.register(AdvancedProfiler, "advanced")
registry.register(SimpleProfiler, "simple")
registry.register(PyTorchProfiler, "pytorch")
registry.register(XLAProfiler, "xla")


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf | str) -> Profiler:
    if isinstance(conf, str):
        return conf

    return registry.from_config(conf)
