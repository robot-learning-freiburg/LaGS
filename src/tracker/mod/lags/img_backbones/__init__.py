# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

from mmdet.models.backbones.resnet import ResNet
from omegaconf import OmegaConf
from torch import nn

from .... import config, utils
from ....config.registry import Registry
from .vovnet import VoVNet, VoVNetSpec

registry = Registry("lags.img_backbone")
registry.register(VoVNet)
registry.register(ResNet)


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> nn.Module:
    conf = config.utils.copy(conf, readonly=False)
    compile_opts = conf.pop("compile", False)

    model = registry.from_config_resolve(conf)
    model = utils.torch.compile(model, compile_opts)

    return model
