# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping

from omegaconf import OmegaConf
from torch import nn

from ... import config
from ..log import get_logger

log = get_logger(__name__)


def compile(
    model: nn.Module,
    opts: bool | OmegaConf | None = None,
    **kwargs: dict[str, Any],
) -> nn.Module:
    # pylint: disable=redefined-builtin
    """
    Compile a PyTorch model with the given options.

    Args:
        model (nn.Module): The PyTorch model to compile.
        opts (bool | OmegaConf): Compilation options. If False, no compilation is done.
            If True, default options are used. If an OmegaConf object, it contains
            specific compilation options.
        **kwargs: Additional keyword arguments for the compilation method.

    Returns:
        nn.Module: The compiled PyTorch model.
    """

    if opts is None and not kwargs:
        return model

    if isinstance(opts, bool) and not opts:
        return model

    if not isinstance(opts, Mapping) or not OmegaConf.is_config(opts):
        opts = {}

    opts = config.utils.copy(opts, readonly=False)

    if not opts.pop("enabled", True):
        return model

    log.info("compiling '%s' with options: %s", model.__class__.__qualname__, {**opts})
    # Note: model.compile() seems to be preferred over torch.compile().
    model.compile(**opts, **kwargs)

    return model
