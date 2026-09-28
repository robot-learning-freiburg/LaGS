# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable
from typing import Any

import lightning as L
from omegaconf import OmegaConf

from .. import utils
from . import callback, logger, plugin, profiler


def _build_item_list(
    build_fn: Callable, base: OmegaConf, field: str, *args, **kwargs
) -> list[Any]:
    items = base.pop(field, {})
    items = items if items is not None else {}

    return [build_fn(v, *args, **kwargs) for v in items.values() if v is not None]


def build(conf: OmegaConf, *args, **kwargs) -> L.Trainer:
    trainer_conf = utils.copy(conf.trainer, readonly=False)

    # build loggers
    loggers = kwargs.pop("logger", [])
    if not isinstance(loggers, list):
        loggers = [loggers]

    loggers += _build_item_list(
        logger.build, trainer_conf, "logger", paths=conf.paths, meta=conf.meta
    )

    # save config to hparams
    hparams = OmegaConf.to_container(conf, resolve=False)
    for log in loggers:
        log.log_hyperparams(hparams)

    # build callbacks
    callbacks = kwargs.pop("callbacks", [])
    callbacks += _build_item_list(callback.build, trainer_conf, "callbacks")

    # build plugins
    plugins = kwargs.pop("plugins", [])
    plugins += _build_item_list(plugin.build, trainer_conf, "plugins")

    # get default root dir
    default_root_dir = kwargs.pop("default_root_dir", conf.paths.current_run)
    default_root_dir = trainer_conf.pop("default_root_dir", default_root_dir)

    prof = trainer_conf.pop("profiler", None)
    prof = profiler.build(prof) if prof is not None else None

    assert prof is None or "profiler" not in kwargs, "profiler is already set in kwargs"
    prof = kwargs.pop("profiler", prof)

    # We need to reload the dataloaders every epoch to ensure that the epoch
    # counter is in sync with the dataloaders. Therfore, we set the default
    # value to 1. Users can still override this value, setting it to 0 to
    # disable reloading.
    if "reload_dataloaders_every_n_epochs" not in trainer_conf:
        trainer_conf.reload_dataloaders_every_n_epochs = 1

    # build trainer
    return L.Trainer(
        *args,
        default_root_dir=default_root_dir,
        logger=loggers,
        callbacks=callbacks,
        plugins=plugins,
        profiler=prof,
        **kwargs,
        **trainer_conf,
    )
