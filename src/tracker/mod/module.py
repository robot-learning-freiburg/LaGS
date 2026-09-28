# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from typing import IO, Any

import lightning as L
import timm
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from omegaconf import OmegaConf

from .. import config
from ..config.registry import Registry

registry = Registry("mod.module")


def get(key: str) -> Any:
    return registry.get(key)


def build(
    model: OmegaConf,
    optimizer: OmegaConf,
    lr_scheduler: OmegaConf,
) -> L.LightningModule:
    cls = registry.get(model.type)

    return cls(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )


def load(
    path: str | Path | IO,
    model: OmegaConf,
    optimizer: OmegaConf,
    lr_scheduler: OmegaConf,
    map_location=None,
    strict: bool | None = None,
) -> L.LightningModule:
    cls = registry.get(model.type)

    return cls.load_from_checkpoint(
        checkpoint_path=path,
        map_location=map_location,
        strict=strict,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )


class Base(L.LightningModule):
    def __init__(
        self,
        model: OmegaConf,
        optimizer: OmegaConf,
        lr_scheduler: OmegaConf,
    ):
        super().__init__()

        self.conf = model
        self.optimizer_conf = optimizer
        self.lr_scheduler_conf = lr_scheduler

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optim_cfg = config.utils.copy(self.optimizer_conf, readonly=False)

        # temporarily add resolver for dynamic parameters
        def _resolver(param: str):
            if param == "estimated_steps":
                return self.trainer.estimated_stepping_batches

            if param == "steps_per_epoch":
                return self.trainer.estimated_stepping_batches / self.trainer.max_epochs

            raise ValueError(f"unsupported parameter '{param}' for optimizer resolver")

        with config.utils.context_resolver("optimizer", _resolver):
            # build parameter groups
            groups_cfg = optim_cfg.pop("param_groups", None)
            if groups_cfg is not None:
                params = config.types.optimizer.get_grouped_params(self, groups_cfg)
            else:
                params = self.parameters()

            # build optimizer
            optimizer = config.types.optimizer.build(optim_cfg, params)

            # build lr scheduler
            scheduler = config.types.lr_scheduler.build(
                self.lr_scheduler_conf, optimizer
            )

        if scheduler is None:
            return {"optimizer": optimizer}

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def lr_scheduler_step(self, scheduler: Any, metric: Any | None) -> None:
        # special stuff for timm schedulers
        if isinstance(scheduler, timm.scheduler.scheduler.Scheduler):
            # timm uses different methods for different scheduler intervals...
            # so figure out which interval we're dealing with and select the
            # method based on it
            conf = self.trainer.lr_scheduler_configs
            conf = next(c for c in conf if c.scheduler == scheduler)

            if conf.interval == "step":
                scheduler.step_update(num_updates=self.global_step, metric=metric)
            else:
                scheduler.step(epoch=self.current_epoch, metric=metric)

        # basic pytorch scheduler
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric=metric)
