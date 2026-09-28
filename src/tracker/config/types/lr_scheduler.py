# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings
from typing import Any, Optional

import timm.scheduler
import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import ChainedScheduler, LRScheduler
from torch.optim.lr_scheduler import SequentialLR as _SequentialLR
from torch.optim.optimizer import Optimizer

from ..registry import Registry

registry = Registry("lr_scheduler")
registry.register_from_module(torch.optim.lr_scheduler, LRScheduler)
registry.register_from_module(
    timm.scheduler, timm.scheduler.scheduler.Scheduler, "timm"
)


# Workaround to prevent a deprecation warning in SequentialLR.step().
# Internally, this function can in some instances call its schedulers with
# scheduler.step(epoch=...), which is deprecated. Ignore it here since it is
# pytorch internal.
class SequentialLR(_SequentialLR):
    # pylint: disable=abstract-method

    def step(self):
        with warnings.catch_warnings(action="ignore", category=UserWarning):
            super().step()


class Fused(LRScheduler):
    """
    A fused learning rate scheduler that uses a warm-up scheduler for a specified
    number of steps and retains the last learning rate after the warm-up phase.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        steps: int,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            scheduler (LRScheduler): Scheduler used for warm-up.
            steps: Number of warm-up steps.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.scheduler = scheduler
        self.steps = steps

        self._initialized = False

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Get the current learning rate from the warm-up scheduler during the warm-up phase.
        After the warm-up phase, return the last learning rate of the warm-up scheduler.
        """
        if self.last_epoch < self.steps:
            return self.scheduler.get_last_lr()

        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, epoch=None):
        """
        Step the warm-up scheduler during the warm-up phase.
        After the warm-up phase, stop modifying the learning rate.
        """
        # Avoid stepping the base scheduler on initialization
        if not self._initialized:
            self._initialized = True
            return

        if self.last_epoch < self.steps:
            self.scheduler.step(epoch)

        self.last_epoch += 1

    def state_dict(self) -> dict[str, Any]:
        ignored = {
            "optimizer",
            "scheduler",
            "_initialized",
        }

        state = {k: v for k, v in self.__dict__.items() if k not in ignored}
        state["scheduler"] = self.scheduler.state_dict()

        return state

    def load_state_dict(self, state_dict: dict[str, Any]):
        scheduler = state_dict.pop("scheduler", None)
        if scheduler is not None:
            self.scheduler.load_state_dict(scheduler)

        self.__dict__.update(state_dict)


class Stepped(LRScheduler):
    """
    A scheduler that steps the base scheduler only every `n` steps.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        interval: int,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            base_scheduler (LRScheduler): The base learning rate scheduler to wrap.
            step_interval (int): Number of steps to wait before stepping the base scheduler.
            last_epoch (int): The index of the last epoch. Default: -1.
        """
        self.base_scheduler = scheduler
        self.step_interval = interval

        self._initialized = False

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Get the current learning rate from the base scheduler.
        """
        return self.base_scheduler.get_last_lr()

    def step(self, epoch=None):
        """
        Step the base scheduler only every `n` steps.
        """
        # Avoid stepping the base scheduler on initialization
        if not self._initialized:
            self._initialized = True
            return

        self.last_epoch += 1
        if self.last_epoch % self.step_interval == self.step_interval - 1:
            self.base_scheduler.step(epoch)

    def state_dict(self) -> dict[str, Any]:
        ignored = {
            "optimizer",
            "base_scheduler",
            "_initialized",
        }

        state = {k: v for k, v in self.__dict__.items() if k not in ignored}
        state["base_scheduler"] = self.base_scheduler.state_dict()

        return state

    def load_state_dict(self, state_dict: dict[str, Any]):
        scheduler = state_dict.pop("base_scheduler", None)
        if scheduler is not None:
            self.base_scheduler.load_state_dict(scheduler)

        self.__dict__.update(state_dict)


@registry.register(key="ChainedScheduler", force=True)
def _chained_scheduler(optimizer, schedulers, **kwargs) -> ChainedScheduler:
    schedulers = [
        registry.from_config_resolve(cfg, optimizer=optimizer) for cfg in schedulers
    ]

    return ChainedScheduler(schedulers=schedulers, optimizer=optimizer, **kwargs)


@registry.register(key="SequentialLR", force=True)
def _sequential_lr(optimizer, schedulers, **kwargs) -> SequentialLR:
    schedulers = [
        registry.from_config_resolve(cfg, optimizer=optimizer) for cfg in schedulers
    ]

    return SequentialLR(optimizer=optimizer, schedulers=schedulers, **kwargs)


@registry.register(key="Fused")
def _fused(optimizer, scheduler, **kwargs) -> Fused:
    scheduler = registry.from_config_resolve(scheduler, optimizer=optimizer)

    return Fused(optimizer=optimizer, scheduler=scheduler, **kwargs)


@registry.register(key="Stepped")
def _stepped(optimizer, scheduler, **kwargs) -> Fused:
    scheduler = registry.from_config_resolve(scheduler, optimizer=optimizer)

    return Stepped(optimizer=optimizer, scheduler=scheduler, **kwargs)


def get(key: str) -> Any:
    return registry.get(key)


def build(
    conf: Optional[OmegaConf],
    optimizer: Optimizer,
) -> LRScheduler:
    if conf is None:
        return None

    if "scheduler" not in conf:
        return None

    return {
        "scheduler": registry.from_config_resolve(conf.scheduler, optimizer=optimizer),
        **OmegaConf.to_container(conf.config, resolve=True),
    }
