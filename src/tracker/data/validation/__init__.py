# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Sequence

import lightning as L
from omegaconf import OmegaConf

from ...utils.types import MetaDict, Sample
from . import base_metrics, nuscenes, occupancy
from .metric import EmptyValidationMetrics, ValidationMetrics
from .registry import registry


class ValidationCallback(L.Callback):
    def __init__(self, metrics: ValidationMetrics):
        self.metrics = metrics

    def setup(self, trainer, pl_module, stage):
        self.metrics.to(pl_module.device)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: MetaDict,
        batch: Sample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.metrics.update(sample=batch, preds=outputs)

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        # Compute the validation metrics. Note: This already synchronizes state
        # across all distributed processes. Hence, we don't need to synchronize
        # again below.
        metrics = self.metrics.compute()
        self.metrics.reset()

        # Log the metrics.
        if metrics:
            pl_module.log_dict(metrics, sync_dist=False)


def get(key: str) -> Any:
    return registry.get(key)


def build_metrics(conf: OmegaConf | None) -> ValidationMetrics:
    if conf is None:
        return None

    return registry.from_config(conf)


def build_callbacks(conf: OmegaConf | None) -> list[ValidationCallback]:
    if conf is None:
        return []

    if not isinstance(conf, Sequence):
        conf = [conf]

    metrics = [build_metrics(cfg) for cfg in conf]
    metrics = [m for m in metrics if m is not None]

    return [ValidationCallback(m) for m in metrics]
