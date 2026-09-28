# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import lightning as L

from ...utils.types import MetaDict, Sample


class PredictionWriter(L.Callback):
    def on_predict_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: MetaDict,
        batch: Sample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        pass

    def on_predict_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        pass
