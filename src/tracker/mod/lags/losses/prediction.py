# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Dict

import torch
from mmdet.registry import MODELS
from omegaconf import OmegaConf
from torch import nn

from ....utils.types import PackedTensor
from . import utils


class PredictionLoss(nn.Module):
    def __init__(self, future_len: int, trajectory_loss: OmegaConf):
        super().__init__()

        self.fut_len = future_len

        trajectory_loss = OmegaConf.to_container(trajectory_loss, resolve=True)
        self.loss_traj = MODELS.build(trajectory_loss)

    def trajectory_loss(self, preds, targets, mask) -> Dict[str, torch.Tensor]:
        loss = self.loss_traj(
            preds * mask.unsqueeze(-1),
            targets * mask.unsqueeze(-1),
        )

        return {"loss_for": loss}

    def forward(
        self,
        preds,
        targets,
        assignments,
    ) -> Dict[str, torch.Tensor]:
        n = self.fut_len

        # convert to relative motion
        target_offsets = targets.center.offsets

        target_motion = targets.center.data
        target_motion = target_motion[:, 1 : n + 1] - target_motion[:, :n]
        target_motion = PackedTensor(data=target_motion, offsets=target_offsets)

        target_valid = targets.valid.data
        target_valid = target_valid[:, 1 : n + 1] & target_valid[:, :n]
        target_valid = PackedTensor(data=target_valid, offsets=target_offsets)

        # map targets to predictions
        mask = assignments >= 0
        target_motion = utils.map_targets(
            assignments, target_motion.unbind(), default=0.0, mask=mask
        )
        target_valid = utils.map_targets(
            assignments, target_valid.unbind(), default=False, mask=mask
        )

        # compute actual loss
        return self.trajectory_loss(
            preds[..., :2],
            target_motion[..., :2],
            target_valid,
        )
