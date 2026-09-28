# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from ....utils.types import MetaDict
from ..utils import denormalize_bbox
from .base import Assigner, registry


@registry.register
class DistanceAssigner(Assigner):
    # pylint: disable=too-few-public-methods

    def __init__(self, threshold: float) -> None:
        super().__init__()

        self.threshold = threshold

    @torch.no_grad()
    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> torch.Tensor:
        device = preds.boxes.device
        num_gt, num_pred = targets.boxes.shape[0], preds.boxes.shape[0]

        target_boxes = targets.boxes.detach()
        pred_boxes = preds.boxes.detach()

        # if there are no labels or no predictions, return
        if num_gt == 0 or num_pred == 0:
            return torch.full((num_pred,), -1, device=device, dtype=torch.long)

        # denormalize the predicted boxes
        pred_boxes = denormalize_bbox(pred_boxes)

        pred_centers = pred_boxes[:, :3].unsqueeze(0)
        gt_centers = target_boxes[:, :3].unsqueeze(0)

        distance = torch.cdist(pred_centers, gt_centers)
        distance = distance.squeeze(0)

        # assign indices
        min_distance, indices = distance.min(dim=1)
        indices[min_distance > self.threshold] = -1

        return indices
