# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from .base import OccupancyPredictor, registry


@registry.register
class DirectVolumeSemanticPredictor(OccupancyPredictor):
    """
    Direct volume semantic predictor for occupancy prediction. This module uses
    the provided semantic volume predictions without additional processing.

    Only provides semantic occupancy predictions, not panoptic or instance
    predictions.
    """

    @torch.no_grad()
    def process_single(
        self,
        volume_semantics: torch.Tensor | None,  # [c, d, h, w]
        instance_ids: torch.Tensor | None,  # [n]
        instance_class_scores: torch.Tensor | None,  # [n, c]
        instance_mask_scores: torch.Tensor | None,  # [n, d, h, w]
        instance_valid: torch.Tensor | None,  # [n]
        semantic_class_scores: torch.Tensor | None,  # [m, c]
        semantic_mask_scores: torch.Tensor | None,  # [m, d, h, w]
        semantic_valid: torch.Tensor | None,  # [m]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:  # [d, h, w], [d, h, w]
        return torch.argmax(volume_semantics, dim=0), None
