# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import abc
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from .....config.registry import Registry
from .....utils.iters import zip_optional
from .....utils.torch import stack_optional
from .....utils.types import MetaDict

registry = Registry("lags.occupancy_predictor")


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> "OccupancyPredictor":
    return registry.from_config(conf)


class OccupancyPredictor(nn.Module, abc.ABC):
    @torch.no_grad()
    def forward(
        self,
        volume_semantics: torch.Tensor | None = None,
        instance_ids: torch.Tensor | None = None,
        instance_class_scores: torch.Tensor | None = None,
        instance_mask_scores: torch.Tensor | None = None,
        instance_valid: torch.Tensor | None = None,
        semantic_class_scores: torch.Tensor | None = None,
        semantic_mask_scores: torch.Tensor | None = None,
        semantic_valid: torch.Tensor | None = None,
    ) -> MetaDict:
        """
        Inference for occupancy prediction.

        Args:
            volume_semantics (torch.Tensor): Semantic occupancy predictions,
                shape [b, c, d, h, w].
            instance_ids (torch.Tensor): Instance IDs for each instance query,
                shape [b, n].
            instance_class_scores (torch.Tensor): Class scores for each
                instance query, shape [b, n, c].
            instance_mask_scores (torch.Tensor): Occupancy mask scores for each
                instance query, shape [b, n, d, h, w].
            instance_valid (torch.Tensor): Mask indicating valid instance
                queries, shape [b, n].
            semantic_class_scores (torch.Tensor): Class scores for each
                semantic query, shape [b, m, c].
            semantic_mask_scores (torch.Tensor): Occupancy mask scores for each
                semantic query, shape [b, m, d, h, w].
            semantic_valid (torch.Tensor): Mask indicating valid semantic
                queries, shape [b, m].

        where:
            - b: batch size
            - c: number of classes
            - d, h, w: dimensions of the occupancy grid
            - n: number of instance queries
            - m: number of semantic queries

        Returns:
            MetaDict: A dictionary containing semantic or panoptic occupancy predictions.
        """
        # generate instance IDs if not provided (e.g., in detection mode)
        if instance_ids is None and instance_mask_scores is not None:
            device = instance_mask_scores.device
            b, n, _, _, _ = instance_mask_scores.shape

            instance_ids = torch.arange(b * n, device=device).view(b, n)

        # decode per batch
        preds = zip_optional(
            volume_semantics,
            instance_ids,
            instance_class_scores,
            instance_mask_scores,
            instance_valid,
            semantic_class_scores,
            semantic_mask_scores,
            semantic_valid,
        )
        preds = [self.process_single(*p) for p in preds]

        semantics, instance_ids = zip(*preds)
        semantics = stack_optional(semantics, dim=0)
        instance_ids = stack_optional(instance_ids, dim=0)

        # collect outputs
        occupancy = MetaDict()
        if semantics is not None:
            occupancy.semantics = semantics
        if instance_ids is not None:
            occupancy.instance_ids = instance_ids

        return occupancy

    @abc.abstractmethod
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
        raise NotImplementedError()
