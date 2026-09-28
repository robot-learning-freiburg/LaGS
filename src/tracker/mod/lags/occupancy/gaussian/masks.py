# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Sequence

import torch
from omegaconf import OmegaConf

from ...utils import build_class_list, build_label_map
from ..utils import masks_to_volume
from .aggregator import GaussianSemanticAggregator
from .base import OccupancyPredictor, registry


@registry.register
class SemanticGaussianPredictor(OccupancyPredictor):
    def __init__(
        self,
        voxel_size: Sequence[float] = (0.4, 0.4, 0.4),
        voxel_range: Sequence[float] = (-40.0, -40.0, -1.0, 40.0, 40.0, 10.0),
        scale_multiplier: float = 5.0,
    ):
        """
        Basic semantic mask predictor for occupancy prediction. This module uses
        the provided semantic Gaussian predictions to compute occupancy semantics.
        """
        super().__init__()

        self.aggregator = GaussianSemanticAggregator(
            voxel_size=voxel_size,
            voxel_range=voxel_range,
            scale_multiplier=scale_multiplier,
        )

    @torch.no_grad()
    def process_single(
        self,
        volume_semantics: torch.Tensor | None,  # [c, d, h, w]
        instance_ids: torch.Tensor | None,  # [n]
        instance_class_scores: torch.Tensor | None,  # [n, c]
        instance_mask_scores: torch.Tensor | None,  # [n, d, h, w]
        instance_valid: torch.Tensor | None,  # [n]
        semantic_gauss_logits: torch.Tensor | None = None,  # [n_gaussians, n_classes]
        semantic_gauss_centers: torch.Tensor | None = None,  # [n_gaussians, 3]
        semantic_gauss_scales: torch.Tensor | None = None,  # [n_gaussians, 3]
        semantic_gauss_rotations: torch.Tensor | None = None,  # [n_gaussians, 4]
        semantic_gauss_opacities: torch.Tensor | None = None,  # [n_gaussians]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:  # [d, h, w], [d, h, w]
        semantics = self.aggregator(
            logits=semantic_gauss_logits[None, None, ...],
            centers=semantic_gauss_centers[None, None, ...],
            scales=semantic_gauss_scales[None, None, ...],
            rotations=semantic_gauss_rotations[None, None, ...],
            opacities=semantic_gauss_opacities[None, None, ...],
            mask=None,
        )
        semantics = semantics.argmax(dim=0)

        return semantics, None


@registry.register
class PanopticGaussianPredictor(OccupancyPredictor):
    # pylint: disable=too-many-instance-attributes

    # buffers
    instance_to_target_map: torch.Tensor
    semantic_to_target_map: torch.Tensor
    classes_thing: torch.Tensor

    def __init__(
        self,
        target_labels: OmegaConf,
        instance_labels: Sequence[str],
        semantic_labels: Sequence[str],
        instance_class_score_threshold: float = 0.2,
        instance_occupancy_score_threshold: float = 0.5,
        instance_overlap_threshold: float = 0.7,
        voxel_size: Sequence[float] = (0.4, 0.4, 0.4),
        voxel_range: Sequence[float] = (-40.0, -40.0, -1.0, 40.0, 40.0, 10.0),
        scale_multiplier: float = 5.0,
    ):
        super().__init__()

        self.target_labels = target_labels
        self.instance_labels = instance_labels
        self.semantic_labels = semantic_labels

        self.instance_class_score_threshold = instance_class_score_threshold
        self.instance_occupancy_score_threshold = instance_occupancy_score_threshold
        self.instance_overlap_threshold = instance_overlap_threshold

        # mapping from instance/box labels to semantic occupancy labels
        inst_to_tgt = build_label_map(instance_labels, target_labels.all)
        self.register_buffer("instance_to_target_map", inst_to_tgt, persistent=False)

        # mapping from semantic query labels to semantic occupancy labels
        sem_to_tgt = build_label_map(semantic_labels, target_labels.all)
        self.register_buffer("semantic_to_target_map", sem_to_tgt, persistent=False)

        # class IDs for thing classes
        classes_thing = build_class_list(target_labels.all, target_labels.thing)
        self.register_buffer("classes_thing", classes_thing, persistent=False)

        # class ID for free space
        self.class_free = target_labels.all.index(target_labels.free[0])

        self.aggregator = GaussianSemanticAggregator(
            voxel_size=voxel_size,
            voxel_range=voxel_range,
            scale_multiplier=scale_multiplier,
        )

    @torch.no_grad()
    def process_single(
        self,
        volume_semantics: torch.Tensor | None,  # [c, d, h, w]
        instance_ids: torch.Tensor | None,  # [n]
        instance_class_scores: torch.Tensor | None,  # [n, c]
        instance_mask_scores: torch.Tensor | None,  # [n, d, h, w]
        instance_valid: torch.Tensor | None,  # [n]
        semantic_gauss_logits: torch.Tensor | None = None,  # [n_gaussians, n_classes]
        semantic_gauss_centers: torch.Tensor | None = None,  # [n_gaussians, 3]
        semantic_gauss_scales: torch.Tensor | None = None,  # [n_gaussians, 3]
        semantic_gauss_rotations: torch.Tensor | None = None,  # [n_gaussians, 4]
        semantic_gauss_opacities: torch.Tensor | None = None,  # [n_gaussians]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:  # [d, h, w], [d, h, w]
        # pylint: disable=too-many-locals

        # prepare class and mask scores
        instance_class_scores = instance_class_scores.sigmoid()
        instance_mask_scores = instance_mask_scores.sigmoid()

        # compute class labels and scores
        instance_scores, instance_labels = instance_class_scores.max(dim=-1)

        # map semantic and instance class labels to target occupancy classes
        instance_labels = self.instance_to_target_map[instance_labels]

        # filter out invalid and low-confidence predictions
        instance_threshold = self.instance_class_score_threshold

        instance_valid = instance_valid if instance_valid is not None else True
        instance_valid = instance_valid & (instance_labels >= 0)
        instance_valid = instance_valid & (instance_scores > instance_threshold)

        # apply filters
        instance_ids = instance_ids[instance_valid]
        instance_scores = instance_scores[instance_valid]
        instance_labels = instance_labels[instance_valid]
        instance_mask_scores = instance_mask_scores[instance_valid]

        # generate occupancy volume for instance predictions
        vol_instance, iids = masks_to_volume(
            class_scores=instance_scores,
            class_labels=instance_labels,
            instance_ids=instance_ids,
            instance_scores=instance_mask_scores,
            default_class=self.class_free,
            default_iid=-1,  # no instance IDs
            occupancy_score_threshold=self.instance_occupancy_score_threshold,
            overlap_threshold=self.instance_overlap_threshold,
        )

        # generate occupancy volume for semantic predictions
        vol_semantic, _, _ = self.aggregator(
            logits=semantic_gauss_logits[None, None, ...],
            centers=semantic_gauss_centers[None, None, ...],
            scales=semantic_gauss_scales[None, None, ...],
            rotations=semantic_gauss_rotations[None, None, ...],
            opacities=semantic_gauss_opacities[None, None, ...],
            mask=None,
        )

        # drop inserted batch and layer dimensions
        vol_semantic = vol_semantic[0, 0]

        # convert semantic volume to target labels
        vol_semantic = vol_semantic.argmax(dim=-1)
        vol_semantic = self.semantic_to_target_map[vol_semantic]

        # reshape to [z, y, x]
        vol_semantic = vol_semantic.view(vol_instance.shape)

        # combine instance-mask- and semantic-mask-based volume predictions
        is_stuff = torch.isin(vol_semantic, self.classes_thing, invert=True)
        is_thing = torch.isin(vol_instance, self.classes_thing)
        semantics = torch.where(is_stuff, vol_semantic, self.class_free)
        semantics = torch.where(is_thing, vol_instance, semantics)

        return semantics, iids
