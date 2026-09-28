# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Sequence

import torch
from omegaconf import OmegaConf

from ...utils import build_class_list, build_label_map
from ..utils import masks_to_volume
from .base import OccupancyPredictor, registry


@registry.register
class PanopticHybridPredictor(OccupancyPredictor):
    # buffers
    instance_to_target_map: torch.Tensor
    semantic_to_target_map: torch.Tensor
    classes_thing: torch.Tensor

    def __init__(
        self,
        target_labels: OmegaConf,
        instance_labels: Sequence[str],
        semantic_labels: Sequence[str],
        class_score_threshold: float = 0.2,
        occupancy_score_threshold: float = 0.5,
        overlap_threshold: float = 0.7,
    ):
        super().__init__()

        self.target_labels = target_labels
        self.instance_labels = instance_labels
        self.semantic_labels = semantic_labels

        self.class_score_threshold = class_score_threshold
        self.occupancy_score_threshold = occupancy_score_threshold
        self.overlap_threshold = overlap_threshold

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
        # prepare class and mask scores
        instance_class_scores = instance_class_scores.sigmoid()
        instance_mask_scores = instance_mask_scores.sigmoid()

        # direct volume-based semantic occupancy prediction
        volume_semantics = torch.argmax(volume_semantics, dim=0)

        # map semantic class scores to target occupancy classes
        volume_semantics = self.semantic_to_target_map[volume_semantics]

        # compute classes and scores of instances
        instance_class_scores, class_labels = instance_class_scores.max(dim=-1)

        # map instance class labels to target occupancy classes
        class_labels = self.instance_to_target_map[class_labels]

        # mask out invalid classes after mapping
        instance_valid = instance_valid if instance_valid is not None else True
        instance_valid = instance_valid & (class_labels >= 0)

        # mask out low-confidence instances
        instance_valid &= instance_class_scores > self.class_score_threshold

        # apply masks
        instance_class_scores = instance_class_scores[instance_valid]
        class_labels = class_labels[instance_valid]
        instance_ids = instance_ids[instance_valid]
        instance_mask_scores = instance_mask_scores[instance_valid]

        # convert instance occupancy masks to volume-based predictions
        mask_semantics, mask_iid = masks_to_volume(
            class_scores=instance_class_scores,
            class_labels=class_labels,
            instance_ids=instance_ids,
            instance_scores=instance_mask_scores,
            default_class=-1,
            default_iid=-1,
            occupancy_score_threshold=self.occupancy_score_threshold,
            overlap_threshold=self.overlap_threshold,
        )

        # combine volume- and mask-based semantic predictions
        # - take stuff predictions from volume-based predictions
        # - take thing predictions from mask-based predictions
        is_stuff = torch.isin(volume_semantics, self.classes_thing, invert=True)
        is_thing = torch.isin(mask_semantics, self.classes_thing)
        semantics = torch.where(is_stuff, volume_semantics, self.class_free)
        semantics = torch.where(is_thing, mask_semantics, semantics)

        return semantics, mask_iid
