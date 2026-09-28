# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Mapping, Sequence

import torch
from omegaconf import OmegaConf

from ...utils import build_class_list, build_label_map
from ..utils import masks_to_volume
from .base import OccupancyPredictor, registry


@registry.register
class BasicSemanticMaskPredictor(OccupancyPredictor):
    def __init__(self, class_score_threshold: float = 0.0):
        """
        Basic semantic mask predictor for occupancy prediction. This module uses
        the provided semantic mask predictions to compute occupancy semantics.

        Expects that the semantic masks cover all classes, including free space.

        Args:
            class_score_threshold (float): Threshold for class scores to filter
                out low-confidence predictions.
        """
        super().__init__()

        self.class_score_threshold = class_score_threshold

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
        class_scores = semantic_class_scores.sigmoid()
        mask_scores = semantic_mask_scores.sigmoid()

        # compute class labels and scores of the semantic queries
        class_confidence, _class_labels = class_scores.max(dim=-1)

        # filter out low-confidence predictions
        valid = semantic_valid if semantic_valid is not None else True
        valid = valid & (class_confidence > self.class_score_threshold)

        class_scores = class_scores[valid]
        mask_scores = mask_scores[valid]

        # compute class scores by aggregating over all selected masks
        semantics = torch.einsum("mc,mdhw->cdhw", class_scores, mask_scores)
        semantics = semantics.argmax(dim=0)

        return semantics, None


@registry.register
class MergingSemanticMaskPredictor(OccupancyPredictor):
    def __init__(
        self,
        labels: OmegaConf,
        class_score_threshold: float = 0.2,
        occupancy_score_threshold: float = 0.5,
        overlap_threshold: float = 0.7,
        exclude_free: bool = True,
    ):
        super().__init__()

        self.class_score_threshold = class_score_threshold
        self.occupancy_score_threshold = occupancy_score_threshold
        self.overlap_threshold = overlap_threshold
        self.exclude_free = exclude_free

        self.class_free = labels.all.index(labels.free[0])

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
        class_scores = semantic_class_scores.sigmoid()
        mask_scores = semantic_mask_scores.sigmoid()

        # compute class labels and scores of the semantic queries
        class_scores, class_labels = class_scores.max(dim=-1)

        # filter out low-confidence predictions
        valid = semantic_valid if semantic_valid is not None else True
        valid = valid & (class_scores > self.class_score_threshold)

        # exclude free space class if requested
        if self.exclude_free:
            valid = valid & (class_labels != self.class_free)

        class_scores = class_scores[valid]
        class_labels = class_labels[valid]
        mask_scores = mask_scores[valid]

        # merge masks across all selected queries
        semantics, _ = masks_to_volume(
            class_scores=class_scores,
            class_labels=class_labels,
            instance_ids=torch.zeros_like(class_labels, dtype=torch.long),
            instance_scores=mask_scores,
            default_class=self.class_free,
            default_iid=-1,  # no instance IDs
            occupancy_score_threshold=self.occupancy_score_threshold,
            overlap_threshold=self.overlap_threshold,
        )

        return semantics, None


@registry.register
class PanopticMaskPredictor(OccupancyPredictor):
    # pylint: disable=too-many-instance-attributes

    # buffers
    instance_to_target_map: torch.Tensor
    semantic_to_target_map: torch.Tensor
    classes_thing: torch.Tensor
    occupancy_score_threshold: torch.Tensor

    def __init__(
        self,
        target_labels: OmegaConf,
        instance_labels: Sequence[str],
        semantic_labels: Sequence[str],
        instance_class_score_threshold: float = 0.2,
        semantic_class_score_threshold: float = 0.2,
        occupancy_score_threshold: float | dict[str, float] = 0.5,
        overlap_threshold: float = 0.0,
        exclude_free: bool = True,
    ):
        # pylint: disable=too-many-locals
        super().__init__()

        self.target_labels = target_labels
        self.instance_labels = instance_labels
        self.semantic_labels = semantic_labels

        self.instance_class_score_threshold = instance_class_score_threshold
        self.semantic_class_score_threshold = semantic_class_score_threshold
        self.overlap_threshold = overlap_threshold
        self.exclude_free = exclude_free

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

        # per-class occupancy score thresholds [num_classes]
        occ_th = self._build_occupancy_score_threshold(occupancy_score_threshold)
        self.register_buffer("occupancy_score_threshold", occ_th, persistent=False)

    def _build_occupancy_score_threshold(
        self, threshold: float | dict[str, float]
    ) -> torch.Tensor:
        num_classes = len(self.target_labels.all)

        default = threshold if isinstance(threshold, float) else 0.5

        per_class = torch.full((num_classes,), default, dtype=torch.float)
        if isinstance(threshold, Mapping):
            for cls_str, val in threshold.items():
                per_class[self.target_labels.all.index(cls_str)] = val

        return per_class

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
        # pylint: disable=too-many-locals

        # prepare class and mask scores
        instance_class_scores = instance_class_scores.sigmoid()
        instance_mask_scores = instance_mask_scores.sigmoid()

        semantic_class_scores = semantic_class_scores.sigmoid()
        semantic_mask_scores = semantic_mask_scores.sigmoid()

        # compute class labels and scores
        instance_scores, instance_labels = instance_class_scores.max(dim=-1)
        semantic_scores, semantic_labels = semantic_class_scores.max(dim=-1)

        # map semantic and instance class labels to target occupancy classes
        instance_labels = self.instance_to_target_map[instance_labels]
        semantic_labels = self.semantic_to_target_map[semantic_labels]

        # filter out invalid and low-confidence predictions
        instance_threshold = self.instance_class_score_threshold
        semantic_threshold = self.semantic_class_score_threshold

        instance_valid = instance_valid if instance_valid is not None else True
        instance_valid = instance_valid & (instance_labels >= 0)
        instance_valid = instance_valid & (instance_scores > instance_threshold)

        semantic_valid = semantic_valid if semantic_valid is not None else True
        semantic_valid = semantic_valid & (semantic_labels >= 0)
        semantic_valid = semantic_valid & (semantic_scores > semantic_threshold)

        # filter out thing predictions from semantic predictions
        semantic_valid &= torch.isin(semantic_labels, self.classes_thing, invert=True)

        # filter out non-thing predictions from instance predictions
        # NOTE: This should be a no-op if the instance labels are already
        #       restricted to thing classes.
        instance_valid &= torch.isin(instance_labels, self.classes_thing, invert=False)

        # exclude free space class if requested
        if self.exclude_free:
            semantic_valid &= semantic_labels != self.class_free

        # apply filters
        instance_ids = instance_ids[instance_valid]
        instance_scores = instance_scores[instance_valid]
        instance_labels = instance_labels[instance_valid]
        instance_mask_scores = instance_mask_scores[instance_valid]

        semantic_scores = semantic_scores[semantic_valid]
        semantic_labels = semantic_labels[semantic_valid]
        semantic_mask_scores = semantic_mask_scores[semantic_valid]

        # dummy instance IDs for semantic predictions
        semantic_ids = torch.full_like(semantic_labels, -1, dtype=torch.long)

        # concatenate instance and semantic predictions
        object_ids = torch.cat((instance_ids, semantic_ids), dim=0)
        class_scores = torch.cat((instance_scores, semantic_scores), dim=0)
        class_labels = torch.cat((instance_labels, semantic_labels), dim=0)
        mask_scores = torch.cat((instance_mask_scores, semantic_mask_scores), dim=0)

        return masks_to_volume(
            class_scores=class_scores,
            class_labels=class_labels,
            instance_ids=object_ids,
            instance_scores=mask_scores,
            default_class=self.class_free,
            default_iid=-1,  # no instance IDs
            occupancy_score_threshold=self.occupancy_score_threshold,
            overlap_threshold=self.overlap_threshold,
        )


@registry.register
class PanopticSplitMaskPredictor(OccupancyPredictor):
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
        semantic_class_score_threshold: float = 0.2,
        instance_occupancy_score_threshold: float = 0.5,
        semantic_occupancy_score_threshold: float = 0.5,
        instance_overlap_threshold: float = 0.0,
        semantic_overlap_threshold: float = 0.0,
        exclude_free: bool = True,
        pre_filter_classes: bool = False,
    ):
        super().__init__()

        self.target_labels = target_labels
        self.instance_labels = instance_labels
        self.semantic_labels = semantic_labels

        self.instance_class_score_threshold = instance_class_score_threshold
        self.semantic_class_score_threshold = semantic_class_score_threshold

        self.instance_occupancy_score_threshold = instance_occupancy_score_threshold
        self.semantic_occupancy_score_threshold = semantic_occupancy_score_threshold

        self.instance_overlap_threshold = instance_overlap_threshold
        self.semantic_overlap_threshold = semantic_overlap_threshold

        self.exclude_free = exclude_free
        self.pre_filter_classes = pre_filter_classes

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
        # pylint: disable=too-many-locals

        # prepare class and mask scores
        instance_class_scores = instance_class_scores.sigmoid()
        instance_mask_scores = instance_mask_scores.sigmoid()

        semantic_class_scores = semantic_class_scores.sigmoid()
        semantic_mask_scores = semantic_mask_scores.sigmoid()

        # compute class labels and scores
        instance_scores, instance_labels = instance_class_scores.max(dim=-1)
        semantic_scores, semantic_labels = semantic_class_scores.max(dim=-1)

        # map semantic and instance class labels to target occupancy classes
        instance_labels = self.instance_to_target_map[instance_labels]
        semantic_labels = self.semantic_to_target_map[semantic_labels]

        # filter out invalid and low-confidence predictions
        instance_threshold = self.instance_class_score_threshold
        semantic_threshold = self.semantic_class_score_threshold

        instance_valid = instance_valid if instance_valid is not None else True
        instance_valid = instance_valid & (instance_labels >= 0)
        instance_valid = instance_valid & (instance_scores > instance_threshold)

        semantic_valid = semantic_valid if semantic_valid is not None else True
        semantic_valid = semantic_valid & (semantic_labels >= 0)
        semantic_valid = semantic_valid & (semantic_scores > semantic_threshold)

        # filter out thing predictions from semantic predictions
        if self.pre_filter_classes:
            semantic_valid &= torch.isin(
                semantic_labels, self.classes_thing, invert=True
            )

        # filter out non-thing predictions from instance predictions
        # NOTE: This should be a no-op if the instance labels are already
        #       restricted to thing classes.
        if self.pre_filter_classes:
            instance_valid &= torch.isin(
                instance_labels, self.classes_thing, invert=False
            )

        # exclude free space class if requested
        if self.exclude_free:
            semantic_valid &= semantic_labels != self.class_free

        # apply filters
        instance_ids = instance_ids[instance_valid]
        instance_scores = instance_scores[instance_valid]
        instance_labels = instance_labels[instance_valid]
        instance_mask_scores = instance_mask_scores[instance_valid]

        semantic_scores = semantic_scores[semantic_valid]
        semantic_labels = semantic_labels[semantic_valid]
        semantic_mask_scores = semantic_mask_scores[semantic_valid]

        # dummy instance IDs for semantic predictions
        semantic_ids = torch.full_like(semantic_labels, -1, dtype=torch.long)

        # generate occupancy volume for semantic predictions
        vol_semantic, _ = masks_to_volume(
            class_scores=semantic_scores,
            class_labels=semantic_labels,
            instance_ids=semantic_ids,
            instance_scores=semantic_mask_scores,
            default_class=self.class_free,
            default_iid=-1,  # no instance IDs
            occupancy_score_threshold=self.semantic_occupancy_score_threshold,
            overlap_threshold=self.semantic_overlap_threshold,
        )

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

        # combine instance-mask- and semantic-mask-based volume predictions
        is_stuff = torch.isin(vol_semantic, self.classes_thing, invert=True)
        is_thing = torch.isin(vol_instance, self.classes_thing)
        semantics = torch.where(is_stuff, vol_semantic, self.class_free)
        semantics = torch.where(is_thing, vol_instance, semantics)

        return semantics, iids
