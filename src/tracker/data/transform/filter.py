# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, List, Literal, Mapping

import torch

from ...utils.math import interesect_box_aabb
from ...utils.types import MetaDict, PackedArray, PackedTensor, Sample
from .registry import registry as transform
from .transform import Transform


@transform.register
class FilterInstancesByRange(Transform):
    """
    Filter instance boxes by range.
    """

    # pylint: disable-next=redefined-builtin
    def __init__(self, range: List[float], check: Literal["center", "box"] = "box"):
        # range needs to be either of size 4 (2D) or 6 (3D)"
        assert len(range) == 4 or len(range) == 6

        range = torch.tensor(range)

        # break up range
        self.ndim = range.shape[0] // 2
        self.aabb_min = range[: self.ndim]
        self.aabb_max = range[self.ndim :]

        self.check = check

        if check == "center":
            self._get_range_mask = self._range_mask_center
        elif check == "box":
            self._get_range_mask = self._range_mask_box
        else:
            raise ValueError(f"unknown check type '{check}'")

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"filter.{self.__class__.__name__}",
            "range": self.aabb_min.tolist() + self.aabb_max.tolist(),
            "check": self.check,
        }

    def _range_mask_center(self, boxes: torch.Tensor) -> torch.Tensor:
        """Simplified check by box center only"""
        center = boxes[:, 0 : 0 + self.ndim]

        mask = (center > self.aabb_min[None, :]) & (center < self.aabb_max[None, :])
        mask = mask.all(1)

        return mask

    def _range_mask_box(self, boxes: torch.Tensor) -> torch.Tensor:
        """Accurate check by box intersection"""

        return interesect_box_aabb(
            box_center=boxes[:, 0 : 0 + self.ndim],
            box_dim=boxes[:, 3 : 3 + self.ndim],
            box_rot=boxes[:, 6],
            aabb_min=self.aabb_min,
            aabb_max=self.aabb_max,
        )

    def apply(self, sample: Sample) -> Sample:
        if "labels" not in sample or "boxes" not in sample.labels:
            return sample

        labels: MetaDict = sample.labels
        assert sample.batch_size == 1
        assert labels.boxes.batch_size == 1
        assert labels.instance_ids.batch_size == 1
        assert labels.class_ids.batch_size == 1
        assert labels.class_names.batch_size == 1

        # compute mask
        mask = self._get_range_mask(labels.boxes.data)

        # apply mask
        labels.boxes = PackedTensor(labels.boxes.data[mask, :])
        labels.instance_ids = PackedTensor(labels.instance_ids.data[mask])
        labels.class_ids = PackedTensor(labels.class_ids.data[mask])
        labels.class_names = PackedArray(labels.class_names.data[mask.numpy()])

        if "trajectories" in labels:
            t = labels.trajectories

            assert t.center.batch_size == 1
            assert t.valid.batch_size == 1

            t.center = PackedTensor(t.center.data[mask, :])
            t.valid = PackedTensor(t.valid.data[mask])

        return sample


@transform.register
class FilterInstancesByClass(Transform):
    """
    Filter instance boxes by class names.
    """

    def __init__(self, classes: List[str]):
        self.classes = classes

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {"type": f"filter.{self.__class__.__name__}", "classes": self.classes}

    def apply(self, sample: Sample) -> Sample:
        if "labels" not in sample or "boxes" not in sample.labels:
            return sample

        labels: MetaDict = sample.labels
        assert sample.batch_size == 1
        assert labels.boxes.batch_size == 1
        assert labels.instance_ids.batch_size == 1
        assert labels.class_ids.batch_size == 1
        assert labels.class_names.batch_size == 1

        # compute mask
        mask = [name in self.classes for name in labels.class_names.data]
        mask = torch.tensor(mask, dtype=torch.bool)

        # apply mask
        labels.boxes = PackedTensor(labels.boxes.data[mask, :])
        labels.instance_ids = PackedTensor(labels.instance_ids.data[mask])
        labels.class_ids = PackedTensor(labels.class_ids.data[mask])
        labels.class_names = PackedArray(labels.class_names.data[mask.numpy()])

        if "trajectories" in labels:
            t = labels.trajectories

            assert t.center.batch_size == 1
            assert t.valid.batch_size == 1

            t.center = PackedTensor(t.center.data[mask, :])
            t.valid = PackedTensor(t.valid.data[mask])

        return sample


@transform.register
class DropSamplesWithoutGtBoxes(Transform):
    """
    Filter out samples with no ground-truth bounding box labels.

    Note: For this to work, the dataset wrapper needs to be set to actively
    re-sample None-type samples (this can be done by setting
    `on_none='resample"`).
    """

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {"type": f"filter.{self.__class__.__name__}"}

    def apply(self, sample: Sample) -> Sample:
        if sample.labels.boxes.data.shape[0] == 0:
            return None

        return sample
