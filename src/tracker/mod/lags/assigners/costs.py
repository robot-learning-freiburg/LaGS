# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from abc import ABC, abstractmethod
from typing import Any

import mmcv.ops
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from ....config.registry import Registry
from ....utils.log import get_logger
from ....utils.types import MetaDict

log = get_logger(__name__)

registry = Registry("lags.assigner.costs")

# Temporary instrumentation: set LOG_MASK_COVERAGE=1 to log the masked voxel
# count (valid/camera) that drives the instance-mask cost-matrix memory peak.
_LOG_MASK_COVERAGE = bool(int(os.environ.get("LOG_MASK_COVERAGE", "0")))


def _log_mask_coverage(
    name: str, mask: torch.Tensor, num_preds: int, num_targets: int
) -> None:
    n_valid = int(mask.sum())
    n_total = int(mask.numel())
    log.info(
        "mask=%s coverage=%d/%d (%.1f%%) num_preds=%d num_targets=%d",
        name,
        n_valid,
        n_total,
        100.0 * n_valid / max(n_total, 1),
        num_preds,
        num_targets,
    )


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> "MatchingCost":
    return registry.from_config(conf)


class MatchingCost(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> torch.Tensor:
        raise NotImplementedError()


@registry.register
class BevDistanceCost(MatchingCost):
    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        weight: float = 1.0,
        pc_range: tuple[float] | None = None,
        p: float = 1,
    ) -> None:
        super().__init__()

        self.weight = weight
        self.p = p

        if pc_range is not None:
            pc_range = torch.as_tensor(pc_range, dtype=torch.float32)
            self.register_buffer("pc_range", pc_range, persistent=False)
        else:
            self.pc_range = None

    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> torch.Tensor:
        pred_center = preds.boxes[..., 0:2]
        target_center = targets.boxes[..., 0:2]

        if self.pc_range is not None:
            pred_center -= self.pc_range[0:2]
            pred_center /= self.pc_range[3:5] - self.pc_range[0:2]

            target_center -= self.pc_range[0:2]
            target_center /= self.pc_range[3:5] - self.pc_range[0:2]

        return torch.cdist(pred_center, target_center, p=self.p) * self.weight


@registry.register
class Box3dL1Cost(MatchingCost):
    def __init__(
        self, weight: float = 1.0, pc_range: tuple[float] | None = None
    ) -> None:
        super().__init__()

        self.weight = weight

        if pc_range is not None:
            pc_range = torch.as_tensor(pc_range, dtype=torch.float32)
            self.register_buffer("pc_range", pc_range, persistent=False)
        else:
            self.pc_range = None

    def _normalize(self, boxes: torch.Tensor) -> torch.Tensor:
        pos = boxes[..., 0:3]

        if self.pc_range is not None:
            pos = (pos - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

        size = boxes[..., 3:6]
        size = size.log()

        rot = boxes[..., 6:7]
        rot_s = torch.sin(rot)
        rot_c = torch.cos(rot)

        return torch.cat((pos, size, rot_s, rot_c), dim=-1)

    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> torch.Tensor:
        pred_boxes = self._normalize(preds.boxes)
        target_boxes = self._normalize(targets.boxes)

        return torch.cdist(pred_boxes, target_boxes, p=1) * self.weight


@registry.register
class IoU3DCost(MatchingCost):
    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()

        self.weight = weight

    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> torch.Tensor:
        pred_boxes = preds.boxes[:, 0:7]
        target_boxes = targets.boxes[:, 0:7]

        iou = mmcv.ops.iou3d.boxes_iou3d(pred_boxes, target_boxes)
        cost = 1 - iou

        return cost * self.weight


@registry.register
class FocalCost(MatchingCost):
    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()

        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> torch.Tensor:
        alpha, gamma, eps = self.alpha, self.gamma, self.eps

        pred_cls = preds.class_scores
        target_cls = targets.class_ids

        pred_cls = pred_cls.sigmoid()

        neg_cost = -(1 - pred_cls + eps).log() * (1 - alpha) * pred_cls.pow(gamma)
        pos_cost = -(pred_cls + eps).log() * alpha * (1 - pred_cls).pow(gamma)

        return (pos_cost[:, target_cls] - neg_cost[:, target_cls]) * self.weight


@registry.register
class InstanceMaskCrossEntropyCost(MatchingCost):
    def __init__(
        self,
        mask: str = "valid",
        weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.mask = mask if mask != "none" else None
        self.weight = weight

    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> torch.Tensor:
        masks_pred = preds.occupancy.instance_scores  # [num_preds, z, y, x]
        masks_target = targets.occupancy.instance_masks  # [num_targets, z, y, x]

        # apply mask
        if self.mask is not None:
            mask = targets.occupancy.masks[self.mask]
            if _LOG_MASK_COVERAGE:
                _log_mask_coverage(
                    self.mask, mask, masks_pred.shape[0], masks_target.shape[0]
                )
            masks_pred = masks_pred[:, mask]
            masks_target = masks_target[:, mask]
        else:
            masks_pred = masks_pred.flatten(1)
            masks_target = masks_target.flatten(1)

        # prepare
        masks_target = masks_target.to(masks_pred.dtype)
        n = masks_target.shape[-1]

        # compute positive and negative costs
        pos_cost = F.binary_cross_entropy_with_logits(
            masks_pred, torch.ones_like(masks_pred), reduction="none"
        )

        neg_cost = F.binary_cross_entropy_with_logits(
            masks_pred, torch.zeros_like(masks_pred), reduction="none"
        )

        # compute the total cost matrix
        pos_cost = torch.einsum("nc,mc->nm", pos_cost, masks_target)
        neg_cost = torch.einsum("nc,mc->nm", neg_cost, (1 - masks_target))
        cost = (pos_cost + neg_cost) / n

        return cost * self.weight


@registry.register
class InstanceMaskFocalCost(MatchingCost):
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2,
        eps: float = 1e-12,
        mask: str = "valid",
        max_value: float = 1e3,
        weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.mask = mask if mask != "none" else None
        self.weight = weight
        self.max_value = max_value

    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> torch.Tensor:
        masks_pred = preds.occupancy.instance_scores  # [num_preds, z, y, x]
        masks_target = targets.occupancy.instance_masks  # [num_targets, z, y, x]

        # apply mask
        if self.mask is not None:
            mask = targets.occupancy.masks[self.mask]
            masks_pred = masks_pred[:, mask]
            masks_target = masks_target[:, mask]
        else:
            masks_pred = masks_pred.flatten(1)
            masks_target = masks_target.flatten(1)

        # compute focal cost
        n = masks_target.shape[-1]

        masks_target = masks_target.to(masks_pred.dtype)
        masks_pred = masks_pred.sigmoid()

        # compute positive and negative costs
        pos_cost = (1 - masks_pred).pow(self.gamma) * masks_pred.log()
        pos_cost = -self.alpha * pos_cost

        neg_cost = masks_pred.pow(self.gamma) * (1 - masks_pred).log()
        neg_cost = -(1 - self.alpha) * neg_cost

        # ensure the costs are finite
        pos_cost = torch.nan_to_num(pos_cost, nan=self.max_value, posinf=self.max_value)
        neg_cost = torch.nan_to_num(neg_cost, nan=self.max_value, posinf=self.max_value)

        # compute the total cost matrix
        pos_cost = torch.einsum("nc,mc->nm", pos_cost, masks_target)
        neg_cost = torch.einsum("nc,mc->nm", neg_cost, (1 - masks_target))
        cost = (pos_cost + neg_cost) / n

        # clamp the cost to a maximum value
        cost = cost.clamp(max=self.max_value)

        return cost * self.weight


@registry.register
class InstanceMaskDiceCost(MatchingCost):
    def __init__(
        self,
        naive: bool = True,
        mask: str = "valid",
        eps: float = 1e-3,
        weight: float = 1.0,
    ) -> None:
        super().__init__()

        self.naive = naive
        self.mask = mask if mask != "none" else None
        self.eps = eps
        self.weight = weight

    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> torch.Tensor:
        masks_pred = preds.occupancy.instance_scores  # [num_preds, z, y, x]
        masks_target = targets.occupancy.instance_masks  # [num_targets, z, y, x]

        # apply mask
        if self.mask is not None:
            mask = targets.occupancy.masks[self.mask]
            masks_pred = masks_pred[:, mask]
            masks_target = masks_target[:, mask]
        else:
            masks_pred = masks_pred.flatten(1)
            masks_target = masks_target.flatten(1)

        # compute dice cost
        masks_target = masks_target.to(masks_pred.dtype)
        masks_pred = masks_pred.sigmoid()

        numer = 2 * torch.einsum("nc,mc->nm", masks_pred, masks_target)

        if not self.naive:
            # Note: masks_target is either 0 or 1, so we don't need to square it
            masks_pred = masks_pred**2

        denom = masks_pred.sum(-1)[:, None] + masks_target.sum(-1)[None, :]

        cost = 1 - (numer + self.eps) / (denom + self.eps)

        return cost * self.weight
