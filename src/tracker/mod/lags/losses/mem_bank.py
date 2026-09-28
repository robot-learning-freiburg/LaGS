# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Dict, Tuple

import torch
from mmdet.registry import MODELS
from omegaconf import OmegaConf
from torch import distributed as dist
from torch import nn

from ....utils.torch import amp
from ....utils.torch.dist import all_reduce
from ..utils import normalize_bbox
from . import utils


class MemBankLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        code_weights: Tuple[float],
        bg_cls_weight: float,
        sync_cls_avg_factor: bool,
        cls_loss: OmegaConf,
        bbox_loss: OmegaConf,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.bg_cls_weight = bg_cls_weight
        self.sync_cls_avg_factor = sync_cls_avg_factor

        cls_loss = OmegaConf.to_container(cls_loss, resolve=True)
        bbox_loss = OmegaConf.to_container(bbox_loss, resolve=True)

        self.loss_cls = MODELS.build(cls_loss)
        self.loss_bbox = MODELS.build(bbox_loss)

        code_weights = torch.as_tensor(code_weights)
        self.register_buffer("code_weights", code_weights, persistent=False)

    def classification_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        num_pos: int,
        num_neg: int,
    ) -> torch.Tensor:
        """
        Compute the classification loss.

        Args:
            preds (torch.Tensor): The predicted class scores. Shape: [b, num_preds, num_classes].
            targets (torch.Tensor): The target class labels. Shape: [b, num_preds].
            weights (torch.Tensor): The weights for the target classes. Shape: [b, num_preds].
            num_pos (int): The number of positive samples.
            num_neg (int): The number of negative samples.

        Returns:
            torch.Tensor: The classification loss.
        """
        # flatten the tensors across batch dimensions
        preds = preds.flatten(end_dim=-2)
        targets = targets.flatten()
        weights = weights.flatten()

        # construct weighted avg_factor to match with the official DETR repo
        avg_factor = num_pos * 1.0 + num_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            avg_factor = preds.new_tensor([avg_factor])
            avg_factor = all_reduce(avg_factor, op=dist.ReduceOp.AVG)

        avg_factor = max(avg_factor, 1)

        # compute the actual loss
        preds = amp.upcast(preds, dtype=torch.float32)
        weights = amp.upcast(weights, dtype=torch.float32)

        loss = self.loss_cls(preds, targets, weights, avg_factor=avg_factor)
        loss = torch.nan_to_num(loss)

        return loss

    def box_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        num_pos: int,
    ) -> torch.Tensor:
        """
        Compute the box regression loss.

        Args:
            preds (torch.Tensor): The predicted bounding box coordinates. Shape: [b, num_preds, n].
            targets (torch.Tensor): The target bounding box coordinates. Shape: [b, num_preds, m].
            weights (torch.Tensor): The weights for the target boxes. Shape: [b, num_preds].
            num_pos (int): The number of positive samples.

        Returns:
            torch.Tensor: The box regression loss.
        """
        # flatten the tensors across batch dimensions
        preds = preds.flatten(end_dim=-2)
        targets = targets.flatten(end_dim=-2)
        weights = weights.flatten()

        # normalize the targets
        targets = normalize_bbox(targets)

        # padded targets may contain NaN values after normalization, filter them out
        mask = torch.isfinite(targets).all(dim=-1)

        preds = preds[mask]
        targets = targets[mask]
        weights = weights[mask]

        # compute box property weights
        weights = weights[:, None] * self.code_weights

        # compute the actual loss
        loss = self.loss_bbox(preds, targets, weights, avg_factor=num_pos)
        loss = torch.nan_to_num(loss)

        return loss

    def forward(self, track_instances, targets, assignments) -> Dict[str, torch.Tensor]:
        # get predictions
        pred_class = track_instances.cache_logits.unsqueeze(0)
        pred_boxes = track_instances.cache_bboxes.unsqueeze(0)

        # map targets to predictions
        valid = assignments >= 0

        target_class = utils.map_targets(
            assignments, targets.class_ids.unbind(), default=-1, mask=valid
        )
        target_class_mask = torch.ones_like(target_class, dtype=torch.float)

        target_boxes = utils.map_targets(
            assignments, targets.boxes.unbind(), default=0.0, mask=valid
        )
        target_boxes_mask = valid.float()

        num_pos = valid.sum().item()
        num_neg = valid.numel() - num_pos

        # compute class refinement loss
        loss_cls = self.classification_loss(
            pred_class,
            target_class,
            target_class_mask,
            num_pos=num_pos,
            num_neg=num_neg,
        )

        # compute box refinement loss
        loss_bbox = self.box_loss(
            pred_boxes,
            target_boxes,
            target_boxes_mask,
            num_pos=num_pos,
        )

        return {
            "loss_mem_cls": loss_cls,
            "loss_mem_bbox": loss_bbox,
        }
