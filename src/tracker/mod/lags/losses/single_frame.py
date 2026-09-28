# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import einops
import torch
from mmdet.models.losses import FocalLoss
from mmdet.registry import MODELS
from omegaconf import OmegaConf
from torch import distributed as dist
from torch import nn

from ....utils.torch import amp
from ....utils.torch.dist import all_reduce
from ....utils.types import MetaDict
from ..utils import normalize_bbox
from . import utils


class SingleFrameLoss(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        num_classes: int,
        interm_loss: bool,
        code_weights: tuple[float],
        bg_cls_weight: float,
        sync_cls_avg_factor: bool,
        cls_loss: OmegaConf,
        bbox_loss: OmegaConf,
        dice_mask_loss: OmegaConf,
        base_mask_loss: OmegaConf,
        occupancy_mask: str | None = "valid",
        mask_pos_weight: float = 10.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.interm_loss = interm_loss
        self.bg_cls_weight = bg_cls_weight
        self.sync_cls_avg_factor = sync_cls_avg_factor

        cls_loss = OmegaConf.to_container(cls_loss, resolve=True)
        bbox_loss = OmegaConf.to_container(bbox_loss, resolve=True)
        dice_mask_loss = OmegaConf.to_container(dice_mask_loss, resolve=True)
        base_mask_loss = OmegaConf.to_container(base_mask_loss, resolve=True)

        self.loss_cls = MODELS.build(cls_loss)
        self.loss_bbox = MODELS.build(bbox_loss)
        self.loss_mask_dice = MODELS.build(dice_mask_loss)
        self.loss_mask_base = MODELS.build(base_mask_loss)
        self.occupancy_mask = occupancy_mask if occupancy_mask != "none" else None
        self.mask_pos_weight = mask_pos_weight

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

        # NOTE: MMCV's FocalLoss does not support bfloat16 inputs, so we need
        # to upcast the predictions and weights to float32.
        preds = amp.upcast(preds, dtype=torch.float32)
        weights = amp.upcast(weights, dtype=torch.float32)

        # compute the actual loss
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

        # compute the average number of gt boxes accross all gpus for normalization
        num_pos = preds.new_tensor([num_pos])
        num_pos = torch.clamp(all_reduce(num_pos, op=dist.ReduceOp.AVG), min=1).item()

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

    @torch.autocast("cuda", enabled=False)
    def mask_loss(
        self,
        preds: torch.Tensor,  # [num_pos, z, y, x]
        targets: torch.Tensor,  # [num_pos, z, y, x]
        mask: torch.Tensor | None,  # [num_pos, z, y, x]
        voxel_weights: torch.Tensor | None = None,  # [num_pos, z, y, x]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=too-many-locals

        # prepare tensors
        preds = preds.flatten(start_dim=1).to(dtype=torch.float32)
        targets = targets.flatten(start_dim=1)
        mask = mask.flatten(start_dim=1) if mask is not None else True

        # filter out predictions that have no occupied voxels
        valid = (targets & mask).sum(dim=-1) > 0
        mask = mask & valid[:, None]
        mask = mask.expand_as(targets)

        # Optional per-voxel importance weights (from voxel subsampling). With
        # ``voxel_weights is None`` this collapses to the original loss exactly.
        if voxel_weights is not None:
            voxel_weights = voxel_weights.flatten(start_dim=1).to(preds.dtype)
            mask_w = mask * voxel_weights
        else:
            mask_w = mask

        # compute averages for normalization accross all gpus
        num_pos = valid.sum().float()
        num_pos = torch.clamp(all_reduce(num_pos, op=dist.ReduceOp.AVG), min=1)

        # weighted voxel count (== n_valid in expectation under subsampling)
        num_voxels = mask_w.sum().float()
        num_voxels = torch.clamp(all_reduce(num_voxels, op=dist.ReduceOp.AVG), min=1)

        # compute per-voxel weights
        weight = torch.where(targets, self.mask_pos_weight, 1.0)
        weight = weight * mask_w

        # compute base mask loss
        loss_base = self.loss_mask_base(
            preds,
            targets,
            weight=weight,
            avg_factor=num_voxels,
        )

        # compute dice loss
        if voxel_weights is not None:
            loss_dice = utils.weighted_naive_dice_loss(
                torch.where(mask, preds, 0),
                torch.where(mask, targets, 0),
                voxel_weight=mask_w,
                avg_factor=num_pos,
                eps=self.loss_mask_dice.eps,
                loss_weight=self.loss_mask_dice.loss_weight,
            )
        else:
            loss_dice = self.loss_mask_dice(
                pred=torch.where(mask, preds, 0) if mask is not None else preds,
                target=torch.where(mask, targets, 0) if mask is not None else targets,
                avg_factor=num_pos,
            )

        return loss_base, loss_dice

    def layer_loss(
        self,
        preds: MetaDict,
        targets: MetaDict,
        assignments: torch.Tensor,
        layer: int,
    ) -> dict[str, torch.Tensor]:
        # pylint: disable=too-many-locals
        """
        Compute the loss for a specific decoder layer.

        Args:
            preds (MetaDict): The predictions for the current layer.
            targets (MetaDict): The targets for the current layer.
            assignments (torch.Tensor): The assignments for the current layer.
            layer (int): The index of the current decoder layer.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the loss values.
        """
        n_classes = self.num_classes
        _b, _n_layers, n_preds = assignments.shape

        # get targets for the current layer
        assign = assignments[:, layer]  # [b, n_preds]

        valid = assign >= 0

        target_class = targets.class_ids.unbind()
        target_class = utils.map_targets(assign, target_class, n_classes, mask=valid)
        target_class_mask = torch.ones_like(target_class, dtype=torch.float)

        target_boxes = targets.boxes.unbind()
        target_boxes = utils.map_targets(assign, target_boxes, 0.0, mask=valid)
        target_boxes_mask = valid.float()

        target_masks = targets.occupancy.instance_masks.unbind()
        target_masks = utils.map_targets(assign, target_masks, False, mask=valid)

        # get predictions for the current layer
        pred_class = preds.class_scores[:, layer]  # [b, n_preds, n_classes]
        pred_boxes = preds.boxes[:, layer]  # [b, n_preds, box_dim]
        pred_masks = preds.occupancy.instance_scores[:, layer]  # [b, n_preds, z, y, x]

        # get mask for occupancy
        # NOTE: the voxel axis may be flattened to a 1D subsample (see
        # LatentGaussianOccupancyTracker voxel subsampling), so use a
        # rank-agnostic repeat pattern.
        mask_occ = self.occupancy_mask
        if mask_occ is not None:
            mask_occ = targets.occupancy.masks[mask_occ]  # [b, *vox]
            mask_occ = einops.repeat(mask_occ, "b ... -> b n ...", n=n_preds)

        # optional per-voxel importance weights from voxel subsampling
        voxel_weights = None
        if "voxel_weights" in targets.occupancy:
            voxel_weights = targets.occupancy.voxel_weights  # [b, *vox]
            voxel_weights = einops.repeat(voxel_weights, "b ... -> b n ...", n=n_preds)

        # get number of positive and negative samples
        num_pos = valid.sum().item()
        num_neg = valid.numel() - num_pos

        # compute losses
        loss_cls = self.classification_loss(
            pred_class,
            target_class,
            target_class_mask,
            num_pos,
            num_neg,
        )

        loss_bbox = self.box_loss(
            pred_boxes,
            target_boxes,
            target_boxes_mask,
            num_pos,
        )

        loss_mask_base, loss_mask_dice = self.mask_loss(
            pred_masks[valid],  # [num_pos, z, y, x]
            target_masks[valid],  # [num_pos, z, y, x]
            mask=mask_occ[valid] if mask_occ is not None else None,
            voxel_weights=voxel_weights[valid] if voxel_weights is not None else None,
        )

        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_mask_base": loss_mask_base,
            "loss_mask_dice": loss_mask_dice,
        }

    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
        assignments: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _batch_size, num_layers, _num_preds = assignments.shape

        first = 0 if self.interm_loss else num_layers - 1

        losses = {}
        for layer in range(first, num_layers):
            loss = self.layer_loss(preds, targets, assignments, layer)
            losses |= {f"d{layer}/{k}": v for k, v in loss.items()}

        return losses


@MODELS.register_module()
class BinaryFocalLoss(FocalLoss):
    """
    Focal loss for binary classification tasks.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        avg_factor: int | None = None,
        reduction_override: str | None = None,
    ) -> torch.Tensor:
        # NOTE: MMCV's focal loss is intended for multi-class classification.
        #       Specifically, it expects the predictions to be of shape [k,
        #       num_classes] and allows for "no class" labels (with value ==
        #       num_classes). For our mask predictions, we only have one class
        #       (occupied) and model free space as "no class". Hence, we
        #       need to reshape the tensors to [k, 1] and invert the targets.

        return super().forward(
            pred=pred.view(-1, 1),  # [k, "num_classes"=1]
            target=(~target).view(-1).long(),
            weight=weight,
            avg_factor=avg_factor,
            reduction_override=reduction_override,
        )
