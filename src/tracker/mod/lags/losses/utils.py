# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


def weighted_naive_dice_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    voxel_weight: torch.Tensor,
    avg_factor: float,
    eps: float = 1e-3,
    loss_weight: float = 1.0,
) -> torch.Tensor:
    """
    Naive soft-dice loss with per-voxel weights.

    Mirrors mmdet's ``DiceLoss`` (``use_sigmoid=True``, ``activate=True``,
    ``naive_dice=True``, ``reduction='mean'`` with ``avg_factor``) but folds a
    per-voxel weight into the numerator/denominator sums. This is used when the
    voxels have been importance-subsampled, so each kept voxel carries an
    inverse-probability weight and the dice estimate stays unbiased. With
    ``voxel_weight == 1`` everywhere it reduces exactly to the mmdet loss.

    Args:
        pred_logits (torch.Tensor): Predicted logits, shape ``[n, *]``.
        target (torch.Tensor): Target mask, shape ``[n, *]``.
        voxel_weight (torch.Tensor): Per-voxel weight, broadcastable to ``[n, *]``.
        avg_factor (float): Factor to average the summed per-mask loss by.
        eps (float): Numerical-stability epsilon.
        loss_weight (float): Overall loss weight.
    """
    pred = pred_logits.sigmoid().flatten(start_dim=1)
    target = target.flatten(start_dim=1).to(pred.dtype)
    weight = voxel_weight.flatten(start_dim=1).to(pred.dtype)

    a = torch.sum(weight * pred * target, dim=1)
    b = torch.sum(weight * pred, dim=1)
    c = torch.sum(weight * target, dim=1)
    d = (2 * a + eps) / (b + c + eps)

    loss = 1 - d

    return loss.sum() / avg_factor * loss_weight


def map_targets_single(
    assignments: torch.Tensor,
    targets: torch.Tensor,
    default: float = 0.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Map the targets to the predictions based on the assignments for a single
    sample.

    Args:
        assignments (torch.Tensor): The assignments for the predictions.
            Shape: [num_preds].
        tensor (torch.Tensor): The targets to map. Shape: [num_gt, ...].
        default (float): The default value to fill in for unmatched
            predictions. Default is 0.0.
        mask (torch.Tensor | None): An optional mask to apply to the predictions.
            Fields for predictions that are not included in the mask will be
            set to the default value. If provided, it must not include any
            unmatched predictions. The mask should have shape [num_preds] and
            should be a boolean tensor. The default value is None, which means
            that all matched predictions will be considered.

    Returns:
        torch.Tensor: The mapped targets. Shape: [num_preds, ...].
    """
    num_preds, data_shape = assignments.shape[0], targets.shape[1:]
    shape = (num_preds, *data_shape)

    if mask is not None:
        mask = assignments >= 0

    out = torch.full(shape, default, device=targets.device, dtype=targets.dtype)
    out[mask] = targets[assignments[mask]]

    return out


def map_targets(
    assignments: torch.Tensor,
    targets: list[torch.Tensor],
    default: float = 0.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Map the targets to the predictions based on the assignments for a batch
    of samples.

    Args:
        assignments (torch.Tensor): The assignments for the predictions.
            Shape: [batch_size, num_preds].
        targets (list[torch.Tensor]): The targets to map. Each target should
            be a tensor of shape [num_gt, ...]. The length of the list should
            match the batch size.
        default (float): The default value to fill in for unmatched
            predictions. Default is 0.0.
        mask (torch.Tensor | None): An optional mask to apply to the predictions.
            Fields for predictions that are not included in the mask will be
            set to the default value. If provided, it must not include any
            unmatched predictions. The mask should have shape [batch_size, num_preds]
            and should be a boolean tensor. The default value is None, which means
            that all matched predictions will be considered.

    Returns:
        torch.Tensor: The mapped targets. Shape: [batch_size, num_preds, ...].
    """
    if mask is not None:
        mask = assignments >= 0

    out = [
        map_targets_single(a, t, default=default, mask=m)
        for a, t, m in zip(assignments, targets, mask)
    ]

    return torch.stack(out, dim=0)
