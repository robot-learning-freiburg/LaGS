# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from ....utils.types import MetaDict, uncollate
from .base import INDEX_UNMATCHED


@torch.no_grad()
def get_unmatched_preds_mask(assigned_gt_indices: torch.Tensor) -> torch.Tensor:
    """
    Get the mask of the unmatched predictions.

    Args:
        assigned_gt_indices (torch.Tensor): Indices of the matched targets
            for each prediction. Tensor of shape [num_preds].

    Returns:
        torch.Tensor: Mask of the unmatched predictions. Entries are True
            if the corresponding prediction is unmatched, and False
            otherwise. Tensor of shape [num_preds].
    """
    return assigned_gt_indices == INDEX_UNMATCHED


@torch.no_grad()
def get_unmatched_targets_mask(
    assigned_gt_indices: torch.Tensor,
    num_targets: int,
) -> torch.Tensor:
    """
    Get the mask of the unmatched targets.

    Args:
        assigned_gt_indices (torch.Tensor): Indices of the matched targets
            for each prediction. Tensor of shape [num_preds].
        num_targets (int): Number of targets in the current frame.
            Note that assignet_gt_indices < num_targets must hold.

    Returns:
        torch.Tensor: Mask of the unmatched targets. Entries are True
            if the corresponding target is unmatched, and False otherwise.
            Tensor of shape [num_targets].
    """
    device = assigned_gt_indices.device

    indices = assigned_gt_indices[assigned_gt_indices >= 0]

    unmatched = torch.ones(num_targets, device=device, dtype=torch.bool)
    unmatched[indices] = False

    return unmatched


@torch.no_grad()
def filter_preds(preds: MetaDict, mask: torch.Tensor) -> MetaDict:
    """
    Filter predictions based on the given mask.

    Note: All inputs are assumed to be unbatched, representing a single sample
    and, in case of multi-layer decoder predictions, the predictions of a single
    decoder layer.

    Args:
        preds (MetaDict): The predictions to filter.
        mask (torch.Tensor): Mask to apply to the predictions. Entries set to
            True will be kept, and entries set to False will be discarded.
            Tensor of shape [num_preds].

    Returns:
        MetaDict: The filtered predictions.
    """

    out = MetaDict()

    if "boxes" in preds:
        out.boxes = preds.boxes[mask]

    if "class_scores" in preds:
        out.class_scores = preds.class_scores[mask]

    if "instance_ids" in preds:
        out.instance_ids = preds.instance_ids[mask]

    if "occupancy" in preds:
        out.occupancy = MetaDict()

        if "instance_scores" in preds.occupancy:
            out.occupancy.instance_scores = preds.occupancy.instance_scores[mask]

    return out


@torch.no_grad()
def filter_targets(targets: MetaDict, mask: torch.Tensor) -> MetaDict:
    """
    Filter targets based on the given mask.

    Note: All inputs are assumed to be unbatched, representing a single sample.

    Args:
        targets (MetaDict): The targets to filter.
        mask (torch.Tensor): Mask to apply to the targets. Entries set to
            True will be kept, and entries set to False will be discarded.
            Tensor of shape [num_preds].

    Returns:
        MetaDict: The filtered targets.
    """

    out = MetaDict()

    if "boxes" in targets:
        out.boxes = targets.boxes[mask]

    if "class_ids" in targets:
        out.class_ids = targets.class_ids[mask]

    if "instance_ids" in targets:
        out.instance_ids = targets.instance_ids[mask]

    if "occupancy" in targets:
        out.occupancy = MetaDict()

        if "masks" in targets.occupancy:
            out.occupancy.masks = targets.occupancy.masks

        if "instance_masks" in targets.occupancy:
            out.occupancy.instance_masks = targets.occupancy.instance_masks[mask]

    return out


@torch.no_grad()
def infer_num_preds(preds: MetaDict) -> int | None:
    """
    Infer the number of predictions based on the given predictions.

    Args:
        preds (MetaDict): The predictions to infer from.

    Returns:
        int | None: The number of predictions, or None if it could not be
            inferred.
    """
    if "class_scores" in preds:
        return preds.class_scores.shape[0]

    if "instance_ids" in preds:
        return preds.instance_ids.shape[0]

    if "boxes" in preds:
        return preds.boxes.shape[0]

    return None


@torch.no_grad()
def infer_num_targets(targets: MetaDict) -> int | None:
    """
    Infer the number of targets from the given targets.

    Args:
        targets (MetaDict): The targets to infer the number of targets from.

    Returns:
        int | None: The number of targets, or None if it could not be inferred.
    """
    if "class_ids" in targets:
        return targets.class_ids.shape[0]

    if "instance_ids" in targets:
        return targets.instance_ids.shape[0]

    if "boxes" in targets:
        return targets.boxes.shape[0]

    return None


@torch.no_grad()
def split_layer_preds(
    preds: MetaDict,
) -> list[MetaDict]:
    """
    Split the predictions into a list of predictions for each decoder layer.

    Args:
        preds (MetaDict): The predictions to split.

    Returns:
        list[MetaDict]: A list of predictions for each decoder layer.
    """
    # instance_ids are not per layer, so drop them here
    if "instance_ids" in preds:
        preds = preds.copy()
        del preds.instance_ids

    return uncollate(preds)
