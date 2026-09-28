# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


@torch.no_grad()
def masks_to_volume(
    class_scores: torch.Tensor,  # [nq]
    class_labels: torch.Tensor,  # [nq]
    instance_ids: torch.Tensor,  # [nq]
    instance_scores: torch.Tensor,  # [nq, d, h, w]
    default_class: int = -1,
    default_iid: int = -1,
    occupancy_score_threshold: float | torch.Tensor = 0.5,
    overlap_threshold: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    # pylint: disable=too-many-locals
    device = class_scores.device
    num_preds, d, h, w = instance_scores.shape

    # initialize empty occupancy predictions
    occ_cls = torch.full((d, h, w), default_class, dtype=torch.long, device=device)
    occ_iid = torch.full((d, h, w), default_iid, dtype=torch.long, device=device)

    # if we have no predictions, return empty occupancy
    if num_preds == 0:
        return occ_cls, occ_iid

    # compute masks for dominant instances [num_preds, d, h, w]
    indices = torch.arange(num_preds, device=device, dtype=torch.long)

    instance_dom = class_scores[:, None, None, None] * instance_scores
    instance_dom = instance_dom.argmax(dim=0)
    instance_dom = instance_dom[None, ...] == indices[:, None, None, None]

    # compute masks for actually occupied space [num_preds, d, h, w]
    # threshold can be a scalar or a [num_classes] tensor indexed by class label
    if isinstance(occupancy_score_threshold, torch.Tensor):
        occ_threshold = occupancy_score_threshold[class_labels]  # [num_preds]
        instance_occ = instance_scores > occ_threshold[:, None, None, None]
    else:
        instance_occ = instance_scores > occupancy_score_threshold

    # compute mask assigned to each instance [num_preds, d, h, w]
    instance_mask = instance_dom & instance_occ

    # compute IoU between assigned dominant mask and predicted occupancy
    # NOTE: We consider the predicted occupancy as the "union" as we don't
    #       care about any regions outside the predicted occupancy.
    #       Specifically, a mask could predict "unoccupied" but still be
    #       the dominant mask for some area. We only want to drop masks
    #       that are only dominant in small regions of their predicted
    #       occupancy.
    im_inter = instance_mask.view(num_preds, -1).sum(dim=-1)
    im_occup = instance_occ.view(num_preds, -1).sum(dim=-1)
    im_overlap = torch.where(im_occup > 0, im_inter / im_occup, 0)  # [num_preds]

    # filter out instances that do not have enough overlap, e.g., false
    # positives that are only dominant in small regions
    im_included = im_overlap > overlap_threshold

    instance_mask = instance_mask[im_included]  # [num_final, d, h, w]
    class_labels = class_labels[im_included]  # [num_final]
    instance_ids = instance_ids[im_included]  # [num_final]

    # aggregate final semantic occupancy predictions
    for cls, iid, imsk in zip(class_labels, instance_ids, instance_mask):
        occ_cls[imsk] = cls
        occ_iid[imsk] = iid

    return occ_cls, occ_iid
