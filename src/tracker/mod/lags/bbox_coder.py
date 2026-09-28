# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from ...utils.types import PackedTensor
from .utils import denormalize_bbox


class TrackNMSFreeCoder:
    # pylint: disable=too-few-public-methods

    def __init__(self, post_center_range, max_num=100):
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.code_size = 10

    def decode(self, preds):  # pylint: disable=too-many-locals
        cls = preds["all_cls_scores"][:, -1]  # [B, Q, C]
        bbox = preds["all_bbox_preds"][:, -1]  # [B, Q, 10]

        num_query = cls.shape[1]

        track_instances = preds.get("track_instances")
        tracking = track_instances is not None

        # per-query class score / label
        probs = cls.sigmoid()
        box_scores, box_labels = probs.max(dim=-1)  # [B, Q] each

        # ranking score: track score in tracking mode, max class score otherwise
        if tracking:
            rank = track_instances.scores.view(1, num_query).to(box_scores)
        else:
            rank = box_scores

        # optional validity: score invalid queries out instead of removing them
        masks = preds.get("all_masks")
        if masks is not None:
            if masks.dim() == 1:
                masks = masks[None]
            rank = rank.masked_fill(~masks, float("-inf"))

        # optional motion forecasting, stored unbatched in the tracking path
        motion = preds.get("all_motion_forecasting")
        if motion is not None and motion.dim() == 3:
            motion = motion[None]

        # top-N candidates per sample, ranked
        num_keep = min(self.max_num, num_query)
        topk_scores, idx = rank.topk(num_keep, dim=-1)  # [B, N]

        sel_bbox = bbox.take_along_dim(idx[..., None], dim=1)  # [B, N, 10]
        sel_labels = box_labels.take_along_dim(idx, dim=1)  # [B, N]
        sel_scores = topk_scores  # [B, N]

        if tracking:
            sel_obj = track_instances.obj_idxes.view(1, num_query).take_along_dim(
                idx, dim=1
            )  # [B, N]

        if motion is not None:
            sel_motion = motion.take_along_dim(
                idx[:, :, None, None], dim=1
            )  # [B, N, T, 2]

        # denormalize to physical boxes [B, N, 9]: cx cy cz w l h rot vx vy
        sel_boxes = denormalize_bbox(sel_bbox)

        # keep boxes whose center is inside the post-decode range and that came
        # from a valid query (finite ranking score)
        rng = torch.tensor(self.post_center_range, device=sel_boxes.device)
        lo, hi = rng[:3], rng[3:]
        centers = sel_boxes[..., :3]
        keep = (centers >= lo).all(dim=-1) & (centers <= hi).all(dim=-1)  # [B, N]
        keep = keep & topk_scores.isfinite()

        # ragged pack, sharing one set of offsets across all fields
        counts = keep.sum(dim=1)  # [B]
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        flat_keep = keep.reshape(-1)  # [B * N]

        def pack(field):
            data = field.reshape(-1, *field.shape[2:])[flat_keep]
            return PackedTensor(data, offsets)

        out = {
            "boxes": pack(sel_boxes),
            "scores": pack(sel_scores),
            "labels": pack(sel_labels),
        }

        if tracking:
            out["instances"] = pack(sel_obj)

        if motion is not None:
            out["forecasting"] = pack(sel_motion)

        return out
