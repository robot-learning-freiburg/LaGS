# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * DETR3D (https://github.com/WangYueFt/detr3d), Copyright (c) 2021 Wang, Yue, licensed under MIT,
# * MMDetection3D (https://github.com/open-mmlab/mmdetection3d), Copyright (c) OpenMMLab, licensed under Apache-2.0.
# See the LICENSES/ directory for full license texts.

import torch

from .... import utils


def denormalize_bbox(boxes: torch.Tensor) -> torch.Tensor:
    # center
    cx = boxes[..., 0:1]
    cy = boxes[..., 1:2]
    cz = boxes[..., 4:5]

    # size
    w = boxes[..., 2:3].exp()
    l = boxes[..., 3:4].exp()
    h = boxes[..., 5:6].exp()

    # rotation
    rot_sine, rot_cosine = boxes[..., 6:7], boxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # convert yaw to nuscenes format
    rot = -rot - torch.pi / 2.0
    rot = utils.math.normalize_angle(rot)

    # velocity
    vx = boxes[..., 8:9]
    vy = boxes[..., 9:10]

    return torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)


def normalize_bbox(boxes: torch.Tensor) -> torch.Tensor:
    # center
    cx = boxes[..., 0:1]
    cy = boxes[..., 1:2]
    cz = boxes[..., 2:3]

    # size
    w = boxes[..., 3:4].log()
    l = boxes[..., 4:5].log()
    h = boxes[..., 5:6].log()

    # rotation
    rot = boxes[..., 6:7]

    # convert yaw to nuscenes format
    rot = -rot - torch.pi / 2.0
    rot = utils.math.normalize_angle(rot)

    # velocity
    vx = boxes[..., 7:8]
    vy = boxes[..., 8:9]

    return torch.cat((cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1)
