# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from typing import Literal

import torch

from ....utils.torch import amp


@torch.autocast("cuda", enabled=False)
def unproject_image_rays(
    img_shape: torch.Size | tuple[int],
    tgt_shape: torch.Size | tuple[int],
    depth_range: tuple[float, float],
    num_depth_bins: int,
    unproject_tx: torch.Tensor,
    device: torch.device,
    eps: float = 1e-5,
    scaling: Literal["linear", "quadratic", "sid"] = "linear",
) -> torch.Tensor:
    # pylint: disable=too-many-locals

    *_, img_h, img_w = img_shape
    *_, h, w = tgt_shape
    b, n, _4, _4 = unproject_tx.shape
    d = num_depth_bins

    # ensure that we compute in at least 32-bit precision
    unproject_tx = amp.upcast(unproject_tx, dtype=torch.float32)

    # build 3D coordinates in image space
    coords_h = torch.arange(h, device=device, dtype=torch.float)
    coords_h = coords_h * img_h / h

    coords_w = torch.arange(w, device=device, dtype=torch.float)
    coords_w = coords_w * img_w / w

    if scaling == "linear":
        depth_bin_size = depth_range[1] - depth_range[0]
        depth_bin_size = depth_bin_size / num_depth_bins

        coords_d = torch.arange(num_depth_bins, device=device, dtype=torch.float)
        coords_d = coords_d * depth_bin_size + depth_range[0]

    elif scaling == "quadratic":
        depth_bin_size = depth_range[1] - depth_range[0]
        depth_bin_size = depth_bin_size / (num_depth_bins * (1 + num_depth_bins))

        coords_d = torch.arange(num_depth_bins, device=device, dtype=torch.float)
        coords_d = coords_d * (coords_d + 1) * depth_bin_size + depth_range[0]

    elif scaling == "sid":
        depth_scale = torch.log((depth_range[1] - 1) / depth_range[0])
        depth_offs = math.log(depth_range[0])

        coords_d = torch.arange(num_depth_bins, device=device, dtype=torch.float)
        coords_d = torch.exp(depth_offs + coords_d / (num_depth_bins - 1) * depth_scale)

    else:
        raise ValueError(f"Unknown scaling type: {scaling}")

    # combine independent coordinates into grid/volume
    coords = torch.meshgrid((coords_w, coords_h, coords_d), indexing="ij")
    coords = torch.stack(coords, dim=-1)  # [w, h, d, 3]

    # convert to homogeneous coordinates
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), dim=-1)

    # transform coordinates into lidar/ego frame
    coords[..., :2] *= torch.clamp(coords[..., 2:3], min=eps)  # multiply by z
    coords = coords.view(1, 1, w, h, d, 4, 1)
    coords = coords.expand(b, n, w, h, d, 4, 1)

    unproject_tx = unproject_tx.view(b, n, 1, 1, 1, 4, 4)
    unproject_tx = unproject_tx.expand(b, n, w, h, d, 4, 4)

    coords = torch.matmul(unproject_tx, coords)  # [b, n, w, h, d, 4, 1]
    coords = coords.squeeze(-1)[..., :3]  # [b, n, w, h, d, 3]

    return coords
