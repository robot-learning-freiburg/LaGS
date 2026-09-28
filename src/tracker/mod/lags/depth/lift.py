# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Literal

import einops
import torch
from torch import nn

from ....ops.bev_pool_v2 import bev_pool_v2
from ..utils import unproject_image_rays


def _prepare_bev_pool_v2(
    voxel_size: torch.Tensor,
    voxel_range: torch.Tensor,
    grid_size: torch.Tensor,
    img_shape: torch.Size | tuple[int],
    tgt_shape: torch.Size | tuple[int],
    depth_range: tuple[float, float],
    num_depth_bins: int,
    unproject_tx: torch.Tensor,
    device: torch.device,
    eps: float = 1e-5,
    scaling: Literal["linear", "quadratic", "sid"] = "linear",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # pylint: disable=too-many-locals
    range_min = voxel_range[:3]

    # compute ray/frustum coordinates in ego/lidar frame
    coords = unproject_image_rays(
        img_shape=img_shape,
        tgt_shape=tgt_shape,
        depth_range=depth_range,
        num_depth_bins=num_depth_bins,
        unproject_tx=unproject_tx,
        device=device,
        eps=eps,
        scaling=scaling,
    )  # [b, n, w, h, d, 3]

    b, n, w, h, d, _3 = coords.shape
    num_points = b * n * d * h * w

    coords = einops.rearrange(coords, "b n w h d c -> b (n d h w) c")

    # transform coordinates to voxel space
    coords = (coords - range_min) / voxel_size
    coords = coords.to(dtype=torch.int64)

    # add batch index to voxel coordinates
    batch_index = torch.arange(b, device=device, dtype=torch.int64)
    batch_index = batch_index.view(b, 1, 1).expand(b, n * d * h * w, 1)

    coords = torch.cat((coords, batch_index), dim=-1)
    coords = coords.view(-1, 4)  # [b*n*w*h*d, (x, y, z, batch)]

    # compute linear depth indices (depth index for each point)
    ranks_depth = torch.arange(num_points, device=device, dtype=torch.int64)

    # compute linear feature indices (feature index for each point)
    ranks_feat = torch.arange(num_points // d, device=device, dtype=torch.int64)
    ranks_feat = ranks_feat.view(b, n, 1, h, w).expand(b, n, d, h, w).flatten()

    # filter out-of-bounds coordinates
    mask = (coords[..., :3] >= 0) & (coords[..., :3] < grid_size)
    mask = mask.all(dim=-1)

    coords = coords[mask]
    ranks_depth = ranks_depth[mask]
    ranks_feat = ranks_feat[mask]

    # compute voxel ranks (linear voxel index for each point)
    ranks_voxel = (
        coords[..., 3] * grid_size[2] * grid_size[1] * grid_size[0]
        + coords[..., 2] * grid_size[1] * grid_size[0]
        + coords[..., 1] * grid_size[0]
        + coords[..., 0]
    )

    if ranks_voxel.shape[0] == 0:
        interval_starts = torch.zeros(0, device=device, dtype=torch.int64)
        interval_lengths = torch.zeros(0, device=device, dtype=torch.int64)
        return ranks_voxel, ranks_depth, ranks_feat, interval_starts, interval_lengths

    # reorder based on voxel ranks
    order = torch.argsort(ranks_voxel)

    ranks_voxel = ranks_voxel[order]
    ranks_depth = ranks_depth[order]
    ranks_feat = ranks_feat[order]

    # compute voxel intervals
    mask = torch.ones(ranks_voxel.shape[0], device=device, dtype=torch.bool)
    mask[1:] = ranks_voxel[1:] != ranks_voxel[:-1]

    interval_starts = torch.where(mask)[0].int()

    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = ranks_voxel.shape[0] - interval_starts[-1]

    return ranks_voxel, ranks_depth, ranks_feat, interval_starts, interval_lengths


def lift_and_pool(
    depth: torch.Tensor,  # [b, n, d, h, w]
    feats: torch.Tensor,  # [b, n, c, h, w]
    voxel_size: torch.Tensor,
    voxel_range: torch.Tensor,
    img_shape: torch.Size | tuple[int],
    depth_range: tuple[float, float],
    unproject_tx: torch.Tensor,
    eps: float = 1e-5,
    scaling: Literal["linear", "quadratic", "sid"] = "linear",
):
    # pylint: disable=too-many-locals
    device = depth.device
    b, _, d, _, _ = depth.shape
    b, _, c, _, _ = feats.shape

    # compute voxel grid size
    range_min = voxel_range[:3]
    range_max = voxel_range[3:]

    grid_size = (range_max - range_min) / voxel_size
    grid_size = grid_size.to(dtype=torch.int64)
    nx, ny, nz = grid_size.tolist()

    # compute indices and intervals for voxel pooling
    ranks_voxel, ranks_depth, ranks_feat, starts, lengths = _prepare_bev_pool_v2(
        voxel_size=voxel_size,
        voxel_range=voxel_range,
        grid_size=grid_size,
        img_shape=img_shape,
        tgt_shape=feats.shape,
        depth_range=depth_range,
        num_depth_bins=d,
        unproject_tx=unproject_tx,
        device=device,
        eps=eps,
        scaling=scaling,
    )

    if ranks_voxel.shape[0] == 0:
        return torch.zeros((b, c, nz, ny, nx), device=device, dtype=depth.dtype)

    # perform voxel pooling
    feats = einops.rearrange(feats, "b n c h w -> b n h w c")
    feats = bev_pool_v2(
        depth=depth,
        feat=feats,
        ranks_depth=ranks_depth,
        ranks_feat=ranks_feat,
        ranks_bev=ranks_voxel,
        bev_feat_shape=(b, nz, ny, nx, c),
        interval_starts=starts,
        interval_lengths=lengths,
    )

    feats = einops.rearrange(feats, "b z y x c -> b c z y x")
    feats = feats.contiguous()

    return feats


class LiftAndPool(nn.Module):
    # buffers
    voxel_size: torch.Tensor
    voxel_range: torch.Tensor

    def __init__(
        self,
        voxel_size: torch.Tensor | tuple[float, float, float],
        voxel_range: torch.Tensor | tuple[float, float, float, float, float, float],
        depth_range: tuple[float, float],
        depth_scaling: Literal["linear", "quadratic", "sid"] = "linear",
    ):
        super().__init__()

        self.depth_range = depth_range
        self.depth_scaling = depth_scaling

        voxel_size = torch.as_tensor(voxel_size, dtype=torch.float32)
        self.register_buffer("voxel_size", voxel_size, persistent=False)

        voxel_range = torch.as_tensor(voxel_range, dtype=torch.float32)
        self.register_buffer("voxel_range", voxel_range, persistent=False)

    def forward(
        self,
        depth: torch.Tensor,  # [b, n, d, h, w]
        feats: torch.Tensor,  # [b, n, c, h, w]
        img_shape: torch.Size | tuple[int],
        unproject_tx: torch.Tensor,  # [b, n, 4, 4]
    ) -> torch.Tensor:
        return lift_and_pool(
            depth=depth,
            feats=feats,
            voxel_size=self.voxel_size,
            voxel_range=self.voxel_range,
            img_shape=img_shape,
            depth_range=self.depth_range,
            unproject_tx=unproject_tx,
            scaling=self.depth_scaling,
        )  # [b, c, nz, ny, nx]
