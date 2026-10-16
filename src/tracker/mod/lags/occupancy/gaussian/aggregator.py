# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
from torch import nn

from ..... import ops, utils
from .utils import quaternion_to_rotation_matrix

# The chains below are pure pointwise math (plus small matmuls), which eager mode runs
# as one kernel launch per op; compiling fuses them. At 640k voxels the score gating
# alone is worth ~1 ms per step on an RTX 4060 Ti.


@torch.compile
def _inverse_covariance(scales: torch.Tensor, rotations: torch.Tensor):
    """cov_inv = R S^(-2) R^T, from scales and quaternions.

    The op derives the determinant from this matrix itself, so it is not returned
    here; passing a separately computed one would only let the two drift apart.
    """
    m_scale_inv_sq = torch.diag_embed(1.0 / (scales * scales))  # S^(-2)
    m_rot = quaternion_to_rotation_matrix(rotations)  # R

    return torch.matmul(m_rot, torch.matmul(m_scale_inv_sq, m_rot.transpose(-1, -2)))


@torch.compile
def _gate_scores(sem_scores: torch.Tensor, bin_scores: torch.Tensor) -> torch.Tensor:
    """Combine semantic and geometric/occupancy scores into per-class + free scores."""
    sem_score = sem_scores * bin_scores[..., None]
    free_score = 1 - bin_scores[..., None]
    return torch.cat([sem_score, free_score], dim=-1)


class GaussianSemanticAggregator(nn.Module):
    # buffers
    coords_grid: torch.Tensor

    def __init__(
        self,
        voxel_size: list[float],
        voxel_range: list[float],
        scale_multiplier: float = 5.0,
        min_radius: int = 1,
        opacity_weighted: bool = False,
        anisotropic_radii: bool = False,
    ):
        super().__init__()

        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.voxel_range = torch.tensor(voxel_range, dtype=torch.float32)

        self.register_buffer(
            "coords_grid",
            _build_coordinate_grid(voxel_range, voxel_size),
            persistent=False,
        )

        nz, ny, nx, _ = self.coords_grid.shape
        self.aggregator = ops.vsplat3d.Aggregator(
            distance_threshold=scale_multiplier,
            grid_size=(nx, ny, nz),
            voxel_range=self.voxel_range,
            voxel_size=self.voxel_size,
            min_radius=min_radius,
            opacity_weighted=opacity_weighted,
            anisotropic_radii=anisotropic_radii,
        )

    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        logits: torch.Tensor,  # [b, l, n_gaussians, n_classes]
        centers: torch.Tensor,  # [b, l, n_gaussians, 3]
        scales: torch.Tensor,  # [b, l, n_gaussians, 3]
        rotations: torch.Tensor,  # [b, l, n_gaussians, 4]
        opacities: torch.Tensor,  # [b, l, n_gaussians]
        mask: torch.Tensor | None = None,  # [b, z, y, x]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pylint: disable=too-many-locals
        b, n_layers, _n_gaussians, n_classes = logits.shape

        assert b == 1, "Batch size must be 1 for now"

        # get the sampling coordinates (where to splat/aggregate to)
        coords = self.coords_grid[None]
        coords = coords[mask] if mask is not None else coords
        coords = coords.flatten(end_dim=-2)  # [n, 3]
        coords = coords[None, ...]  # [1, n, 3]

        # prepare the logits
        scores = utils.torch.amp.upcast(logits, dtype=torch.float32)
        scores = torch.softmax(scores, dim=-1)  # [b, l, n_gaussians, n_classes]
        scores = scores.to(logits.dtype)

        # handle all gaussian properties in full precision
        centers = utils.torch.amp.upcast(centers, dtype=torch.float32)
        scales = utils.torch.amp.upcast(scales, dtype=torch.float32)
        rotations = utils.torch.amp.upcast(rotations, dtype=torch.float32)
        opacities = utils.torch.amp.upcast(opacities, dtype=torch.float32)

        cov_inv = _inverse_covariance(scales, rotations)

        # Aggregate all layers in one batched call: the op treats the leading dim as
        # independent elements over a shared grid, so the layers (times the batch,
        # once batch size > 1 is supported here) go through one splat together.
        sem_scores, bin_scores, density, prob_sum = self.aggregator(
            sample_points=coords.expand(b * n_layers, -1, -1),
            gaussian_means=centers.flatten(0, 1),
            gaussian_icovs=cov_inv.flatten(0, 1),
            gaussian_scales=scales.flatten(0, 1),
            gaussian_opacities=opacities.flatten(0, 1),
            gaussian_semantics=scores.flatten(0, 1),
            default_value=1.0 / n_classes,
        )

        sem_scores = sem_scores.reshape(b, n_layers, *sem_scores.shape[1:])
        bin_scores = bin_scores.reshape(b, n_layers, *bin_scores.shape[1:])
        density = density.reshape(b, n_layers, *density.shape[1:])
        prob_sum = prob_sum.reshape(b, n_layers, *prob_sum.shape[1:])

        # combine semantic and geometric/occupancy scores
        all_scores = _gate_scores(sem_scores, bin_scores)

        return all_scores, bin_scores, density, prob_sum


class GaussianFeatureAggregator(nn.Module):
    # buffers
    coords_grid: torch.Tensor

    def __init__(
        self,
        voxel_size: list[float],
        voxel_range: list[float],
        scale_multiplier: float = 5.0,
        default_value: float = 0.0,
        min_radius: int = 1,
        opacity_weighted: bool = False,
        anisotropic_radii: bool = False,
    ):
        super().__init__()

        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        self.voxel_range = torch.tensor(voxel_range, dtype=torch.float32)
        self.default_value = default_value

        self.register_buffer(
            "coords_grid",
            _build_coordinate_grid(voxel_range, voxel_size),
            persistent=False,
        )

        nz, ny, nx, _ = self.coords_grid.shape
        self.aggregator = ops.vsplat3d.Aggregator(
            distance_threshold=scale_multiplier,
            grid_size=(nx, ny, nz),
            voxel_range=self.voxel_range,
            voxel_size=self.voxel_size,
            min_radius=min_radius,
            opacity_weighted=opacity_weighted,
            anisotropic_radii=anisotropic_radii,
        )

    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        features: torch.Tensor,  # [b, n_gaussians, n_classes]
        centers: torch.Tensor,  # [b, n_gaussians, 3]
        scales: torch.Tensor,  # [b, n_gaussians, 3]
        rotations: torch.Tensor,  # [b, n_gaussians, 4]
        opacities: torch.Tensor,  # [b, n_gaussians]
        mask: torch.Tensor | None = None,  # [b, z, y, x]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # pylint: disable=too-many-locals
        b, _n_gaussians, _n_channels = features.shape

        assert b == 1, "Batch size must be 1 for now"

        # get the sampling coordinates (where to splat/aggregate to)
        coords = self.coords_grid[None]
        coords = coords[mask] if mask is not None else coords
        coords = coords.flatten(end_dim=-2)  # [n, 3]
        coords = coords[None, ...]  # [1, n, 3]

        # handle all gaussian properties in full precision
        centers = utils.torch.amp.upcast(centers, dtype=torch.float32)
        scales = utils.torch.amp.upcast(scales, dtype=torch.float32)
        rotations = utils.torch.amp.upcast(rotations, dtype=torch.float32)
        opacities = utils.torch.amp.upcast(opacities, dtype=torch.float32)

        cov_inv = _inverse_covariance(scales, rotations)

        # aggregate the gaussians
        features, bin_scores, density, prob_sum = self.aggregator(
            sample_points=coords,
            gaussian_means=centers,
            gaussian_icovs=cov_inv,
            gaussian_scales=scales,
            gaussian_opacities=opacities,
            gaussian_semantics=features,
            default_value=self.default_value,
        )

        # gate features by occupancy scores
        features = features * bin_scores[..., None]

        return features, bin_scores, density, prob_sum


@torch.no_grad()
def _build_coordinate_grid(voxel_range, voxel_size) -> torch.Tensor:
    voxel_range = torch.as_tensor(voxel_range, dtype=torch.float32)
    voxel_size = torch.as_tensor(voxel_size, dtype=torch.float32)

    # compute number of voxels in each dimension using integer division
    # to avoid floating-point precision issues
    nx = ((voxel_range[3] - voxel_range[0]) / voxel_size[0]).round().long().item()
    ny = ((voxel_range[4] - voxel_range[1]) / voxel_size[1]).round().long().item()
    nz = ((voxel_range[5] - voxel_range[2]) / voxel_size[2]).round().long().item()

    # generate integer indices
    grid_x = torch.arange(nx, dtype=torch.float32)
    grid_y = torch.arange(ny, dtype=torch.float32)
    grid_z = torch.arange(nz, dtype=torch.float32)

    # convert indices to coordinates
    # #coordinate = (index + 0.5) * voxel_size + voxel_range_min
    # the +0.5 centers the coordinate in the voxel
    grid_x = (grid_x + 0.5) * voxel_size[0] + voxel_range[0]
    grid_y = (grid_y + 0.5) * voxel_size[1] + voxel_range[1]
    grid_z = (grid_z + 0.5) * voxel_size[2] + voxel_range[2]

    # broadcast to 3D grid
    grid_x = grid_x[None, None, :].expand(nz, ny, nx)
    grid_y = grid_y[None, :, None].expand(nz, ny, nx)
    grid_z = grid_z[:, None, None].expand(nz, ny, nx)

    # stack the grids into a single tensor
    return torch.stack((grid_x, grid_y, grid_z), dim=-1)
