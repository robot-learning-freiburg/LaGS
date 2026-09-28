# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
from torch import nn

from ..... import ops, utils
from .utils import quaternion_to_rotation_matrix


class GaussianSemanticAggregator(nn.Module):
    # buffers
    coords_grid: torch.Tensor

    def __init__(
        self,
        voxel_size: list[float],
        voxel_range: list[float],
        scale_multiplier: float = 5.0,
        min_radius: int = 1,
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

        # direct computation of inverse:
        m_scale_inv_sq = 1.0 / (scales * scales)
        m_scale_inv_sq = torch.diag_embed(m_scale_inv_sq)  # S^(-2)

        m_rot = quaternion_to_rotation_matrix(rotations)  # R

        # cov_inv = R S^(-2) R^T
        cov_inv = torch.matmul(
            m_rot, torch.matmul(m_scale_inv_sq, m_rot.transpose(-1, -2))
        )

        # directly compute (inverse) determinant
        det_inv = 1.0 / torch.prod(scales**2, dim=-1)

        # for each layer, aggregate the gaussians
        out_scores, out_bin_scores, out_density, out_prob_sum = [], [], [], []
        for layer in range(n_layers):
            layer_scores, layer_bin_scores, layer_density, layer_prob_sum = (
                self.aggregator(
                    sample_points=coords,
                    gaussian_means=centers[:, layer],
                    gaussian_icovs=cov_inv[:, layer],
                    gaussian_idets=det_inv[:, layer],
                    gaussian_scales=scales[:, layer],
                    gaussian_opacities=opacities[:, layer],
                    gaussian_semantics=scores[:, layer],
                    default_value=1.0 / n_classes,
                )
            )

            out_scores.append(layer_scores)
            out_bin_scores.append(layer_bin_scores)
            out_density.append(layer_density)
            out_prob_sum.append(layer_prob_sum)

        # stack the results
        sem_scores = torch.stack(out_scores, dim=1)  # [b, l, n, c]
        bin_scores = torch.stack(out_bin_scores, dim=1)  # [b, l, n, 1]
        density = torch.stack(out_density, dim=1)  # [b, l, n]
        prob_sum = torch.stack(out_prob_sum, dim=1)  # [b, l, n]

        # combine semantic and geometric/occupancy scores
        sem_score = sem_scores * bin_scores[..., None]
        free_score = 1 - bin_scores[..., None]
        all_scores = torch.cat([sem_score, free_score], dim=-1)

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

        # direct computation of inverse:
        m_scale_inv_sq = 1.0 / (scales * scales)
        m_scale_inv_sq = torch.diag_embed(m_scale_inv_sq)  # S^(-2)

        m_rot = quaternion_to_rotation_matrix(rotations)  # R

        # cov_inv = R S^(-2) R^T
        cov_inv = torch.matmul(
            m_rot, torch.matmul(m_scale_inv_sq, m_rot.transpose(-1, -2))
        )

        # directly compute (inverse) determinant
        det_inv = 1.0 / torch.prod(scales**2, dim=-1)

        # aggregate the gaussians
        features, bin_scores, density, prob_sum = self.aggregator(
            sample_points=coords,
            gaussian_means=centers,
            gaussian_icovs=cov_inv,
            gaussian_idets=det_inv,
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
