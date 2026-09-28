# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Density tensor ego-motion compensation for temporal sampling.

Transforms density maps from previous ego frame to current ego frame
using grid sampling, enabling density-aware gaussian query sampling.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DensityEgoMotionCompensation(nn.Module):
    """
    Transform density tensor from previous ego frame to current ego frame.

    This module warps a density grid from the previous frame's coordinate
    system to the current frame's coordinate system using F.grid_sample.
    This enables using past density information to guide sampling in
    regions with low coverage.

    For out-of-bounds regions (areas newly entering the view), the density
    is set to zero via padding_mode='zeros', which naturally leads to high
    sampling priority in those regions.
    """

    def __init__(self, voxel_range: list[float], voxel_size: list[float]):
        """
        Args:
            voxel_range: Voxel coordinate range [x_min, y_min, z_min, x_max, y_max, z_max]
            voxel_size: Voxel size [sx, sy, sz]
        """
        super().__init__()

        voxel_range = torch.tensor(voxel_range, dtype=torch.float32)
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)

        self.register_buffer("voxel_range", voxel_range, persistent=False)
        self.register_buffer("voxel_size", voxel_size, persistent=False)

        # Compute grid dimensions from voxel configuration
        vx_min, vx_max = voxel_range[:3], voxel_range[3:]
        grid_extent = vx_max - vx_min
        grid_dims = (grid_extent / voxel_size).int()
        w, h, d = grid_dims[0].item(), grid_dims[1].item(), grid_dims[2].item()

        # Pre-compute ego coordinates grid [d, h, w, 3]
        ego_coords = self._build_ego_coords(d, h, w, vx_min, voxel_size)
        self.register_buffer("ego_coords", ego_coords, persistent=False)

    def forward(
        self,
        density: torch.Tensor,
        transform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Resample density from previous to current ego frame.

        For each voxel in the current frame:
        1. Convert voxel indices to world coordinates (current ego)
        2. Apply inverse transform to get coordinates in previous ego frame
        3. Sample density from previous frame at those coordinates

        Args:
            density: Density tensor [b, 1, d, h, w] in previous ego frame
            transform: Transformation matrix [4, 4] = ego_to_global_curr^-1 @ ego_to_global_prev
                      This transforms points from previous ego frame to current ego frame

        Returns:
            Warped density [b, 1, d, h, w] in current ego frame
        """
        b = density.shape[0]

        # Transform ego coordinates to previous ego frame
        # The given transform maps prev_ego -> curr_ego, so we need its inverse
        with torch.autocast("cuda", enabled=False):
            tx_inv = torch.inverse(transform.to(dtype=torch.float64))
            tx_inv = tx_inv.to(dtype=self.ego_coords.dtype)

        prev_ego = self._transform_coords(self.ego_coords, tx_inv)  # [d, h, w, 3]

        # Convert to normalized grid coordinates [-1, 1] for grid_sample
        sample_grid = self._ego_to_sample_grid(prev_ego)  # [d, h, w, 3]
        sample_grid = sample_grid.unsqueeze(0).expand(b, -1, -1, -1, -1)

        # Resample density from previous frame
        # padding_mode='zeros' ensures new regions get zero density (high sampling priority)
        warped = F.grid_sample(
            density,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return warped

    @staticmethod
    def _build_ego_coords(
        d: int,
        h: int,
        w: int,
        vx_min: torch.Tensor,
        voxel_size: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build pre-computed ego coordinate grid.

        Args:
            d, h, w: Grid dimensions (depth, height, width)
            vx_min: Voxel range minimum [x_min, y_min, z_min]
            voxel_size: Voxel size [sx, sy, sz]

        Returns:
            Ego coordinates [d, h, w, 3]
        """
        # Build voxel index grid
        # Note: density is stored as [b, c, d, h, w] where d=z, h=y, w=x
        z = torch.arange(d, dtype=torch.float32)
        y = torch.arange(h, dtype=torch.float32)
        x = torch.arange(w, dtype=torch.float32)

        # Create meshgrid with z, y, x order matching density layout
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")

        # Stack as (x, y, z) for coordinate conversion
        grid = torch.stack([xx, yy, zz], dim=-1)  # [d, h, w, 3]

        # Convert to ego coordinates (voxel centers)
        # ego_coords = voxel_index * voxel_size + voxel_min + voxel_size/2
        return grid * voxel_size + vx_min + voxel_size / 2

    def _transform_coords(
        self,
        coords: torch.Tensor,
        transform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply 4x4 transformation to 3D coordinates.

        Args:
            coords: Coordinates [d, h, w, 3]
            transform: Transformation matrix [4, 4]

        Returns:
            Transformed coordinates [d, h, w, 3]
        """
        shape = coords.shape[:-1]
        coords_flat = coords.view(-1, 3)

        # Convert to homogeneous coordinates
        ones = torch.ones(
            coords_flat.shape[0], 1, device=coords.device, dtype=coords.dtype
        )
        coords_homo = torch.cat((coords_flat, ones), dim=-1)  # [n, 4]

        # Apply transformation
        transformed = torch.einsum("ij,nj->ni", transform, coords_homo)

        # Return xyz only
        return transformed[..., :3].view(*shape, 3)

    def _ego_to_sample_grid(self, world_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to normalized grid_sample coordinates [-1, 1].

        Args:
            world_coords: World coordinates [d, h, w, 3]

        Returns:
            Normalized coordinates [d, h, w, 3] in [-1, 1]
        """
        vx_min = self.voxel_range[:3]
        vx_max = self.voxel_range[3:]

        # Normalize to [0, 1]
        normalized = (world_coords - vx_min) / (vx_max - vx_min)

        # Convert to [-1, 1] for grid_sample
        return normalized * 2 - 1
