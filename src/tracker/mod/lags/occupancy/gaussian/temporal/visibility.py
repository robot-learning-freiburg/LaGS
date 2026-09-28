# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Configurable visibility computation for FoV/visibility losses.

Registry-based visibility computation supporting multiple depth backends:
- none: No-op, returns (None, None)
- fov: FoV check only, no occlusion computation
- depth-bins: Uses DepthNet's binned depth distribution
- rendered-depth: Renders depth from decoded gaussians via gsplat

All backends produce per-gaussian visibility scores [0, 1] and in-FoV masks.

Usage:
    Two-phase interface to avoid redundant computation across streams:

    # Phase 1: Prepare depth surface (once per frame)
    vis_context = vis_module.prepare(depth_dist=..., gaussians=..., ...)

    # Phase 2: Compute per-stream visibility (per stream)
    visibility, in_fov = vis_module(centers, scales, rotations, ..., **vis_context)
"""

import abc
import math
from typing import Any, Literal

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from ......config.registry import Registry
from ......utils.torch import amp
from ....losses.gaussian_depth import decompose_projection_matrix, render_gaussian_depth
from ..utils import quaternion_to_rotation_matrix

# ---------------------------------------------------------------------------
# Visibility computation utilities
# ---------------------------------------------------------------------------


def compute_viewing_aligned_extent(
    scales: torch.Tensor,  # [b, n_gaussians, 3]
    rotations: torch.Tensor,  # [b, n_gaussians, 4] quaternions wxyz
    tx_project: torch.Tensor,  # [b, n_cams, 4, 4]
) -> torch.Tensor:
    """
    Compute the extent of each gaussian along camera viewing directions.

    For an ellipsoid with semi-axes (sx, sy, sz) rotated by R, the extent
    along direction d is: 1 / ||S^{-1} @ R^T @ d||

    This gives the distance from center to surface along the viewing ray,
    which is what we need for depth tolerance computation.

    Args:
        scales: Gaussian scales [b, n_gaussians, 3]
        rotations: Gaussian rotations as quaternions (wxyz) [b, n_gaussians, 4]
        tx_project: Ego-to-image transforms [b, n_cams, 4, 4]

    Returns:
        extents: Per-gaussian per-camera extent [b, n_cams, n_gaussians]
    """
    # Camera z-axis in ego frame: row 2 of tx_project gives depth
    # depth = tx_project[2, :] @ point_homo, so the direction is tx_project[2, 0:3]
    cam_z_axes = tx_project[:, :, 2, 0:3]  # [b, n_cams, 3]
    cam_z_axes = F.normalize(cam_z_axes, dim=-1)

    # Convert quaternions to rotation matrices [b, n_gaussians, 3, 3]
    rot_matrices = quaternion_to_rotation_matrix(rotations)

    # Transform camera z-axis to each gaussian's local frame: R^T @ cam_z
    # rot_matrices: [b, n_gaussians, 3, 3]
    # cam_z_axes: [b, n_cams, 3]
    # Want: for each (b, cam, gaussian): R_gaussian^T @ z_cam
    # = einsum('bnji,bcj->bcni', rot_matrices, cam_z_axes)
    local_z = torch.einsum("bnji,bcj->bcni", rot_matrices, cam_z_axes)
    # local_z: [b, n_cams, n_gaussians, 3]

    # Compute extent: 1 / ||S^{-1} @ local_z||
    # ||S^{-1} @ v||^2 = sum((v_i / s_i)^2)
    # scales: [b, n_gaussians, 3] -> [b, 1, n_gaussians, 3]
    inv_scales_sq = 1.0 / (scales.unsqueeze(1) ** 2 + 1e-8)

    # ||S^{-1} @ local_z||^2
    extent_inv_sq = (local_z**2 * inv_scales_sq).sum(dim=-1)  # [b, n_cams, n_gaussians]
    extent = 1.0 / (extent_inv_sq.sqrt() + 1e-6)

    return extent


@torch.autocast("cuda", enabled=False)
def project_points_to_cameras(
    centers: torch.Tensor,  # [b, n_gaussians, 3]
    tx_project: torch.Tensor,  # [b, n_cams, 4, 4]
    img_shape: tuple[int, int],  # (h, w)
    min_depth: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project 3D points to camera image planes.

    Args:
        centers: Gaussian centers in ego frame [b, n_gaussians, 3]
        tx_project: Ego-to-image transforms [b, n_cams, 4, 4]
        img_shape: Image shape (h, w)
        min_depth: Minimum valid depth (points closer are considered out of FoV)

    Returns:
        uv: Normalized image coords [-1, 1] for grid_sample [b, n_cams, n_gaussians, 2]
        depth: Depth of each point in each camera [b, n_cams, n_gaussians]
        in_fov: Boolean mask for points in field of view [b, n_cams, n_gaussians]
    """
    # pylint: disable=too-many-locals

    b, n_gaussians, _ = centers.shape
    h, w = img_shape

    # Upcast for precision
    centers = amp.upcast(centers, dtype=torch.float32)
    tx_project = amp.upcast(tx_project, dtype=torch.float32)

    # Convert to homogeneous coordinates [b, n_gaussians, 4]
    ones = torch.ones(b, n_gaussians, 1, device=centers.device, dtype=centers.dtype)
    centers_homo = torch.cat([centers, ones], dim=-1)

    # Project to all cameras: [b, n_cams, 4, 4] @ [b, n_gaussians, 4] -> [b, n_cams, n_gaussians, 4]
    projected = torch.einsum("bcij,bnj->bcni", tx_project, centers_homo)

    # Extract depth (z) - this is the z coordinate in camera frame
    depth = projected[..., 2]  # [b, n_cams, n_gaussians]

    # Check valid depth BEFORE perspective division to avoid numerical issues
    valid_depth = depth > min_depth

    # Perspective division: only divide where depth is valid, else set to 0
    # This avoids division by near-zero values causing huge UV coordinates
    safe_depth = torch.where(valid_depth, depth, torch.ones_like(depth))
    uv = projected[..., :2] / safe_depth.unsqueeze(-1)  # [b, n_cams, n_gaussians, 2]

    # Normalize to [-1, 1] for grid_sample
    uv = uv / torch.tensor([w, h], device=uv.device, dtype=uv.dtype)
    uv = uv * 2 - 1

    # Set invalid points to out-of-bounds UV
    uv = torch.where(valid_depth.unsqueeze(-1), uv, torch.full_like(uv, 10.0))

    # Compute in-FoV mask: valid depth AND within image bounds
    in_fov = (
        valid_depth
        & (uv[..., 0] > -1)
        & (uv[..., 0] < 1)
        & (uv[..., 1] > -1)
        & (uv[..., 1] < 1)
    )

    return uv, depth, in_fov


def compute_bin_depths(
    num_bins: int,
    depth_range: tuple[float, float],
    scaling: Literal["linear", "quadratic", "sid"] = "linear",
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute the depth value for each bin center.

    Args:
        num_bins: Number of depth bins
        depth_range: (min_depth, max_depth)
        scaling: Depth scaling mode
        device: Target device

    Returns:
        bin_depths: Depth value for each bin [num_bins]
    """
    min_depth, max_depth = depth_range

    if scaling == "linear":
        bin_size = (max_depth - min_depth) / num_bins
        bin_depths = torch.arange(num_bins, device=device, dtype=torch.float32)
        bin_depths = bin_depths * bin_size + min_depth

    elif scaling == "quadratic":
        bin_size = (max_depth - min_depth) / (num_bins * (1 + num_bins))
        bin_depths = torch.arange(num_bins, device=device, dtype=torch.float32)
        bin_depths = bin_depths * (bin_depths + 1) * bin_size + min_depth

    elif scaling == "sid":
        depth_scale = torch.log(torch.tensor((max_depth - 1) / min_depth))
        depth_offs = math.log(min_depth)
        bin_depths = torch.arange(num_bins, device=device, dtype=torch.float32)
        bin_depths = torch.exp(depth_offs + bin_depths / (num_bins - 1) * depth_scale)

    else:
        raise ValueError(f"Unknown scaling type: {scaling}")

    return bin_depths


def compute_visibility_from_depth(
    centers: torch.Tensor,  # [b, n_gaussians, 3]
    scales: torch.Tensor,  # [b, n_gaussians, 3]
    rotations: torch.Tensor,  # [b, n_gaussians, 4] quaternions wxyz
    depth_dist: torch.Tensor,  # [b, n_cams, n_bins, h, w] - softmax over bins
    tx_project: torch.Tensor,  # [b, n_cams, 4, 4]
    bin_depths: torch.Tensor,  # [n_bins]
    img_shape: tuple[int, int],  # original image shape (h, w)
    depth_tolerance: float = 0.0,
    scale_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute visibility for each gaussian by integrating depth distributions.

    For each gaussian at depth d_gauss, visibility is the probability that
    the observed surface is at or beyond that depth (minus tolerance):

        visibility = P(surface_depth >= d_gauss - tolerance)

    The tolerance accounts for the gaussian's extent toward the camera. For elongated
    gaussians (e.g., streets perpendicular to view), the center may be far from the
    actual visible surface. Using scale-aware tolerance prevents these from being
    incorrectly marked as occluded.

    Args:
        centers: Gaussian centers in ego frame [b, n_gaussians, 3]
        scales: Gaussian scales [b, n_gaussians, 3]
        rotations: Gaussian rotations as quaternions (wxyz) [b, n_gaussians, 4]
        depth_dist: Depth distribution (softmax over bins) [b, n_cams, n_bins, h, w]
        tx_project: Ego-to-image transforms [b, n_cams, 4, 4]
        bin_depths: Depth value for each bin [n_bins]
        img_shape: Original image shape (h, w) for projection normalization
        depth_tolerance: Base tolerance (meters) added to viewing-aligned extent
        scale_factor: Multiplier for viewing-aligned extent

    Returns:
        visibility: Per-gaussian visibility scores [b, n_gaussians] in [0, 1]
        in_fov: Per-gaussian in-FoV mask (visible in at least one camera) [b, n_gaussians]
    """
    # pylint: disable=too-many-locals
    n_cams, n_bins = depth_dist.shape[1], depth_dist.shape[2]

    # Compute per-gaussian per-camera depth tolerance using viewing-aligned extent
    # extent gives the distance from center to surface along each camera's view direction
    viewing_extent = compute_viewing_aligned_extent(scales, rotations, tx_project)
    # viewing_extent: [b, n_cams, n_gaussians]
    per_cam_tolerance = viewing_extent * scale_factor + depth_tolerance

    # Project gaussian centers to all cameras
    uv, depth_gauss, in_fov = project_points_to_cameras(centers, tx_project, img_shape)

    # uv: [b, n_cams, n_gaussians, 2]
    # depth_gauss: [b, n_cams, n_gaussians]
    # in_fov: [b, n_cams, n_gaussians]

    # Sample depth distribution at projected locations
    # grid_sample expects [b, c, h, w] and grid [b, h_out, w_out, 2]
    # We have depth_dist [b, n_cams, n_bins, h, w], need to sample per camera

    cam_visibilities = []

    for cam_idx in range(n_cams):
        # Get this camera's depth distribution [b, n_bins, h, w]
        cam_depth_dist = depth_dist[:, cam_idx]

        # Get UV coordinates for this camera [b, n_gaussians, 2]
        cam_uv = uv[:, cam_idx]

        # Reshape UV for grid_sample: [b, 1, n_gaussians, 2]
        cam_uv = cam_uv.unsqueeze(1)

        # Sample: [b, n_bins, 1, n_gaussians] -> [b, n_bins, n_gaussians]
        sampled_dist = F.grid_sample(
            cam_depth_dist,
            cam_uv,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampled_dist = sampled_dist.squeeze(2)  # [b, n_bins, n_gaussians]
        sampled_dist = sampled_dist.permute(0, 2, 1)  # [b, n_gaussians, n_bins]

        # Get gaussian depths for this camera [b, n_gaussians]
        cam_depth_gauss = depth_gauss[:, cam_idx]

        # Compute which bins are at or in front of the gaussian's effective front depth
        # bin_depths: [n_bins], cam_depth_gauss: [b, n_gaussians]
        # per_cam_tolerance: [b, n_cams, n_gaussians] - viewing-aligned extent per camera
        bin_depths_expanded = bin_depths.view(1, 1, n_bins)
        cam_tolerance = per_cam_tolerance[:, cam_idx]  # [b, n_gaussians]
        effective_front_depth = cam_depth_gauss - cam_tolerance

        # Mask: bins that are in front of the gaussian's front surface [b, n_gaussians, n_bins]
        mask = bin_depths_expanded < effective_front_depth[..., None]

        # Visibility = one minus sum of probabilities in "in front" bins
        cam_vis = 1.0 - (sampled_dist * mask.float()).sum(dim=-1)  # [b, n_gaussians]

        # Zero out gaussians not in FoV for this camera
        cam_vis = cam_vis * in_fov[:, cam_idx].float()

        cam_visibilities.append(cam_vis)

    # Stack across cameras [b, n_gaussians, n_cams]
    visibilities = torch.stack(cam_visibilities, dim=-1)

    # Max visibility across cameras (visible in at least one camera)
    visibility = visibilities.max(dim=-1).values

    # In-FoV: visible in at least one camera [b, n_gaussians]
    any_in_fov = in_fov.any(dim=1)

    return visibility, any_in_fov


# ---------------------------------------------------------------------------
# Registry and visibility computation classes
# ---------------------------------------------------------------------------

registry = Registry("lags.losses.gaussian_fov_visibility")


class FoVVisibilityComputation(nn.Module, abc.ABC):
    """
    Base class for FoV visibility computation.

    Two-phase interface:
    - prepare(): Compute depth surface once per frame (before stream loop)
    - forward(): Compute per-gaussian visibility for a batch of query gaussians
    """

    @abc.abstractmethod
    def prepare(self, **kwargs) -> dict[str, Any]:
        """
        Prepare depth context for the current frame.

        Called once before iterating over streams. Each implementation
        picks the kwargs it needs and ignores the rest.

        Returns:
            vis_context: Dict of tensors to pass to forward() via **kwargs
        """

    @abc.abstractmethod
    def forward(
        self,
        centers: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        tx_project: torch.Tensor,
        img_shape: tuple[int, ...],
        **vis_context,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute visibility for each gaussian.

        Args:
            centers: Gaussian centers in ego frame [b, n_gaussians, 3]
            scales: Gaussian scales [b, n_gaussians, 3]
            rotations: Gaussian rotations (wxyz) [b, n_gaussians, 4]
            tx_project: Ego-to-image transforms [b, n_cams, 4, 4]
            img_shape: Original image shape (..., h, w)
            **vis_context: Context from prepare()

        Returns:
            visibility: [b, n_gaussians] in [0, 1]
            in_fov: [b, n_gaussians] bool
        """


@registry.register(key="none")
class NoneVisibility(FoVVisibilityComputation):
    """
    No-op visibility computation that returns None.

    Useful when using opacity-only confidence trackers that don't need
    visibility computation. Avoids expensive depth rendering/sampling.
    """

    def prepare(self, **kwargs) -> dict[str, Any]:
        return {}

    def forward(
        self,
        centers: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        tx_project: torch.Tensor,
        img_shape: tuple[int, ...],
        **vis_context,
    ) -> tuple[None, None]:
        return None, None


@registry.register(key="fov")
class FoVOnlyVisibility(FoVVisibilityComputation):
    """
    FoV-only visibility that skips depth/occlusion computation.

    Projects gaussians to cameras and returns:
    - visibility: 1.0 for in-FoV gaussians, 0.0 for out-of-FoV
    - in_fov: boolean mask

    Useful when you only care about whether gaussians are in the camera's
    field of view, without expensive depth-based occlusion checks.
    """

    def prepare(self, **kwargs) -> dict[str, Any]:
        return {}

    def forward(
        self,
        centers: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        tx_project: torch.Tensor,
        img_shape: tuple[int, ...],
        **vis_context,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=unused-argument
        *_, h, w = img_shape

        # Project points to cameras to get in_fov mask
        _uv, _depth, in_fov = project_points_to_cameras(centers, tx_project, (h, w))

        # In-FoV if visible in at least one camera
        any_in_fov = in_fov.any(dim=1)  # [b, n_gaussians]

        # Visibility = 1.0 for in-FoV, 0.0 otherwise
        visibility = any_in_fov.float()

        return visibility, any_in_fov


@registry.register(key="depth-bins")
class DepthBinsVisibility(FoVVisibilityComputation):
    """
    Visibility via DepthNet's binned depth distribution.

    Integrates the predicted depth distribution at each gaussian's projected
    location to determine the probability of occlusion. This is the original
    visibility computation method.

    prepare(): Passes through the depth_dist tensor.
    forward(): Integrates depth distribution at projected gaussian locations.
    """

    bin_depths: torch.Tensor

    def __init__(
        self,
        depth_bins: int,
        depth_range: tuple[float, float] | list[float],
        depth_scaling: str = "linear",
        depth_tolerance: float = 0.5,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.depth_tolerance = depth_tolerance
        self.scale_factor = scale_factor

        bin_depths = compute_bin_depths(
            num_bins=depth_bins,
            depth_range=tuple(depth_range),
            scaling=depth_scaling,
        )
        self.register_buffer("bin_depths", bin_depths, persistent=False)

    def prepare(self, *, depth_dist: torch.Tensor, **kwargs) -> dict[str, Any]:
        # pylint: disable=arguments-differ
        return {"depth_dist": depth_dist}

    def forward(
        self,
        centers: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        tx_project: torch.Tensor,
        img_shape: tuple[int, ...],
        *,
        depth_dist: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=arguments-differ, unused-argument

        *_, h, w = img_shape

        return compute_visibility_from_depth(
            centers=centers,
            scales=scales,
            rotations=rotations,
            depth_dist=depth_dist,
            tx_project=tx_project,
            bin_depths=self.bin_depths,
            img_shape=(h, w),
            depth_tolerance=self.depth_tolerance,
            scale_factor=self.scale_factor,
        )


@registry.register(key="rendered-depth")
class RenderedDepthVisibility(FoVVisibilityComputation):
    """
    Visibility via gaussian-rendered depth comparison.

    Renders depth from all decoded gaussians using gsplat, then determines
    visibility by comparing each query gaussian's depth against the rendered
    depth surface. Uses viewing-aligned extent for scale-aware tolerance and
    sigmoid smoothing for continuous [0, 1] output.

    prepare(): Renders depth from all decoded gaussians (once per frame).
    forward(): Compares query gaussian depths against rendered depth surface.
    """

    def __init__(
        self,
        render_scale: float = 0.25,
        alpha_threshold: float = 0.1,
        depth_tolerance: float = 2.5,
        scale_factor: float = 1.0,
        smoothing: float = 1.0,
        near_plane: float = 0.1,
        far_plane: float = 100.0,
    ):
        """
        Args:
            render_scale: Resolution scale for depth rendering (relative to image)
            alpha_threshold: Minimum alpha for a surface to count as occluder
            depth_tolerance: Base depth margin (meters) added to viewing-aligned extent
            scale_factor: Multiplier for viewing-aligned extent contribution
            smoothing: Sigmoid temperature for soft visibility transition (meters)
            near_plane: gsplat near clipping plane
            far_plane: gsplat far clipping plane
        """
        super().__init__()
        self.render_scale = render_scale
        self.alpha_threshold = alpha_threshold
        self.depth_tolerance = depth_tolerance
        self.scale_factor = scale_factor
        self.smoothing = smoothing
        self.near_plane = near_plane
        self.far_plane = far_plane

    @torch.no_grad()
    @torch.autocast("cuda", enabled=False)
    def prepare(
        self,
        *,
        gaussians: dict,
        tx_project: torch.Tensor,
        img_shape: tuple[int, ...],
        **kwargs,
    ) -> dict[str, Any]:
        """
        Render depth from all decoded gaussians.

        Args:
            gaussians: Dict[str, MetaDict] of all decoded gaussian streams.
                Each MetaDict has centers, scales, rotations, opacities with
                shape [b, num_layers, n, ...]. We use the last layer ([:, -1]).
            tx_project: Ego-to-image transforms [b, n_cams, 4, 4]
            img_shape: Original image shape (..., h, w)
        """
        # pylint: disable=arguments-differ, too-many-locals

        # Concatenate all gaussian streams
        all_centers = torch.cat([gs.centers[:, -1] for gs in gaussians.values()], dim=1)
        all_scales = torch.cat([gs.scales[:, -1] for gs in gaussians.values()], dim=1)
        all_rotations = torch.cat(
            [gs.rotations[:, -1] for gs in gaussians.values()], dim=1
        )
        all_opacities = torch.cat(
            [gs.opacities[:, -1] for gs in gaussians.values()], dim=1
        )

        *_, h, w = img_shape
        render_h = int(h * self.render_scale)
        render_w = int(w * self.render_scale)

        batch_size = all_centers.shape[0]
        rendered_depths = []
        rendered_alphas = []

        tx_project = amp.upcast(tx_project, dtype=torch.float64)

        for b_idx in range(batch_size):
            viewmats, intrinsics = decompose_projection_matrix(tx_project[b_idx])
            viewmats, intrinsics = viewmats.float(), intrinsics.float()

            # Scale intrinsics for render resolution
            intrinsics_scaled = intrinsics.clone()
            intrinsics_scaled[:, 0, :] *= self.render_scale
            intrinsics_scaled[:, 1, :] *= self.render_scale

            depth_map, alpha_map = render_gaussian_depth(
                centers=all_centers[b_idx],
                scales=all_scales[b_idx],
                rotations=all_rotations[b_idx],
                opacities=all_opacities[b_idx],
                viewmats=viewmats,
                intrinsics=intrinsics_scaled,
                width=render_w,
                height=render_h,
                near_plane=self.near_plane,
                far_plane=self.far_plane,
            )

            rendered_depths.append(depth_map)
            rendered_alphas.append(alpha_map)

        # [b, n_cams, render_h, render_w]
        rendered_depth = torch.stack(rendered_depths, dim=0)
        rendered_alpha = torch.stack(rendered_alphas, dim=0)

        return {
            "rendered_depth": rendered_depth,
            "rendered_alpha": rendered_alpha,
        }

    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        centers: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        tx_project: torch.Tensor,
        img_shape: tuple[int, ...],
        *,
        rendered_depth: torch.Tensor,
        rendered_alpha: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute visibility by comparing gaussian depth to rendered depth surface.

        Uses original img_shape for projection (tx_project gives original-resolution
        pixel coords). The [-1, 1] normalized UVs work correctly with grid_sample
        on the rendered depth tensor regardless of its resolution.
        """
        # pylint: disable=arguments-differ, unused-argument, too-many-locals

        # Upcast for precision
        centers = amp.upcast(centers, dtype=torch.float32)
        scales = amp.upcast(scales, dtype=torch.float32)
        rotations = amp.upcast(rotations, dtype=torch.float32)
        tx_project = amp.upcast(tx_project, dtype=torch.float32)

        *_, h, w = img_shape
        n_cams = tx_project.shape[1]

        # Compute per-gaussian per-camera tolerance using viewing-aligned extent
        viewing_extent = compute_viewing_aligned_extent(scales, rotations, tx_project)
        per_cam_tolerance = viewing_extent * self.scale_factor + self.depth_tolerance

        # Project query gaussian centers to cameras (using original image dimensions)
        uv, depth_gauss, in_fov = project_points_to_cameras(centers, tx_project, (h, w))

        cam_visibilities = []
        for cam_idx in range(n_cams):
            # Reshape UV for grid_sample: [b, 1, n_gaussians, 2]
            cam_uv = uv[:, cam_idx].unsqueeze(1)

            # Sample rendered depth at projected locations
            cam_rendered_depth = rendered_depth[:, cam_idx].unsqueeze(
                1
            )  # [b, 1, rh, rw]
            sampled_depth = (
                F.grid_sample(
                    cam_rendered_depth,
                    cam_uv,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                .squeeze(1)
                .squeeze(1)
            )  # [b, n_gaussians]

            # Sample rendered alpha at projected locations
            cam_rendered_alpha = rendered_alpha[:, cam_idx].unsqueeze(1)
            sampled_alpha = (
                F.grid_sample(
                    cam_rendered_alpha,
                    cam_uv,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
                .squeeze(1)
                .squeeze(1)
            )  # [b, n_gaussians]

            cam_depth_gauss = depth_gauss[:, cam_idx]
            cam_tolerance = per_cam_tolerance[:, cam_idx]

            # Sigmoid-smoothed visibility:
            # depth_margin > 0 means gaussian is in front of or at the surface
            # depth_margin < 0 means gaussian is behind the surface
            depth_margin = sampled_depth + cam_tolerance - cam_depth_gauss
            smooth_vis = torch.sigmoid(depth_margin / self.smoothing)

            # Where no confident surface is rendered (low alpha), default to visible
            cam_vis = torch.where(
                sampled_alpha >= self.alpha_threshold,
                smooth_vis,
                torch.ones_like(smooth_vis),
            )

            # Zero out for out-of-FoV gaussians
            cam_vis = cam_vis * in_fov[:, cam_idx].float()

            cam_visibilities.append(cam_vis)

        # Max visibility across cameras (visible in at least one)
        visibilities = torch.stack(cam_visibilities, dim=-1)
        visibility = visibilities.max(dim=-1).values

        # In-FoV: visible in at least one camera
        any_in_fov = in_fov.any(dim=1)

        return visibility, any_in_fov


def build(conf: OmegaConf | dict | None = None) -> FoVVisibilityComputation | None:
    """
    Build FoV visibility computation from config.

    Backwards compatible: if no 'type' field, defaults to 'depth-bins'.
    """
    if conf is None:
        return NoneVisibility()

    if not OmegaConf.is_config(conf):
        conf = OmegaConf.create(conf)

    # Backwards compatibility: default to depth-bins if no type field
    if "type" not in conf:
        conf = OmegaConf.merge(OmegaConf.create({"type": "depth-bins"}), conf)

    return registry.from_config(conf)
