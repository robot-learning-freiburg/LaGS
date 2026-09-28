# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Gaussian depth rendering loss using gsplat.

Supervises gaussian placement by comparing rendered depth against
sparse ground truth depth (e.g., from LiDAR).
"""

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from ....utils.torch import amp


def decompose_projection_matrix(
    ego_to_image: torch.Tensor,  # [n_cams, 4, 4]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose ego_to_image projection matrix into viewmat and intrinsics.

    Uses RQ decomposition to extract K (intrinsics) and R,t (extrinsics).

    Args:
        ego_to_image: Combined projection matrix [n_cams, 4, 4]

    Returns:
        viewmats: [n_cams, 4, 4] camera extrinsics (ego-to-camera)
        intrinsics: [n_cams, 3, 3] camera intrinsic matrices
    """
    # pylint: disable=too-many-locals
    n_cams = ego_to_image.shape[0]
    device = ego_to_image.device
    dtype = ego_to_image.dtype

    viewmats = torch.zeros(n_cams, 4, 4, device=device, dtype=dtype)
    intrinsics = torch.zeros(n_cams, 3, 3, device=device, dtype=dtype)

    for cam_idx in range(n_cams):
        proj = ego_to_image[cam_idx]  # [4, 4]
        proj_3x4 = proj[:3, :]  # [3, 4]
        proj_3x3 = proj_3x4[:, :3]  # [3, 3]

        # RQ decomposition via QR on flipped transpose
        # pylint: disable-next=not-callable
        q_mat, r_mat = torch.linalg.qr(proj_3x3.T.flip(0, 1))
        r_mat = r_mat.flip(0, 1).T
        q_mat = q_mat.flip(0, 1).T

        # Ensure positive diagonal for K
        diag_signs = torch.diag(torch.sign(torch.diag(r_mat)))
        k_mat = r_mat @ diag_signs
        rot_mat = diag_signs @ q_mat

        # Ensure proper rotation (det = 1)
        if torch.det(rot_mat) < 0:
            k_mat[:, -1] *= -1
            rot_mat[-1, :] *= -1

        # Normalize so K[2,2] = 1
        k_mat = k_mat / k_mat[2, 2]

        # Extract translation
        # pylint: disable-next=not-callable
        t_vec = torch.linalg.solve(k_mat, proj_3x4[:, 3])

        # Build viewmat
        viewmats[cam_idx, :3, :3] = rot_mat
        viewmats[cam_idx, :3, 3] = t_vec
        viewmats[cam_idx, 3, 3] = 1.0

        intrinsics[cam_idx] = k_mat

    return viewmats, intrinsics


def render_gaussian_depth(
    centers: torch.Tensor,  # [n, 3]
    scales: torch.Tensor,  # [n, 3]
    rotations: torch.Tensor,  # [n, 4] wxyz
    opacities: torch.Tensor,  # [n]
    viewmats: torch.Tensor,  # [n_cams, 4, 4]
    intrinsics: torch.Tensor,  # [n_cams, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.1,
    far_plane: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Render depth from gaussians using gsplat.

    Args:
        centers: Gaussian centers [n, 3]
        scales: Gaussian scales [n, 3]
        rotations: Gaussian rotations (wxyz) [n, 4]
        opacities: Gaussian opacities [n]
        viewmats: Camera extrinsics [n_cams, 4, 4]
        intrinsics: Camera intrinsics [n_cams, 3, 3]
        width: Output image width
        height: Output image height
        near_plane: Near clipping plane
        far_plane: Far clipping plane

    Returns:
        depth: Rendered depth [n_cams, h, w]
        alpha: Accumulated opacity [n_cams, h, w]
    """
    # pylint: disable=import-outside-toplevel
    from gsplat import rasterization

    # Dummy colors (required by gsplat even for depth-only)
    colors = torch.ones(
        centers.shape[0],
        1,
        device=centers.device,
        dtype=torch.float32,
    )

    renders, alphas, _ = rasterization(
        means=centers.float().contiguous(),
        quats=rotations.float().contiguous(),
        scales=scales.float().contiguous(),
        opacities=opacities.float().contiguous(),
        colors=colors,
        viewmats=viewmats.float().contiguous(),
        Ks=intrinsics.float().contiguous(),
        width=width,
        height=height,
        render_mode="ED",  # Expected depth
        near_plane=near_plane,
        far_plane=far_plane,
    )

    # renders: [n_cams, h, w, 1], alphas: [n_cams, h, w, 1]
    return renders[..., 0], alphas[..., 0]


def downsample_sparse_depth_nearest(
    depth: torch.Tensor,
    mask: torch.Tensor,
    target_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Downsample sparse depth to target size using min pooling.

    Args:
        depth: Sparse depth map [n_cams, h, w]
        mask: Validity mask [n_cams, h, w]
        target_size: Target (h, w)

    Returns:
        Downsampled depth and mask
    """
    # Set invalid to inf for min pooling
    depth_masked = torch.where(mask, depth, torch.inf)

    # Compute kernel size
    h_in, w_in = depth.shape[-2:]
    h_out, w_out = target_size
    kernel_h = h_in // h_out
    kernel_w = w_in // w_out

    # Min pool
    depth_down = -F.max_pool2d(
        -depth_masked.unsqueeze(1),
        kernel_size=(kernel_h, kernel_w),
    ).squeeze(1)

    # Update mask
    mask_down = depth_down != torch.inf
    depth_down = torch.where(mask_down, depth_down, 0.0)

    return depth_down, mask_down


class GaussianDepthLoss(nn.Module):
    """
    Loss that supervises gaussian depth rendering against sparse GT depth.

    Renders depth from gaussians using gsplat and compares against
    ground truth depth (e.g., from LiDAR projection).
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        render_scale: float = 0.25,
        loss_type: Literal["l1", "l2", "smooth_l1"] = "l1",
        alpha_threshold: float = 1e-5,
        weight: float = 1.0,
        depth_range: tuple[float, float] = (1.0, 60.0),
        normalize_by_depth: bool = False,
        use_alpha_weighting: bool = False,
        opacity_pull_weight: float = 0.0,
        opacity_pull_sigma: float = 2.0,
    ):
        """
        Args:
            render_scale: Scale factor for rendering resolution (relative to image)
            loss_type: Type of regression loss
            alpha_threshold: Minimum alpha (coverage) for valid comparison
            weight: Loss weight
            depth_range: Valid depth range (min, max) in meters
            normalize_by_depth: Weight loss inversely by depth (closer = more important)
            use_alpha_weighting: If True, weight depth loss by (detached) rendered alpha
                to focus on well-covered regions.
            opacity_pull_weight: Weight for opacity pull loss that encourages high opacity
                where depth matches well. Set to 0 to disable (default).
            opacity_pull_sigma: Sigma for soft depth matching in opacity pull loss.
                Controls how forgiving the match is (in meters). Lower = stricter.
        """
        super().__init__()

        assert not (normalize_by_depth and use_alpha_weighting)

        self.render_scale = render_scale
        self.alpha_threshold = alpha_threshold
        self.weight = weight
        self.min_depth, self.max_depth = depth_range
        self.normalize_by_depth = normalize_by_depth
        self.use_alpha_weighting = use_alpha_weighting
        self.opacity_pull_weight = opacity_pull_weight
        self.opacity_pull_sigma = opacity_pull_sigma

        if loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif loss_type == "l2":
            self.loss_fn = F.mse_loss
        elif loss_type == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    @torch.autocast("cuda", enabled=False)
    def forward(
        self,
        centers: torch.Tensor,  # [b, n, 3]
        scales: torch.Tensor,  # [b, n, 3]
        rotations: torch.Tensor,  # [b, n, 4]
        opacities: torch.Tensor,  # [b, n]
        ego_to_image: torch.Tensor,  # [b, n_cams, 4, 4]
        image_shape: tuple[int, int],  # (h, w)
        gt_depth: torch.Tensor,  # [b, n_cams, h, w]
        gt_mask: torch.Tensor,  # [b, n_cams, h, w]
    ) -> torch.Tensor:
        """
        Compute gaussian depth loss.

        Args:
            centers, scales, rotations, opacities: Gaussian parameters
            ego_to_image: Projection matrices [b, n_cams, 4, 4]
            image_shape: Original image shape (h, w)
            gt_depth: Ground truth depth (sparse, e.g., from LiDAR)
            gt_mask: Valid mask for ground truth depth

        Returns:
            Scalar loss tensor
        """
        # pylint: disable=too-many-locals

        # Upcast for numerical stability
        centers = amp.upcast(centers, dtype=torch.float32)
        scales = amp.upcast(scales, dtype=torch.float32)
        rotations = amp.upcast(rotations, dtype=torch.float32)
        opacities = amp.upcast(opacities, dtype=torch.float32)
        ego_to_image = amp.upcast(ego_to_image, dtype=torch.float32)
        gt_depth = amp.upcast(gt_depth, dtype=torch.float32)

        batch_size = centers.shape[0]
        img_h, img_w = image_shape
        render_h = int(img_h * self.render_scale)
        render_w = int(img_w * self.render_scale)

        total_loss_depth = torch.tensor(0.0, device=centers.device)
        total_loss_opacity = torch.tensor(0.0, device=centers.device)
        total_valid = 0

        ego_to_image = amp.upcast(ego_to_image, dtype=torch.float64)

        for b_idx in range(batch_size):
            # Decompose projection matrices
            viewmats, intrinsics = decompose_projection_matrix(ego_to_image[b_idx])
            viewmats, intrinsics = viewmats.float(), intrinsics.float()

            # Scale intrinsics for render resolution
            intrinsics_scaled = intrinsics.clone()
            intrinsics_scaled[:, 0, :] *= self.render_scale
            intrinsics_scaled[:, 1, :] *= self.render_scale

            # Render depth from gaussians
            rendered_depth, rendered_alpha = render_gaussian_depth(
                centers=centers[b_idx],
                scales=scales[b_idx],
                rotations=rotations[b_idx],
                opacities=opacities[b_idx],
                viewmats=viewmats,
                intrinsics=intrinsics_scaled,
                width=render_w,
                height=render_h,
            )

            # Downsample GT depth to render resolution
            gt_depth_down, gt_mask_down = downsample_sparse_depth_nearest(
                gt_depth[b_idx],
                gt_mask[b_idx],
                target_size=(render_h, render_w),
            )

            # Build valid mask: good coverage AND valid gt AND in range
            valid_mask = (
                (rendered_alpha > self.alpha_threshold)
                & gt_mask_down
                & (gt_depth_down > self.min_depth)
                & (gt_depth_down < self.max_depth)
            )

            pred = rendered_depth[valid_mask]
            target = gt_depth_down[valid_mask]
            depth_error = (pred - target).abs()

            n_valid = valid_mask.sum()
            if n_valid > 0:
                if self.normalize_by_depth:
                    # Weight inversely by depth
                    weights = 1.0 / (target + 1.0)
                    batch_loss = (weights * depth_error).sum()
                elif self.use_alpha_weighting:
                    # Weight by detached alpha (unnormalized to match scale of default branch)
                    alpha_weights = rendered_alpha[valid_mask].detach()
                    batch_loss = (alpha_weights * depth_error).sum()
                else:
                    batch_loss = self.loss_fn(pred, target, reduction="sum")

                total_loss_depth = total_loss_depth + batch_loss
                total_valid = total_valid + n_valid

            # Opacity pull loss: encourage high opacity where depth matches well
            if self.opacity_pull_weight > 0.0 and n_valid > 0:
                match_quality = torch.exp(-depth_error / self.opacity_pull_sigma)

                # Pull opacity up where depth matches well
                opacity_pull = (1.0 - rendered_alpha[valid_mask]) * match_quality
                opacity_pull_loss = opacity_pull.sum()

                total_loss_opacity = total_loss_opacity + opacity_pull_loss

        # Normalize by total valid pixels
        total_valid = torch.clamp(total_valid, min=1)

        total_loss_depth = total_loss_depth / total_valid
        losses = {"depth_loss": total_loss_depth * self.weight}

        if self.opacity_pull_weight > 0.0:
            total_loss_opacity = total_loss_opacity / total_valid
            losses |= {
                "opacity_pull_loss": total_loss_opacity * self.opacity_pull_weight
            }

        return losses
