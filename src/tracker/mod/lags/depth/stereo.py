# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * BEVStereo (https://github.com/Megvii-BaseDetection/BEVStereo), Copyright (c) Megvii, licensed under MIT,
# * BEVDet (https://github.com/HuangJunJie2017/BEVDet), Copyright (c) Phigent Robotics, licensed under Apache-2.0.
# See the LICENSES/ directory for full license texts.

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint as ckpt

from ....utils.types import MetaDict
from .aspp import Aspp
from .depthnet import BasicBlock, DepthNet


@dataclass
class StereoState:
    """State carried between frames for stereo depth estimation."""

    features: torch.Tensor  # [b*ncams, cv_channels, h, w]
    ego_to_global: torch.Tensor  # [4, 4]
    ego_to_image: torch.Tensor  # [b, ncams, 4, 4]

    def detach(self):
        return StereoState(
            features=self.features.detach(),
            ego_to_global=self.ego_to_global.detach(),
            ego_to_image=self.ego_to_image.detach(),
        )


class StereoDepthNet(DepthNet):
    """
    Stereo depth estimation network adapted from BEVStereo/BEVDet4D.

    Extends DepthNet with temporal stereo: warps previous-frame image features
    to the current viewpoint and constructs a cost volume that provides geometric
    depth cues. The cost volume is concatenated with monocular depth features
    before the depth convolution head.

    On the first frame (no previous features), a zero cost volume is used,
    gracefully degrading to monocular-only depth estimation.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        context_channels: int,
        depth_bins: int,
        depth_range: tuple[float, float],
        depth_scaling: Literal["linear", "quadratic", "sid"] = "linear",
        cost_volume_channels: int = 64,
        stereo_bias: float = 5.0,
        use_aspp: bool = True,
        aspp_hidden_channels: int | None = None,
        use_checkpointing: bool = False,
    ):
        # pylint: disable=too-many-locals

        # Initialize parent (creates reduce_conv, SE gating, context_conv, depth_conv)
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            context_channels=context_channels,
            depth_bins=depth_bins,
            use_aspp=use_aspp,
            aspp_hidden_channels=aspp_hidden_channels,
            use_checkpointing=use_checkpointing,
        )

        self.depth_bins = depth_bins
        self.depth_range = depth_range
        self.depth_scaling = depth_scaling
        self.cost_volume_channels = cost_volume_channels
        self.stereo_bias = stereo_bias

        # Project features to lower dim for cost volume matching
        self.feature_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, cost_volume_channels, 1, bias=False),
            nn.BatchNorm2d(cost_volume_channels),
            nn.ReLU(inplace=True),
        )

        # Refine cost volume at depth feature resolution
        # Input/Output: [B*N, depth_bins, H, W]
        self.cost_volume_net = nn.Sequential(
            nn.Conv2d(depth_bins, depth_bins, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depth_bins),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_bins, depth_bins, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(depth_bins),
            nn.ReLU(inplace=True),
        )

        # Rebuild depth_conv: first BasicBlock takes extra channels from cost volume
        stereo_input_channels = hidden_channels + depth_bins
        downsample = nn.Sequential(
            nn.Conv2d(stereo_input_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )

        depth_conv_list = [
            BasicBlock(stereo_input_channels, hidden_channels, downsample=downsample),
            BasicBlock(hidden_channels, hidden_channels),
            BasicBlock(hidden_channels, hidden_channels),
        ]

        if use_aspp:
            aspp = Aspp(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                hidden_channels=aspp_hidden_channels or hidden_channels,
            )
            depth_conv_list.append(aspp)

        self.depth_conv = nn.Sequential(
            *depth_conv_list,
            nn.Conv2d(hidden_channels, depth_bins, kernel_size=1, stride=1, padding=0),
        )

        # Cache for cost volume features (set during forward, retrieved after)
        self._cv_features: torch.Tensor | None = None

    def get_cv_features(self) -> torch.Tensor:
        """Return the latest cost volume features for storage in StereoState."""
        assert self._cv_features is not None
        return self._cv_features

    def _create_frustum(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a frustum grid of depth hypotheses in pixel coordinates.

        Args:
            h: Feature height
            w: Feature width
            device: Target device

        Returns:
            Frustum coordinates [D, H, W, 4] as (u*d, v*d, d, 1)
        """
        d = self.depth_bins

        # Create depth bins (same logic as unproject_image_rays)
        if self.depth_scaling == "linear":
            depth_bin_size = (self.depth_range[1] - self.depth_range[0]) / d
            coords_d = torch.arange(d, device=device, dtype=torch.float32)
            coords_d = coords_d * depth_bin_size + self.depth_range[0]

        elif self.depth_scaling == "quadratic":
            depth_bin_size = (self.depth_range[1] - self.depth_range[0]) / (d * (1 + d))
            coords_d = torch.arange(d, device=device, dtype=torch.float32)
            coords_d = coords_d * (coords_d + 1) * depth_bin_size + self.depth_range[0]

        elif self.depth_scaling == "sid":
            depth_scale = torch.log(
                torch.tensor(
                    (self.depth_range[1] - 1) / self.depth_range[0], device=device
                )
            )
            depth_offs = math.log(self.depth_range[0])
            coords_d = torch.arange(d, device=device, dtype=torch.float32)
            coords_d = torch.exp(depth_offs + coords_d / (d - 1) * depth_scale)

        else:
            raise ValueError(f"Unknown depth_scaling: {self.depth_scaling}")

        # Create pixel coordinates at the feature resolution
        coords_w = torch.linspace(0.5, w - 0.5, w, device=device, dtype=torch.float32)
        coords_h = torch.linspace(0.5, h - 0.5, h, device=device, dtype=torch.float32)

        # Build meshgrid: [D, H, W]
        grid_d, grid_h, grid_w = torch.meshgrid(
            coords_d, coords_h, coords_w, indexing="ij"
        )

        # Create homogeneous coordinates (u*d, v*d, d, 1)
        frustum = torch.stack(
            [
                grid_w * grid_d,  # u * d
                grid_h * grid_d,  # v * d
                grid_d,  # d
                torch.ones_like(grid_d),  # 1
            ],
            dim=-1,
        )  # [D, H, W, 4]

        return frustum

    @torch.no_grad()
    def _compute_warping_grid(
        self,
        frustum: torch.Tensor,
        image_to_ego_curr: torch.Tensor,
        ego_to_image_prev: torch.Tensor,
        ego_to_global_curr: torch.Tensor,
        ego_to_global_prev: torch.Tensor,
        h_img: int,
        w_img: int,
    ) -> torch.Tensor:
        """
        Compute the sampling grid for warping previous frame features.

        Transform chain:
          current_pixel(u,v,d) → current_ego → global → prev_ego → prev_pixel

        Args:
            frustum: [D, H, W, 4] frustum in current pixel coords (u*d, v*d, d, 1)
            image_to_ego_curr: [b, ncams, 4, 4] current frame unprojection
            ego_to_image_prev: [b, ncams, 4, 4] previous frame projection
            ego_to_global_curr: [4, 4] current ego-to-global
            ego_to_global_prev: [4, 4] previous ego-to-global
            h_img: Image height at feature resolution (for grid normalization)
            w_img: Image width at feature resolution (for grid normalization)

        Returns:
            grid: [b*ncams, D*H, W, 2] normalized sampling grid for grid_sample
        """
        # pylint: disable=too-many-locals
        b, n = image_to_ego_curr.shape[:2]

        # Compute ego motion: current_ego → prev_ego
        with torch.autocast("cuda", enabled=False):
            ego_to_global_curr = ego_to_global_curr.to(dtype=torch.float64)
            ego_to_global_prev = ego_to_global_prev.to(dtype=torch.float64)

            # pylint: disable-next=not-callable
            ego_motion = torch.linalg.inv(ego_to_global_prev) @ ego_to_global_curr

            ego_motion = ego_motion.to(dtype=torch.float32)

        # Full warp transform: curr_pixel → prev_pixel
        # ego_to_image_prev @ ego_motion @ image_to_ego_curr
        # Shape: [b, n, 4, 4]
        warp_tx = ego_to_image_prev @ ego_motion.unsqueeze(0).unsqueeze(0)
        warp_tx = warp_tx @ image_to_ego_curr

        # Apply transform to frustum
        # frustum: [D, H, W, 4] → [1, 1, D*H*W, 4]
        d, h, w, _4 = frustum.shape
        pts = frustum.reshape(1, 1, d * h * w, 4, 1)

        # warp_tx: [b, n, 1, 4, 4]
        warp_tx = warp_tx.unsqueeze(2)

        # Batched matmul: [b, n, D*H*W, 4, 1]
        warped = warp_tx @ pts
        warped = warped.squeeze(-1)  # [b, n, D*H*W, 4]

        # Perspective divide
        z = warped[..., 2:3]
        neg_mask = z < 1e-3
        warped_uv = warped[..., :2] / z.clamp(min=1e-5)

        # Normalize to [-1, 1] for grid_sample
        warped_uv[..., 0] = warped_uv[..., 0] / (w_img - 1) * 2 - 1
        warped_uv[..., 1] = warped_uv[..., 1] / (h_img - 1) * 2 - 1

        # Mask points behind the previous camera
        neg_mask = neg_mask.expand_as(warped_uv)
        warped_uv[neg_mask] = -2.0  # Out of bounds for grid_sample

        # Reshape for grid_sample: [b*n, D*H, W, 2]
        grid = warped_uv.reshape(b * n, d * h, w, 2)

        return grid

    def _compute_cost_volume(
        self,
        curr_features: torch.Tensor,
        prev_features: torch.Tensor,
        grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute stereo matching cost volume.

        Args:
            curr_features: [B*N, cv_ch, H, W] current frame features
            prev_features: [B*N, cv_ch, H, W] previous frame features
            grid: [B*N, D*H, W, 2] sampling grid

        Returns:
            cost_volume: [B*N, D, H, W] soft depth probability from stereo
        """
        bn, c, h, w = curr_features.shape
        d = self.depth_bins

        # Warp previous features to current viewpoint
        warped = F.grid_sample(
            prev_features,
            grid,
            align_corners=True,
            padding_mode="zeros",
            mode="bilinear",
        )  # [B*N, cv_ch, D*H, W]

        # Reshape warped features: [B*N, cv_ch, D, H, W]
        warped = warped.view(bn, c, d, h, w)

        # Current features expanded: [B*N, cv_ch, 1, H, W] → broadcast to [B*N, cv_ch, D, H, W]
        curr_expanded = curr_features.unsqueeze(2)

        # Group-wise L1 cost: sum of per-channel absolute difference
        cost_volume = (curr_expanded - warped).abs().sum(dim=1)  # [B*N, D, H, W]

        # Penalize invalid regions (where warped features are zero = out of view)
        if self.stereo_bias != 0:
            invalid = warped[:, 0, ...] == 0  # [B*N, D, H, W]
            cost_volume = cost_volume + invalid.float() * self.stereo_bias

        # Convert to probability: negate and softmax over depth
        cost_volume = (-cost_volume).softmax(dim=1)

        return cost_volume

    def forward(
        self,
        x: torch.Tensor,
        img_meta: MetaDict,
        stereo_meta: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional stereo cost volume.

        Args:
            x: Image features [*, C, H, W] (same as DepthNet)
            img_meta: Camera metadata (same as DepthNet)
            stereo_meta: Optional dict with keys:
                - prev_features: [B*N, cv_ch, H, W]
                - prev_ego_to_image: [B, N, 4, 4]
                - curr_image_to_ego: [B, N, 4, 4]
                - curr_ego_to_global: [4, 4]
                - prev_ego_to_global: [4, 4]

        Returns:
            depth: [*, D, H, W] depth probability distribution
            context: [*, C_ctx, H, W] context features for voxel lifting
        """
        # pylint: disable=too-many-locals
        *b, c, h, w = x.shape

        # === Shared monocular processing (inherited from DepthNet) ===

        # Prepare camera parameters
        cam_params = self._collect_camera_params(img_meta.transforms, x.dtype)
        cam_params = cam_params.view(-1, cam_params.shape[-1])
        cam_params = self.bn(cam_params)

        # Prepare image features
        x = x.view(-1, c, h, w)  # [B*N, c, h, w]
        x = self.reduce_conv(x)

        # Context branch (identical to DepthNet)
        context_se = self.context_mlp(cam_params)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        # Monocular depth features with SE gating
        depth_se = self.depth_mlp(cam_params)[..., None, None]
        depth = self.depth_se(x, depth_se)

        # === Stereo cost volume ===

        # Project features for cost volume matching
        cv_features = self.feature_proj(x)  # [B*N, cv_ch, H, W]

        # Store for next frame (detached, no gradient flow across frames)
        self._cv_features = cv_features.detach()

        if stereo_meta is not None and stereo_meta.get("prev_features") is not None:
            # Compute warping grid
            frustum = self._create_frustum(h, w, device=x.device)

            grid = self._compute_warping_grid(
                frustum=frustum,
                image_to_ego_curr=stereo_meta["curr_image_to_ego"],
                ego_to_image_prev=stereo_meta["prev_ego_to_image"],
                ego_to_global_curr=stereo_meta["curr_ego_to_global"],
                ego_to_global_prev=stereo_meta["prev_ego_to_global"],
                h_img=h,
                w_img=w,
            )

            # Compute and refine cost volume at feature resolution
            cost_volume = self._compute_cost_volume(
                curr_features=cv_features,
                prev_features=stereo_meta["prev_features"],
                grid=grid,
            )  # [B*N, D, H, W]
            cost_volume = self.cost_volume_net(cost_volume)  # [B*N, D, H, W]
        else:
            # First frame: zero cost volume
            cost_volume = torch.zeros(
                depth.shape[0],
                self.depth_bins,
                h,
                w,
                device=x.device,
                dtype=x.dtype,
            )

        # Concatenate monocular depth features with cost volume
        depth = torch.cat([depth, cost_volume], dim=1)  # [B*N, hidden_ch + D, H, W]

        # Process through depth convolution head
        if self.use_checkpointing:
            depth = ckpt.checkpoint(self.depth_conv, depth, use_reentrant=False)
        else:
            depth = self.depth_conv(depth)

        depth = depth.softmax(dim=-3)

        # Reshape to original batch size
        depth = depth.view(*b, -1, h, w)
        context = context.view(*b, -1, h, w)

        return depth, context
