# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from typing import Literal

import einops
import torch
from torch import nn
from torch.nn import functional as F

from ....utils.torch import amp


def downsample_sparse_depth(
    depth: torch.Tensor, mask: torch.Tensor, factor: int
) -> torch.Tensor:
    """
    Downsample sparse depth map.

    Downsamples a sparse depth map by a given factor using min pooling over
    valid pixels.

    Args:
        depth (torch.Tensor): Sparse depth map of shape (..., H, W).
        mask (torch.Tensor): Depth validity mask of shape (..., H, W).
        factor (int): Downsampling factor.

    Returns:
        torch.Tensor: Downsampled depth map of shape (..., H // factor, W // factor).
        torch.Tensor: Downsampled mask of shape (..., H // factor, W // factor).
    """
    *b, h, w = depth.shape

    assert (
        h % factor == 0 and w % factor == 0
    ), "Height and width must be divisible by the downsampling factor."

    # set invalid pixels to NaN
    depth = torch.where(mask, depth, torch.inf)

    # perform min pooling over the downsampled region
    depth = depth.view(-1, h // factor, factor, w // factor, factor)
    depth = einops.rearrange(depth, "... h dh w dw -> (... h w) (dh dw)")

    # take the mean of the valid pixels
    depth = torch.min(depth, dim=-1).values

    # reshape to the original leading dimensions
    depth = depth.view(*b, h // factor, w // factor)

    # get the downsampled mask
    mask = depth != torch.inf

    # reset the invalid pixels to zero
    depth = torch.where(mask, depth, 0.0)

    return depth, mask


def get_gt_depth_bins(
    depth: torch.Tensor,
    mask: torch.Tensor,
    depth_bins: int,
    depth_range: tuple[float, float] = (1.0, 60.0),
    scaling: Literal["linear", "sid"] = "linear",
) -> torch.Tensor:
    """
    Convert depth map to depth bins.

    Args:
        depth (torch.Tensor): Depth map of shape (..., H, W).
        mask (torch.Tensor): Depth validity mask of shape (..., H, W).
        depth_bins (int): Number of depth bins.
        depth_range (tuple[float, float]): Depth range (min, max).
        scaling (str): Scaling method. Either "linear" or "sid".
            "linear" scales depth linearly to the range [0, depth_bins].
            "sid" uses Spacing Increasing Discretization (SID) as introduced in
                `STS: Surround-view Temporal Stereo for Multi-view 3D
                Detection` to scale depth.

    Returns:
        torch.Tensor: Depth bins of shape (..., H, W, depth_bins).
        torch.Tensor: Depth mask of shape (..., H, W).
    """
    min_depth, max_depth = depth_range

    # scale depth to the range [0, depth_bins]
    if scaling == "linear":
        depth = ((depth - min_depth) / (max_depth - min_depth)) * depth_bins

    elif scaling == "sid":
        depth = torch.log(depth) - math.log(min_depth)
        depth = depth * (depth_bins - 1) / math.log((max_depth - 1.0) / min_depth)

    # add "invalid" depth bin as the first bin
    depth = depth + 1.0

    # set invalid depth bins to zero
    depth = torch.where(mask & (0 < depth) & (depth < depth_bins), depth, 0.0)

    # convert to integer bin indices
    depth = depth.long()
    depth = F.one_hot(depth, num_classes=depth_bins + 1)  # pylint: disable=not-callable

    # remove the "invalid" bin
    mask = depth[..., 0] == 0
    depth = depth[..., 1:]

    return depth.float(), mask


def prepare_depth_gt(
    depth: torch.Tensor,
    mask: torch.Tensor,
    depth_bins: int,
    depth_range: tuple[float, float] = (1.0, 60.0),
    scaling: Literal["linear", "sid"] = "linear",
    downsampling_factor: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare ground truth depth map and mask.

    Args:
        depth (torch.Tensor): Depth map of shape (..., H, W).
        mask (torch.Tensor): Depth validity mask of shape (..., H, W).
        depth_bins (int): Number of depth bins.
        depth_range (tuple[float, float]): Depth range (min, max).
        scaling (str): Scaling method. Either "linear" or "sid".
        downsampling_factor (int): Downsampling factor for the depth map.

    Returns:
        torch.Tensor: Depth bins of shape (..., H//factor, W//factor, depth_bins).
        torch.Tensor: Depth mask of shape (..., H//factor, W//factor).
    """
    # downsample the depth map
    depth, mask = downsample_sparse_depth(depth, mask, factor=downsampling_factor)

    # convert to depth bins
    depth, mask = get_gt_depth_bins(
        depth,
        mask,
        depth_bins=depth_bins,
        depth_range=depth_range,
        scaling=scaling,
    )

    return depth, mask


class DepthLoss(nn.Module):
    """
    Loss for binned depth prediction.
    """

    def __init__(
        self,
        range: tuple[float, float] = (1.0, 60.0),
        scaling: Literal["linear", "sid"] = "linear",
        weight: float = 1.0,
    ):
        # pylint: disable=redefined-builtin
        super().__init__()

        self.depth_range = range
        self.scaling = scaling
        self.weight = weight

    @torch.autocast("cuda", enabled=False)
    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the depth loss.

        Args:
            preds (torch.Tensor): Predicted depth map of shape (..., depth_bins, H//d, W//d).
            targets (torch.Tensor): Ground truth depth map of shape (..., H, W).
            mask (torch.Tensor): Depth validity mask of shape (..., H, W).

        Returns:
            torch.Tensor: Depth loss.
        """
        depth_bins, h_pred, w_pred = preds.shape[-3:]
        h_target, w_target = targets.shape[-2:]

        # ensure that the aspect ratio is preserved
        assert h_target // h_pred == w_target // w_pred

        # downsample the target depth map and convert it to depth bins
        targets, mask = prepare_depth_gt(
            targets,
            mask,
            depth_bins=depth_bins,
            depth_range=self.depth_range,
            scaling=self.scaling,
            downsampling_factor=h_target // h_pred,
        )

        # permute and reshape to [_, h_pred * w_pred, depth_bins]
        preds = preds.view(-1, depth_bins, h_pred * w_pred)
        preds = preds.permute(0, 2, 1)

        targets = targets.view(-1, h_pred * w_pred, depth_bins)
        mask = mask.view(-1, h_pred * w_pred)

        # filter out pixels without valid ground truth depth
        preds = preds[mask]
        targets = targets[mask]

        # NOTE: We cannot use `F.binary_cross_entropy_with_logits` here because
        # the targets are one-hot encoded depth bins and we use a softmax (not
        # sigmoid) to generate the predictions. Unfortunately, that means that
        # we have to fall back to the more unstable binary_cross_entropy. To
        # avoid any problems, we explicitly disable autocasting here and ensure
        # tensors are fp32. Also: We disable it for the whole function to keep
        # the target conversion in higher precision as well (though not sure if
        # that would make much of a difference).
        preds = amp.upcast(preds, dtype=torch.float32)
        preds = torch.clamp(preds, min=1e-9, max=1.0 - 1e-9)
        targets = amp.upcast(targets, dtype=torch.float32)

        # compute the depth loss
        loss = F.binary_cross_entropy(preds, targets, reduction="sum")
        loss = loss / max(1.0, mask.sum())

        return loss * self.weight
