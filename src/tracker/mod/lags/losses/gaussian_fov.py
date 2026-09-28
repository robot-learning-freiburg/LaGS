# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Field-of-view and visibility losses for fresh gaussian predictions.

These losses encourage fresh gaussians to focus on well-observed regions:
- FoV loss: Hard penalty for predicting gaussians outside the camera field of view
- Visibility loss: Soft penalty for predicting in occluded/low-visibility areas
"""

from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from ....config.registry import Registry

registry = Registry("lags.losses.gaussian_fov")


class FreshGaussianLoss(nn.Module):
    """Base class for fresh gaussian losses."""

    def forward(
        self,
        opacities: torch.Tensor,
        visibility: torch.Tensor,
        in_fov: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for fresh gaussians.

        Args:
            opacities: Fresh gaussian opacities [b, n_fresh]
            visibility: Fresh gaussian visibility scores [b, n_fresh] in [0, 1]
            in_fov: In-FoV mask for fresh gaussians [b, n_fresh] bool

        Returns:
            Scalar loss value
        """
        raise NotImplementedError()


@registry.register(key="fov")
class FreshGaussianFoVLoss(FreshGaussianLoss):
    """
    Hard penalty for fresh gaussians predicted outside the field of view.

    Fresh gaussians should only represent what we can currently observe.
    Predicting outside FoV indicates hallucination - the model is predicting
    geometry it cannot verify from current observations.

    Loss: mean(opacity * (1 - in_fov))
    """

    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight: Loss weight multiplier
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        opacities: torch.Tensor,
        visibility: torch.Tensor,
        in_fov: torch.Tensor,
    ) -> torch.Tensor:
        # Penalize opacity of gaussians outside FoV
        outside_fov = ~in_fov
        loss = (opacities * outside_fov.float()).mean()
        return self.weight * loss


@registry.register(key="visibility")
class FreshGaussianVisibilityLoss(FreshGaussianLoss):
    """
    Soft penalty for fresh gaussians in low-visibility (occluded) areas.

    Even if a gaussian is within the FoV, if it's occluded (low visibility),
    our observation of it is unreliable. This loss encourages fresh gaussians
    to focus their opacity on well-observed regions.

    Loss: mean(opacity * max(0, threshold - visibility) * in_fov)
    """

    def __init__(
        self,
        weight: float = 0.5,
        threshold: float = 0.3,
    ):
        """
        Args:
            weight: Loss weight multiplier
            threshold: Visibility below this is penalized
        """
        super().__init__()
        self.weight = weight
        self.threshold = threshold

    def forward(
        self,
        opacities: torch.Tensor,
        visibility: torch.Tensor,
        in_fov: torch.Tensor,
    ) -> torch.Tensor:
        # Only penalize in-FoV gaussians with low visibility
        # (out-of-FoV is handled by FoV loss)
        vis_deficit = (self.threshold - visibility).clamp(min=0)
        loss = (opacities * vis_deficit * in_fov.float()).mean()
        return self.weight * loss


class FreshGaussianLosses(nn.Module):
    """
    Combined losses for fresh gaussians.

    Wraps multiple loss components configured via registry.
    """

    def __init__(self, losses: nn.ModuleDict):
        """
        Args:
            losses: Dictionary of loss name -> loss module
        """
        super().__init__()
        self.losses = losses

    def forward(
        self,
        opacities: torch.Tensor,
        visibility: torch.Tensor,
        in_fov: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all configured losses.

        Args:
            opacities: Fresh gaussian opacities [b, n_fresh]
            visibility: Fresh gaussian visibility scores [b, n_fresh]
            in_fov: In-FoV mask for fresh gaussians [b, n_fresh] bool

        Returns:
            Dictionary of loss name -> loss value
        """
        return {
            name: loss(opacities, visibility, in_fov)
            for name, loss in self.losses.items()
        }


def build(conf: OmegaConf | list | None, **kwargs: Any) -> FreshGaussianLosses | None:
    """
    Build fresh gaussian losses from config.

    Config can be either:
    - A list of loss configs: [{"type": "fov", "weight": 1.0}, {"type": "visibility", ...}]
    - A dict with "losses" key containing the list

    Each loss config should have a "type" key matching a registered loss.
    """
    if conf is None:
        return None

    # Handle both list and dict formats
    if isinstance(conf, (list, tuple)) or OmegaConf.is_list(conf):
        loss_configs = conf
    else:
        loss_configs = [conf]

    if not loss_configs:
        return None

    # Build each loss from registry
    losses = {}
    for loss_conf in loss_configs:
        loss = registry.from_config(loss_conf, **kwargs)

        # Use type as key, or generate unique key if duplicate
        key = loss_conf.get("type", "unknown")
        if key in losses:
            raise ValueError(f"Duplicate fresh gaussian loss type: {key}")

        losses[key] = loss

    return FreshGaussianLosses(nn.ModuleDict(losses))
