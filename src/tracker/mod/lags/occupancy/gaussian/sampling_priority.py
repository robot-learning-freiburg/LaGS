# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Sampling priority computation for gaussian query sampling.

This module provides registry-based priority computers that determine
where to sample new gaussian queries based on feature magnitude and
optionally past density coverage.
"""

import abc
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from .....config.registry import Registry

registry = Registry("lags.occupancy.gaussian.sampling_priority")


class SamplingPriorityComputer(nn.Module, abc.ABC):
    """Base class for computing sampling priorities."""

    @abc.abstractmethod
    def forward(
        self,
        voxel_feats: torch.Tensor,
        past_density: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute sampling priorities for gaussian query sampling.

        Args:
            voxel_feats: Voxel features [b, c, d, h, w]
            past_density: Optional density from previous frame [b, d, h, w]
                         (after ego-motion compensation)

        Returns:
            Sampling priorities [b, d, h, w] - higher values = more likely to sample
        """
        raise NotImplementedError()


def build(conf: OmegaConf | None = None, **kwargs: Any) -> SamplingPriorityComputer:
    """Build a sampling priority computer from config."""
    if conf is None:
        conf = OmegaConf.create({"type": "none"})
    return registry.from_config(conf, **kwargs)


@registry.register(key="feature-magnitude")
class FeatureMagnitudePriority(SamplingPriorityComputer):
    """
    Default: use feature magnitude only.

    This preserves backward compatibility by computing sampling priority
    as the norm of features across the channel dimension.
    """

    def __init__(
        self,
        ord: int = 2,
        normalization: str | None = None,
        top_percentile: float = 1.0,
        gamma: float = 1.0,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            ord: Order of the norm (default: 2 for L2 norm)
            normalization: Optional normalization method ("max", "minmax", "percentile", "log").
                If None, returns raw feature magnitudes (backward compatible).
            top_percentile: Fraction of top features to keep (0-1). Values outside
                this top fraction get zero priority. Remaining values are re-normalized to [0, 1].
                Only applies when normalization="percentile".
                E.g., top_percentile=0.25 keeps only the top 25% of features.
            gamma: Power exponent applied to normalized priorities. Values > 1 sharpen
                the distribution (more concentrated on peaks); < 1 flatten it.
            epsilon: Small value for numerical stability
        """
        # pylint: disable=redefined-builtin
        super().__init__()

        self.ord = ord
        self.normalization = normalization
        self.top_percentile = top_percentile
        self.gamma = gamma
        self.epsilon = epsilon

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize feature magnitudes to [0, 1] range per batch.

        Args:
            x: Feature magnitudes [b, d, h, w]

        Returns:
            Normalized tensor [b, d, h, w] in range [0, 1]
        """
        if self.normalization is None:
            return x

        b = x.shape[0]
        x_flat = x.view(b, -1)

        if self.normalization == "max":
            x_max = x_flat.max(dim=-1, keepdim=True).values
            normalized = x_flat / (x_max + self.epsilon)

        elif self.normalization == "minmax":
            x_min = x_flat.min(dim=-1, keepdim=True).values
            x_max = x_flat.max(dim=-1, keepdim=True).values
            normalized = (x_flat - x_min) / (x_max - x_min + self.epsilon)

        elif self.normalization == "percentile":
            # Rank-based normalization (robust to outliers)
            ranks = torch.argsort(torch.argsort(x_flat, dim=-1), dim=-1).float()
            normalized = ranks / (x_flat.shape[-1] - 1 + self.epsilon)

            # Apply top_percentile threshold: keep only top fraction, rest get zero
            # E.g., top_percentile=0.25 keeps top 25% (percentile >= 0.75)
            if self.top_percentile < 1.0:
                threshold = 1.0 - self.top_percentile
                normalized = torch.clamp(normalized - threshold, min=0.0)
                normalized = normalized * (1.0 / self.top_percentile)

        elif self.normalization == "log":
            # Log-space normalization
            x_log = torch.log1p(x_flat)
            x_min = x_log.min(dim=-1, keepdim=True).values
            x_max = x_log.max(dim=-1, keepdim=True).values
            normalized = (x_log - x_min) / (x_max - x_min + self.epsilon)

        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

        normalized = normalized**self.gamma

        return normalized.view_as(x)

    def forward(
        self,
        voxel_feats: torch.Tensor,
        past_density: torch.Tensor | None = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        # Compute vector norm across channel dimension
        # pylint: disable=not-callable
        feat_mag = torch.linalg.vector_norm(voxel_feats.detach(), dim=1, ord=self.ord)
        return self._normalize(feat_mag)


@registry.register(key="temporal-density-aware")
class DensityAwarePriority(SamplingPriorityComputer):
    """
    Combine feature magnitude with exponential suppression of past rendered coverage.

    High priority requires BOTH:
    - High feature magnitude (salient regions in current frame)
    - Low past rendered coverage (sparse regions needing more Gaussians)

    Uses multiplicative combination: priority = feat_norm * exp(-beta * prob_sum)

    `past_density` is expected to be the opacity-weighted, scale-normalized rendered
    coverage (prob_sum = Σ opacity_i * denom_i * power_i from the previous frame's
    splatting), which directly reflects how strongly each voxel is already explained
    by the current Gaussian set. Tight, opaque Gaussians suppress strongly; large or
    transparent ones suppress weakly, leaving room for new queries.

    Feature normalization maps magnitudes to [0, 1] before combination.
    Density suppression is soft (exponential), so no region is ever fully excluded.
    """

    def __init__(
        self,
        feature_normalization: str = "percentile",
        feature_top_percentile: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            feature_normalization: How to normalize features ("max", "minmax", "percentile", "log")
            feature_top_percentile: Fraction of top features to keep (0-1). Values outside
                this top fraction get zero priority. Remaining values are re-normalized to [0, 1].
                Only applies when feature_normalization="percentile".
                E.g., feature_top_percentile=0.25 keeps only the top 25% of features.
            beta: Exponential decay rate for coverage suppression in exp(-beta * prob_sum).
                Controls how aggressively already-rendered regions are down-weighted:
                - beta=0: no suppression (priority = feat_norm only)
                - beta=1: a voxel at the center of one opacity=1 unit-scale Gaussian
                  is suppressed by exp(-denom) where denom ≈ 0.063 for a 1m Gaussian
                - Higher beta: sharper suppression; well-rendered regions are sampled
                  much less often relative to poorly-covered ones.
            gamma: Power exponent applied to normalized feature priorities. Values > 1 sharpen
                the distribution (more concentrated on peaks); < 1 flatten it.
            epsilon: Small value for numerical stability
        """
        super().__init__()

        self.feature_normalization = feature_normalization
        self.feature_top_percentile = feature_top_percentile
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon

    def _normalize_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize feature magnitudes to [0, 1] range per batch.

        Args:
            x: Feature magnitudes [b, d, h, w]

        Returns:
            Normalized tensor [b, d, h, w] in range [0, 1]
        """
        b = x.shape[0]
        x_flat = x.view(b, -1)

        if self.feature_normalization == "max":
            x_max = x_flat.max(dim=-1, keepdim=True).values
            normalized = x_flat / (x_max + self.epsilon)

        elif self.feature_normalization == "minmax":
            x_min = x_flat.min(dim=-1, keepdim=True).values
            x_max = x_flat.max(dim=-1, keepdim=True).values
            normalized = (x_flat - x_min) / (x_max - x_min + self.epsilon)

        elif self.feature_normalization == "percentile":
            # Rank-based normalization (robust to outliers)
            ranks = torch.argsort(torch.argsort(x_flat, dim=-1), dim=-1).float()
            normalized = ranks / (x_flat.shape[-1] - 1 + self.epsilon)

            # Apply top_percentile threshold: keep only top fraction, rest get zero
            # E.g., top_percentile=0.25 keeps top 25% (percentile >= 0.75)
            if self.feature_top_percentile < 1.0:
                threshold = 1.0 - self.feature_top_percentile
                normalized = torch.clamp(normalized - threshold, min=0.0)
                normalized = normalized * (1.0 / self.feature_top_percentile)

        elif self.feature_normalization == "log":
            # Log-space normalization
            x_log = torch.log1p(x_flat)
            x_min = x_log.min(dim=-1, keepdim=True).values
            x_max = x_log.max(dim=-1, keepdim=True).values
            normalized = (x_log - x_min) / (x_max - x_min + self.epsilon)

        else:
            raise ValueError(
                f"Unknown feature normalization: {self.feature_normalization}"
            )

        normalized = normalized**self.gamma

        return normalized.view_as(x)

    def forward(
        self,
        voxel_feats: torch.Tensor,
        past_density: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Compute and normalize feature magnitude
        feat_mag = torch.norm(voxel_feats.detach(), dim=1)  # [b, d, h, w]
        feat_norm = self._normalize_features(feat_mag)

        if past_density is None:
            # First frame - no past density, use features only
            return feat_norm

        return feat_norm * torch.exp(-self.beta * past_density)
