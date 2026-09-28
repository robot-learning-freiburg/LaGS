# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pruning operations for temporal Gaussian instances."""

import abc
from collections import defaultdict
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from ......config.registry import Registry
from .storage import GaussianInstances, GaussianStreamInstances

registry = Registry("lags.occupancy.gaussian.pruning")


class GaussianPruning(nn.Module):
    """Base class for Gaussian pruning operations."""

    @abc.abstractmethod
    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """
        Prune gaussians based on specific criteria.

        Args:
            gaussians: Input gaussian instances

        Returns:
            Pruned gaussian instances
        """
        raise NotImplementedError()


@registry.register(key="range")
class RangePruning(GaussianPruning):
    """Prune gaussians outside a specified range."""

    def __init__(self, voxel_range: list[float], margin: float = 0.0):
        """
        Initialize range-based pruning.

        Args:
            voxel_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            margin: Margin to add around voxel range boundaries (in meters).
                    Gaussians within voxel_range ± margin are kept.
        """
        super().__init__()

        voxel_range = torch.as_tensor(voxel_range, dtype=torch.float32)

        # Expand range by margin
        voxel_min = voxel_range[:3] - margin
        voxel_max = voxel_range[3:] + margin
        self.register_buffer("voxel_min", voxel_min, persistent=False)
        self.register_buffer("voxel_max", voxel_max, persistent=False)

    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """Prune gaussians outside voxel range (including margin)."""
        pruned_streams = {}

        for stream_name, stream_data in gaussians.streams.items():
            # Assume batch size of 1, squeeze batch dimension
            centers = stream_data.centers.squeeze(0)  # [n, 3]

            # Check voxel range bounds (with margin)
            in_range_mask = torch.all(centers >= self.voxel_min, dim=-1)
            in_range_mask &= torch.all(centers <= self.voxel_max, dim=-1)

            # Apply mask to all fields and add batch dimension back
            pruned_streams[stream_name] = GaussianStreamInstances(
                query=stream_data.query.squeeze(0)[in_range_mask].unsqueeze(0),
                query_coords=stream_data.query_coords.squeeze(0)[
                    in_range_mask
                ].unsqueeze(0),
                logits=stream_data.logits.squeeze(0)[in_range_mask].unsqueeze(0),
                centers=stream_data.centers.squeeze(0)[in_range_mask].unsqueeze(0),
                scales=stream_data.scales.squeeze(0)[in_range_mask].unsqueeze(0),
                rotations=stream_data.rotations.squeeze(0)[in_range_mask].unsqueeze(0),
                opacities=stream_data.opacities.squeeze(0)[in_range_mask].unsqueeze(0),
                age=stream_data.age.squeeze(0)[in_range_mask].unsqueeze(0),
                confidence=stream_data.confidence.squeeze(0)[in_range_mask].unsqueeze(
                    0
                ),
                instance_ids=stream_data.instance_ids.squeeze(0)[
                    in_range_mask
                ].unsqueeze(0),
            )

        return GaussianInstances(streams=pruned_streams)


@registry.register(key="opacity")
class OpacityPruning(GaussianPruning):
    """Prune gaussians below a minimum opacity threshold."""

    def __init__(self, threshold: float):
        """
        Initialize opacity-based pruning.

        Args:
            threshold: Minimum opacity threshold (typically 0.0 to 1.0)
        """
        super().__init__()

        self.threshold = threshold

    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """Prune gaussians with opacity below threshold."""
        pruned = {}

        for stream_name, stream_data in gaussians.streams.items():
            # Assume batch size of 1, squeeze batch dimension
            opacities = stream_data.opacities.squeeze(0)  # [n]
            valid_mask = opacities >= self.threshold  # [n]

            # Apply mask to all fields and add batch dimension back
            pruned[stream_name] = GaussianStreamInstances(
                query=stream_data.query.squeeze(0)[valid_mask].unsqueeze(0),
                query_coords=stream_data.query_coords.squeeze(0)[valid_mask].unsqueeze(
                    0
                ),
                logits=stream_data.logits.squeeze(0)[valid_mask].unsqueeze(0),
                centers=stream_data.centers.squeeze(0)[valid_mask].unsqueeze(0),
                scales=stream_data.scales.squeeze(0)[valid_mask].unsqueeze(0),
                rotations=stream_data.rotations.squeeze(0)[valid_mask].unsqueeze(0),
                opacities=stream_data.opacities.squeeze(0)[valid_mask].unsqueeze(0),
                age=stream_data.age.squeeze(0)[valid_mask].unsqueeze(0),
                confidence=stream_data.confidence.squeeze(0)[valid_mask].unsqueeze(0),
                instance_ids=stream_data.instance_ids.squeeze(0)[valid_mask].unsqueeze(
                    0
                ),
            )

        return GaussianInstances(streams=pruned)


@registry.register(key="age")
class AgePruning(GaussianPruning):
    """Prune gaussians older than a maximum age."""

    def __init__(self, threshold: int):
        """
        Initialize age-based pruning.

        Args:
            threshold: Maximum age in frames. Gaussians with age > threshold are pruned.
        """
        super().__init__()

        self.threshold = threshold

    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """Prune gaussians older than threshold."""
        pruned = {}

        for name, data in gaussians.streams.items():
            # Assume batch size of 1, squeeze batch dimension
            age = data.age.squeeze(0)  # [n]
            valid_mask = age <= self.threshold  # [n]

            # Apply mask to all fields and add batch dimension back
            pruned[name] = GaussianStreamInstances(
                query=data.query.squeeze(0)[valid_mask].unsqueeze(0),
                query_coords=data.query_coords.squeeze(0)[valid_mask].unsqueeze(0),
                logits=data.logits.squeeze(0)[valid_mask].unsqueeze(0),
                centers=data.centers.squeeze(0)[valid_mask].unsqueeze(0),
                scales=data.scales.squeeze(0)[valid_mask].unsqueeze(0),
                rotations=data.rotations.squeeze(0)[valid_mask].unsqueeze(0),
                opacities=data.opacities.squeeze(0)[valid_mask].unsqueeze(0),
                age=data.age.squeeze(0)[valid_mask].unsqueeze(0),
                confidence=data.confidence.squeeze(0)[valid_mask].unsqueeze(0),
                instance_ids=data.instance_ids.squeeze(0)[valid_mask].unsqueeze(0),
            )

        return GaussianInstances(streams=pruned)


@registry.register(key="scale")
class ScalePruning(GaussianPruning):
    """Prune gaussians smaller than the specified scale."""

    def __init__(self, threshold: float):
        """
        Initialize scale-based pruning.

        Args:
            threshold: Minimum scale per dimension.
        """
        super().__init__()

        self.threshold = threshold

    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """Prune gaussians older than threshold."""
        pruned = {}

        for name, data in gaussians.streams.items():
            # Assume batch size of 1, squeeze batch dimension
            scales = data.scales.squeeze(0)  # [n, 3]
            valid_mask = torch.any(scales >= self.threshold, dim=-1)  # [n]

            # Apply mask to all fields and add batch dimension back
            pruned[name] = GaussianStreamInstances(
                query=data.query.squeeze(0)[valid_mask].unsqueeze(0),
                query_coords=data.query_coords.squeeze(0)[valid_mask].unsqueeze(0),
                logits=data.logits.squeeze(0)[valid_mask].unsqueeze(0),
                centers=data.centers.squeeze(0)[valid_mask].unsqueeze(0),
                scales=data.scales.squeeze(0)[valid_mask].unsqueeze(0),
                rotations=data.rotations.squeeze(0)[valid_mask].unsqueeze(0),
                opacities=data.opacities.squeeze(0)[valid_mask].unsqueeze(0),
                age=data.age.squeeze(0)[valid_mask].unsqueeze(0),
                confidence=data.confidence.squeeze(0)[valid_mask].unsqueeze(0),
                instance_ids=data.instance_ids.squeeze(0)[valid_mask].unsqueeze(0),
            )

        return GaussianInstances(streams=pruned)


@registry.register(key="topk_opacity")
class TopKOpacityPruning(GaussianPruning):
    """Retain only the top K gaussians with highest opacity."""

    def __init__(self, k: int | dict[str, int]):
        """
        Initialize top-K opacity-based pruning.

        Args:
            k: Number of gaussians to retain per stream. Can be either:
               - int: Same budget for all streams
               - dict[str, int]: Per-stream budgets (e.g., {"coarse": 100, "medium": 200}).
                 Streams not in dict are kept unmodified.
        """
        super().__init__()

        if isinstance(k, int):
            # All streams get the same budget
            self.k_per_stream = defaultdict(lambda: k)
        else:
            # Per-stream budgets, unspecified streams return None (keep all)
            self.k_per_stream = defaultdict(lambda: None, k)

    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """Retain only top K gaussians by opacity per batch."""
        pruned = {}

        for name, data in gaussians.streams.items():
            k = self.k_per_stream[name]

            if k is None:
                # No budget specified for this stream, keep all
                pruned[name] = data
                continue

            opacities = data.opacities  # [b, n]
            batch_size, num_gaussians = opacities.shape

            if num_gaussians <= k:
                # Already within budget, keep all
                pruned[name] = data
                continue

            # Get indices of top-k opacities per batch element
            # topk returns (values, indices) where indices are [b, k]
            _, top_indices = torch.topk(opacities, k=k, dim=1, largest=True)

            # Use advanced indexing to gather top-k elements
            # batch_indices: [b, 1] repeated to [b, k]
            batch_indices = (
                torch.arange(batch_size, device=opacities.device)
                .unsqueeze(1)
                .expand(-1, k)
            )

            # Apply indices to all fields
            pruned[name] = GaussianStreamInstances(
                query=data.query[batch_indices, top_indices],
                query_coords=data.query_coords[batch_indices, top_indices],
                logits=data.logits[batch_indices, top_indices],
                centers=data.centers[batch_indices, top_indices],
                scales=data.scales[batch_indices, top_indices],
                rotations=data.rotations[batch_indices, top_indices],
                opacities=data.opacities[batch_indices, top_indices],
                age=data.age[batch_indices, top_indices],
                confidence=data.confidence[batch_indices, top_indices],
                instance_ids=data.instance_ids[batch_indices, top_indices],
            )

        return GaussianInstances(streams=pruned)


@registry.register(key="topk_opacity_random")
class TopKOpacityRandomPruning(GaussianPruning):
    """Retain K gaussians sampled with probability proportional to opacity."""

    def __init__(self, k: int | dict[str, int]):
        """
        Initialize randomized top-K opacity-based pruning.

        Args:
            k: Number of gaussians to retain per stream. Can be either:
               - int: Same budget for all streams
               - dict[str, int]: Per-stream budgets (e.g., {"coarse": 100, "medium": 200}).
                 Streams not in dict are kept unmodified.
        """
        super().__init__()

        if isinstance(k, int):
            self.k_per_stream = defaultdict(lambda: k)
        else:
            self.k_per_stream = defaultdict(lambda: None, k)

    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """Sample K gaussians weighted by opacity per batch."""
        pruned = {}

        for name, data in gaussians.streams.items():
            k = self.k_per_stream[name]

            if k is None:
                pruned[name] = data
                continue

            opacities = data.opacities  # [b, n]
            batch_size, num_gaussians = opacities.shape

            if num_gaussians <= k:
                pruned[name] = data
                continue

            # Sample K indices with probability proportional to opacity
            weights = opacities.clamp(min=0.0)
            sampled_indices = torch.multinomial(weights, k, replacement=False)

            batch_indices = (
                torch.arange(batch_size, device=opacities.device)
                .unsqueeze(1)
                .expand(-1, k)
            )

            pruned[name] = GaussianStreamInstances(
                query=data.query[batch_indices, sampled_indices],
                query_coords=data.query_coords[batch_indices, sampled_indices],
                logits=data.logits[batch_indices, sampled_indices],
                centers=data.centers[batch_indices, sampled_indices],
                scales=data.scales[batch_indices, sampled_indices],
                rotations=data.rotations[batch_indices, sampled_indices],
                opacities=data.opacities[batch_indices, sampled_indices],
                age=data.age[batch_indices, sampled_indices],
                confidence=data.confidence[batch_indices, sampled_indices],
                instance_ids=data.instance_ids[batch_indices, sampled_indices],
            )

        return GaussianInstances(streams=pruned)


@registry.register(key="topk_confidence")
class TopKConfidencePruning(GaussianPruning):
    """Retain only the top K gaussians with highest confidence."""

    def __init__(self, k: int | dict[str, int]):
        """
        Initialize top-K confidence-based pruning.

        Args:
            k: Number of gaussians to retain per stream. Can be either:
               - int: Same budget for all streams
               - dict[str, int]: Per-stream budgets (e.g., {"coarse": 100, "fine": 200}).
                 Streams not in dict are kept unmodified.
        """
        super().__init__()

        if isinstance(k, int):
            # All streams get the same budget
            self.k_per_stream = defaultdict(lambda: k)
        else:
            # Per-stream budgets, unspecified streams return None (keep all)
            self.k_per_stream = defaultdict(lambda: None, k)

    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """Retain only top K gaussians by confidence per batch."""
        pruned = {}

        for name, data in gaussians.streams.items():
            k = self.k_per_stream[name]

            if k is None:
                # No budget specified for this stream, keep all
                pruned[name] = data
                continue

            confidence = data.confidence  # [b, n]
            batch_size, num_gaussians = confidence.shape

            if num_gaussians <= k:
                # Already within budget, keep all
                pruned[name] = data
                continue

            # Get indices of top-k confidence per batch element
            # topk returns (values, indices) where indices are [b, k]
            _, top_indices = torch.topk(confidence, k=k, dim=1, largest=True)

            # Use advanced indexing to gather top-k elements
            # batch_indices: [b, 1] repeated to [b, k]
            batch_indices = (
                torch.arange(batch_size, device=confidence.device)
                .unsqueeze(1)
                .expand(-1, k)
            )

            # Apply indices to all fields
            pruned[name] = GaussianStreamInstances(
                query=data.query[batch_indices, top_indices],
                query_coords=data.query_coords[batch_indices, top_indices],
                logits=data.logits[batch_indices, top_indices],
                centers=data.centers[batch_indices, top_indices],
                scales=data.scales[batch_indices, top_indices],
                rotations=data.rotations[batch_indices, top_indices],
                opacities=data.opacities[batch_indices, top_indices],
                age=data.age[batch_indices, top_indices],
                confidence=data.confidence[batch_indices, top_indices],
                instance_ids=data.instance_ids[batch_indices, top_indices],
            )

        return GaussianInstances(streams=pruned)


@registry.register(key="topk_confidence_random")
class TopKConfidenceRandomPruning(GaussianPruning):
    """Retain K gaussians sampled with probability proportional to confidence."""

    def __init__(self, k: int | dict[str, int]):
        """
        Initialize randomized top-K confidence-based pruning.

        Args:
            k: Number of gaussians to retain per stream. Can be either:
               - int: Same budget for all streams
               - dict[str, int]: Per-stream budgets (e.g., {"coarse": 100, "fine": 200}).
                 Streams not in dict are kept unmodified.
        """
        super().__init__()

        if isinstance(k, int):
            self.k_per_stream = defaultdict(lambda: k)
        else:
            self.k_per_stream = defaultdict(lambda: None, k)

    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """Sample K gaussians weighted by confidence per batch."""
        pruned = {}

        for name, data in gaussians.streams.items():
            k = self.k_per_stream[name]

            if k is None:
                pruned[name] = data
                continue

            confidence = data.confidence  # [b, n]
            batch_size, num_gaussians = confidence.shape

            if num_gaussians <= k:
                pruned[name] = data
                continue

            # Sample K indices with probability proportional to confidence
            weights = confidence.clamp(min=0.0)
            sampled_indices = torch.multinomial(weights, k, replacement=False)

            batch_indices = (
                torch.arange(batch_size, device=confidence.device)
                .unsqueeze(1)
                .expand(-1, k)
            )

            pruned[name] = GaussianStreamInstances(
                query=data.query[batch_indices, sampled_indices],
                query_coords=data.query_coords[batch_indices, sampled_indices],
                logits=data.logits[batch_indices, sampled_indices],
                centers=data.centers[batch_indices, sampled_indices],
                scales=data.scales[batch_indices, sampled_indices],
                rotations=data.rotations[batch_indices, sampled_indices],
                opacities=data.opacities[batch_indices, sampled_indices],
                age=data.age[batch_indices, sampled_indices],
                confidence=data.confidence[batch_indices, sampled_indices],
                instance_ids=data.instance_ids[batch_indices, sampled_indices],
            )

        return GaussianInstances(streams=pruned)


def _rank_hybrid_score(
    opacities: torch.Tensor,
    confidence: torch.Tensor,
    mode: str = "sum",
) -> torch.Tensor:
    """
    Compute a scale-invariant hybrid score from opacity and confidence.

    Rank-normalizes each signal independently to avoid scale imbalance
    (opacity values are typically much lower than confidence values), then
    combines them according to ``mode``.

    Args:
        opacities: [b, n] opacity values
        confidence: [b, n] confidence values
        mode: How to combine the two rank signals.
            ``"sum"``  – rank_opacity + rank_confidence. Keeps gaussians that
            score well on *both* criteria (default, backwards-compatible).
            ``"max"``  – max(rank_opacity, rank_confidence). Keeps a gaussian
            if it ranks highly on *either* criterion: currently visible (high
            opacity) OR well-established out-of-FOV (high confidence).

    Returns:
        [b, n] combined rank scores (higher = better)
    """
    rank_opacity = torch.argsort(torch.argsort(opacities, dim=1), dim=1).float()
    rank_confidence = torch.argsort(torch.argsort(confidence, dim=1), dim=1).float()
    if mode == "max":
        return torch.max(rank_opacity, rank_confidence)
    return rank_opacity + rank_confidence


@registry.register(key="topk_hybrid")
class TopKHybridPruning(GaussianPruning):
    """Retain only the top K gaussians by combined opacity+confidence rank."""

    def __init__(self, k: int | dict[str, int], hybrid_mode: str = "sum"):
        """
        Initialize top-K hybrid pruning.

        Args:
            k: Number of gaussians to retain per stream. Can be either:
               - int: Same budget for all streams
               - dict[str, int]: Per-stream budgets (e.g., {"coarse": 100, "fine": 200}).
                 Streams not in dict are kept unmodified.
            hybrid_mode: Rank combination mode passed to ``_rank_hybrid_score``.
                ``"sum"`` (default) or ``"max"``.
        """
        super().__init__()

        if isinstance(k, int):
            self.k_per_stream = defaultdict(lambda: k)
        else:
            self.k_per_stream = defaultdict(lambda: None, k)

        self.hybrid_mode = hybrid_mode

    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """Retain only top K gaussians by hybrid rank score per batch."""
        pruned = {}

        for name, data in gaussians.streams.items():
            k = self.k_per_stream[name]

            if k is None:
                pruned[name] = data
                continue

            score = _rank_hybrid_score(
                data.opacities, data.confidence, self.hybrid_mode
            )  # [b, n]
            batch_size, num_gaussians = score.shape

            if num_gaussians <= k:
                pruned[name] = data
                continue

            _, top_indices = torch.topk(score, k=k, dim=1, largest=True)

            batch_indices = (
                torch.arange(batch_size, device=score.device).unsqueeze(1).expand(-1, k)
            )

            pruned[name] = GaussianStreamInstances(
                query=data.query[batch_indices, top_indices],
                query_coords=data.query_coords[batch_indices, top_indices],
                logits=data.logits[batch_indices, top_indices],
                centers=data.centers[batch_indices, top_indices],
                scales=data.scales[batch_indices, top_indices],
                rotations=data.rotations[batch_indices, top_indices],
                opacities=data.opacities[batch_indices, top_indices],
                age=data.age[batch_indices, top_indices],
                confidence=data.confidence[batch_indices, top_indices],
                instance_ids=data.instance_ids[batch_indices, top_indices],
            )

        return GaussianInstances(streams=pruned)


@registry.register(key="topk_hybrid_random")
class TopKHybridRandomPruning(GaussianPruning):
    """Retain K gaussians sampled with probability proportional to combined rank score."""

    def __init__(
        self,
        k: int | dict[str, int],
        temperature: float = 1.0,
        hybrid_mode: str = "sum",
    ):
        """
        Initialize randomized top-K hybrid pruning.

        Args:
            k: Number of gaussians to retain per stream. Can be either:
               - int: Same budget for all streams
               - dict[str, int]: Per-stream budgets (e.g., {"coarse": 100, "fine": 200}).
                 Streams not in dict are kept unmodified.
            temperature: Exponent applied to rank scores before sampling. Higher
                values sharpen the distribution toward high-ranked gaussians
                (temperature=1 is linear, temperature→∞ approaches deterministic topk).
            hybrid_mode: Rank combination mode passed to ``_rank_hybrid_score``.
                ``"sum"`` (default) or ``"max"``.
        """
        super().__init__()

        if isinstance(k, int):
            self.k_per_stream = defaultdict(lambda: k)
        else:
            self.k_per_stream = defaultdict(lambda: None, k)

        self.temperature = temperature
        self.hybrid_mode = hybrid_mode

    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """Sample K gaussians weighted by hybrid_rank_score^temperature."""
        pruned = {}

        for name, data in gaussians.streams.items():
            k = self.k_per_stream[name]

            if k is None:
                pruned[name] = data
                continue

            score = _rank_hybrid_score(
                data.opacities, data.confidence, self.hybrid_mode
            )  # [b, n]
            batch_size, num_gaussians = score.shape

            if num_gaussians <= k:
                pruned[name] = data
                continue

            weights = score.clamp(min=1.0) ** self.temperature
            sampled_indices = torch.multinomial(weights, k, replacement=False)

            batch_indices = (
                torch.arange(batch_size, device=score.device).unsqueeze(1).expand(-1, k)
            )

            pruned[name] = GaussianStreamInstances(
                query=data.query[batch_indices, sampled_indices],
                query_coords=data.query_coords[batch_indices, sampled_indices],
                logits=data.logits[batch_indices, sampled_indices],
                centers=data.centers[batch_indices, sampled_indices],
                scales=data.scales[batch_indices, sampled_indices],
                rotations=data.rotations[batch_indices, sampled_indices],
                opacities=data.opacities[batch_indices, sampled_indices],
                age=data.age[batch_indices, sampled_indices],
                confidence=data.confidence[batch_indices, sampled_indices],
                instance_ids=data.instance_ids[batch_indices, sampled_indices],
            )

        return GaussianInstances(streams=pruned)


@registry.register(key="confidence")
class ConfidencePruning(GaussianPruning):
    """Prune gaussians below a minimum confidence threshold.

    Includes min_age protection to prevent fresh gaussians from being pruned
    before they have a chance to accumulate confidence.

    Includes opacity_threshold to keep high-opacity gaussians regardless of
    confidence (useful for newly visible areas).
    """

    def __init__(
        self,
        threshold: float,
        min_age: int = 0,
        opacity_threshold: float = 1e8,
    ):
        """
        Initialize confidence-based pruning.

        Args:
            threshold: Minimum confidence threshold (0.0 to 1.0).
                Gaussians with confidence < threshold are pruned.
            min_age: Minimum age (in frames) before a gaussian can be pruned.
                Gaussians with age < min_age are always kept regardless of
                confidence. Default is 0 (no age protection).
            opacity_threshold: Opacity threshold (0.0 to 1.0). Gaussians with
                opacity >= opacity_threshold are kept regardless of confidence.
        """
        super().__init__()

        self.threshold = threshold
        self.min_age = min_age
        self.opacity_threshold = opacity_threshold

    def forward(self, gaussians: GaussianInstances) -> GaussianInstances:
        """Prune gaussians with confidence below threshold (respecting min_age and opacity)."""
        pruned = {}

        for name, data in gaussians.streams.items():
            # Assume batch size of 1, squeeze batch dimension
            confidence = data.confidence.squeeze(0)  # [n]
            age = data.age.squeeze(0)  # [n]
            opacity = data.opacities.squeeze(0)  # [n]

            # Keep if confidence >= threshold OR age < min_age OR opacity >= opacity_threshold
            valid_mask = (confidence >= self.threshold) | (age < self.min_age)
            if self.opacity_threshold is not None:
                valid_mask = valid_mask | (opacity >= self.opacity_threshold)

            # Apply mask to all fields and add batch dimension back
            pruned[name] = GaussianStreamInstances(
                query=data.query.squeeze(0)[valid_mask].unsqueeze(0),
                query_coords=data.query_coords.squeeze(0)[valid_mask].unsqueeze(0),
                logits=data.logits.squeeze(0)[valid_mask].unsqueeze(0),
                centers=data.centers.squeeze(0)[valid_mask].unsqueeze(0),
                scales=data.scales.squeeze(0)[valid_mask].unsqueeze(0),
                rotations=data.rotations.squeeze(0)[valid_mask].unsqueeze(0),
                opacities=data.opacities.squeeze(0)[valid_mask].unsqueeze(0),
                age=data.age.squeeze(0)[valid_mask].unsqueeze(0),
                confidence=data.confidence.squeeze(0)[valid_mask].unsqueeze(0),
                instance_ids=data.instance_ids.squeeze(0)[valid_mask].unsqueeze(0),
            )

        return GaussianInstances(streams=pruned)


def build(conf: OmegaConf, **kwargs: Any) -> GaussianPruning:
    """Build a pruning operation from config."""
    return registry.from_config(conf, **kwargs)


def build_sequence(
    confs: list[OmegaConf] | OmegaConf, **kwargs: Any
) -> list[GaussianPruning]:
    """
    Build a sequence of pruning operations from a list of configs.

    Args:
        confs: List of pruning configs or a single config
        **kwargs: Additional arguments passed to each pruning operation

    Returns:
        ModuleList of pruning operations to be applied sequentially
    """
    if isinstance(confs, (list, tuple)) or OmegaConf.is_list(confs):
        return nn.ModuleList([build(conf, **kwargs) for conf in confs])

    # Single config - wrap in list
    return nn.ModuleList([build(confs, **kwargs)])
