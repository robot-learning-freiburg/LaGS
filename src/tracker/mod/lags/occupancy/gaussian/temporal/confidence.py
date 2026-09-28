# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Confidence tracking for temporal gaussians.

Confidence represents "how reliably do we know this gaussian's properties" and
accumulates over time based on observation quality.

Multiple confidence tracker implementations are available via the registry:
- visibility: obs_quality = visibility * dist * fov
- opacity: obs_quality = opacity * dist * fov
- visibility-opacity: obs_quality = visibility * opacity * dist * fov
- max-visibility-opacity: obs_quality = max(visibility, opacity) * dist * fov
- learned: predicts confidence from query embeddings using MLP
"""

import abc
from typing import Any, Sequence

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from ......config.registry import Registry

registry = Registry("lags.occupancy.gaussian.temporal.confidence")


class ConfidenceTrackerBase(nn.Module):
    """
    Base class for confidence trackers.

    Confidence is updated based on observation quality:
        confidence = confidence + alpha * obs_quality * (1 - confidence)

    This gives:
    - Confidence grows toward 1 with good observations
    - Growth slows as confidence increases (diminishing returns)
    - No decay when not observed (maintains confidence)

    When dynamic_labels is provided, dynamic-class gaussians receive
    a per-frame multiplicative confidence decay to reflect the fact that
    their properties may become stale (EMC only compensates for ego-motion,
    not object motion).
    """

    # buffers
    is_dynamic_class: torch.Tensor | None

    def __init__(
        self,
        depth_scale: float = 20.0,
        alpha: float = 0.8,
        decay_rate: float = 0.0,
        global_decay_rate: float = 0.0,
        gaussian_labels: Sequence[str] | None = None,
        dynamic_labels: Sequence[str] | None = None,
    ):
        """
        Args:
            depth_scale: Distance at which observation quality is halved (meters).
            alpha: Rate for confidence update.
            decay_rate: Per-frame confidence decay for dynamic-class gaussians.
            global_decay_rate: Per-frame confidence decay applied to ALL gaussians,
                regardless of class. Makes confidence track current observation quality
                rather than only accumulated history: gaussians that stop being observed
                gradually lose confidence and eventually get pruned. A value of 0.05–0.1
                is recommended when using topk_confidence for pruning.
            gaussian_labels: Full list of gaussian class names.
            dynamic_labels: Subset of gaussian_labels that are dynamic
                (e.g., vehicles, pedestrians). Required when decay_rate > 0.
        """
        super().__init__()
        self.depth_scale = depth_scale
        self.alpha = alpha
        self.decay_rate = decay_rate
        self.global_decay_rate = global_decay_rate

        if (
            gaussian_labels is not None
            and dynamic_labels is not None
            and len(dynamic_labels) > 0
        ):
            num_classes = len(gaussian_labels)
            dynamic_class_indices = [gaussian_labels.index(n) for n in dynamic_labels]
            is_dynamic_class = torch.zeros(num_classes, dtype=torch.bool)
            for idx in dynamic_class_indices:
                is_dynamic_class[idx] = True
            self.register_buffer("is_dynamic_class", is_dynamic_class, persistent=False)
        else:
            self.is_dynamic_class = None

    def compute_distance_factor(self, centers: torch.Tensor) -> torch.Tensor:
        """
        Compute distance-based quality factor.

        Args:
            centers: [b, n, 3] gaussian centers in ego frame

        Returns:
            [b, n] distance factors in (0, 1], closer = higher
        """
        depth = torch.norm(centers, dim=-1)  # [b, n]
        return 1.0 / (1.0 + depth / self.depth_scale)

    @abc.abstractmethod
    def compute_observation_quality(
        self,
        centers: torch.Tensor,
        visibility: torch.Tensor | None = None,
        in_fov: torch.Tensor | None = None,
        opacity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute observation quality for each gaussian.

        Args:
            centers: [b, n, 3] gaussian centers in ego frame
            visibility: [b, n] visibility scores in [0, 1]
            in_fov: [b, n] boolean mask for in field-of-view
            opacity: [b, n] opacity values in [0, 1] (optional, depends on tracker type)

        Returns:
            [b, n] observation quality scores in [0, 1]
        """
        raise NotImplementedError

    def _apply_dynamic_decay(
        self,
        confidence: torch.Tensor,
        logits: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Apply per-frame confidence decay for dynamic-class gaussians.

        Args:
            confidence: [b, n] confidence values after EMA update
            logits: [b, n, c] predicted class logits

        Returns:
            [b, n] confidence with decay applied to dynamic-class gaussians
        """
        if self.decay_rate <= 0 or self.is_dynamic_class is None or logits is None:
            return confidence

        pred_class = logits.detach().argmax(dim=-1)  # [b, n]
        is_dynamic = self.is_dynamic_class[pred_class]  # [b, n]
        decay = torch.where(is_dynamic, 1.0 - self.decay_rate, 1.0)
        return confidence * decay

    def forward(
        self,
        confidence: torch.Tensor,
        centers: torch.Tensor | None = None,
        visibility: torch.Tensor | None = None,
        in_fov: torch.Tensor | None = None,
        opacity: torch.Tensor | None = None,
        query: torch.Tensor | None = None,  # pylint: disable=unused-argument
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Update confidence based on current observation.

        Args:
            confidence: [b, n] current confidence values
            centers: [b, n, 3] gaussian centers in ego frame
            visibility: [b, n] visibility scores in [0, 1] (optional)
            in_fov: [b, n] boolean mask for in field-of-view (optional)
            opacity: [b, n] opacity values in [0, 1] (optional)
            query: [b, n, d] query embeddings (optional, used by learned tracker)
            logits: [b, n, c] predicted class logits (optional, for dynamic decay)

        Returns:
            [b, n] updated confidence values
        """
        obs_quality = self.compute_observation_quality(
            centers, visibility, in_fov, opacity
        )
        if self.global_decay_rate > 0:
            confidence = confidence * (1 - self.global_decay_rate)
        confidence = confidence + self.alpha * obs_quality * (1 - confidence)
        return self._apply_dynamic_decay(confidence, logits)

    def predict(self, query: torch.Tensor) -> torch.Tensor:
        """
        Direct prediction for loss computation.

        Only implemented by learned trackers. Heuristic trackers raise an error.

        Args:
            query: [b, n, d] query embeddings

        Returns:
            [b, n] predicted confidence values
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support learned confidence loss. "
            "Remove 'loss' from confidence config or use type: learned"
        )


@registry.register(key="visibility")
class VisibilityConfidenceTracker(ConfidenceTrackerBase):
    """
    Confidence based on visibility only.

    obs_quality = visibility * distance_factor * in_fov
    """

    # pylint: disable=abstract-method

    def compute_observation_quality(
        self,
        centers: torch.Tensor,
        visibility: torch.Tensor | None = None,
        in_fov: torch.Tensor | None = None,
        opacity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist = self.compute_distance_factor(centers)
        return visibility * dist * in_fov.float()


@registry.register(key="opacity")
class OpacityConfidenceTracker(ConfidenceTrackerBase):
    """
    Confidence based on decoded opacity only.

    obs_quality = opacity * distance_factor * in_fov
    """

    # pylint: disable=abstract-method

    def compute_observation_quality(
        self,
        centers: torch.Tensor,
        visibility: torch.Tensor | None = None,
        in_fov: torch.Tensor | None = None,
        opacity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist = self.compute_distance_factor(centers)
        return opacity * dist * in_fov.float()


@registry.register(key="visibility-opacity")
class VisibilityOpacityConfidenceTracker(ConfidenceTrackerBase):
    """
    Confidence based on product of visibility and opacity.

    obs_quality = visibility * opacity * distance_factor * in_fov
    """

    # pylint: disable=abstract-method

    def compute_observation_quality(
        self,
        centers: torch.Tensor,
        visibility: torch.Tensor | None = None,
        in_fov: torch.Tensor | None = None,
        opacity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist = self.compute_distance_factor(centers)
        return visibility * opacity * dist * in_fov.float()


@registry.register(key="max-visibility-opacity")
class MaxVisibilityOpacityConfidenceTracker(ConfidenceTrackerBase):
    """
    Confidence based on max of visibility and opacity.

    obs_quality = max(visibility, opacity) * distance_factor * in_fov
    """

    # pylint: disable=abstract-method

    def compute_observation_quality(
        self,
        centers: torch.Tensor,
        visibility: torch.Tensor | None = None,
        in_fov: torch.Tensor | None = None,
        opacity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dist = self.compute_distance_factor(centers)
        return torch.maximum(visibility, opacity) * dist * in_fov.float()


@registry.register(key="learned")
class LearnedConfidenceTracker(ConfidenceTrackerBase):
    """
    Learned confidence predictor.

    Predicts confidence from query embeddings using a small MLP.
    Supervised with binary correctness (pred_class == gt_class).
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        depth_scale: float = 20.0,
        alpha: float = 0.8,
        decay_rate: float = 0.0,
        global_decay_rate: float = 0.0,
        gaussian_labels: Sequence[str] | None = None,
        dynamic_labels: Sequence[str] | None = None,
    ):
        """
        Args:
            embed_dim: Dimension of query embeddings.
            hidden_dim: Hidden dimension of MLP.
            num_layers: Number of MLP layers.
            dropout: Dropout rate.
            depth_scale: Inherited from base (unused).
            alpha: EMA blending factor for confidence update.
            decay_rate: Per-frame confidence decay for dynamic-class gaussians.
            gaussian_labels: Full list of gaussian class names.
            dynamic_labels: Subset of gaussian_labels that are dynamic.
        """
        super().__init__(
            depth_scale=depth_scale,
            alpha=alpha,
            decay_rate=decay_rate,
            global_decay_rate=global_decay_rate,
            gaussian_labels=gaussian_labels,
            dynamic_labels=dynamic_labels,
        )

        layers = []
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def compute_observation_quality(
        self,
        centers: torch.Tensor | None = None,
        visibility: torch.Tensor | None = None,
        in_fov: torch.Tensor | None = None,
        opacity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Not used by learned tracker."""
        raise NotImplementedError(
            "LearnedConfidenceTracker uses query, not observation quality"
        )

    def forward(
        self,
        confidence: torch.Tensor,
        centers: torch.Tensor | None = None,
        visibility: torch.Tensor | None = None,
        in_fov: torch.Tensor | None = None,
        opacity: torch.Tensor | None = None,
        query: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict confidence from query embeddings."""
        assert query is not None, "LearnedConfidenceTracker requires query argument"
        predicted = torch.sigmoid(self.predict(query))  # [b, n]
        # EMA update: blend previous confidence with new prediction
        if self.global_decay_rate > 0:
            confidence = confidence * (1 - self.global_decay_rate)
        confidence = confidence + self.alpha * predicted * (1 - confidence)
        return self._apply_dynamic_decay(confidence, logits)

    def predict(self, query: torch.Tensor) -> torch.Tensor:
        """Direct prediction (logits) for loss computation."""
        return self.mlp(query).squeeze(-1)  # Return logits, not sigmoid


class LearnedConfidenceLoss(nn.Module):
    """BCE loss for learned confidence prediction."""

    def __init__(self, weight: float = 1.0):
        """
        Args:
            weight: Loss weight multiplier.
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predicted_logits: torch.Tensor,
        is_correct: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCE loss for learned confidence.

        Args:
            predicted_logits: [b, n] predicted confidence logits (pre-sigmoid)
            is_correct: [b, n] binary correctness targets
            valid_mask: [b, n] mask for valid (in-bounds) positions

        Returns:
            Scalar loss value
        """
        if not valid_mask.any():
            return predicted_logits.new_zeros(())

        loss = F.binary_cross_entropy_with_logits(
            predicted_logits[valid_mask],
            is_correct[valid_mask],
            reduction="mean",
        )
        return loss * self.weight


def sample_gt_semantic(
    centers: torch.Tensor,
    gt_semantic: torch.Tensor,
    voxel_range: torch.Tensor | list[float],
) -> torch.Tensor:
    """
    Sample GT semantic class at gaussian center positions.

    Args:
        centers: [b, n, 3] gaussian centers in world coordinates
        gt_semantic: [b, d, h, w] GT semantic volume
        voxel_range: [6] tensor or list [x_min, y_min, z_min, x_max, y_max, z_max]

    Returns:
        [b, n] sampled class IDs (255 for out-of-bounds)
    """
    b, n, _ = centers.shape

    # Convert voxel_range to tensor if needed
    if not isinstance(voxel_range, torch.Tensor):
        voxel_range = torch.tensor(
            voxel_range, device=centers.device, dtype=centers.dtype
        )

    voxel_min = voxel_range[:3]
    voxel_max = voxel_range[3:]

    # Convert world coords to grid coords [-1, 1]
    grid_coords = (centers - voxel_min) / (voxel_max - voxel_min) * 2 - 1

    # Detect out-of-bounds positions
    out_of_bounds = (torch.abs(grid_coords) > 1.0).any(dim=-1)  # [b, n]

    # grid_sample expects [b, c, d, h, w] input and [b, d_out, h_out, w_out, 3] grid
    # We have [b, n] points, so reshape to [b, 1, 1, n, 3]
    gt_5d = gt_semantic.unsqueeze(1).float()  # [b, 1, d, h, w]
    grid = grid_coords.view(b, 1, 1, n, 3)  # [b, 1, 1, n, 3]

    sampled = F.grid_sample(
        gt_5d,
        grid,
        mode="nearest",
        padding_mode="border",
        align_corners=False,
    )  # [b, 1, 1, 1, n]
    sampled = sampled.view(b, n).long()

    # Mark out-of-bounds as ignore index (255)
    return torch.where(out_of_bounds, torch.full_like(sampled, 255), sampled)


def compute_correctness_targets(
    logits: torch.Tensor,
    centers: torch.Tensor,
    gt_semantic: torch.Tensor,
    voxel_range: torch.Tensor | list[float],
    class_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute correctness targets for learned confidence.

    Args:
        logits: [b, n, c] predicted semantic logits (in gaussian class space)
        centers: [b, n, 3] gaussian centers in world coordinates
        gt_semantic: [b, d, h, w] GT semantic volume (in occupancy class space)
        voxel_range: [6] tensor or list [x_min, y_min, z_min, x_max, y_max, z_max]
        class_map: [num_occupancy_classes] tensor mapping occupancy classes to
            gaussian classes. If None, assumes 1:1 mapping.

    Returns:
        is_correct: [b, n] binary correctness (1 if pred matches GT)
        valid_mask: [b, n] mask for valid positions (in-bounds)
    """
    pred_class = logits.argmax(dim=-1)  # [b, n] in gaussian class space

    # Sample GT in occupancy class space
    gt_class_occ = sample_gt_semantic(centers, gt_semantic, voxel_range)  # [b, n]

    # Check for out-of-bounds (marked as 255)
    in_bounds = gt_class_occ != 255

    # Map to gaussian class space if class_map provided
    if class_map is not None:
        # Clamp to valid range before indexing (out-of-bounds already handled)
        gt_class_clamped = gt_class_occ.clamp(0, class_map.shape[0] - 1)
        gt_class = class_map[gt_class_clamped]  # [b, n] in gaussian class space
        # Note: Classes not in gaussian subset are mapped to -1, i.e., we treat
        # them as unmodeled space and discourage gaussians there.
    else:
        gt_class = gt_class_occ

    # Compute correctness
    is_correct = (pred_class == gt_class).float()

    return is_correct, in_bounds


def build(conf: OmegaConf, **kwargs: Any) -> ConfidenceTrackerBase:
    """Build a confidence tracker from config."""
    return registry.from_config(conf, **kwargs)
