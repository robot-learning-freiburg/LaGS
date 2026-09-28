# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ....utils.types import MetaDict
from ..occupancy.gaussian.utils import quaternion_to_rotation_matrix
from ..utils import build_class_list


class TemporalGaussianConsistencyLoss(nn.Module):
    """
    Enforces that temporal gaussian queries decode to their
    EMC-transformed targets from the previous frame.

    The loss encourages temporal consistency by supervising:
        decode(query_t) ≈ EMC(gaussian_{t-1})

    This prevents the model from learning arbitrary mappings and
    ensures query features maintain meaningful geometric relationships.

    When dynamic_labels is provided, applies weighting to only
    penalize static classes, since dynamic objects may have moved between
    frames and the EMC transformation only compensates for ego-motion.

    Two static weighting modes are available:
    - "soft": Weight by (1 - P(dynamic)) using softmax probabilities
    - "hard": Binary mask where argmax class determines static/dynamic
    """

    # pylint: disable=too-many-instance-attributes

    # buffers:
    dynamic_class_indices: torch.Tensor | None
    is_dynamic_class: torch.Tensor | None

    def __init__(
        self,
        center_weight: float = 1.0,
        scale_weight: float = 1.0,
        rotation_weight: float = 1.0,
        opacity_weight: float = 1.0,
        logits_weight: float = 0.0,
        gaussian_labels: Sequence[str] | None = None,
        dynamic_labels: Sequence[str] | None = None,
        static_weighting_mode: str = "soft",
        static_class_weight: float = 1.0,
        dynamic_class_weight: float = 1.0,
        visibility_boost: float = 0.0,
        confidence_boost: float = 0.0,
    ):
        """
        Args:
            center_weight: Weight for center (position) loss component.
            scale_weight: Weight for scale loss component.
            rotation_weight: Weight for rotation loss component.
            opacity_weight: Weight for opacity loss component.
            logits_weight: Weight for logits (classification) loss component.
            gaussian_labels: Ordered list of gaussian class names.
            dynamic_labels: Subset of gaussian_labels that are dynamic
                (e.g., vehicles, pedestrians). When provided along with
                gaussian_labels, the loss is weighted to focus on static classes.
            static_weighting_mode: How to weight static vs dynamic classes:
                - "soft": Weight by (1 - P(dynamic)) using softmax probabilities.
                    Provides smooth gradients but dynamic objects still contribute.
                - "hard": Binary mask based on argmax class. Gaussians classified
                    as dynamic (argmax in dynamic_labels) get zero weight.
                    Sharper boundary but no gradient through class decision.
                Default: "soft" for backward compatibility.
            static_class_weight: Weight multiplier for static class gaussians.
                Default 1.0.
            dynamic_class_weight: Weight multiplier for dynamic class gaussians
                (only applied when matched to GT objects).
            visibility_boost: Boost factor for non-visible gaussians. When > 0,
                the loss weight is multiplied by (1 + visibility_boost * (1 - visibility))
                for static gaussians. This encourages stronger consistency for
                occluded/out-of-view gaussians. Default 0.0 (no visibility weighting).
            confidence_boost: Boost factor for high-confidence gaussians. When > 0,
                the loss weight is multiplied by (1 + confidence_boost * confidence).
                High confidence gaussians (well-observed over time) get stronger
                consistency enforcement. Default 0.0 (no confidence weighting).
        """
        # pylint: disable=too-many-locals
        super().__init__()

        self.center_weight = center_weight
        self.scale_weight = scale_weight
        self.rotation_weight = rotation_weight
        self.opacity_weight = opacity_weight
        self.logits_weight = logits_weight
        self.static_class_weight = static_class_weight
        self.dynamic_class_weight = dynamic_class_weight
        self.visibility_boost = visibility_boost
        self.confidence_boost = confidence_boost

        if static_weighting_mode not in ("soft", "hard"):
            raise ValueError(
                f"static_weighting_mode must be 'soft' or 'hard', got '{static_weighting_mode}'"
            )
        self.static_weighting_mode = static_weighting_mode

        if gaussian_labels is not None and dynamic_labels is not None:
            dynamic_class_indices = build_class_list(gaussian_labels, dynamic_labels)
            self.register_buffer(
                "dynamic_class_indices", dynamic_class_indices, persistent=False
            )

            # For hard mode: precompute a boolean mask for dynamic classes
            # is_dynamic_class[c] = True if class c is dynamic
            num_classes = len(gaussian_labels)
            is_dynamic_class = torch.zeros(num_classes, dtype=torch.bool)
            is_dynamic_class[dynamic_class_indices] = True
            self.register_buffer("is_dynamic_class", is_dynamic_class, persistent=False)
        else:
            self.dynamic_class_indices = None
            self.is_dynamic_class = None

    def _compute_static_weights_soft(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute soft weights for static classes based on class probabilities.

        Args:
            logits: Class logits [b, n_temporal, num_classes]

        Returns:
            Static weights [b, n_temporal] where weight = 1 - P(dynamic)
        """
        # Convert logits to probabilities
        probs = torch.softmax(logits.detach(), dim=-1)  # [b, n_temporal, num_classes]

        # Sum probabilities of dynamic classes
        dynamic_probs = probs[..., self.dynamic_class_indices].sum(dim=-1)

        # Static weight = 1 - P(dynamic)
        static_weights = 1.0 - dynamic_probs  # [b, n_temporal]

        return static_weights

    def _compute_static_weights_hard(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute hard (binary) mask for static classes based on argmax class.

        Args:
            logits: Class logits [b, n_temporal, num_classes]

        Returns:
            Static mask [b, n_temporal] where weight = 0 if argmax is dynamic, 1 otherwise
        """
        # Get predicted class via argmax
        pred_classes = logits.detach().argmax(dim=-1)  # [b, n_temporal]

        # Check if predicted class is dynamic
        # pylint: disable=unsubscriptable-object
        is_dynamic = self.is_dynamic_class[pred_classes]  # [b, n_temporal]

        # Static weight = 1 for static, 0 for dynamic
        static_weights = (~is_dynamic).float()  # [b, n_temporal]

        return static_weights

    def _compute_static_weights(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute static weights based on configured weighting mode.

        Args:
            logits: Class logits [b, n_temporal, num_classes]

        Returns:
            Static weights [b, n_temporal]
        """
        if self.static_weighting_mode == "hard":
            return self._compute_static_weights_hard(logits)
        return self._compute_static_weights_soft(logits)

    def _compute_dynamic_supervision_weights(
        self,
        logits: torch.Tensor,
        matched_instance_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Compute weights for dynamic object supervision.

        Weight = static_class_weight for:
        - Static gaussians (EMC targets are correct)

        Weight = dynamic_class_weight for:
        - Dynamic gaussians matched to objects (object-motion targets are correct)

        Weight = 0 for:
        - Dynamic gaussians not matched to any object (no valid target)

        Args:
            logits: Class logits [b, n_temporal, num_classes]
            matched_instance_ids: Instance IDs per gaussian [b, n_temporal], -1 = unmatched

        Returns:
            Weights [b, n_temporal]
        """
        if matched_instance_ids is None:
            # Fallback to static-only weighting (scaled by static_class_weight)
            return self._compute_static_weights(logits) * self.static_class_weight

        # Get predicted class via argmax
        pred_classes = logits.detach().argmax(dim=-1)  # [b, n_temporal]

        # Check if predicted class is dynamic
        # pylint: disable=unsubscriptable-object
        is_dynamic = self.is_dynamic_class[pred_classes]  # [b, n_temporal]

        # Check if gaussian is matched to an object
        has_match = matched_instance_ids >= 0  # [b, n_temporal]

        # Compute weights:
        # - Static gaussians: static_class_weight
        # - Dynamic gaussians with match: dynamic_class_weight
        # - Dynamic gaussians without match: 0
        is_static = ~is_dynamic
        is_dynamic_matched = is_dynamic & has_match

        weights = (
            is_static.float() * self.static_class_weight
            + is_dynamic_matched.float() * self.dynamic_class_weight
        )

        return weights

    @torch.compile(mode="default", dynamic=True)
    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
        visibility: torch.Tensor,
        confidence: torch.Tensor | None,
        matched_instance_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute temporal consistency loss between predictions and EMC-transformed targets.

        Args:
            preds: Decoded gaussian predictions with shape [b, n_temporal, ...]:
                - centers: [b, n_temporal, 3]
                - scales: [b, n_temporal, 3]
                - rotations: [b, n_temporal, 4]
                - opacities: [b, n_temporal]
                - logits: [b, n_temporal, num_classes] (optional, for static weighting)
            targets: EMC-transformed targets from prev frame (same shapes as preds).
                For static gaussians, these are EMC targets. For dynamic gaussians
                matched to objects, these should be object-motion targets.
            visibility: Per-gaussian visibility scores [b, n_temporal] in [0, 1].
                When visibility_boost > 0, non-visible gaussians receive higher
                consistency weight to preserve their properties.
            confidence: Per-gaussian confidence scores [b, n_temporal] in [0, 1].
                When confidence_boost > 0, high-confidence gaussians receive higher
                consistency weight.
            matched_instance_ids: Instance ID for each gaussian [b, n_temporal].
                -1 means unmatched. When provided, enables dynamic object supervision:
                dynamic gaussians get weight=1 if matched, weight=0 if unmatched.

        Returns:
            Dict of loss components:
                - loss_center: L2 loss on gaussian centers
                - loss_scale: L2 loss on gaussian scales
                - loss_rotation: MSE loss on rotation matrices
                - loss_opacity: L2 loss on opacities
                - loss_logits: KL divergence on class probabilities
        """
        # pylint: disable=too-many-locals
        losses = {}

        # Compute weights based on whether dynamic supervision is enabled
        if self.dynamic_class_indices is not None:
            static_weights = self._compute_dynamic_supervision_weights(
                preds.logits, matched_instance_ids
            )
        else:
            static_weights = torch.ones_like(preds.centers[..., 0])

        # Apply visibility boost: non-visible static gaussians get higher weight
        if self.visibility_boost > 0 and visibility is not None:
            occlusion = 1 - visibility.detach()
            static_weights = static_weights * (1 + self.visibility_boost * occlusion)

        # Apply confidence boost: high-confidence gaussians get higher weight
        if self.confidence_boost > 0 and confidence is not None:
            static_weights = static_weights * (
                1 + self.confidence_boost * confidence.detach()
            )

        # Center loss (L2/MSE)
        if self.center_weight > 0:
            diff = (preds.centers - targets.centers) ** 2  # [b, n, 3]
            loss_center = (diff.mean(dim=-1) * static_weights).mean()
            losses["loss_center"] = self.center_weight * loss_center

        # Scale loss (L2/MSE)
        if self.scale_weight > 0:
            diff = (preds.scales - targets.scales) ** 2  # [b, n, 3]
            loss_scale = (diff.mean(dim=-1) * static_weights).mean()
            losses["loss_scale"] = self.scale_weight * loss_scale

        # Rotation loss (rotation matrix MSE)
        # Convert quaternions to rotation matrices and compare directly.
        # This provides better gradient signal for axis correction compared
        # to quaternion dot product, which under-penalizes axis errors for
        # small rotation angles.
        if self.rotation_weight > 0:
            r_pred = quaternion_to_rotation_matrix(preds.rotations)
            r_target = quaternion_to_rotation_matrix(targets.rotations)
            diff = (r_pred - r_target) ** 2  # [b, n, 3, 3]
            loss_rotation = (diff.mean(dim=(-2, -1)) * static_weights).mean()
            losses["loss_rotation"] = self.rotation_weight * loss_rotation

        # Opacity loss (L2/MSE)
        if self.opacity_weight > 0:
            diff = (preds.opacities - targets.opacities) ** 2  # [b, n]
            loss_opacity = (diff * static_weights).mean()
            losses["loss_opacity"] = self.opacity_weight * loss_opacity

        # Logits loss (KL divergence)
        if self.logits_weight > 0:
            log_probs_pred = F.log_softmax(preds.logits, dim=-1)
            probs_target = F.softmax(targets.logits, dim=-1)
            # KL divergence: sum over classes, then apply static weights
            kl_div = F.kl_div(log_probs_pred, probs_target, reduction="none")
            kl_div = kl_div.sum(dim=-1)  # [b, n]
            loss_logits = (kl_div * static_weights).mean()
            losses["loss_logits"] = self.logits_weight * loss_logits

        return losses
