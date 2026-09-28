# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Visibility-based gating for temporal gaussian updates.

Controls how much the transformer can modify temporal gaussian queries based on
their current visibility. Low visibility gaussians are preserved (gate → 0),
while high visibility gaussians can be updated (gate → 1).
"""

from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from ......config.registry import Registry

registry = Registry("lags.occupancy.gaussian.visibility_gate")


class VisibilityGate(nn.Module):
    """Base class for visibility-based gating modules."""

    def forward(
        self,
        pre_transformer: torch.Tensor,
        post_transformer: torch.Tensor,
        visibility: torch.Tensor,
        in_fov: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply visibility-based gating to transformer outputs.

        Args:
            pre_transformer: Query features before transformer [b, l, n, c]
            post_transformer: Query features after transformer [b, l, n, c]
            visibility: Per-query visibility scores [b, n] in [0, 1]
            in_fov: Per-query in-FoV mask [b, n] bool
            confidence: Per-query confidence scores [b, n] in [0, 1], or None

        Returns:
            gated_output: Gated query features [b, l, n, c]
            gate_values: Gate values for logging [b, n]
        """
        raise NotImplementedError()


@registry.register(key="simple")
class SimpleVisibilityGate(VisibilityGate):
    """
    Simple gate: directly use visibility as gate value.

    gate = clamp(visibility, min_gate, max_gate)
    output = gate * post_transformer + (1 - gate) * pre_transformer
    """

    def __init__(
        self,
        min_gate: float = 0.0,
        max_gate: float = 1.0,
        confidence_damping: float = 0.0,
    ):
        """
        Args:
            min_gate: Minimum gate value (always allow some update)
            max_gate: Maximum gate value
            confidence_damping: How much confidence dampens the gate (0 = no effect, 1 = full)
        """
        super().__init__()
        self.min_gate = min_gate
        self.max_gate = max_gate
        self.confidence_damping = confidence_damping

    def forward(
        self,
        pre_transformer: torch.Tensor,
        post_transformer: torch.Tensor,
        visibility: torch.Tensor,
        in_fov: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Gate = visibility, clamped to [min, max]
        gate = visibility.clamp(self.min_gate, self.max_gate)

        # Dampen gate based on confidence (high confidence → preserve)
        if confidence is not None and self.confidence_damping > 0:
            gate = gate * (1 - self.confidence_damping * confidence)

        # For out-of-FoV queries, use minimum gate (preserve)
        gate = torch.where(in_fov, gate, torch.full_like(gate, self.min_gate))

        # Apply gate: [b, n] -> [b, 1, n, 1] for broadcasting
        gate_expanded = gate.unsqueeze(1).unsqueeze(-1)

        output = (
            gate_expanded * post_transformer + (1 - gate_expanded) * pre_transformer
        )

        return output, gate


@registry.register(key="learned_hybrid")
class LearnedHybridVisibilityGate(VisibilityGate):
    """
    Learned hybrid gate: learnable scale and bias on visibility.

    gate = sigmoid(scale * visibility + bias + in_fov_bias * in_fov)
    output = gate * post_transformer + (1 - gate) * pre_transformer

    This allows the model to calibrate how visibility maps to update strength.
    """

    def __init__(
        self,
        min_gate: float = 0.0,
        max_gate: float = 1.0,
        init_scale: float = 4.0,
        init_bias: float = -2.0,
        init_confidence_scale: float | None = None,
    ):
        """
        Args:
            min_gate: Minimum gate value after sigmoid
            max_gate: Maximum gate value after sigmoid
            init_scale: Initial scale for visibility (higher = sharper transition)
            init_bias: Initial bias (negative = default toward preservation)
            init_confidence_scale: Initial scale for confidence
                (negative = pushes toward preservation)
        """
        super().__init__()
        self.min_gate = min_gate
        self.max_gate = max_gate

        # Learnable parameters
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.bias = nn.Parameter(torch.tensor(init_bias))
        self.in_fov_bias = nn.Parameter(torch.tensor(0.0))

        if init_confidence_scale is not None:
            self.confidence_scale = nn.Parameter(torch.tensor(init_confidence_scale))
        else:
            self.confidence_scale = None

    def forward(
        self,
        pre_transformer: torch.Tensor,
        post_transformer: torch.Tensor,
        visibility: torch.Tensor,
        in_fov: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute gate logits
        gate_logits = self.scale * visibility + self.bias
        gate_logits = gate_logits + self.in_fov_bias * in_fov.float()

        # Confidence pushes toward preservation (negative scale → lower gate)
        if self.confidence_scale is not None and confidence is not None:
            gate_logits = gate_logits + self.confidence_scale * confidence

        # Sigmoid and rescale to [min_gate, max_gate]
        gate = torch.sigmoid(gate_logits)
        gate = self.min_gate + (self.max_gate - self.min_gate) * gate

        # Apply gate: [b, n] -> [b, 1, n, 1] for broadcasting
        gate_expanded = gate.unsqueeze(1).unsqueeze(-1)

        output = (
            gate_expanded * post_transformer + (1 - gate_expanded) * pre_transformer
        )

        return output, gate


@registry.register(key="learned")
class LearnedVisibilityGate(VisibilityGate):
    """
    Fully learned gate: MLP on query features + visibility + in_fov.

    This is the most expressive but also most expensive option.
    The gate can learn complex relationships between query content and visibility.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 64,
        min_gate: float = 0.0,
        max_gate: float = 1.0,
        use_confidence: bool = False,
    ):
        """
        Args:
            embed_dim: Dimension of query features
            hidden_dim: Hidden dimension for gate MLP
            min_gate: Minimum gate value
            max_gate: Maximum gate value
            use_confidence: Whether to include confidence in the gate input
        """
        super().__init__()
        self.min_gate = min_gate
        self.max_gate = max_gate
        self.use_confidence = use_confidence

        # Gate MLP: [query_diff, visibility, in_fov, optional confidence] -> gate
        # Using query difference to capture "how much did transformer change things"
        # diff + visibility + in_fov + confidence (optional)
        input_dim = embed_dim + 2 + int(bool(self.use_confidence))

        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Initialize final layer to produce ~0.5 gate initially
        nn.init.zeros_(self.gate_mlp[-1].bias)
        nn.init.normal_(self.gate_mlp[-1].weight, std=0.01)

    def forward(
        self,
        pre_transformer: torch.Tensor,
        post_transformer: torch.Tensor,
        visibility: torch.Tensor,
        in_fov: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=too-many-locals

        # Use last layer features for gate computation
        pre_last = pre_transformer[:, -1]  # [b, n, c]
        post_last = post_transformer[:, -1]  # [b, n, c]

        # Compute difference (normalized)
        diff = post_last - pre_last
        diff_norm = diff / (diff.norm(dim=-1, keepdim=True) + 1e-6)

        # Concatenate inputs
        inputs = [diff_norm, visibility.unsqueeze(-1), in_fov.float().unsqueeze(-1)]
        if self.use_confidence:
            inputs += [confidence.unsqueeze(-1)]

        gate_input = torch.cat(inputs, dim=-1)  # [b, n, c+3]

        # Compute gate
        gate_logits = self.gate_mlp(gate_input).squeeze(-1)  # [b, n]

        # Sigmoid and rescale
        gate = torch.sigmoid(gate_logits)
        gate = self.min_gate + (self.max_gate - self.min_gate) * gate

        # Apply gate: [b, n] -> [b, 1, n, 1] for broadcasting
        gate_expanded = gate.unsqueeze(1).unsqueeze(-1)

        output = (
            gate_expanded * post_transformer + (1 - gate_expanded) * pre_transformer
        )

        return output, gate


def build(conf: OmegaConf | None, **kwargs: Any) -> VisibilityGate | None:
    """Build a visibility gate module from config."""
    if conf is None:
        return None

    return registry.from_config(conf, **kwargs)
