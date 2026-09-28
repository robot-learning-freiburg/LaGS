# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Hierarchical gaussian aggregation with multi-scale fusion.

Aggregates gaussians from multiple hierarchical streams (coarse/medium/fine)
with appropriate scale parameters, then fuses them FPN-style.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .aggregator import GaussianFeatureAggregator


class HierarchicalGaussianAggregator(nn.Module):
    """
    Aggregates hierarchical gaussians with multi-scale fusion.

    Each stream (coarse/medium/fine) is aggregated with its own scale multiplier
    to match the receptive field of that hierarchy level. Results are then
    fused in a feature-pyramid-network (FPN) style: coarse → medium → fine.

    Uses GaussianFeatureAggregator for each stream to aggregate gaussian features
    into a voxel grid.

    Args:
        voxel_size: Voxel size for aggregation grid
        voxel_range: Voxel coordinate range
        streams: Names of streams (e.g., ['coarse', 'medium', 'fine'])
        scale_multiplier: Scale multipliers per stream
        fusion_mode: How to fuse multi-scale features
            - 'learned': Summation with learned fusion weights (default)
            - 'sum': Simple summation
            - 'concat': Concatenate and project (requires num_channels)
        num_channels: Number of feature channels. Required for concat fusion mode.
    """

    def __init__(
        self,
        voxel_size: list[float],
        voxel_range: list[float],
        streams: list[str],
        scale_multiplier: dict[str, float] | float = 3.0,
        fusion_mode: str = "learned",
        num_channels: int | None = None,
        min_radius: int = 1,
    ):
        super().__init__()

        # Ensure streams are in a consistent order
        streams = sorted(streams)

        self.streams = streams
        self.num_streams = len(streams)
        self.fusion_mode = fusion_mode

        if isinstance(scale_multiplier, (int, float)):
            scale_multiplier = {n: scale_multiplier for n in streams}

        # Create aggregator per stream with appropriate scale
        self.aggregators = nn.ModuleDict(
            {
                name: GaussianFeatureAggregator(
                    voxel_size=voxel_size,
                    voxel_range=voxel_range,
                    scale_multiplier=scale_multiplier[name],
                    min_radius=min_radius,
                )
                for name in streams
            }
        )

        # Fusion module
        if fusion_mode == "learned":
            # Learnable weights per stream
            self.fusion_weights = nn.Parameter(torch.ones(self.num_streams))
        elif fusion_mode == "concat":
            # Concatenate and project
            if num_channels is None:
                raise ValueError(
                    "num_channels must be specified for concat fusion mode"
                )

            # Projections to map concatenated features back to original dimensions
            # features: [num_streams * num_channels] -> [num_channels]
            self.concat_proj_features = nn.Linear(
                self.num_streams * num_channels, num_channels
            )
        elif fusion_mode != "sum":
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    def forward(
        self,
        gaussians_per_stream: dict[str, dict[str, torch.Tensor]],
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aggregate hierarchical gaussians and fuse multi-scale.

        Args:
            gaussians_per_stream: Dict mapping stream names to gaussian parameters:
                {
                    'coarse': {
                        'features': [b, n_queries, n_channels],
                        'centers': [b, n_queries, 3],
                        'scales': [b, n_queries, 3],
                        'rotations': [b, n_queries, 4],
                        'opacities': [b, n_queries],
                    },
                    'medium': {...},
                    'fine': {...},
                }
            mask: Optional mask for valid voxels [b, z, y, x]

        Returns:
            features: Fused features [b, n_voxels, n_channels]
            bin_scores: Fused occupancy scores [b, n_voxels, 1]
            density: Fused density [b, n_voxels]
            prob_sum: Fused opacity-weighted density [b, n_voxels]
        """
        # Aggregate each stream
        outputs = {}
        for stream in self.streams:
            gaussians = gaussians_per_stream[stream]

            features, bin_scores, density, prob_sum = self.aggregators[stream](
                features=gaussians["features"],
                centers=gaussians["centers"],
                scales=gaussians["scales"],
                rotations=gaussians["rotations"],
                opacities=gaussians["opacities"],
                mask=mask,
            )

            outputs[stream] = {
                "features": features,
                "bin_scores": bin_scores,
                "density": density,
                "prob_sum": prob_sum,
            }

        # Fuse multi-scale
        if self.fusion_mode == "learned":
            # Weighted sum with learned weights
            weights = F.softmax(self.fusion_weights, dim=0)

            features = sum(
                w * outputs[name]["features"] for w, name in zip(weights, self.streams)
            )
            bin_scores = sum(
                w * outputs[name]["bin_scores"]
                for w, name in zip(weights, self.streams)
            )
            density = sum(
                w * outputs[name]["density"] for w, name in zip(weights, self.streams)
            )
            prob_sum = sum(
                w * outputs[name]["prob_sum"] for w, name in zip(weights, self.streams)
            )

        elif self.fusion_mode == "sum":
            # Simple summation
            features = sum(outputs[name]["features"] for name in self.streams)

            # Average bin scores, density, and prob_sum over streams
            bin_scores = sum(outputs[name]["bin_scores"] for name in self.streams)
            bin_scores = bin_scores / self.num_streams

            density = sum(outputs[name]["density"] for name in self.streams)
            density = density / self.num_streams

            prob_sum = sum(outputs[name]["prob_sum"] for name in self.streams)
            prob_sum = prob_sum / self.num_streams

        elif self.fusion_mode == "concat":
            # Concatenate and project
            # Concatenate features along channel dimension
            features = [outputs[name]["features"] for name in self.streams]
            features = torch.cat(features, dim=-1)
            features = self.concat_proj_features(features)

            # Average bin scores, density, and prob_sum over streams
            bin_scores = sum(outputs[name]["bin_scores"] for name in self.streams)
            bin_scores = bin_scores / self.num_streams

            density = sum(outputs[name]["density"] for name in self.streams)
            density = density / self.num_streams

            prob_sum = sum(outputs[name]["prob_sum"] for name in self.streams)
            prob_sum = prob_sum / self.num_streams

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        return features, bin_scores, density, prob_sum
