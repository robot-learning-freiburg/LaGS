# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Stateful transformer operations that can modify query state and features.

This module extends the base Operation class to support operations that:
1. Transform query features (like regular ops)
2. Update query state (positions, position embeddings, masks, etc.)
3. Update shared features dict (for cross-stream visibility)
4. Enable position refinement, downsampling, cross-stream communication

Usage:
    class MyStatefulOp(StatefulOperation):
        def forward(self, query, query_pos=None, query_coords=None, **kwargs):
            # Process features
            new_query = self.attn(query, query_pos=query_pos, ...)

            # Update inputs (stream state)
            new_inputs = {
                'query': new_query,
                'query_pos': query_pos,
                'query_coords': new_coords,
                **kwargs,
            }

            # Update features (for cross-stream access)
            new_features = {
                'feats': new_query,
                'feats_pos': query_pos,
                'feats_coords': new_coords,
            }

            return new_inputs, new_features

Standard input keys:
    - 'query': Query features [b, n, c] (required)
    - 'query_pos': Position embeddings [b, n, c]
    - 'query_coords': Coordinates [b, n, 3]
    - Any other custom keys

Standard feature keys:
    - 'feats': Features for cross-stream access
    - 'feats_pos': Position embeddings for cross-stream access
    - 'feats_coords': Coordinates for cross-stream access
"""

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ..head import LearnedSinePosEnc3d
from .base import AttentionLayer, StatefulOperation, attention_ops


@attention_ops.register(key="UpdatePositionEmbeddings")
class UpdatePositionEmbeddingsOp(StatefulOperation):
    """
    Recompute position embeddings from current coordinates.

    Use this after operations that modify query_coords (like RefinePositions)
    to update the position embeddings to match the new coordinates.

    Config example:
        - type: UpdatePositionEmbeddings
          pos_encoder_cfg:
            num_feats: 128
    """

    def __init__(
        self,
        embed_dim: int,
        num_feats: int = 128,
        args_group: str | Sequence[str] | None = None,
    ):
        super().__init__(
            args_group=(
                args_group if args_group is not None else "UpdatePositionEmbeddings"
            )
        )

        # Position encoder
        self.pos_encoder = LearnedSinePosEnc3d(
            embed_dim=embed_dim,
            num_feats=num_feats,
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        query_coords: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Recompute position embeddings from coordinates.
        """
        assert (
            query_coords is not None
        ), "UpdatePositionEmbeddings requires query_coords"

        # Recompute position embeddings from current coordinates
        new_query_pos = self.pos_encoder(query_coords)

        # Return updated inputs (only embeddings changed)
        new_inputs = {
            "query": query,
            "query_pos": new_query_pos,
            "query_coords": query_coords,
            **kwargs,
        }

        return new_inputs, features


@attention_ops.register(key="RefinePositions")
class RefinePositionsOp(StatefulOperation):
    """
    Refine gaussian positions based on learned features.

    This operation predicts small position deltas to adaptively move
    gaussians toward better locations based on attention features.
    Does NOT update position embeddings - use UpdatePositionEmbeddings after this.

    Supports arbitrary coordinate frames with per-dimension or uniform scaling.

    Config examples:
        # Uniform scaling with [0, 1] range (default)
        - type: RefinePositions
          max_delta: 0.1
          coord_range: [0.0, 1.0]

        # Per-dimension scaling with custom range
        - type: RefinePositions
          max_delta: [2.0, 4.0, 1.0]  # x, y, z deltas
          coord_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
            # [x_min, y_min, z_min, x_max, y_max, z_max]

        # Uniform scaling with custom range
        - type: RefinePositions
          max_delta: 0.05
          coord_range: [-50.0, -50.0, -5.0, 50.0, 50.0, 5.0]
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        max_delta: float | Sequence[float] = 0.1,
        coord_range: Sequence[float] = (0.0, 1.0),
        args_group: str | Sequence[str] | None = None,
    ):
        """
        Args:
            embed_dim: Feature dimension
            hidden_dim: Hidden dimension for delta predictor (default: embed_dim // 2)
            num_layers: Number of layers in delta predictor
            max_delta: Maximum position delta. Can be:
                - float: uniform across all dimensions
                - [dx, dy, dz]: per-dimension deltas
            coord_range: Coordinate range. Can be:
                - [min, max]: uniform range for all dimensions
                - [x_min, y_min, z_min, x_max, y_max, z_max]: per-dimension ranges
        """
        super().__init__(
            args_group=args_group if args_group is not None else "RefinePositions"
        )

        max_delta = torch.as_tensor(max_delta, dtype=torch.float32).expand(3)
        self.register_buffer("max_delta", max_delta, persistent=False)

        coord_range = torch.as_tensor(coord_range, dtype=torch.float32)
        if coord_range.shape == (2,):
            coord_range = torch.tensor([0, 1]).view(2, 1).expand(2, 3).reshape(-1)

        self.register_buffer("coord_range", coord_range, persistent=False)

        assert max_delta.shape == (3,)
        assert coord_range.shape == (6,)

        if hidden_dim is None:
            hidden_dim = embed_dim // 2

        # Delta predictor
        layers = [
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ]

        self.delta_predictor = nn.Sequential(
            *layers,
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),  # bounded to [-1, 1]
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        query_coords: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Predict position deltas and update coordinates.
        Position embeddings are NOT updated - use UpdatePositionEmbeddings for that.
        """
        assert query_coords is not None, "RefinePositions requires query_coords"

        # Predict position deltas
        delta = self.delta_predictor(query)  # [b, n, 3] in [-1, 1]
        delta = delta * self.max_delta  # scale by max_delta per dimension

        # Update positions (clamp to valid range)
        new_coords = query_coords + delta
        new_coords = torch.clamp(
            new_coords,
            self.coord_range[:3],  # min values
            self.coord_range[3:],  # max values
        )

        # Return updated inputs (coords changed, embeddings unchanged)
        new_inputs = {
            "query": query,
            "query_pos": query_pos,  # unchanged
            "query_coords": new_coords,
            **kwargs,
        }

        return new_inputs, features


@attention_ops.register(key="UpdateQueryImageReferencePoints")
class UpdateQueryImageReferencePointsOp(StatefulOperation):
    """
    Recompute image reference points from current coordinates.

    Use this after operations that modify query_coords (like RefinePositions)
    to update the image reference points to match the new coordinates.
    """

    def __init__(
        self,
        embed_dim: int,  # pylint: disable=unused-argument
        features: str | Sequence[str],
        args_group: str | Sequence[str] | None = None,
    ):
        super().__init__(
            args_group=(
                args_group
                if args_group is not None
                else "UpdateQueryImageReferencePoints"
            )
        )

        if isinstance(features, str):
            features = [features]
        self.features = features

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        query_coords: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        for key in self.features:
            feats = features.get(key, None)
            if feats is None:
                continue

            update_fn = feats["reference_points_fn"]
            ref, ref_mask = update_fn(query_coords)

            feats["reference_points"] = ref
            feats["reference_points_mask"] = ref_mask

        inputs = {
            "query": query,
            "query_pos": query_pos,
            "query_coords": query_coords,
            **kwargs,
        }

        return inputs, features


@attention_ops.register(key="CrossStreamAggregation")
class CrossStreamAggregationOp(StatefulOperation):
    """
    Aggregate information from a finer stream into a coarser stream.

    Each coarse query (superpoint) aggregates features from nearby
    fine queries within a spatial radius.

    Config example:
        - type: CrossStreamAggregation
          context_stream: fine  # name of stream to aggregate from
          radius: 0.1
          num_heads: 8
          aggregation_type: attention  # or 'pool', 'max'
    """

    def __init__(
        self,
        embed_dim: int,
        context_stream: str,  # name of context stream
        radius: float = 0.1,
        num_heads: int = 8,
        aggregation_type: str = "attention",  # 'attention', 'pool', 'max'
        args_group: str | Sequence[str] | None = None,
    ):
        super().__init__(
            args_group=(
                args_group if args_group is not None else "CrossStreamAggregation"
            )
        )

        self.context_stream = context_stream
        self.radius = radius
        self.aggregation_type = aggregation_type

        if aggregation_type == "attention":
            self.cross_attn = AttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=0.0,
            )

        elif aggregation_type == "pool":
            self.spatial_net = nn.Sequential(
                nn.Linear(3, embed_dim // 4),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim // 4, 1),
            )
            self.feature_fusion = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(inplace=True),
            )

    def forward(
        self,
        query: torch.Tensor,  # coarse features [b, n_coarse, c]
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        query_coords: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Aggregate fine stream features into coarse stream.

        Expects features[context_stream] to contain:
            - 'feats': fine features [b, n_fine, c]
            - 'feats_coords': fine coords [b, n_fine, 3]
            - 'feats_pos': fine position embeddings [b, n_fine, c]
        """
        # pylint: disable=too-many-locals

        assert features is not None
        assert self.context_stream in features
        assert query_coords is not None, "CrossStreamAggregation requires query_coords"

        context = features[self.context_stream]
        context_features = context["feats"]
        context_coords = context["feats_coords"]
        context_pos = context.get("feats_pos")

        if self.aggregation_type == "attention":
            # Full cross-attention with spatial bias
            spatial_bias = self._compute_spatial_bias(query_coords, context_coords)

            updated_query = self.cross_attn(
                query=query,
                feats=context_features,
                query_pos=query_pos,
                feats_pos=context_pos,
                attn_mask=spatial_bias,
            )

        elif self.aggregation_type == "pool":
            # Compute pairwise distances [b, n_coarse, n_fine]
            dists = torch.cdist(query_coords, context_coords)

            # Create spatial mask (only pool from nearby points)
            spatial_mask = dists < self.radius  # [b, n_coarse, n_fine]

            # Compute spatial weights
            rel_pos = context_coords.unsqueeze(1) - query_coords.unsqueeze(2)
            spatial_weights = self.spatial_net(rel_pos).squeeze(
                -1
            )  # [b, n_coarse, n_fine]

            # Apply mask and softmax
            spatial_weights = spatial_weights.masked_fill(~spatial_mask, float("-inf"))
            spatial_weights = torch.softmax(spatial_weights, dim=-1)

            # Weighted aggregation
            aggregated = torch.bmm(
                spatial_weights, context_features
            )  # [b, n_coarse, c]

            # Fuse with original features
            updated_query = self.feature_fusion(torch.cat([query, aggregated], dim=-1))

        else:  # 'max'
            # Max pooling over spatial neighborhood
            dists = torch.cdist(query_coords, context_coords)
            spatial_mask = dists < self.radius

            aggregated = []
            for i in range(query_coords.shape[1]):
                mask_i = spatial_mask[:, i, :]  # [b, n_fine]
                masked_feats = context_features * mask_i.unsqueeze(-1)
                max_pooled = masked_feats.max(dim=1, keepdim=True)[0]
                aggregated.append(max_pooled)

            aggregated = torch.cat(aggregated, dim=1)  # [b, n_coarse, c]
            updated_query = query + aggregated

        # Return updated inputs
        new_inputs = {
            "query": updated_query,
            "query_pos": query_pos,
            "query_coords": query_coords,
            **kwargs,
        }

        return new_inputs, features

    def _compute_spatial_bias(self, query_coords, context_coords):
        """Compute spatial bias for attention (closer points get higher scores)."""
        dists = torch.cdist(query_coords, context_coords)  # [b, n_q, n_c]
        # Gaussian kernel
        spatial_bias = -dists.pow(2) / (2 * self.radius**2)
        return spatial_bias


@attention_ops.register(key="CrossStreamPropagation")
class CrossStreamPropagationOp(StatefulOperation):
    """
    Propagate information from a coarser stream to a finer stream.

    Each fine query attends to relevant coarse superpoints for
    global context.

    Config example:
        - type: CrossStreamPropagation
          context_stream: coarse  # name of stream to propagate from
          num_heads: 8
          k_nearest: 4
          propagation_type: knn  # or 'broadcast', 'route'
    """

    def __init__(
        self,
        embed_dim: int,
        context_stream: str,  # name of context stream
        num_heads: int = 8,
        k_nearest: int = 4,
        propagation_type: str = "knn",  # 'broadcast', 'knn', 'route'
        args_group: str | Sequence[str] | None = None,
    ):
        super().__init__(
            args_group=(
                args_group if args_group is not None else "CrossStreamPropagation"
            )
        )

        self.context_stream = context_stream
        self.k_nearest = k_nearest
        self.propagation_type = propagation_type

        self.cross_attn = AttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
        )

        if propagation_type == "route":
            self.router = nn.Sequential(
                nn.Linear(embed_dim + 3, embed_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim // 2, 1),
                nn.Sigmoid(),
            )

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        query: torch.Tensor,  # fine features [b, n_fine, c]
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        query_coords: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Propagate coarse stream features to fine stream.

        Expects features[context_stream] to contain:
            - 'feats': coarse features [b, n_coarse, c]
            - 'feats_coords': coarse coords [b, n_coarse, 3]
            - 'feats_pos': coarse position embeddings [b, n_coarse, c]
        """
        # pylint: disable=too-many-locals

        assert features is not None
        assert self.context_stream in features
        assert query_coords is not None, "CrossStreamPropagation requires query_coords"

        context = features[self.context_stream]
        context_features = context["feats"]
        context_coords = context["feats_coords"]
        context_pos = context.get("feats_pos")

        if self.propagation_type == "broadcast":
            # All fine points attend to all coarse points
            propagated = self.cross_attn(
                query=query,
                feats=context_features,
                query_pos=query_pos,
                feats_pos=context_pos,
            )

        elif self.propagation_type == "knn":
            # K-nearest neighbor attention
            _b, n_fine, _c = query.shape
            n_coarse = context_features.shape[1]

            # Find k-nearest coarse points for each fine point
            dists = torch.cdist(query_coords, context_coords)  # [b, n_fine, n_coarse]
            knn_indices = torch.topk(
                dists, self.k_nearest, dim=-1, largest=False
            ).indices

            # Create attention mask
            attn_mask = torch.ones(
                n_fine, n_coarse, device=query_coords.device
            ) * float("-inf")
            for i in range(n_fine):
                attn_mask[i, knn_indices[0, i]] = 0.0

            propagated = self.cross_attn(
                query=query,
                feats=context_features,
                query_pos=query_pos,
                feats_pos=context_pos,
                attn_mask=attn_mask,
            )

        elif self.propagation_type == "route":
            # Learnable routing
            rel_pos = query_coords.unsqueeze(2) - context_coords.unsqueeze(1)
            query_expanded = query.unsqueeze(2).expand(
                -1, -1, context_coords.shape[1], -1
            )

            router_input = torch.cat([query_expanded, rel_pos], dim=-1)
            route_scores = self.router(router_input).squeeze(
                -1
            )  # [b, n_fine, n_coarse]

            # Use as attention mask
            attn_mask = torch.log(route_scores + 1e-8)

            propagated = self.cross_attn(
                query=query,
                feats=context_features,
                query_pos=query_pos,
                feats_pos=context_pos,
                attn_mask=attn_mask,
            )

        else:
            raise ValueError(f"Unknown propagation_type: {self.propagation_type}")

        # Fuse propagated features with original
        updated_query = self.fusion(torch.cat([query, propagated], dim=-1))

        # Return updated inputs
        new_inputs = {
            "query": updated_query,
            "query_pos": query_pos,
            "query_coords": query_coords,
            **kwargs,
        }

        return new_inputs, features
