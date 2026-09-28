# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping

import einops
import torch
from omegaconf import OmegaConf
from torch import nn

from .base import TransformerLayerSequence
from .multistream import MultiStreamTransformerLayerSequence


class UnifiedTrackingTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        operations: list[OmegaConf | Mapping[str, Any]],
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.decoder = TransformerLayerSequence(
            num_layers=num_layers,
            embed_dim=embed_dim,
            operations=operations,
            return_intermediate=True,
            use_checkpointing=use_checkpointing,
        )

        # initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if hasattr(m, "weight") and m.weight.ndim > 1:
                nn.init.xavier_uniform_(m.weight)

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        instance_query: torch.Tensor,
        instance_query_embs: torch.Tensor,
        instance_reference_points: torch.Tensor,
        semantic_query: torch.Tensor,
        semantic_query_embs: torch.Tensor,
        semantic_reference_points: torch.Tensor,
        image_features: torch.Tensor,
        image_features_mask: torch.Tensor,
        image_features_embs: torch.Tensor,
        voxel_features: tuple[torch.Tensor],
        keypoint_features: torch.Tensor | None,
        keypoint_embs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=too-many-locals
        b, _n, _c, _h, _w = image_features.shape

        # Concatenate instance and semantic queries.
        num_semantic_queries = semantic_query.shape[0]

        query = torch.cat((semantic_query, instance_query), dim=0)
        query_embs = torch.cat((semantic_query_embs, instance_query_embs), dim=0)
        reference_points = torch.cat(
            (semantic_reference_points, instance_reference_points), dim=0
        )

        # Attention layers expect batch-size to come second and queries first.
        # So we need to re-arrange things below.

        # Re-arrange and collect keys and values across all frames and image
        # dimensions.
        image_features = {
            "feats": einops.rearrange(image_features, "b n c h w -> b (n h w) c"),
            "feats_pos": einops.rearrange(
                image_features_embs, "b n c h w -> b (n h w) c"
            ),
            "key_padding_mask": einops.rearrange(
                image_features_mask, "b n h w -> b (n h w)"
            ),
        }

        # Re-organize voxel features.
        num_levels = len(voxel_features)
        reference_points = reference_points[None, :, None, :]
        reference_points = reference_points.repeat(b, 1, num_levels, 1)

        # Expand queries to cover the whole batch.
        query = query[None, :, :].repeat(b, 1, 1)  # [b, n_queries, q_dim]
        query_embs = query_embs[None, :, :].repeat(b, 1, 1)  # [b, n_queries, q_dim]

        # Run the transformer decoder.
        features = {
            "img": image_features,
            "voxel": {
                "value": voxel_features,
                "reference_points": reference_points,
            },
        }

        if keypoint_features is not None:
            features |= {
                "keypoint": {
                    "feats": keypoint_features,
                    "feats_pos": keypoint_embs,
                },
            }

        query = self.decoder(
            query=query,
            query_pos=query_embs,
            features=features,
        )  # [b, n_layers, n_queries, q_dim]

        # Split the queries back into instance and semantic.
        q_semantic = query[:, :, :num_semantic_queries, :]
        q_instance = query[:, :, num_semantic_queries:, :]

        return q_semantic, q_instance


class DisjointTrackingTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        semantic_num_layers: int,
        instance_num_layers: int,
        semantic_operations: list[OmegaConf | Mapping[str, Any]],
        instance_operations: list[OmegaConf | Mapping[str, Any]],
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.semantic_decoder = TransformerLayerSequence(
            num_layers=semantic_num_layers,
            embed_dim=embed_dim,
            operations=semantic_operations,
            return_intermediate=True,
            use_checkpointing=use_checkpointing,
        )

        self.instance_decoder = TransformerLayerSequence(
            num_layers=instance_num_layers,
            embed_dim=embed_dim,
            operations=instance_operations,
            return_intermediate=True,
            use_checkpointing=use_checkpointing,
        )

        # initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if hasattr(m, "weight") and m.weight.ndim > 1:
                nn.init.xavier_uniform_(m.weight)

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        instance_query: torch.Tensor,
        instance_query_embs: torch.Tensor,
        instance_reference_points: torch.Tensor,
        semantic_query: torch.Tensor,
        semantic_query_embs: torch.Tensor,
        semantic_reference_points: torch.Tensor,
        image_features: torch.Tensor,
        image_features_mask: torch.Tensor,
        image_features_embs: torch.Tensor,
        voxel_features: tuple[torch.Tensor],
        keypoint_features: torch.Tensor | None,
        keypoint_embs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=too-many-locals
        b, _n, _c, _h, _w = image_features.shape

        # Attention layers expect batch-size to come second and queries first.
        # So we need to re-arrange things below.

        # Re-arrange and collect keys and values across all frames and image
        # dimensions.
        image_features = {
            "feats": einops.rearrange(image_features, "b n c h w -> b (n h w) c"),
            "feats_pos": einops.rearrange(
                image_features_embs, "b n c h w -> b (n h w) c"
            ),
            "key_padding_mask": einops.rearrange(
                image_features_mask, "b n h w -> b (n h w)"
            ),
        }

        # Re-organize voxel features.
        num_levels = len(voxel_features)

        semantic_reference_points = semantic_reference_points[None, :, None, :]
        semantic_reference_points = semantic_reference_points.repeat(
            b, 1, num_levels, 1
        )

        instance_reference_points = instance_reference_points[None, :, None, :]
        instance_reference_points = instance_reference_points.repeat(
            b, 1, num_levels, 1
        )

        # Expand queries to cover the whole batch.
        semantic_query = semantic_query[None, :, :].repeat(b, 1, 1)
        semantic_query_embs = semantic_query_embs[None, :, :].repeat(b, 1, 1)

        instance_query = instance_query[None, :, :].repeat(b, 1, 1)
        instance_query_embs = instance_query_embs[None, :, :].repeat(b, 1, 1)

        # Run the transformer decoder for semantic queries.
        features = {
            "img": image_features,
            "voxel": {
                "value": voxel_features,
                "reference_points": semantic_reference_points,
            },
        }

        if keypoint_features is not None:
            features |= {
                "keypoint": {
                    "feats": keypoint_features,
                    "feats_pos": keypoint_embs,
                },
            }

        semantic_query = self.semantic_decoder(
            query=semantic_query,
            query_pos=semantic_query_embs,
            features=features,
        )  # [b, n_layers, n_queries, q_dim]

        # Run the transformer decoder for instance queries.
        features = {
            "img": image_features,
            "voxel": {
                "value": voxel_features,
                "reference_points": instance_reference_points,
            },
            "semantic": {
                "feats": semantic_query[:, -1, ...],  # last layer of semantic query
            },
        }

        if keypoint_features is not None:
            features |= {
                "keypoint": {
                    "feats": keypoint_features,
                    "feats_pos": keypoint_embs,
                },
            }

        instance_query = self.instance_decoder(
            query=instance_query,
            query_pos=instance_query_embs,
            features=features,
        )  # [b, n_layers, n_queries, q_dim]

        return semantic_query, instance_query


class DualStreamTrackingTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        operations: list[OmegaConf | Mapping[str, Any]],
    ) -> None:
        super().__init__()

        self.decoder = MultiStreamTransformerLayerSequence(
            num_layers=num_layers,
            embed_dim=embed_dim,
            operations=operations,
            return_intermediate=True,
        )

        # initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if hasattr(m, "weight") and m.weight.ndim > 1:
                nn.init.xavier_uniform_(m.weight)

                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        instance_query: torch.Tensor,
        instance_query_embs: torch.Tensor,
        instance_reference_points: torch.Tensor,
        semantic_query: torch.Tensor,
        semantic_query_embs: torch.Tensor,
        semantic_reference_points: torch.Tensor,
        image_features: torch.Tensor,
        image_features_mask: torch.Tensor,
        image_features_embs: torch.Tensor,
        voxel_features: tuple[torch.Tensor],
        keypoint_features: torch.Tensor | None,
        keypoint_embs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=too-many-locals
        b, _n, _c, _h, _w = image_features.shape

        # Attention layers expect batch-size to come second and queries first.
        # So we need to re-arrange things below.

        # Re-arrange and collect keys and values across all frames and image
        # dimensions.
        image_features = {
            "feats": einops.rearrange(image_features, "b n c h w -> b (n h w) c"),
            "feats_pos": einops.rearrange(
                image_features_embs, "b n c h w -> b (n h w) c"
            ),
            "key_padding_mask": einops.rearrange(
                image_features_mask, "b n h w -> b (n h w)"
            ),
        }

        # Re-organize voxel features.
        num_levels = len(voxel_features)

        semantic_reference_points = semantic_reference_points[None, :, None, :]
        semantic_reference_points = semantic_reference_points.repeat(
            b, 1, num_levels, 1
        )

        instance_reference_points = instance_reference_points[None, :, None, :]
        instance_reference_points = instance_reference_points.repeat(
            b, 1, num_levels, 1
        )

        # Expand queries to cover the whole batch.
        semantic_query = semantic_query[None, :, :].repeat(b, 1, 1)
        semantic_query_embs = semantic_query_embs[None, :, :].repeat(b, 1, 1)

        instance_query = instance_query[None, :, :].repeat(b, 1, 1)
        instance_query_embs = instance_query_embs[None, :, :].repeat(b, 1, 1)

        # Run the transformer decoder.
        inputs = {
            "semantic": {
                "query": semantic_query,
                "query_pos": semantic_query_embs,
            },
            "instance": {
                "query": instance_query,
                "query_pos": instance_query_embs,
            },
        }

        features = {
            "img": image_features,
            "voxel-semantic": {
                "value": voxel_features,
                "reference_points": semantic_reference_points,
            },
            "voxel-instance": {
                "value": voxel_features,
                "reference_points": instance_reference_points,
            },
        }

        if keypoint_features is not None:
            features |= {
                "keypoint": {
                    "feats": keypoint_features,
                    "feats_pos": keypoint_embs,
                },
            }

        query = self.decoder(inputs=inputs, features=features)

        return query["semantic"], query["instance"]
