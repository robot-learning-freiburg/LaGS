# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping

import einops
import torch
from omegaconf import OmegaConf
from torch import nn

from .base import TransformerLayerSequence


class SemanticTransformer(nn.Module):
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
        query: torch.Tensor,
        features: torch.Tensor,
        features_mask: torch.Tensor,
        query_embs: torch.Tensor,
        features_embs: torch.Tensor,
        voxel_features: tuple[torch.Tensor],
        reference_points: torch.Tensor,
    ) -> torch.Tensor:
        b, _n, _c, _h, _w = features.shape

        # Attention layers expect batch-size to come second and queries first.
        # So we need to re-arrange things below.

        # Re-arrange and collect keys and values across all frames and image
        # dimensions.
        img_features = {
            "feats": einops.rearrange(features, "b n c h w -> b (n h w) c"),
            "feats_pos": einops.rearrange(features_embs, "b n c h w -> b (n h w) c"),
            "key_padding_mask": einops.rearrange(features_mask, "b n h w -> b (n h w)"),
        }

        # Re-organize voxel features.
        num_levels = len(voxel_features)
        reference_points = reference_points[None, :, None, :]
        reference_points = reference_points.repeat(b, 1, num_levels, 1)

        voxel_features = {
            "value": voxel_features,
            "reference_points": reference_points,
        }

        # Expand queries to cover the whole batch.
        query = query[None, :, :].repeat(b, 1, 1)  # [b, n_queries, q_dim]
        query_embs = query_embs[None, :, :].repeat(b, 1, 1)  # [b, n_queries, q_dim]

        # Run the transformer decoder.
        return self.decoder(
            query=query,
            query_pos=query_embs,
            features={
                "img": img_features,
                "voxel": voxel_features,
            },
        )  # [b, n_layers, n_queries, q_dim]
