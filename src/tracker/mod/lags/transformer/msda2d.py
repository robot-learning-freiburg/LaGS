# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * OccFormer (https://github.com/zhangyp15/OccFormer), licensed under Apache-2.0,
# * MMDetection3D (https://github.com/open-mmlab/mmdetection3d), Copyright (c) OpenMMLab, licensed under Apache-2.0.
# See the LICENSES/ directory for full license texts.

import math
from typing import Any, Mapping, Sequence

import einops
import torch
from torch import nn
from torch.nn import functional as F

from .base import Operation, attention_ops


@torch.compile(fullgraph=True, mode="reduce-overhead")
def _sample_and_aggregate(
    features: Sequence[torch.Tensor],  # list[num_levels] of [b, embed_dim, h, w]
    weights: torch.Tensor,  # [b, num_query, num_heads, num_levels, num_points]
    points: torch.Tensor,  # [b, num_query, num_heads, num_levels, num_points, (x,y)]
):
    # pylint: disable=too-many-locals
    b, num_query, num_heads, _num_levels, num_points = weights.shape

    sampled = []
    for level, feat in enumerate(features):
        b, embed_dim, h, w = feat.shape

        feat = feat.view(b, num_heads, embed_dim // num_heads, h, w)
        feat = feat.view(b * num_heads, embed_dim // num_heads, h, w)

        grid = points[:, :, :, level, :, :]  # [b, q, k, p, (x,y)]
        grid = einops.rearrange(grid, "b q k p d -> (b k) q p d")

        feat = F.grid_sample(feat, grid, mode="bilinear", align_corners=False)
        feat = feat.view(b, num_heads, embed_dim // num_heads, num_query, num_points)

        sampled.append(feat)

    feats = torch.stack(sampled, dim=-2)
    feats = einops.rearrange(feats, "b h c q l p -> b q h c (l p)")

    weights = einops.rearrange(weights, "b q h l p -> b q h (l p)")

    output = feats * weights[:, :, :, None, :]
    output = output.sum(dim=-1)
    output = einops.rearrange(output, "b q h c -> b q (h c)")

    return output


class MultiScaleDeformableAttention2d(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        input_dim: int | None = None,
        batch_first: bool = False,
        with_output_proj: bool = True,
    ):
        super().__init__()

        input_dim = input_dim or embed_dim

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first

        assert embed_dim % num_heads == 0

        attn_size = num_heads * num_levels * num_points

        self.sampling_offsets = nn.Linear(embed_dim, attn_size * 2)
        self.attention_weights = nn.Linear(embed_dim, attn_size)
        self.value_proj = nn.Conv2d(input_dim, embed_dim, kernel_size=1)

        if with_output_proj:
            self.output_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.output_proj = None

        self._init_weights()

    def _init_weights(self):
        # initialize sampling offsets as grid
        with torch.no_grad():
            thetas = torch.arange(self.num_heads, dtype=torch.float32)
            thetas = thetas * 2.0 * math.pi / self.num_heads

            grid_init = torch.stack((thetas.cos(), thetas.sin()), dim=-1)
            grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
            grid_init = grid_init.view(self.num_heads, 1, 1, 2)
            grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)

            for i in range(self.num_points):
                grid_init[:, :, i, :] *= i + 1

            self.sampling_offsets.bias.copy_(grid_init.view(-1))

        torch.nn.init.constant_(self.sampling_offsets.weight, 0)

        # initialize attention weights
        torch.nn.init.constant_(self.attention_weights.weight, 0)
        torch.nn.init.constant_(self.attention_weights.bias, 0)

        # initialize value projection
        torch.nn.init.xavier_uniform_(self.value_proj.weight, gain=1.0)
        torch.nn.init.constant_(self.value_proj.bias, 0)

        # initialize output projection
        if self.output_proj is not None:
            torch.nn.init.xavier_uniform_(self.output_proj.weight, gain=1.0)
            torch.nn.init.constant_(self.output_proj.bias, 0)

    @torch.compile(mode="reduce-overhead")
    def forward(
        self,
        query: torch.Tensor,  # [b, num_query, embed_dim] if batch_first
        value: Sequence[torch.Tensor],  # list of [b, c, h, w]
        reference_points: torch.Tensor,  # [b, num_query, num_level, (x,y)] in range (-1, 1)
        key_padding_mask: Sequence[torch.Tensor] | None = None,  # [b, h, w]
    ) -> torch.Tensor:
        # ensure everything is batch-first
        if not self.batch_first:
            query = query.permute(1, 0, 2)

        b, num_query, _ = query.shape

        # apply value projection
        value = [self.value_proj(v) for v in value]

        # mask padded values
        if key_padding_mask is not None:
            value = [
                v.masked_fill(m[:, None, :, :], 0.0)
                for v, m in zip(value, key_padding_mask)
            ]

        # compute sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            b, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )

        # normalize sampling offsets (note: shape is in (y,x) order)
        dims = [torch.tensor(v.shape[-2:][::-1]) for v in value]
        dims = torch.stack(dims, dim=0).to(device=query.device)  # [num_levels, (x,y)]
        sampling_offsets = sampling_offsets / dims[None, None, None, :, None, :]

        # compute attention weights
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            b, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(dim=-1)
        attention_weights = attention_weights.view(
            b, num_query, self.num_heads, self.num_levels, self.num_points
        )

        # compute sampling locations
        sampling_locations = reference_points[:, :, None, :, None, :]
        sampling_locations = sampling_locations + sampling_offsets

        # sample and aggregate values
        output = _sample_and_aggregate(
            features=value,
            weights=attention_weights,
            points=sampling_locations,
        )

        # apply output projection
        if self.output_proj is not None:
            output = self.output_proj(output)

        # bring everything back to original order
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


class MultiScaleDeformableAttention2dLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        dropout: float,
    ):
        super().__init__()

        self.attn = MultiScaleDeformableAttention2d(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            batch_first=True,
        )

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        value: Sequence[torch.Tensor],
        reference_points: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query_upd = self.attn(
            query=query + query_pos if query_pos is not None else query,
            value=value,
            reference_points=reference_points,
            key_padding_mask=key_padding_mask,
        )

        query_upd = self.drop(query_upd)

        return self.norm(query + query_upd)


@attention_ops.register(key="MultiScaleDeformableCrossAttention2d")
class MultiScaleDeformableCrossAttention2dOp(Operation):
    def __init__(
        self,
        embed_dim: int,
        features: str,
        num_heads: int,
        num_levels: int,
        num_points: int,
        dropout: float,
        args_group: str | Sequence[str] | None = None,
    ):
        if args_group is None:
            args_group = "MultiScaleDeformableCrossAttention2d"

        super().__init__(args_group=args_group)

        self.features = features

        self.op = MultiScaleDeformableAttention2dLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        assert features is not None
        assert self.features in features

        return self.op(
            query=query,
            query_pos=query_pos,
            **features[self.features],
        )
