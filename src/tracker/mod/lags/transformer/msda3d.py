# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * OccFormer (https://github.com/zhangyp15/OccFormer), licensed under Apache-2.0,
# * MMDetection3D (https://github.com/open-mmlab/mmdetection3d), Copyright (c) OpenMMLab, licensed under Apache-2.0.
# See the LICENSES/ directory for full license texts.

"""Multi-scale deformable attention for 3D volumes (pure PyTorch version)."""

import math
from typing import Any, Mapping, Sequence

import einops
import torch
from torch import nn
from torch.nn import functional as F

from .base import Operation, attention_ops


@torch.compile(fullgraph=True, mode="reduce-overhead")
def _sample_and_aggregate(
    features: Sequence[torch.Tensor],  # list[num_levels] of [b, embed_dim, z, y, x]
    weights: torch.Tensor,  # [b, num_query, num_heads, num_levels, num_points]
    points: torch.Tensor,  # [b, num_query, num_heads, num_levels, num_points, (x,y,z)]
):
    # pylint: disable=too-many-locals
    b, num_query, num_heads, _num_levels, num_points = weights.shape

    sampled = []
    for level, feat in enumerate(features):
        b, embed_dim, z, y, x = feat.shape

        feat = feat.view(b, num_heads, embed_dim // num_heads, z, y, x)
        feat = feat.view(b * num_heads, embed_dim // num_heads, z, y, x)

        grid = points[:, :, :, level, :, :]  # [b, q, h, p, (x,y,z)]
        grid = einops.rearrange(grid, "b q h p d -> (b h) 1 q p d")

        feat = F.grid_sample(feat, grid, mode="bilinear", align_corners=False)
        feat = feat.squeeze(2)  # [b*num_heads, embed_dim//num_heads, q, p]
        feat = feat.view(b, num_heads, embed_dim // num_heads, num_query, num_points)

        sampled.append(feat)

    feats = torch.stack(sampled, dim=-2)
    feats = einops.rearrange(feats, "b h c q l p -> b q h c (l p)")

    weights = einops.rearrange(weights, "b q h l p -> b q h (l p)")

    output = feats * weights[:, :, :, None, :]
    output = output.sum(dim=-1)
    output = einops.rearrange(output, "b q h c -> b q (h c)")

    return output


class MultiScaleDeformableAttention3d(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        batch_first: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.batch_first = batch_first

        assert embed_dim % num_heads == 0

        attn_size = num_heads * num_levels * num_points

        self.sampling_offsets = nn.Linear(embed_dim, attn_size * 3)
        self.attention_weights = nn.Linear(embed_dim, attn_size)
        self.value_proj = nn.Conv3d(embed_dim, embed_dim, kernel_size=1)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        # initialize sampling offsets as grid
        with torch.no_grad():
            thetas = torch.arange(self.num_heads, dtype=torch.float32)
            thetas = thetas * 2.0 * math.pi / self.num_heads

            grid_init = (thetas.cos(), thetas.sin(), (thetas.sin() + thetas.cos()) / 2)
            grid_init = torch.stack(grid_init, dim=-1)
            grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True)[0]
            grid_init = grid_init.view(self.num_heads, 1, 1, 3)
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
        torch.nn.init.xavier_uniform_(self.output_proj.weight, gain=1.0)
        torch.nn.init.constant_(self.output_proj.bias, 0)

    @torch.compile
    def forward(
        self,
        query: torch.Tensor,  # [b, num_query, embed_dim] if batch_first
        value: Sequence[torch.Tensor],  # list of [b, embed_dim, z, y, x]
        reference_points: torch.Tensor,  # [b, num_query, num_level, (x,y,z)] in range (-1, 1)
        key_padding_mask: Sequence[torch.Tensor] | None = None,  # [b, z, y, x]
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
                v.masked_fill(m[:, None, :, :, :], 0.0)
                for v, m in zip(value, key_padding_mask)
            ]

        # compute sampling offsets
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            b, num_query, self.num_heads, self.num_levels, self.num_points, 3
        )

        # normalize sampling offsets (note: shape is in (z,y,x) order)
        dims = [torch.tensor(v.shape[-3:][::-1]) for v in value]
        dims = torch.stack(dims, dim=0).to(device=query.device)  # [num_levels, (x,y,z)]
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
        output = self.output_proj(output)

        # bring everything back to original order
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


class MultiScaleDeformableAttention3dLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        dropout: float,
    ):
        super().__init__()

        self.attn = MultiScaleDeformableAttention3d(
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


@attention_ops.register(key="MultiScaleDeformableCrossAttention3d")
class MultiScaleDeformableCrossAttention3dOp(Operation):
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
            args_group = "MultiScaleDeformableCrossAttention3d"

        super().__init__(args_group=args_group)

        self.features = features

        self.op = MultiScaleDeformableAttention3dLayer(
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


@attention_ops.register(key="MultiScaleDeformableSelfAttention3d")
class MultiScaleDeformableSelfAttention3dOp(Operation):
    # buffers
    reference_points: torch.Tensor  # [num_query, (x,y,z)] normalized to (-1, 1)

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_points: int,
        shapes: Sequence[tuple[int, int, int] | torch.Size],
        dropout: float,
        args_group: str | Sequence[str] | None = None,
    ):
        if args_group is None:
            args_group = "MultiScaleDeformableSelfAttention3d"

        super().__init__(args_group=args_group)

        self.embed_dim = embed_dim
        self.shapes = shapes

        # compute per-level strides
        strides = [0] + [math.prod(s) for s in shapes]
        strides = torch.tensor(strides, dtype=torch.long).cumsum(dim=0)
        self.strides = strides

        num_levels = len(shapes)
        assert num_levels > 0

        self.op = MultiScaleDeformableAttention3dLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )

        # pre-compute reference points in range (-1, 1)
        reference_points = self._compute_reference_points(shapes)
        self.register_buffer("reference_points", reference_points, persistent=False)

    def _compute_reference_points(self, shapes: Sequence[torch.Size]) -> torch.Tensor:
        reference_points = []
        for shape in shapes:
            nz, ny, nx = shape

            # reference points are in range (-1, 1)
            x = (torch.arange(nx) + 0.5) / nx * 2 - 1
            y = (torch.arange(ny) + 0.5) / ny * 2 - 1
            z = (torch.arange(nz) + 0.5) / nz * 2 - 1

            z, y, x = torch.meshgrid(z, y, x, indexing="ij")
            grid = torch.stack((x, y, z), dim=-1)  # [nz, ny, nx, (x,y,z)]

            grid = grid.view(-1, 3)  # [nz*ny*nx, (x,y,z)]
            reference_points.append(grid)

        return torch.cat(reference_points, dim=0)  # [num_query, (x,y,z)]

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        b, num_query, _ = query.shape
        num_levels = len(self.shapes)

        assert num_query == self.reference_points.shape[0]

        # build feature levels (value) from query
        # (transform from [b, num_query, embed_dim] to [b, embed_dim, z, y, x])
        value = [query[:, a:b, :] for a, b in zip(self.strides[:-1], self.strides[1:])]
        value = [v.permute(0, 2, 1) for v in value]
        value = [v.view(v.shape[0], v.shape[1], *s) for v, s in zip(value, self.shapes)]

        # expand reference points to [b, num_query, num_levels, (x,y,z)]
        reference_points = self.reference_points  # [num_query, (x,y,z)]
        reference_points = reference_points[None, :, None, :]
        reference_points = reference_points.expand(b, -1, num_levels, -1)

        return self.op(
            query=query,
            query_pos=query_pos,
            value=value,
            reference_points=reference_points,
        )
