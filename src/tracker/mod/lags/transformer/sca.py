# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .base import Operation, attention_ops
from .msda2d import MultiScaleDeformableAttention2d


class SpatialCrossAttention(nn.Module):
    """
    Multi-image multi-scale deformable cross-attention module.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        input_dim: int | None = None,
        batch_first: bool = False,
    ):
        super().__init__()

        self.batch_first = batch_first

        self.msda = MultiScaleDeformableAttention2d(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            input_dim=input_dim,
            batch_first=True,
            with_output_proj=False,
        )

        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        torch.nn.init.xavier_uniform_(self.output_proj.weight, gain=1.0)
        torch.nn.init.constant_(self.output_proj.bias, 0)

    def _compress_query(
        self,
        query: torch.Tensor,  # [b, num_query, embed_dim] if batch_first
        reference_points: torch.Tensor,  # [b, num_cams, num_query, 2]
        reference_points_mask: torch.Tensor,  # [b, num_cams, num_query]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # pylint: disable=too-many-locals
        b, num_cams, num_query, _ = reference_points.shape

        reference_points = reference_points.view(b * num_cams, num_query, 2)
        reference_points_mask = reference_points_mask.view(b * num_cams, num_query)

        indices = reference_points_mask.nonzero()  # [nonzero_elements, (b*nc, nq)]

        num_points = torch.bincount(indices[:, 0], minlength=b * num_cams)
        max_num_points = num_points.max()

        cpct_query = query.new_zeros(b * num_cams, max_num_points, query.shape[-1])
        cpct_pts = reference_points.new_zeros(b * num_cams, max_num_points, 2)

        n = 0
        for i in range(b * num_cams):
            batch = i // num_cams
            k = num_points[i]

            cpct_query[i, 0:k] = query[batch, indices[n : n + k, 1]]
            cpct_pts[i, 0:k] = reference_points[i, indices[n : n + k, 1]]

            n += k

        return indices, num_points, cpct_query, cpct_pts

    def _decompress_query(
        self,
        query: torch.Tensor,  # [b * num_cams, max_num_q, embed_dim]
        indices: torch.Tensor,  # [nonzero_elements, (b*nc, nq)]
        num_points: torch.Tensor,  # [b * num_cams]
        reference_points_mask: torch.Tensor,  # [b, num_cams, num_query]
    ) -> torch.Tensor:
        _, _, embed_dim = query.shape
        b, num_cams, num_query = reference_points_mask.shape

        # sum up queries across all cameras
        output = torch.zeros(b, num_query, embed_dim, device=query.device)

        n = 0
        for i in range(b * num_cams):
            batch = i // num_cams
            k = num_points[i]

            output[batch, indices[n : n + k, 1]] += query[i, 0:k]

            n += k

        # average over all cameras
        counts = reference_points_mask.sum(dim=1).clamp(min=1)  # [b, num_query]
        output = output / counts[..., None]

        return output

    @torch.compile
    def forward(
        self,
        query: torch.Tensor,  # [b, num_query, embed_dim] if batch_first
        value: Sequence[torch.Tensor],  # seq [b, num_cams, c, h_i, w_i]
        reference_points: torch.Tensor,  # [b, num_cams, num_query, 2]
        reference_points_mask: torch.Tensor,  # [b, num_cams, num_query]
        value_mask: Sequence[torch.Tensor] | None = None,  # seq [b, num_cams, h_i, w_i]
    ) -> torch.Tensor:
        # ensure everything is batch-first
        if not self.batch_first:
            query = query.permute(1, 0, 2)

        # On a standard surround-view camera setup (like nuScnes), reference
        # points fall only into one or at most two images. That means, we can
        # significantly reduce the number of reference points we need to sample
        # from.
        indices, num_points, query, points = self._compress_query(
            query=query,
            reference_points=reference_points,
            reference_points_mask=reference_points_mask,
        )

        # perform deformable attention
        points = points[:, :, None, :]
        points = points.expand(-1, -1, self.msda.num_levels, -1)

        value = [v.flatten(0, 1) for v in value]  # [b * num_cams, c, h_i, w_i]
        value_mask = [m.flatten(0, 1) for m in value_mask] if value_mask else None

        query = self.msda(
            query=query,
            value=value,
            reference_points=points,
            key_padding_mask=value_mask,
        )

        # sum up queries across all cameras
        output = self._decompress_query(
            query=query,
            indices=indices,
            num_points=num_points,
            reference_points_mask=reference_points_mask,
        )  # [b, num_query, embed_dim]

        # apply output projection
        output = self.output_proj(output)

        # bring everything back to original order
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output


class SpatialCrossAttentionLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        dropout: float,
    ):
        super().__init__()

        self.attn = SpatialCrossAttention(
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
        reference_points_mask: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        value_mask: Sequence[torch.Tensor] | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        query_upd = self.attn(
            query=query + query_pos if query_pos is not None else query,
            value=value,
            value_mask=value_mask,
            reference_points=reference_points,
            reference_points_mask=reference_points_mask,
        )

        query_upd = self.drop(query_upd)

        return self.norm(query + query_upd)


@attention_ops.register(key="SpatialCrossAttention")
class SpatialCrossAttentionOp(Operation):
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
            args_group = "SpatialCrossAttention"

        super().__init__(args_group)

        self.features = features

        self.op = SpatialCrossAttentionLayer(
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
