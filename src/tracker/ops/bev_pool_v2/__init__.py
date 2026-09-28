# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * BEVDet (https://github.com/HuangJunJie2017/BEVDet), Copyright (c) Phigent Robotics, licensed under Apache-2.0.
# See the LICENSES/ directory for full license texts.

"""BEV pooling operations."""

from typing import Any

import torch
from torch.autograd import Function

from . import bev_pool_v2_ext as _ext

_pool_forward = torch.ops.bev_pool_v2.pool_forward
_pool_backward = torch.ops.bev_pool_v2.pool_backward

# TODO: register fakes


class QuickCumsum(Function):
    # pylint: disable=abstract-method
    """
    BEVPoolv2 implementation for Lift-Splat-Shoot view transformation.

    Please refer to the `paper <https://arxiv.org/abs/2211.17111>`
    """

    @staticmethod
    def forward(
        ctx: Any,
        depth: torch.Tensor,
        feat: torch.Tensor,
        ranks_depth: torch.Tensor,
        ranks_feat: torch.Tensor,
        ranks_bev: torch.Tensor,
        bev_feat_shape: tuple[int] | torch.Size,
        interval_starts: torch.Tensor,
        interval_lengths: torch.Tensor,
    ):
        # pylint: disable=arguments-differ

        depth = depth.contiguous().float()
        feat = feat.contiguous().float()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        ranks_bev = ranks_bev.int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        out = feat.new_zeros(bev_feat_shape)

        _pool_forward(
            depth,
            feat,
            out,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )

        ctx.save_for_backward(depth, feat, ranks_bev, ranks_depth, ranks_feat)

        return out

    @staticmethod
    def backward(ctx: Any, out_grad: torch.Tensor):
        # pylint: disable=arguments-differ

        depth, feat, ranks_bev, ranks_depth, ranks_feat = ctx.saved_tensors

        order = ranks_feat.argsort()
        ranks_feat = ranks_feat[order]
        ranks_depth = ranks_depth[order]
        ranks_bev = ranks_bev[order]

        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

        depth = depth.contiguous()
        feat = feat.contiguous()
        ranks_depth = ranks_depth.contiguous()
        ranks_feat = ranks_feat.contiguous()
        ranks_bev = ranks_bev.contiguous()
        interval_lengths = interval_lengths.contiguous()
        interval_starts = interval_starts.contiguous()

        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        out_grad = out_grad.contiguous()

        _pool_backward(
            out_grad,
            depth_grad,
            feat_grad,
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            interval_lengths,
            interval_starts,
        )

        return depth_grad, feat_grad, None, None, None, None, None, None, None, None


def bev_pool_v2(
    depth: torch.Tensor,
    feat: torch.Tensor,
    ranks_depth: torch.Tensor,
    ranks_feat: torch.Tensor,
    ranks_bev: torch.Tensor,
    bev_feat_shape: tuple[int] | torch.Size,  # [b, z, y, x, c]
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
) -> torch.Tensor:
    return QuickCumsum.apply(
        depth,
        feat,
        ranks_depth,
        ranks_feat,
        ranks_bev,
        bev_feat_shape,
        interval_starts,
        interval_lengths,
    )  # [b, z, y, x, c]
