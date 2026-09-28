# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * PETR (https://github.com/megvii-research/PETR), Copyright (c) 2022 megvii-research, licensed under Apache-2.0.
# See the LICENSES/ directory for full license texts.

import torch
from torch import nn


class MultiFrameSinePosEnc(nn.Module):
    """
    Sinusoidal positional encodings, extended for multi-frame images.

    See https://arxiv.org/abs/1706.03762.
    """

    def __init__(
        self,
        num_feats: int,
        temperature: float = 10000.0,
        normalize: bool = False,
        scale: float = 2 * torch.pi,
        eps: float = 1e-6,
        offset: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        # Note: the mask really shouldn't require gradients. We do stuff
        # in-place for simplicity later, so better to make sure of that here.
        assert not mask.requires_grad

        b, n, h, w = mask.shape

        # compute index grid via mask
        mask = ~mask
        idx_n = torch.cumsum(mask, dim=1, dtype=torch.float32)
        idx_y = torch.cumsum(mask, dim=2, dtype=torch.float32)
        idx_x = torch.cumsum(mask, dim=3, dtype=torch.float32)

        # normalize indices
        if self.normalize:
            idx_n = (idx_n + self.offset) / (idx_n[:, -1:, :, :] + self.eps)
            idx_n *= self.scale

            idx_y = (idx_y + self.offset) / (idx_y[:, :, -1:, :] + self.eps)
            idx_y *= self.scale

            idx_x = (idx_x + self.offset) / (idx_x[:, :, :, -1:] + self.eps)
            idx_x *= self.scale

        # compute feature time divisor
        d = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        d = self.temperature ** (2 * (d // 2) / self.num_feats)

        # compute actual time values
        pos_n = idx_n[..., None] / d
        pos_y = idx_y[..., None] / d
        pos_x = idx_x[..., None] / d

        # compute fourier series
        pos_n = torch.stack((pos_n[..., 0::2].sin(), pos_n[..., 1::2].cos()), dim=4)
        pos_n = pos_n.view(b, n, h, w, -1)

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4)
        pos_x = pos_x.view(b, n, h, w, -1)

        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4)
        pos_y = pos_y.view(b, n, h, w, -1)

        # combine everything
        pos = torch.cat((pos_n, pos_y, pos_x), dim=4)
        pos = pos.permute(0, 1, 4, 2, 3).contiguous()  # [b, n, 3*num_feats, h, w]

        return pos


def embedding1d(
    coords: torch.Tensor,
    num_feats: int = 128,
    temperature: float = 10000,
    scale: float = 2 * torch.pi,
) -> torch.Tensor:
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=coords.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)

    coords = coords * scale
    emb = coords[..., 0, None] / dim_t

    emb = torch.stack((emb[..., 0::2].sin(), emb[..., 1::2].cos()), dim=-1)
    emb = emb.flatten(-2)

    return emb


def embedding3d(
    coords: torch.Tensor,
    num_feats: int = 128,
    temperature: float = 10000,
    scale: float = 2 * torch.pi,
) -> torch.Tensor:
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=coords.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)

    coords = coords * scale
    px = coords[..., 0, None] / dim_t
    py = coords[..., 1, None] / dim_t
    pz = coords[..., 2, None] / dim_t

    px = torch.stack((px[..., 0::2].sin(), px[..., 1::2].cos()), dim=-1)
    px = px.flatten(-2)

    py = torch.stack((py[..., 0::2].sin(), py[..., 1::2].cos()), dim=-1)
    py = py.flatten(-2)

    pz = torch.stack((pz[..., 0::2].sin(), pz[..., 1::2].cos()), dim=-1)
    pz = pz.flatten(-2)

    return torch.cat((py, px, pz), dim=-1)
