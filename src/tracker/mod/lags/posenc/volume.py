# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from ....config.registry import Registry
from .fourier import embedding3d

registry = Registry("posenc.volume")


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> nn.Module:
    return registry.from_config(conf)


@registry.register
class LearnedGridPositionalEncoding3d(nn.Module):
    def __init__(
        self,
        embed_dim: tuple[int, int, int],
        shape: tuple[int, int, int],
    ):

        super().__init__()

        self.shape = shape
        self.embed_dim = embed_dim

        nz, ny, nx = shape
        cz, cy, cx = embed_dim

        self.embed_x = nn.Embedding(nx, embedding_dim=cx)
        self.embed_y = nn.Embedding(ny, embedding_dim=cy)
        self.embed_z = nn.Embedding(nz, embedding_dim=cz)

        self._init_weights()

    def _init_weights(self):
        nn.init.uniform_(self.embed_x.weight)
        nn.init.uniform_(self.embed_y.weight)
        nn.init.uniform_(self.embed_z.weight)

    def forward(self):
        nz, ny, nx = self.shape
        cz, cy, cx = self.embed_dim

        emb_x = self.embed_x.weight.view(1, 1, nx, cx).expand(nz, ny, nx, cx)
        emb_y = self.embed_y.weight.view(1, ny, 1, cy).expand(nz, ny, nx, cy)
        emb_z = self.embed_z.weight.view(nz, 1, 1, cz).expand(nz, ny, nx, cz)

        emb = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        emb = emb.unsqueeze(0)  # [1, nz, ny, nx, (x,y,z)]

        return emb


@registry.register
class FourierGridPositionalEncoding3d(nn.Module):
    # pylint: disable=too-many-instance-attributes

    # buffers
    grid: torch.Tensor  # [nz, ny, nx, (x,y,z)]

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        shape: tuple[int, int, int],
        voxel_range: tuple[float, float, float, float, float, float],
        target_range: tuple[float, float, float, float, float, float],
        pos_temp: float = 10000.0,
        pos_scale: float = 2 * torch.pi,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.shape = shape
        self.voxel_range = voxel_range
        self.target_range = target_range

        self.pos_dim = pos_dim
        self.pos_temp = pos_temp
        self.pos_scale = pos_scale

        self.register_buffer("grid", self._build_grid(), persistent=False)

        self.adapter = nn.Sequential(
            nn.Linear(3 * pos_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )

    def _build_grid(self):
        nz, ny, nx = self.shape

        # create grid coordinates for voxel centers
        x = torch.arange(nx, dtype=torch.float32) + 0.5
        y = torch.arange(ny, dtype=torch.float32) + 0.5
        z = torch.arange(nz, dtype=torch.float32) + 0.5

        # map grid coordinates to voxel range
        x_min, y_min, z_min, x_max, y_max, z_max = self.voxel_range
        x = x / nx * (x_max - x_min) + x_min
        y = y / ny * (y_max - y_min) + y_min
        z = z / nz * (z_max - z_min) + z_min

        # normalize to [0, 1] using target range
        x_min, y_min, z_min, x_max, y_max, z_max = self.target_range
        x = (x - x_min) / (x_max - x_min)
        y = (y - y_min) / (y_max - y_min)
        z = (z - z_min) / (z_max - z_min)

        x = x.view(1, 1, nx, 1).expand(nz, ny, nx, 1)
        y = y.view(1, ny, 1, 1).expand(nz, ny, nx, 1)
        z = z.view(nz, 1, 1, 1).expand(nz, ny, nx, 1)

        return torch.cat((x, y, z), dim=-1)

    def forward(self):
        coords = self.grid.clone()  # [nz, ny, nx, (x,y,z)]
        coords = coords.unsqueeze(0)  # [1, nz, ny, nx, (x,y,z)]

        # apply fourier encoding
        embs = embedding3d(
            coords=coords,
            num_feats=self.pos_dim,
            temperature=self.pos_temp,
            scale=self.pos_scale,
        )

        embs = self.adapter(embs)

        return embs  # [1, nz, ny, nx, embed_dim]
