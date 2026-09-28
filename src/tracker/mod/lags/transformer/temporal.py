# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping

import torch
from omegaconf import OmegaConf
from torch import nn

from .base import TransformerLayerSequence


class TemporalTransformer(nn.Module):
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
            return_intermediate=False,
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
        target: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor,
        key_pos: torch.Tensor,
        key_padding_mask_self: torch.Tensor | None = None,
        key_padding_mask_cross: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.decoder(
            query=target,
            query_pos=query_pos,
            features={
                "memory": {
                    "feats": memory,
                    "feats_pos": key_pos,
                    "key_padding_mask": key_padding_mask_cross,
                }
            },
            kwargs_group={
                "SelfAttention": {
                    "key_padding_mask": key_padding_mask_self,
                },
            },
        )  # [b, n_queries, emb_dim]
