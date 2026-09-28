# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Age embedding modules for temporal Gaussian queries."""

import abc
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from ......config.registry import Registry

registry = Registry("lags.occupancy.gaussian.age_embedding")


class AgeEmbedding(nn.Module):
    """Base class for age embedding modules."""

    @abc.abstractmethod
    def forward(self, age: torch.Tensor) -> torch.Tensor:
        """
        Compute age embeddings.

        Args:
            age: Tensor of shape [b, n] containing ages (0, 1, 2, ...)

        Returns:
            Age embeddings of shape [b, n, embed_dim]
        """
        raise NotImplementedError()


@registry.register(key="learned_continuous")
class LearnedContinuousAgeEmbedding(AgeEmbedding):
    """
    Learned continuous age embedding using MLP.

    Maps normalized age through a small MLP to produce embeddings.
    Handles arbitrary ages through interpolation and doesn't suffer
    from sparse age distribution issues.
    """

    def __init__(
        self,
        embed_dim: int,
        max_age: int = 10,
        hidden_dim: int = 64,
        init_scale: float = 0.01,
    ):
        """
        Initialize continuous age embedding.

        Args:
            embed_dim: Dimension of output embeddings
            max_age: Maximum age for normalization
            hidden_dim: Hidden dimension of MLP
            init_scale: Initialization scale for weights (small to not corrupt features)
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_age = max_age

        # Small MLP to map normalized age to embedding
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Initialize with small weights to avoid corrupting refined features
        nn.init.normal_(self.encoder[0].weight, 0, init_scale)
        nn.init.zeros_(self.encoder[0].bias)
        nn.init.normal_(self.encoder[2].weight, 0, init_scale)
        nn.init.zeros_(self.encoder[2].bias)

        # Learnable scale for blending with query features
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, age: torch.Tensor) -> torch.Tensor:
        """
        Compute age embeddings.

        Args:
            age: [b, n] tensor of ages

        Returns:
            Age embeddings [b, n, embed_dim] scaled for gentle addition
        """
        # Normalize age to [0, 1]
        age_normalized = age.float().unsqueeze(-1) / self.max_age  # [b, n, 1]

        # Pass through encoder
        age_embed = self.encoder(age_normalized)  # [b, n, embed_dim]

        # Apply learned scale
        return self.scale * age_embed


@registry.register(key="binary")
class BinaryAgeEmbedding(AgeEmbedding):
    """
    Simple binary age embedding.

    Learned embedding for two states: t == 0 (current) and t >= 1 (past).
    """

    def __init__(self, embed_dim: int):
        """
        Initialize binary age embedding.

        Args:
            embed_dim: Dimension of output embeddings
        """
        super().__init__()

        self.embed_dim = embed_dim

        # Embedding layer for t >= 1
        self.embedding = nn.Embedding(2, embed_dim)
        nn.init.normal_(self.embedding.weight, 0, 0.01)

    def forward(self, age: torch.Tensor) -> torch.Tensor:
        """
        Compute binary age embeddings.

        Args:
            age: [b, n] tensor of ages

        Returns:
            Age embeddings [b, n, embed_dim]
        """
        return self.embedding(torch.clamp(age, max=1))


def build(conf: OmegaConf, **kwargs: Any) -> AgeEmbedding:
    """Build an age embedding module from config."""
    if conf is None:
        return None

    return registry.from_config(conf, **kwargs)
