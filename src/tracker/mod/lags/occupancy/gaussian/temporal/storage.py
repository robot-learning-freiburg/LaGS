# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass
from typing import Self

import torch
from tensordict import tensorclass


@tensorclass
class GaussianStreamInstances:
    # pylint: disable=too-few-public-methods

    # query features
    query: torch.Tensor
    query_coords: torch.Tensor

    # decoded properties
    logits: torch.Tensor
    centers: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor

    # state
    age: torch.Tensor
    confidence: torch.Tensor  # [b, n] accumulated confidence in [0, 1]
    instance_ids: torch.Tensor  # [b, n] instance IDs (-1 = unassigned)

    def detach(self) -> Self:
        return GaussianStreamInstances(
            query=self.query.detach(),
            query_coords=self.query_coords.detach(),
            logits=self.logits.detach(),
            centers=self.centers.detach(),
            scales=self.scales.detach(),
            rotations=self.rotations.detach(),
            opacities=self.opacities.detach(),
            age=self.age.detach(),
            confidence=self.confidence.detach(),
            instance_ids=self.instance_ids.detach(),
        )

    @staticmethod
    def cat(
        instances: list[Self],
        dim: int = 1,
    ) -> Self:
        """Concatenate a list of GaussianStreamInstances along a given dimension."""
        return GaussianStreamInstances(
            query=torch.cat([inst.query for inst in instances], dim=dim),
            query_coords=torch.cat([inst.query_coords for inst in instances], dim=dim),
            logits=torch.cat([inst.logits for inst in instances], dim=dim),
            centers=torch.cat([inst.centers for inst in instances], dim=dim),
            scales=torch.cat([inst.scales for inst in instances], dim=dim),
            rotations=torch.cat([inst.rotations for inst in instances], dim=dim),
            opacities=torch.cat([inst.opacities for inst in instances], dim=dim),
            age=torch.cat([inst.age for inst in instances], dim=dim),
            confidence=torch.cat([inst.confidence for inst in instances], dim=dim),
            instance_ids=torch.cat([inst.instance_ids for inst in instances], dim=dim),
        )


@dataclass
class GaussianInstances:
    streams: dict[str, GaussianStreamInstances]

    def detach(self) -> Self:
        return GaussianInstances(
            streams={k: v.detach() for k, v in self.streams.items()}
        )

    @staticmethod
    def cat(
        instances: list[Self],
        dim: int = 1,
    ) -> Self:
        """Concatenate a list of GaussianInstances along a given dimension."""
        streams = instances[0].streams.keys()
        assert all(inst.streams.keys() == streams for inst in instances)

        out = {
            name: GaussianStreamInstances.cat(
                [inst.streams[name] for inst in instances], dim=dim
            )
            for name in streams
        }

        return GaussianInstances(streams=out)


@dataclass
class TemporalState:
    """
    Complete temporal state including gaussians and density maps.

    This bundles the gaussian instances with their associated density map,
    enabling density-aware sampling in subsequent frames.
    """

    gaussians: GaussianInstances
    density_map: torch.Tensor | None = None  # [b, 1, d, h, w]

    def detach(self) -> Self:
        return TemporalState(
            gaussians=self.gaussians.detach(),
            density_map=(
                self.density_map.detach() if self.density_map is not None else None
            ),
        )
