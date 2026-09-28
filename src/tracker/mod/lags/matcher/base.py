# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import abc
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from ....config.registry import Registry
from ....utils.types import MetaDict, uncollate

registry = Registry("lags.matcher")

INDEX_UNMATCHED = -1
INDEX_DISCONTINUED = -2


class Matcher(nn.Module):
    @torch.no_grad()
    def uncollate_preds(
        self,
        preds: MetaDict,
    ) -> list[MetaDict]:
        """
        Split the predictions into a list of dictionaries, each containing the
        predictions for a single batch.

        Args:
            preds (MetaDict): The predictions to split. Predictions stored here
                should either be PackedTensor, PackedArray, or regular tensors
                of shape [b, ...], where b is the batch size. For multi-layer
                inputs, the shape should be [b, n, ...], where n is the number
                of layers.

        Returns:
            list[MetaDict]: A list of dictionaries, each containing the
                predictions for a single batch.
        """
        return uncollate(preds)

    @torch.no_grad()
    def uncollate_targets(
        self,
        targets: MetaDict,
    ) -> list[MetaDict]:
        """
        Split the targets into a list of dictionaries, each containing the
        targets for a single batch.

        Args:
            targets (MetaDict): The targets to split. Targets stored here
                should either be PackedTensor, PackedArray, or regular tensors
                of shape [b, ...], where b is the batch size.

        Returns:
            list[MetaDict]: A list of dictionaries, each containing the targets
                for a single batch.
        """
        return uncollate(targets)

    @abc.abstractmethod
    def match_sample(self, preds: MetaDict, targets: MetaDict) -> torch.Tensor:
        """
        Match the predictions to the targets for a single sample.

        Args:
            preds (MetaDict): The predictions for a single sample.
            targets (MetaDict): The targets for a single sample.

        Returns:
            torch.Tensor: The indices of the matched targets for each prediction.
        """
        raise NotImplementedError()

    @torch.no_grad()
    def match(self, preds: MetaDict, targets: MetaDict) -> torch.Tensor:
        # split the predictions and targets into individual batches
        preds = self.uncollate_preds(preds)
        targets = self.uncollate_targets(targets)

        assert len(preds) == len(targets)

        # perform matching separately for each batch
        matches = [self.match_sample(p, t) for p, t in zip(preds, targets)]
        matches = torch.stack(matches, dim=0)

        return matches

    @torch.no_grad()
    def forward(self, preds: MetaDict, targets: MetaDict) -> MetaDict:
        return self.match(preds, targets)


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf, **kwargs: Any) -> Matcher:
    return registry.from_config(conf, **kwargs)
