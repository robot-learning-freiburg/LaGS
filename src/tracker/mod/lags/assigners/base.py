# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from abc import ABC, abstractmethod
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn

from ....config.registry import Registry
from ....utils.types import MetaDict

registry = Registry("lags.assigner")


class Assigner(nn.Module, ABC):
    # pylint: disable=too-few-public-methods
    """
    Assigner for matching box predictions with ground-truth labels.
    """

    @abstractmethod
    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> torch.Tensor:
        """
        Tries to match ground-truth objects with predicted objects.

        Assigns a ground-truth object index to each prediction. If no match can
        be found for a prediction, the assigned index will be -1.

        Args:
            preds: A dictionary containing the predictions.
            targets: A dictionary containing the ground-truth objects.

        Returns:
            A tensor of shape [N] containing values of -1 to M-1 where N is the
            number of predictions and M the number of targets. Index i contains
            the assignment of predicted box i, meaning the corresponding
            ground-truth index in the `gt_boxes` and `gt_labels` tensors or -1
            if no ground-truth box could be assigned.
        """
        raise NotImplementedError()


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf) -> Assigner:
    return registry.from_config(conf)
