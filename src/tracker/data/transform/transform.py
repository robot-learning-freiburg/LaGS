# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from abc import ABC, abstractmethod
from typing import Any, Mapping

from ...utils.types import Sample


class Transform(ABC):
    """
    Base sample transform class.
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def hparams(self) -> Mapping[str, Any]:
        """
        Any hyperparameters that influence the transformation and result in
        different data being produced.
        """

    @abstractmethod
    def apply(self, sample: Sample) -> Sample | None:
        return sample

    def __call__(self, sample: Sample | None) -> Sample | None:
        if sample is None:
            return sample

        return self.apply(sample)
