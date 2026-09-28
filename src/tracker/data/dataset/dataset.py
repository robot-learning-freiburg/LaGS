# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from abc import ABC, abstractmethod
from typing import Any, Iterator, Mapping

import torch.utils.data as D

from ...utils.types import MetaDict, Sample


class Dataset(D.Dataset[Sample], ABC):
    @property
    @abstractmethod
    def hparams(self) -> Mapping[str, Any]:
        """
        Any dataset hyperparameters that influence the data returned by this
        dataset.
        """

    @property
    def metadata(self) -> MetaDict:
        return None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index) -> Sample:
        pass


class IterableDataset(D.IterableDataset[Sample], ABC):
    @property
    @abstractmethod
    def hparams(self) -> Mapping[str, Any]:
        """
        Any dataset hyperparameters that influence the data returned by this
        dataset.
        """

    @property
    def metadata(self) -> MetaDict:
        return None

    def __getitem__(self, *args, **kwargs):
        raise RuntimeError(f"{self.__class__.__name__} does not support __getitem__()")

    @abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        pass
