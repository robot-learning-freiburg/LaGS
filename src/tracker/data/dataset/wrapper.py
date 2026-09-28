# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Iterator, List, Literal, Mapping

import numpy as np

from ...utils.types import MetaDict, Sample
from ..transform import Transform
from .dataset import Dataset, IterableDataset


def _transform_sample(
    sample: Sample, transforms: List[Transform], epoch: int
) -> Sample:
    sample.meta.epoch = epoch

    for tx in transforms:
        sample = tx(sample)

    # Prune the sample and remove any heavy-weight members like dataset
    # object references. Pushing those across process boundaries is
    # sub-optimal.
    if sample is not None:
        sample.prune()

    return sample


class DatasetWrapper(Dataset):
    def __init__(
        self,
        source: Dataset,
        transforms: List[Transform],
        on_none: Literal["raise", "ignore", "resample"] = "raise",
    ) -> None:
        super().__init__()

        self.source = source
        self.transforms = transforms
        self.on_none = on_none
        self.epoch = 0

        if on_none == "raise":
            self._getitem_impl = self._getitem_raise
        elif on_none == "ignore":
            self._getitem_impl = self._getitem_ignore
        elif on_none == "resample":
            self._getitem_impl = self._getitem_resample
        else:
            raise ValueError(f"unknown value for 'on_none': '{on_none}'")

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "source": self.source.hparams,
            "transforms": [t.hparams for t in self.transforms],
            "on_none": self.on_none,
        }

    @property
    def metadata(self) -> MetaDict:
        return self.source.metadata

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

        if hasattr(self.source, "set_epoch"):
            self.source.set_epoch(epoch)

    def __len__(self):
        return len(self.source)

    def _getitem_raise(self, index: int) -> Sample:
        sample = _transform_sample(self.source[index], self.transforms, self.epoch)

        if sample is None:
            raise RuntimeError("sample is None after transformation")

        return sample

    def _getitem_ignore(self, index: int) -> Sample | None:
        return _transform_sample(self.source[index], self.transforms, self.epoch)

    def _getitem_resample(self, index: int) -> Sample:
        for _ in range(len(self.source)):
            # try to get a valid sample
            sample = _transform_sample(self.source[index], self.transforms, self.epoch)
            if sample is not None:
                return sample

            # if that did not succeed, try another (random) index
            index = np.random.randint(0, len(self.source))

        raise RuntimeError("failed to sample non-None sample")

    def __getitem__(self, index: int) -> Sample | None:
        return self._getitem_impl(index)


# pylint: disable=abstract-method,too-few-public-methods
class IterableDatasetWrapper(IterableDataset):
    def __init__(
        self,
        source: Dataset,
        transforms: List[Transform],
        on_none: Literal["raise", "ignore"] = "raise",
    ) -> None:
        super().__init__()

        assert on_none in ["raise", "ignore"]

        self.source = source
        self.transforms = transforms
        self.on_none = on_none
        self.epoch = 0

        # only expose __len__ if the source also has it
        if getattr(self.source, "__len__", None) is not None:

            def _len(self):
                return len(self.source)

            setattr(self, "__len__", _len)

        # only expose __getitem__ if the source also has it
        if getattr(self.source, "__getitem__", None) is not None:

            def _getitem(self, index) -> Sample:
                sample = _transform_sample(
                    self.source[index], self.transforms, self.epoch
                )

                if sample is None and self.on_none == "raise":
                    raise RuntimeError("sample is None after transformation")

                return sample

            setattr(self, "__getitem__", _getitem)

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "source": self.source.hparams,
            "transforms": [t.hparams for t in self.transforms],
            "on_none": self.on_none,
        }

    @property
    def metadata(self) -> MetaDict:
        return self.source.metadata

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

        if hasattr(self.source, "set_epoch"):
            self.source.set_epoch(epoch)

    def __iter__(self) -> Iterator[Sample]:
        for sample in iter(self.source):
            sample = _transform_sample(sample, self.transforms, self.epoch)

            if sample is None and self.on_none == "raise":
                raise RuntimeError("sample is None after transformation")

            yield sample


def wrap(source: Dataset, transforms: List[Transform], **kwargs) -> Dataset:
    if isinstance(source, IterableDataset):
        return IterableDatasetWrapper(source, transforms, **kwargs)

    return DatasetWrapper(source, transforms, **kwargs)
