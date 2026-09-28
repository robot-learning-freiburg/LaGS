# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from torch import Tensor
from torch.utils.data._utils.collate import collate, default_collate_fn_map

from .metadict import MetaDict


class SampleMetadata(MetaDict):
    dataset_type: str
    sample_id: Any
    sequence_id: Any

    @classmethod
    def create(cls, dataset_type: str, sample_id: Any, sequence_id: Any, **kwargs):
        return cls(
            {
                "dataset_type": dataset_type,
                "sample_id": sample_id,
                "sequence_id": sequence_id,
                **kwargs,
            }
        )


def _sample_metadata_collate_fn(batch, *, collate_fn_map=None):
    # pylint: disable=unused-argument
    return batch


default_collate_fn_map[SampleMetadata] = _sample_metadata_collate_fn


@dataclass
class SampleSource:
    @classmethod
    def create(cls, dataset_type: str, data: Any, meta: MetaDict, sample: Any):
        return cls(dataset_type, data, meta, sample)

    def __init__(self, dataset_type: str, data: Any, meta: MetaDict, sample: Any):
        self.type = dataset_type  # dataset type
        self.data = data  # data source object
        self.meta = meta  # additional parameters/arguments for data loading
        self.sample = sample  # sample data


def _sample_source_collate_fn(batch, *, collate_fn_map=None):
    # pylint: disable=unused-argument
    raise RuntimeError(
        "SampleSource should not be passed into training/validation loop"
    )


default_collate_fn_map[SampleSource] = _sample_source_collate_fn


class Sample(MetaDict):
    """
    Base class for a sample containing arbitrary sample data. Automatically
    keeps track of the batch size.

    Note: Since this is based on MetaDict (which in turn is a MutableMapping),
    the usual automatisms of PyTorch (specifically: memory pinning of contained
    tensors) and Lightning (moving tensors to devices) is already supported out
    of the box.
    """

    batch_size: int
    source: SampleSource | None
    meta: SampleMetadata | List[SampleMetadata]
    is_valid: bool | Tensor

    @classmethod
    def create(
        cls, source: SampleSource, meta: SampleMetadata, is_valid: bool = True, **kwargs
    ):
        return cls({"source": source, "meta": meta, "is_valid": is_valid, **kwargs})

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        super().__init__(data)

        self.batch_size = self.batch_size if "batch_size" in self else 1

    def prune(self):
        """
        Prune the sample. This removes all information related to sample
        loading, like references to dataset objects that should really not be
        passed along outside of the dataloader processes.
        """
        del self.source


def _sample_collate_fn(batch, *, collate_fn_map=None):
    base = collate([x.__dict__ for x in batch], collate_fn_map=collate_fn_map)

    sample = Sample(base)
    sample.batch_size = len(batch)

    return sample


default_collate_fn_map[Sample] = _sample_collate_fn
