# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping

import torch

from .infer import infer_batch_size
from .metadict import MetaDict
from .packedarray import PackedArray
from .packedtensor import PackedTensor


def uncollate(x: Mapping[str, Any], batch_size=None) -> list[MetaDict]:
    """
    Split a dictionary of tensors into a list of dictionaries, each
    representing a single batch.

    Args:
        x (MetaDict): The input dictionary to split.
        batch_size (int, optional): The batch size to use. If None, it will
            be inferred from the input.

    Returns:
        list[MetaDict]: A list of dictionaries, each representing a single
            batch.
    """

    if batch_size is None:
        batch_size = infer_batch_size(x)
        assert batch_size is not None, "Batch size could not be inferred."

    out = [MetaDict() for _ in range(batch_size)]

    for key, value in x.items():
        if isinstance(value, torch.Tensor):
            unbound = value.unbind(dim=0)

        elif isinstance(value, (PackedTensor | PackedArray)):
            unbound = value.unbind()

        elif isinstance(value, Mapping):
            unbound = uncollate(value, batch_size=batch_size)

        elif isinstance(value, list):
            # Custom collate fns (e.g. t3d.Transform, EgoPoses, SampleMetadata)
            # return the batch as a list of length batch_size; that list is
            # already the per-sample (unbound) form, so pass it through.
            unbound = value

        else:
            raise ValueError(f"Unsupported type: {type(value)}")

        assert len(unbound) == batch_size

        for i in range(batch_size):
            out[i][key] = unbound[i]

    return out
