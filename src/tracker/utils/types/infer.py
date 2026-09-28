# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping, Sequence

import torch

from .packedarray import PackedArray
from .packedtensor import PackedTensor


def infer_device(x: Any) -> torch.device:
    """
    Infer the device of a tensor or a collection of tensors.

    Args:
        x: A tensor or a collection of tensors (e.g., list, dict).

    Returns:
        The device of the tensor or None if no tensor is found.
    """

    if isinstance(x, torch.Tensor):
        return x.device

    if isinstance(x, Mapping):
        for value in x.values():
            device = infer_device(value)
            if device is not None:
                return device

    if isinstance(x, Sequence):
        for item in x:
            device = infer_device(item)
            if device is not None:
                return device

    return None


def infer_batch_size(x: Any) -> int | None:
    """
    Infer the batch size from a given object.

    Args:
        x (Any): The object to infer the batch size from.

    Returns:
        int: The inferred batch size.
    """

    if isinstance(x, torch.Tensor):
        return x.shape[0]

    if isinstance(x, PackedTensor):
        return x.batch_size

    if isinstance(x, PackedArray):
        return x.batch_size

    if isinstance(x, Mapping):
        for value in x.values():
            batch_size = infer_batch_size(value)
            if batch_size is not None:
                return batch_size

    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        # A sequence (e.g. produced by a custom collate fn) represents the
        # batch as a list of per-sample elements, so its length is the batch
        # size. Empty sequences carry no such information.
        if len(x) > 0:
            return len(x)

    return None
