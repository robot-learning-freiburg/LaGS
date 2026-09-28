# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import functools
from typing import List, Optional, Self

import torch
from torch.utils.data._utils.collate import default_collate_fn_map

from ... import utils
from .metadict import MetaDict

HANDLED_FUNCTIONS = {}


class PackedTensor:
    """
    Class for storing variable-length tensors (e.g., point clouds) in a single
    packed tensor. The stored tensors must have the same shape except for the
    first dimension.

    This is intended for batch-processing of point clouds.
    """

    @classmethod
    def __torch_function__(cls, func, _types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}

        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __init__(self, data: torch.Tensor, offsets: Optional[torch.Tensor] = None):
        if offsets is None:
            offsets = [0, data.shape[0]]
            offsets = torch.tensor(offsets, device="cpu", dtype=torch.int64)
        else:
            offsets = offsets.to(device="cpu")

        self.data = data
        self.offsets = offsets

    @classmethod
    def from_tensors(cls, tensors: List[torch.Tensor]) -> Self:
        offsets = [0] + [t.shape[0] for t in tensors]
        offsets = torch.tensor(offsets, device="cpu", dtype=torch.int64)
        offsets = torch.cumsum(offsets, dim=0)

        data = torch.cat(tensors, dim=0)

        return cls(data, offsets)

    @property
    def batch_size(self):
        return self.offsets.shape[0] - 1

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={repr(self.data)}, offsets={repr(self.offsets)})"

    def __len__(self) -> int:
        return self.batch_size

    def __iter__(self):
        for i in range(len(self)):
            yield self.get(i)

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def dtype(self) -> torch.dtype:
        return self.data.dtype

    def to(self, *args, **kwargs) -> Self:
        # the dtype for offsets should remain the same, so exclude it here
        # the device for offsets should always be 'cpu'
        to_args = utils.torch.parse_to(*args, **kwargs).to_dict()
        to_args.pop("dtype", None)
        to_args["device"] = torch.device("cpu")

        data = self.data.to(*args, **kwargs)
        offs = self.offsets.to(**to_args)

        return PackedTensor(data, offs)

    def cuda(self) -> Self:
        return self.to(device=torch.device("cuda"))

    def cpu(self) -> Self:
        return self.to(device=torch.device("cpu"))

    def clone(self, *args, **kwargs) -> Self:
        return PackedTensor(
            data=self.data.clone(*args, **kwargs),
            offsets=self.offsets.clone(*args, **kwargs),
        )

    @property
    def indices(self) -> torch.Tensor:
        n = torch.diff(self.offsets.to(device=self.device))
        indices = torch.arange(0, self.batch_size, device=self.device, dtype=torch.long)
        indices = torch.repeat_interleave(indices, n)

        return indices

    def get(self, index) -> torch.Tensor:
        return self.data[self.offsets[index] : self.offsets[index + 1]]

    def padded(
        self,
        value: float = 0,
        with_mask: bool = False,
        with_counts: bool = False,
        device: torch.device | None = None,
    ) -> MetaDict:
        """
        Convert the packed tensor to a padded tensor.

        Args:
            value (float): Value to pad with.
            with_mask (bool): Whether to return a mask indicating the
                valid entries in the padded tensor. If True, the mask will
                be a boolean tensor, where True indicates a valid entry and
                False indicates a padded entry. For a padded tensor of shape
                (batch_size, n, ...), the mask will be of shape (batch_size, n).
            with_counts (bool): Whether to return the counts of valid entries
                for each batch. If True, the counts will be a 1D tensor of
                shape (batch_size,).
            device (torch.device): Device to use for the padded tensor. If None,
                the device of the original tensor will be used.

        Returns:
            torch.Tensor: Padded tensor.
            torch.Tensor: Mask for valid entries (if with_mask is True).
        """
        device = device or self.device

        counts = torch.diff(self.offsets)
        n = counts.max().item()

        padded = torch.full(
            (self.batch_size, n) + self.data.shape[1:],
            fill_value=value,
            dtype=self.data.dtype,
            device=device,
        )

        if with_mask:
            mask = torch.full(
                (self.batch_size, n),
                fill_value=False,
                dtype=torch.bool,
                device=device,
            )

        for i in range(self.batch_size):
            start = self.offsets[i].item()
            end = self.offsets[i + 1].item()

            padded[i, : end - start] = self.data[start:end]

            if with_mask:
                mask[i, : end - start] = True

        out = MetaDict()
        out.data = padded

        if with_mask:
            out.mask = mask

        if with_counts:
            out.counts = counts

        return out

    def unbind(self) -> List[torch.Tensor]:
        """
        Unbind the packed tensor into a list of tensors.

        Returns:
            List[torch.Tensor]: List of tensors.
        """
        return [self.get(i) for i in range(self.batch_size)]

    def __getitem__(self, key: Self) -> Self:
        if isinstance(key, PackedTensor):
            if key.dtype == torch.bool and len(key.data.shape) == 1:  # filter by mask
                assert torch.all(self.offsets == key.offsets)

                mask = key.data

                data = self.data[mask]

                offsets = torch.zeros_like(self.offsets)
                offsets[1:] = torch.cumsum(mask, dim=0)[self.offsets[1:] - 1]

                return PackedTensor(data, offsets)

        raise NotImplementedError()

    def __eq__(self, other: Self | float) -> Self:
        if isinstance(other, (float, int)):
            return PackedTensor(self.data == other, self.offsets)

        if isinstance(other, PackedTensor):
            if not torch.all(self.offsets == other.offsets):
                raise ValueError("PackedTensors must have the same offsets.")

            return PackedTensor(self.data == other.data, self.offsets)

        raise NotImplementedError()

    def __ne__(self, other: Self | float) -> Self:
        if isinstance(other, (float, int)):
            return PackedTensor(self.data != other, self.offsets)

        if isinstance(other, PackedTensor):
            if not torch.all(self.offsets == other.offsets):
                raise ValueError("PackedTensors must have the same offsets.")

            return PackedTensor(self.data != other.data, self.offsets)

        raise NotImplementedError()

    def __lt__(self, other: Self | float) -> Self:
        if isinstance(other, (float, int)):
            return PackedTensor(self.data < other, self.offsets)

        if isinstance(other, PackedTensor):
            if not torch.all(self.offsets == other.offsets):
                raise ValueError("PackedTensors must have the same offsets.")

            return PackedTensor(self.data < other.data, self.offsets)

        raise NotImplementedError()

    def __le__(self, other: Self | float) -> Self:
        if isinstance(other, (float, int)):
            return PackedTensor(self.data <= other, self.offsets)

        if isinstance(other, PackedTensor):
            if not torch.all(self.offsets == other.offsets):
                raise ValueError("PackedTensors must have the same offsets.")

            return PackedTensor(self.data <= other.data, self.offsets)

        raise NotImplementedError()

    def __gt__(self, other: Self | float) -> Self:
        if isinstance(other, (float, int)):
            return PackedTensor(self.data > other, self.offsets)

        if isinstance(other, PackedTensor):
            if not torch.all(self.offsets == other.offsets):
                raise ValueError("PackedTensors must have the same offsets.")

            return PackedTensor(self.data > other.data, self.offsets)

        raise NotImplementedError()

    def __ge__(self, other: Self | float) -> Self:
        if isinstance(other, (float, int)):
            return PackedTensor(self.data >= other, self.offsets)

        if isinstance(other, PackedTensor):
            if not torch.all(self.offsets == other.offsets):
                raise ValueError("PackedTensors must have the same offsets.")

            return PackedTensor(self.data >= other.data, self.offsets)

        raise NotImplementedError()


def _collate_fn(batch, *, collate_fn_map=None):
    # pylint: disable=unused-argument

    data = [x.data for x in batch]
    offsets = [x.offsets for x in batch]

    elem = offsets[0]

    batched_offsets = [torch.tensor([0], dtype=elem.dtype, device=elem.device)]
    for offs in offsets:
        assert offs[0] == 0

        batched_offsets.append(offs[1:] + batched_offsets[-1][-1])

    return PackedTensor(
        data=torch.cat(data, dim=0),
        offsets=torch.cat(batched_offsets, dim=0),
    )


default_collate_fn_map[PackedTensor] = _collate_fn


def implements(torch_function):
    """Register a torch function override for ScalarTensor"""

    def decorator(func):
        functools.update_wrapper(func, torch_function)
        HANDLED_FUNCTIONS[torch_function] = func
        return func

    return decorator


@implements(torch.add)
def add(a, b, alpha=1) -> PackedTensor:
    assert issubclass(type(a), PackedTensor)
    assert issubclass(type(b), PackedTensor)
    assert torch.all(a.offsets == b.offsets)

    data = torch.add(a.data, b.data, alpha=alpha)

    return PackedTensor(data, a.offsets)


@implements(torch.sub)
def sub(a, b, alpha=1) -> PackedTensor:
    assert issubclass(type(a), PackedTensor)
    assert issubclass(type(b), PackedTensor)
    assert torch.all(a.offsets == b.offsets)

    data = torch.add(a.data, b.data, alpha=alpha)

    return PackedTensor(data, a.offsets)


@implements(torch.Tensor.__getitem__)
def getitem(tensor: torch.Tensor, key: PackedTensor) -> PackedTensor:
    if isinstance(key, tuple):
        if len(key) != 1:
            return NotImplemented

        key = key[0]

    if key.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.int,
        torch.long,
    ):
        return NotImplemented

    return PackedTensor(tensor[key.data], key.offsets)
