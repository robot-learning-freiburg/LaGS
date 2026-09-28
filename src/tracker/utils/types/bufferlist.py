# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * PyTorch (https://github.com/pytorch/pytorch), licensed under BSD-3-Clause.
# See the LICENSES/ directory for full license texts.

import operator
from collections import abc
from typing import Any, Iterable, Iterator, Optional, TypeVar, overload

import torch
from torch import nn
from typing_extensions import Self

T = TypeVar("T", bound=nn.Module)


class BufferList(nn.Module):
    """
    Holds module buffers in a list.

    Adapted from torch.nn.ParameterList.
    """

    def __init__(self, values: Optional[Iterable[Any]] = None, persistent=True) -> None:
        super().__init__()

        self._size = 0
        self._persistent = persistent

        if values is not None:
            self += values

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)

        if not -len(self) <= idx < len(self):
            raise IndexError(f"index {idx} is out of range")

        if idx < 0:
            idx += len(self)

        return str(idx)

    @overload
    def __getitem__(self, idx: int) -> Any: ...

    @overload
    def __getitem__(self: T, idx: slice) -> T: ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))

            out = self.__class__()
            for i in range(start, stop, step):
                out.append(self[i])

            return out

        idx = self._get_abs_string_index(idx)
        return getattr(self, idx)

    def __setitem__(self, idx: int, buffer: Any) -> None:
        idx = self._get_abs_string_index(idx)

        return self.register_buffer(idx, buffer, persistent=self._persistent)

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Any]:
        return iter(self[i] for i in range(len(self)))

    def __iadd__(self, buffers: Iterable[Any]) -> Self:
        return self.extend(buffers)

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self, value: Any) -> Self:
        self._size += 1
        self[self._size - 1] = value

        return self

    def extend(self, values: Iterable[Any]) -> Self:
        if not isinstance(values, abc.Iterable) or isinstance(values, torch.Tensor):
            raise TypeError(
                "BufferList.extend should be called with an "
                "iterable, but got " + type(values).__name__
            )

        for value in values:
            self.append(value)

        return self

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in enumerate(self):
            if isinstance(p, torch.Tensor):
                size = "x".join(str(size) for size in p.size())

                if p.device.type in ["cuda"]:
                    device = f" ({p.device})"
                else:
                    device = ""

                parastr = f"Buffer containing: [{p.dtype} of size {size}{device}]"
                child_lines.append("  (" + str(k) + "): " + parastr)
            else:
                child_lines.append(
                    "  (" + str(k) + "): Object of type: " + type(p).__name__
                )

        tmpstr = "\n".join(child_lines)
        return tmpstr

    def __call__(self, *args, **kwargs):
        raise RuntimeError("BufferList should not be called.")

    def forward(self, *args, **kwargs):
        raise RuntimeError("BufferList should not be called.")
