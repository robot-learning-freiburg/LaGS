# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate_fn_map

from ... import utils


class MetaArray:
    def __init__(self, data):
        self.data = np.asarray(data, dtype=object)

    @property
    def shape(self):
        return self.data.shape

    def __getitem__(self, index) -> Any:
        if not isinstance(index, tuple):
            index = (index,)

        ij, k = index[: self.data.ndim], index[self.data.ndim :]

        # check if we are indexing the object exactly
        if len(ij) == self.data.ndim and not any(isinstance(x, slice) for x in ij):
            return self.data[*ij]

        # check if we are creating a view
        if len(k) == 0:
            return type(self)(self.data[*ij])

        # slicing across objects
        if any(isinstance(x, slice) for x in ij):

            @np.vectorize(otypes="O")
            def index_elem(elem):
                return elem[*k]

            data = index_elem(self.data[*ij])
            return type(self)(data)

        return self.data[*ij][*k]

    def __setitem__(self, index, value) -> Any:
        if not isinstance(index, tuple):
            index = (index,)

        ij, k = index[: self.data.ndim], index[self.data.ndim :]

        if len(k) == 0:
            self.data[*ij] = value
        else:
            self.data[*ij][k] = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.data)})"

    def map(self, fn, *args, **kwargs):
        @np.vectorize(otypes="O")
        def apply(data):
            return fn(data, *args, **kwargs)

        return type(self)(apply(self.data, *args, **kwargs))

    def map_(self, fn, *args, **kwargs):
        @np.vectorize(otypes="O")
        def apply(data):
            return fn(data, *args, **kwargs)

        self.data = apply(self.data, *args, **kwargs)

    def reshape(self, *args, **kwargs):
        return type(self)(self.data.reshape(*args, **kwargs))

    @classmethod
    def stack(cls, arrays, dim=0):
        return cls(np.stack([a.data for a in arrays], axis=dim))


class TensorArray(MetaArray):
    def __init__(self, data, device=None, dtype=None):
        data = np.asarray(data, dtype=object)

        n = np.prod(data.shape)

        if device is None:
            device = data.flat[0].device if n > 0 else torch.get_default_device()

        if dtype is None:
            dtype = data.flat[0].dtype if n > 0 else torch.get_default_dtype()

        assert all(x.dtype == dtype for x in data.flat)
        assert all(x.device == device for x in data.flat)

        super().__init__(data)

        self._device = device
        self._dtype = dtype

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    def to(self, *args, **kwargs):
        to = utils.torch.parse_to(*args, **kwargs)

        if to.device is not None:
            self._device = to.device

        if to.dtype is not None:
            self._dtype = to.dtype

        def _apply(tensor):
            return tensor.to(*args, **kwargs)

        data = np.vectorize(_apply, otypes="O")(self.data)
        return TensorArray(data)

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    def collect(self):
        """
        Collect all tensors into a single tensor of shape (*self.shape,
        *tensor_shape).
        """
        # flatten the object array to a list of tensors
        flat = list(self.data.flat)

        elem_shape = flat[0].shape
        assert all(x.shape == elem_shape for x in flat)

        # Stack them and reshape to the full shape
        return torch.stack(flat).view(*self.shape, *elem_shape)


def _collate_fn(batch, *, collate_fn_map=None):
    # pylint: disable=unused-argument
    return type(batch[0]).stack(batch, dim=0)


default_collate_fn_map[MetaArray] = _collate_fn
default_collate_fn_map[TensorArray] = _collate_fn
