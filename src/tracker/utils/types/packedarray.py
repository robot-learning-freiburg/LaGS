# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import List, Optional, Self

import numpy as np
from torch.utils.data._utils.collate import default_collate_fn_map

from .metadict import MetaDict


class PackedArray:
    """
    Class for storing variable-length NumPy NDArrays (e.g., point clouds) in a
    single packed array. The stored arrays must have the same shape except for
    the first dimension.

    This is intended for batch-processing of point clouds and bounding boxes,
    when PackedTensor does not work (e.g., for string types).
    """

    def __init__(self, data: np.ndarray, offsets: Optional[np.ndarray] = None):
        if offsets is None:
            offsets = np.array([0, data.shape[0]], dtype=np.int64)

        self.data = data
        self.offsets = offsets

    @classmethod
    def from_array(cls, tensors: List[np.ndarray]) -> Self:
        offsets = [0] + [t.shape[0] for t in tensors]
        offsets = np.array(offsets, dtype=np.int64)
        offsets = np.cumsum(offsets, axis=0)

        data = np.concatenate(tensors, axis=0)

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

    def copy(self, *args, **kwargs) -> Self:
        return PackedArray(
            data=self.data.clone(*args, **kwargs),
            offsets=self.offsets.clone(*args, **kwargs),
        )

    @property
    def indices(self) -> np.ndarray:
        n = np.diff(self.offsets)
        indices = np.arange(0, self.batch_size, dtype=np.int64)
        indices = np.repeat(indices, n)

        return indices

    def get(self, index) -> np.ndarray:
        return self.data[self.offsets[index] : self.offsets[index + 1]]

    def padded(
        self,
        value: float = 0,
        with_mask: bool = False,
        with_counts: bool = False,
    ) -> MetaDict:
        """
        Convert the packed array to a padded array.

        Args:
            value (float): Value to pad with.
            with_mask (bool): Whether to return a mask indicating the
                valid entries in the padded array. If True, the mask will
                be a boolean array, where True indicates a valid entry and
                False indicates a padded entry. For a padded array of shape
                (batch_size, n, ...), the mask will be of shape (batch_size, n).
            with_counts (bool): Whether to return the counts of valid entries
                for each batch. If True, the counts will be a 1D array of
                shape (batch_size,).

        Returns:
            np.ndarray: Padded array.
            np.ndarray: Mask for valid entries (if with_mask is True).
        """
        counts = np.diff(self.offsets)
        n = counts.max()

        padded = np.full(
            (self.batch_size, n) + self.data.shape[1:],
            fill_value=value,
            dtype=self.data.dtype,
        )

        if with_mask:
            mask = np.full(
                (self.batch_size, n),
                fill_value=False,
                dtype=bool,
            )

        for i in range(self.batch_size):
            start = self.offsets[i]
            end = self.offsets[i + 1]

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

    def unbind(self) -> List[np.ndarray]:
        """
        Unbind the packed array into a list of arrays.

        Returns:
            List[np.ndarray]: List of arrays.
        """
        return [self.get(i) for i in range(self.batch_size)]


def _collate_fn(batch, *, collate_fn_map=None):
    # pylint: disable=unused-argument

    data = [x.data for x in batch]
    offsets = [x.offsets for x in batch]

    elem = offsets[0]

    batched_offsets = [np.array([0], dtype=elem.dtype)]
    for offs in offsets:
        assert offs[0] == 0

        batched_offsets.append(offs[1:] + batched_offsets[-1][-1])

    return PackedArray(
        data=np.concatenate(data, axis=0),
        offsets=np.concatenate(batched_offsets, axis=0),
    )


default_collate_fn_map[PackedArray] = _collate_fn
