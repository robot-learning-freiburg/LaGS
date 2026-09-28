# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import itertools
from collections.abc import Sequence
from typing import Any

from torch.utils.data._utils.collate import default_collate_fn_map


class BatchAsList(Sequence):
    """
    A class to force batching the wrapped data as list.

    This class itself behaves as if it were a single-element list to ensure
    access before and after batching can be done in a similar way.
    """

    def __init__(self, data: Any):
        self.data = [data]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.data)})"

    def __contains__(self, elem):
        return elem in self.data

    def __iter__(self):
        return iter(self.data)

    def __reversed__(self):
        return reversed(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __delitem__(self, index):
        del self.data[index]


def _collate_fn(batch, *, collate_fn_map=None):
    # pylint: disable=unused-argument

    return list(itertools.chain(*[x.data for x in batch]))


default_collate_fn_map[BatchAsList] = _collate_fn
