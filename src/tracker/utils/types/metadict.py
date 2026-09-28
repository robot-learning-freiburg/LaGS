# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import MutableMapping
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import numpy as np


class MetaDict(SimpleNamespace, MutableMapping):
    """
    Dictionary allowing access to members via dot notation (meaning d.key
    returns d['key']).

    This is essentially a wrapper over types.SimpleNamespace to make it
    compatible with collections.abc.Mapping. We do this so that we can access
    entries more ergonomically via dot notation, but retain the automatic
    handling for mappings in pytorch (collate, memory pinning) and lightning
    (moving tensors to devices).
    """

    def __init__(self, data: Optional[Mapping[str, Any]] = None):
        super().__init__(**(data if data is not None else {}))

    def __contains__(self, x):
        return x in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, k):
        return self.__dict__[k]

    def __setitem__(self, k, v):
        self.__dict__[k] = v

    def __delitem__(self, k):
        del self.__dict__[k]

    def __or__(self, other):
        assert isinstance(other, Mapping)

        new = MetaDict(self)
        new |= other

        return new

    def __ior__(self, other):
        assert isinstance(other, Mapping)

        for key, val in other.items():
            self[key] = val

        return self

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def pop(self, key, default=None):
        return self.__dict__.pop(key, default)

    def __array__(self, dtype=None, copy=None):
        if dtype is not None and dtype != np.dtype("O"):
            raise ValueError(f"cannot convert MetaDict to dtype '{dtype}'")

        if copy:
            raise ValueError(
                "copying of MetaDict may be ambiguous and is not supported in __array__()"
            )

        # Create a zero-dimensional object array with 'self' as member.
        #
        # The two-step process is required because namespace interfers and has
        # some bogus "automatic" conversion. So instead, we first force the
        # creation a new zero-dimensional object-type array, which we then fill
        # with our object reference. Note: This is essentially why this
        # __array__ implementation exists.
        a = np.array(None, dtype=object)
        a.put(0, super())

        return a

    def copy(self):
        """
        Returns a shallow copy of the MetaDict.
        """
        return MetaDict(self.__dict__.copy())
