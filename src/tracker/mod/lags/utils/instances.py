# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * Detectron2 (https://github.com/facebookresearch/detectron2), Copyright (c) Facebook, Inc., licensed under Apache-2.0 (via MOTR, TrackOcc).
# See the LICENSES/ directory for full license texts.

import copy
import itertools
from typing import Any, Dict, List, Union

import torch


class Instances:
    def __init__(self, **kwargs: Any):
        """
        Args:
            kwargs: fields to add to this `Instances`.
        """
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError(f"Cannot find field '{name}' in the given Instances!")

        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), f"Adding a field of length {data_len} to a Instances of length {len(self)}"
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields
        Modifying the returned dict will modify this instance.
        """
        return self._fields

    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """

        def to(v):
            return v.to(*args, **kwargs) if hasattr(v, "to") else v

        return Instances(**{k: to(v) for k, v in self._fields.items()})

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.
        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if isinstance(item, int):
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")

            item = slice(item, None, len(self))

        return Instances(**{k: v[item] for k, v in self._fields.items()})

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()

        raise NotImplementedError("Empty Instances does not support __len__!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        ret = Instances()
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError(f"Unsupported type {type(v0)} for concatenation")
            ret.set(k, values)
        return ret

    def clone(self):
        def clone(v):
            return v.clone() if hasattr(v, "clone") else copy.deepcopy(v)

        return Instances(**{k: clone(v) for k, v in self._fields.items()})

    def detach(self):
        def detach(v):
            return v.detach() if hasattr(v, "clone") else copy.deepcopy(v)

        return Instances(**{k: detach(v) for k, v in self._fields.items()})

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += f"num_instances={len(self)}, "
        s += f"fields=[{', '.join((f'{k}: {v}' for k, v in self._fields.items()))}])"
        return s

    __repr__ = __str__
