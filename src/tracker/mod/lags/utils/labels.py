# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Mapping, Sequence

import torch


def build_label_map(
    source: Sequence[str], target: Sequence[str] | Mapping[str, int], default: int = -1
) -> torch.Tensor:
    if isinstance(target, Mapping):
        # if target is a mapping, we need to map source labels to target indices
        def index(n: str) -> int:
            return target.get(n, default)

    else:
        # if target is a sequence, we can directly use the index method
        # but we need to handle the case where the label is not found
        # pylint: disable=unused-variable
        def index(n: str) -> int:
            return target.index(n) if n in target else default

    clsmap = [index(n) for n in source]
    clsmap = torch.tensor(clsmap, dtype=torch.long)

    return clsmap


def build_class_list(target: Sequence[str], subset: Sequence[str]) -> torch.Tensor:
    """
    Build a class list for the target classes based on the subset.

    Args:
        target (Sequence[str]): The full list of target classes.
        subset (Sequence[str]): The subset of classes to include.

    Returns:
        torch.Tensor: A tensor containing the indices of the subset classes in the target.
    """
    cls_list = [target.index(n) for n in subset]
    cls_list = torch.tensor(cls_list, dtype=torch.long)

    return cls_list
