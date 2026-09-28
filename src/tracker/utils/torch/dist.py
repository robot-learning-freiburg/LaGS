# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import torch
from torch import distributed as dist


def all_reduce(
    tensor: torch.Tensor,
    op: dist.ReduceOp.RedOpType,
    group: Any | None = None,
) -> torch.Tensor:
    """
    Perform all-reduce operation on the given tensor.

    This function is a wrapper around the PyTorch distributed all_reduce
    function. It checks if the code is running in a distributed environment
    and, if so, performs the all-reduce operation on the input tensor using the
    specified operation and group. If the code is not running in a distributed
    environment, it simply returns the input tensor without any modification.

    Args:
        tensor (torch.Tensor): The tensor to be reduced.
        op (dist.ReduceOp.RedOpType): The reduction operation to be applied.
        group (Any | None, optional): The group to perform the operation on. Defaults to None.

    Returns:
        torch.Tensor: The reduced tensor.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    dist.all_reduce(tensor, op, group, async_op=False)
    return tensor
