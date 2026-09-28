# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


def upcast(tensor: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Ensure that the tensor is at least in the specified precision, converting
    it if necessary.

    Args:
        tensor (torch.Tensor): The input tensor.
        dtype (torch.dtype): The target data type. Default is torch.float32.

    Returns:
        torch.Tensor: The tensor converted to the specified precision if necessary.
    """
    src_bits = torch.finfo(tensor.dtype).bits
    tgt_bits = torch.finfo(dtype).bits

    if src_bits < tgt_bits:
        return tensor.to(dtype=dtype)

    return tensor
