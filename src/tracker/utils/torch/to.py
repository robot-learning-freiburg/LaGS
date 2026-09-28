# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import asdict, dataclass
from typing import Optional

import torch


@dataclass
class ToArgs:
    device: Optional[torch.device]
    dtype: Optional[torch.dtype]
    non_blocking: bool
    memory_format: Optional[torch.memory_format]
    copy: Optional[bool]

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


def parse_to(*args, **kwargs):
    """
    Parse arguments for the .to() function of tensor-like classes. Useful for
    implementing custom tensor-like classes.
    """
    # torch parse_to does not handle copy, so we deal with it here
    copy = kwargs.pop("copy", None)

    # pylint: disable=protected-access
    device, dtype, non_blocking, memory_format = torch._C._nn._parse_to(*args, **kwargs)

    return ToArgs(
        device=device,
        dtype=dtype,
        non_blocking=non_blocking,
        memory_format=memory_format,
        copy=copy,
    )
