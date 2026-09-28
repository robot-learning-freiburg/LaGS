# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Sequence

import torch


def stack_optional(
    tensors: Sequence[torch.Tensor | None], dim: int = 0
) -> torch.Tensor | None:
    if any(x is None for x in tensors):
        assert all(x is None for x in tensors)
        return None

    return torch.stack(tensors, dim=dim)
