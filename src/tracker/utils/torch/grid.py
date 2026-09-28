# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


def coordinate_grid_2d(b, h, w, device=None):
    """
    Builds a coordinate grid of shape (b, 2, h, w) where b is the batch size
    and (h, w) the 2D dimensions of the grid. The second dimension contains the
    (x, y) coordinates ranging from (0, 0) to (w, h).
    """
    ys = torch.arange(0, h, device=device)
    xs = torch.arange(0, w, device=device)

    ys, xs = torch.meshgrid((ys, xs), indexing="ij")

    ys = ys.view(1, h, w).repeat(b, 1, 1)
    xs = xs.view(1, h, w).repeat(b, 1, 1)

    return torch.stack((xs, ys), dim=1)
