# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Optional per-dataset "visual filter" for prediction voxels.

Renderer-agnostic (shared by the Rerun and viser occupancy viewers). Dense
predictions look noisy when shown in full, so we start from the LiDAR observation
mask (AND FoV when present) and OR back a hand-tuned set of class + height rules
that keep plausible ground/static surfaces the sensor did not directly observe.

All tensors are ``[X, Y, Z]`` (height = last axis), matching the renderers'
post-``permute(2, 1, 0)`` frame, so the ``below_N`` height slices are correct.
"""

import torch


def _mask_xyz(occupancy, name):
    """Fetch a named observation mask permuted to ``[X, Y, Z]``, or ``None``."""
    if "masks" not in occupancy:
        return None
    mask = occupancy.masks.get(name)
    return None if mask is None else mask.permute(2, 1, 0)


def _eq(semantics, labels, name):
    """``semantics == labels.all.index(name)`` or all-False if the class is absent."""
    if name not in labels.all:
        return torch.zeros_like(semantics, dtype=torch.bool)
    return semantics == labels.all.index(name)


def visual_filter_mask(semantics_xyz, occupancy, labels, dataset):
    """Return the ``[X, Y, Z]`` bool region to show for a (dense) prediction."""
    # pylint: disable=too-many-statements
    lidar = _mask_xyz(occupancy, "lidar")
    fov = _mask_xyz(occupancy, "fov")

    mask = (
        lidar.clone()
        if lidar is not None
        else torch.ones_like(semantics_xyz, dtype=torch.bool)
    )
    if fov is not None:
        mask = mask & fov

    below_10 = torch.zeros_like(semantics_xyz, dtype=torch.bool)
    below_10[:, :, :10] = True
    below_8 = torch.zeros_like(semantics_xyz, dtype=torch.bool)
    below_8[:, :, :8] = True
    below_6 = torch.zeros_like(semantics_xyz, dtype=torch.bool)
    below_6[:, :, :6] = True

    if dataset == "nuscenes":
        mask = mask | _eq(semantics_xyz, labels, "others")
        mask = mask | (_eq(semantics_xyz, labels, "driveable_surface") & below_8)
        mask = mask | _eq(semantics_xyz, labels, "other_flat")
        mask = mask | _eq(semantics_xyz, labels, "sidewalk")
        mask = mask | _eq(semantics_xyz, labels, "terrain")
        mask = mask | below_6
    elif dataset == "waymo":
        mask = mask | _eq(semantics_xyz, labels, "others")
        mask = mask | (_eq(semantics_xyz, labels, "pole") & below_10)
        mask = mask | _eq(semantics_xyz, labels, "construction_cone")
        mask = mask | _eq(semantics_xyz, labels, "bicycle")
        mask = mask | _eq(semantics_xyz, labels, "motorcycle")
        mask = mask | _eq(semantics_xyz, labels, "tree_trunk")
        mask = mask | (_eq(semantics_xyz, labels, "road") & below_8)
        mask = mask | _eq(semantics_xyz, labels, "walkable")
        mask = mask | below_6
        if fov is not None:
            mask = mask & fov
    # any other dataset: no tuned heuristic yet -> base mask only.

    return mask
