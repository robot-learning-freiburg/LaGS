# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render panoptic occupancy as batched voxel meshes in viser.

Draws two batched cube meshes -- ``things_batched`` (instance voxels, coloured per
instance id) and ``stuff_batched`` (non-instance occupied voxels, coloured per
semantic class). Two policies share one core:

- :func:`render_occupancy` (predictions): instance voxels are always shown; "stuff"
  is gated by the per-dataset visual filter (or a single ``base_mask``).
- :func:`render_occupancy_gt` (ground truth): an optional ``mask_type`` hard-masks
  both things and stuff, with no heuristic.
"""

import numpy as np
import torch

from scripts.common.datasets import color_for_class
from scripts.vis.common.heuristics import visual_filter_mask
from scripts.vis.viser.utils import mesh
from tracker.utils.types import MetaDict

_THINGS = "occupancy/things"
_STUFF = "occupancy/stuff"


def instance_color(instance_id) -> np.ndarray:
    """Deterministic RGB (in [0, 1]) for an instance id, stable across frames.

    Shared by the voxel and box/trajectory renderers so an instance keeps one
    colour everywhere in the figure.
    """
    torch.manual_seed(hash(int(instance_id)) % (2**32))
    return torch.rand(3).numpy()


def _remove(server, name):
    """Remove a scene entity by name, ignoring it if absent."""
    try:
        server.scene.remove_by_name(name)
    except Exception:  # pylint: disable=broad-exception-caught
        pass


def _region(semantics_xyz, masks, labels, dataset, visual_filter, base_mask):
    """Bool ``[X, Y, Z]`` region gating "stuff" voxels (prediction policy)."""
    if visual_filter:
        occ = MetaDict()
        occ.masks = MetaDict(masks or {})
        return visual_filter_mask(semantics_xyz, occ, labels, dataset)
    if base_mask and base_mask != "none" and masks and base_mask in masks:
        return masks[base_mask].permute(2, 1, 0)
    return torch.ones_like(semantics_xyz, dtype=torch.bool)


def _named_mask(masks, mask_type):
    """Named observation mask permuted to ``[X, Y, Z]`` ('any' = camera OR lidar)."""
    if not masks:
        return None
    if mask_type == "any":
        camera, lidar = masks.get("camera"), masks.get("lidar")
        if camera is None or lidar is None:
            return None
        return (camera | lidar).permute(2, 1, 0)
    mask = masks.get(mask_type)
    return None if mask is None else mask.permute(2, 1, 0)


def _instance_colors(thing_iid):
    """Per-voxel instance colours ([N, 3]) via :func:`instance_color`."""
    colors = np.zeros((len(thing_iid), 3), dtype=np.float32)
    for uid in torch.unique(thing_iid):
        colors[(thing_iid == uid).numpy()] = instance_color(uid)
    return colors


def _add_voxels(server, name, idx_xyz, colors, voxel_size, voxel_range, scene_scale):
    """Add (or clear) a batched cube mesh for voxel indices ``idx_xyz`` ([N, 3])."""
    if len(idx_xyz) == 0:
        _remove(server, name)
        return
    centers = (
        idx_xyz.float() * voxel_size + voxel_range[:3] + voxel_size / 2
    ) * scene_scale
    vertices, faces = mesh.create_cube_mesh_geometry(voxel_size.numpy() * scene_scale)
    rotations = np.tile([1.0, 0.0, 0.0, 0.0], (len(idx_xyz), 1))
    server.scene.add_batched_meshes_simple(
        name=name,
        vertices=vertices,
        faces=faces,
        batched_positions=centers.numpy(),
        batched_wxyzs=rotations,
        batched_colors=np.asarray(colors, dtype=np.float32),
        flat_shading=True,
        lod="off",
    )


def _draw(
    server,
    semantics,
    instances,
    labels,
    stuff_region,
    grid,
    scene_scale,
    *,
    show_things=True,
    show_stuff=True,
):
    """Draw thing/stuff voxel meshes; ``semantics``/``instances`` are ``[X, Y, Z]``."""
    # pylint: disable=too-many-locals,too-many-arguments
    voxel_size, voxel_range = grid

    if instances is not None:
        thing_idx = torch.nonzero(instances != -1, as_tuple=False)
        thing_iid = instances[thing_idx[:, 0], thing_idx[:, 1], thing_idx[:, 2]]
    else:
        thing_idx = torch.zeros((0, 3), dtype=torch.long)
        thing_iid = torch.zeros((0,), dtype=torch.long)
    thing_col = _instance_colors(thing_iid)

    valid = semantics != labels.ignore_index
    if "free" in labels.all:
        valid = valid & (semantics != labels.all.index("free"))
    valid = valid & stuff_region

    stuff_map = semantics.clone()
    if len(thing_idx):
        stuff_map[thing_idx[:, 0], thing_idx[:, 1], thing_idx[:, 2]] = -1
    stuff_map[~valid] = -1
    stuff_idx = torch.nonzero(stuff_map != -1, as_tuple=False)
    stuff_cls = semantics[stuff_idx[:, 0], stuff_idx[:, 1], stuff_idx[:, 2]]
    stuff_col = np.array(
        [color_for_class(labels.all[int(c)]) for c in stuff_cls], dtype=np.float32
    ).reshape(-1, 3)

    if show_things:
        _add_voxels(
            server, _THINGS, thing_idx, thing_col, voxel_size, voxel_range, scene_scale
        )
    else:
        _remove(server, _THINGS)
    if show_stuff:
        _add_voxels(
            server, _STUFF, stuff_idx, stuff_col, voxel_size, voxel_range, scene_scale
        )
    else:
        _remove(server, _STUFF)


def _grid(voxel_size, voxel_range):
    return (
        torch.as_tensor(np.asarray(voxel_size), dtype=torch.float32),
        torch.as_tensor(np.asarray(voxel_range), dtype=torch.float32),
    )


def render_occupancy(
    server,
    semantics,
    instances,
    masks,
    labels,
    dataset,
    *,
    voxel_size,
    voxel_range,
    scene_scale=0.1,
    visual_filter=True,
    base_mask="lidar",
):
    """Prediction occupancy: instances always shown, "stuff" gated by the filter.

    ``semantics`` / ``instances`` are ``[Z, Y, X]`` (``instances`` ``-1`` = none,
    or ``None``); ``masks`` maps mask name -> ``[Z, Y, X]`` bool.
    """
    # pylint: disable=too-many-arguments
    semantics = semantics.cpu().permute(2, 1, 0)  # [X, Y, Z]
    instances = None if instances is None else instances.cpu().permute(2, 1, 0)
    region = _region(semantics, masks, labels, dataset, visual_filter, base_mask)
    _draw(
        server,
        semantics,
        instances,
        labels,
        region,
        _grid(voxel_size, voxel_range),
        scene_scale,
    )


def render_occupancy_gt(
    server,
    semantics,
    instances,
    masks,
    labels,
    *,
    voxel_size,
    voxel_range,
    scene_scale=0.1,
    mask_type="none",
    show_things=True,
    show_stuff=True,
):
    """Ground-truth occupancy: ``mask_type`` hard-masks both things and stuff.

    ``mask_type`` is one of the observation masks ('camera'/'lidar'/'valid'/'any')
    or 'none' (show everything). ``show_things`` / ``show_stuff`` toggle either
    voxel layer. ``semantics`` / ``instances`` are ``[Z, Y, X]``.
    """
    # pylint: disable=too-many-arguments
    semantics = semantics.cpu().permute(2, 1, 0)  # [X, Y, Z]
    instances = None if instances is None else instances.cpu().permute(2, 1, 0)

    if mask_type != "none":
        mask = _named_mask(masks, mask_type)
        if mask is not None:
            semantics = torch.where(mask, semantics, torch.tensor(labels.ignore_index))
            if instances is not None:
                instances = torch.where(mask, instances, torch.tensor(-1))

    region = torch.ones_like(semantics, dtype=torch.bool)
    _draw(
        server,
        semantics,
        instances,
        labels,
        region,
        _grid(voxel_size, voxel_range),
        scene_scale,
        show_things=show_things,
        show_stuff=show_stuff,
    )
