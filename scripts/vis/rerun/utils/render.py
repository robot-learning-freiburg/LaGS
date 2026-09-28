# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rerun rendering of sensors, panoptic occupancy and annotations.

Adapted from ``test/visualize_occupancy_panoptic_rerun.py`` (kept as its own
framework module so the entrypoints stay thin). Differences from the original:

- point clouds are always coloured by intensity (no per-ring debug split);
- free-space voxels are never rendered;
- occupancy voxels are grouped into per-class entities
  (``<root>/stuff/<class>/voxels`` and ``<root>/things/<class>/voxels``), with
  all instance-less "thing" voxels collected under ``<root>/things/invalid``;
- an ``entity_root`` lets the same renderer draw ground truth and predictions
  into separate, independently toggle-able sub-trees;
- an optional ``visual_filter`` callable augments the observation mask.
"""

import numpy as np
import rerun as rr
import torch

from scripts.common.datasets import color_for_class

# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


def visualize_cameras(sample):
    num_cameras = sample.images.data.shape[0]

    # Precompute inverse of the reference ego pose so we can go world -> ego
    # frame. Only available when LiDAR has been loaded.
    ref_pose_inv = None
    if hasattr(sample, "points"):
        ref_pose_inv = np.linalg.inv(sample.points.meta.transforms.pose.matrix.numpy())

    for i in range(num_cameras):
        image = sample.images.data[i]
        camera_name = sample.images.meta.channel[i]

        intrinsic = sample.images.meta.transforms.intrinsic[i]
        intrinsic_matrix = intrinsic.matrix.numpy()[:3, :3]

        extrinsic = sample.images.meta.transforms.extrinsic[i]
        extrinsic_matrix = extrinsic.matrix.numpy()

        img_transforms = sample.images.meta.transforms
        if ref_pose_inv is not None:
            # camera -> world -> reference ego frame
            cam_pose = img_transforms.pose[i].matrix.numpy()
            cam_to_ego = ref_pose_inv @ cam_pose @ extrinsic_matrix
        else:
            cam_to_ego = extrinsic_matrix

        rr.log(
            f"cameras/{camera_name}",
            rr.Transform3D(
                mat3x3=cam_to_ego[:3, :3],
                translation=cam_to_ego[:3, 3],
            ),
        )
        rr.log(f"cameras/{camera_name}", rr.Pinhole(image_from_camera=intrinsic_matrix))
        rr.log(f"cameras/{camera_name}", rr.Image(image.permute(1, 2, 0).numpy()))


# ---------------------------------------------------------------------------
# Point cloud (intensity only)
# ---------------------------------------------------------------------------


def visualize_pointcloud(sample):
    if not hasattr(sample, "points"):
        return

    points = sample.points.data.get(0)

    if "sensor_id" in sample.points:
        sensor_ids = sample.points.sensor_id.get(0).numpy()
        sensor_names = list(sample.points.meta.channel)
    else:
        sensor_ids = np.zeros(len(points), dtype=np.int64)
        sensor_names = ["pointcloud"]

    intensity = points[:, 3].numpy()
    intensity_norm = (
        (intensity / intensity.max() * 255).astype(np.uint8)
        if intensity.max() > 0
        else intensity.astype(np.uint8)
    )
    colors = np.stack([intensity_norm, intensity_norm, intensity_norm], axis=1)

    for idx, name in enumerate(sensor_names):
        mask = sensor_ids == idx
        if not mask.any():
            continue
        rr.log(
            f"pointcloud/{name}",
            rr.Points3D(
                positions=points[mask, :3].numpy(),
                colors=colors[mask],
                radii=0.05,
            ),
        )


# ---------------------------------------------------------------------------
# Occupancy
# ---------------------------------------------------------------------------


def _resolve_mask_filter(occupancy, mask_type):
    """Return a ``[Z, Y, X]`` bool mask or ``None``. Computes 'any' from all masks."""
    if mask_type == "none" or "masks" not in occupancy:
        return None
    masks = occupancy.masks
    if mask_type == "any":
        result = None
        for v in masks.values():
            result = v if result is None else (result | v)
        return result
    return masks.get(mask_type, None)


def _voxel_centers(idx_xyz, voxel_size, voxel_range):
    """World-frame centres for ``[X, Y, Z]`` voxel indices."""
    return (idx_xyz.float() * voxel_size + voxel_range[:3] + voxel_size / 2).numpy()


def _log_voxels(entity, idx_xyz, colors, voxel_size, voxel_range, labels=None):
    if len(idx_xyz) == 0:
        return
    kwargs = {} if labels is None else {"labels": labels}
    rr.log(
        entity,
        rr.Boxes3D(
            centers=_voxel_centers(idx_xyz, voxel_size, voxel_range),
            sizes=torch.tile(voxel_size, (len(idx_xyz), 1)).numpy() - 0.01,
            colors=colors,
            fill_mode="solid",
            **kwargs,
        ),
    )


def visualize_occupancy(
    occupancy,
    labels,
    voxel_size,
    voxel_range,
    entity_root,
    mask_type="none",
    visual_filter=None,
):
    """Render panoptic occupancy as per-class voxel entities (no free space).

    ``occupancy`` is a mapping with ``semantics`` ([Z, Y, X]) and optionally
    ``instance_ids`` and ``masks``. When ``visual_filter(sem_xyz, occupancy,
    labels) -> mask_xyz`` ([X, Y, Z] bool) is given it fully determines the
    region to show (it may read ``occupancy.masks`` itself); otherwise the raw
    observation mask (``mask_type``) gates the scene.
    """
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if "semantics" not in occupancy:
        return

    semantics = occupancy.semantics.permute(2, 1, 0)  # [X, Y, Z]
    instances = (
        occupancy.instance_ids.permute(2, 1, 0) if "instance_ids" in occupancy else None
    )

    if visual_filter is not None:
        region = visual_filter(semantics, occupancy, labels)
    else:
        base_mask = _resolve_mask_filter(occupancy, mask_type)
        if base_mask is not None:
            region = base_mask.permute(2, 1, 0)
        else:
            region = torch.ones_like(semantics, dtype=torch.bool)

    ignore = labels.ignore_index
    free_index = labels.all.index("free") if "free" in labels.all else None

    valid = (semantics != ignore) & region
    occupied = valid & (semantics != free_index) if free_index is not None else valid

    thing_names = set(labels.instance) if "instance" in labels else set()
    is_thing_class = torch.zeros_like(semantics, dtype=torch.bool)
    for name in thing_names:
        is_thing_class |= semantics == labels.all.index(name)

    # Clear stale per-class entities from the previous frame (a class present in
    # frame N but absent in N+1 would otherwise linger).
    rr.log(f"{entity_root}/things", rr.Clear(recursive=True))
    rr.log(f"{entity_root}/stuff", rr.Clear(recursive=True))

    if instances is not None:
        valid_thing = occupied & is_thing_class & (instances != -1)
        invalid_thing = occupied & is_thing_class & (instances == -1)
    else:
        valid_thing = torch.zeros_like(occupied)
        invalid_thing = torch.zeros_like(occupied)

    # Valid "thing" voxels: grouped by semantic class, coloured per instance id.
    thing_idx = torch.nonzero(valid_thing, as_tuple=False)
    if len(thing_idx):
        thing_cls = semantics[thing_idx[:, 0], thing_idx[:, 1], thing_idx[:, 2]]
        thing_iid = instances[thing_idx[:, 0], thing_idx[:, 1], thing_idx[:, 2]]

        id_to_color = {}
        for uid in torch.unique(thing_iid):
            torch.manual_seed(hash(uid.item()) % (2**32))
            id_to_color[uid.item()] = torch.rand(3)

        for cls_id in torch.unique(thing_cls).tolist():
            cls_mask = thing_cls == cls_id
            idx_c = thing_idx[cls_mask]
            iid_c = thing_iid[cls_mask].numpy()
            colors = np.zeros((len(idx_c), 3), dtype=np.float32)
            for uid, col in id_to_color.items():
                colors[iid_c == uid] = col.numpy()
            name = labels.all[cls_id]
            _log_voxels(
                f"{entity_root}/things/{name}/voxels",
                idx_c,
                colors,
                voxel_size,
                voxel_range,
            )

    # Instance-less "thing" voxels: one shared node, coloured red.
    invalid_idx = torch.nonzero(invalid_thing, as_tuple=False)
    if len(invalid_idx):
        colors = np.tile([1.0, 0.0, 0.0], (len(invalid_idx), 1)).astype(np.float32)
        _log_voxels(
            f"{entity_root}/things/invalid/voxels",
            invalid_idx,
            colors,
            voxel_size,
            voxel_range,
        )

    # "Stuff" voxels: occupied, non-thing-class, grouped by semantic class.
    stuff = occupied & ~is_thing_class
    stuff_idx = torch.nonzero(stuff, as_tuple=False)
    if len(stuff_idx):
        stuff_cls = semantics[stuff_idx[:, 0], stuff_idx[:, 1], stuff_idx[:, 2]]
        for cls_id in torch.unique(stuff_cls).tolist():
            idx_c = stuff_idx[stuff_cls == cls_id]
            name = labels.all[cls_id]
            color = np.array(color_for_class(name), dtype=np.float32)
            colors = np.tile(color, (len(idx_c), 1))
            _log_voxels(
                f"{entity_root}/stuff/{name}/voxels",
                idx_c,
                colors,
                voxel_size,
                voxel_range,
            )


def visualize_occupancy_bounds(voxel_range, entity="ground_truth/occupancy/bounds"):
    """Draw a wireframe box around the full occupancy volume."""
    mins = voxel_range[:3]
    maxs = voxel_range[3:]
    center = ((mins + maxs) / 2).numpy()
    size = (maxs - mins).numpy()
    rr.log(
        entity,
        rr.Boxes3D(
            centers=center.reshape(1, 3),
            sizes=size.reshape(1, 3),
            colors=np.array([[255, 255, 0]], dtype=np.uint8),
            fill_mode="majorwireframe",
        ),
    )


# ---------------------------------------------------------------------------
# Boxes + trajectories
# ---------------------------------------------------------------------------


def visualize_boxes(sample, trajectory_history, entity_root="annotations"):
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    if "boxes" not in sample.labels:
        rr.log(f"{entity_root}/boxes/current", rr.Clear(recursive=False))
        rr.log(f"{entity_root}/past_trajectories/paths", rr.Clear(recursive=False))
        rr.log(f"{entity_root}/future_trajectories/paths", rr.Clear(recursive=False))
        rr.log(
            f"{entity_root}/future_trajectories/endpoints", rr.Clear(recursive=False)
        )
        return

    boxes = sample.labels.boxes.get(0)
    centers = boxes[:, :3].numpy()
    sizes = boxes[:, 3:6].numpy()
    rotations = boxes[:, 6].numpy()

    # Convert yaw to quaternion (rotation around Z-axis)
    adjusted_rotations = rotations - np.pi / 2
    cos_half = np.cos(adjusted_rotations / 2)
    sin_half = np.sin(adjusted_rotations / 2)
    quaternions = np.stack(
        [np.zeros_like(rotations), np.zeros_like(rotations), sin_half, cos_half],
        axis=1,
    )

    instance_ids = sample.labels.instance_ids.get(0).numpy()

    # Per-box class names (aligned 1:1 with boxes/instance_ids). Fall back to the
    # class id, or "?" if neither is available.
    if "class_names" in sample.labels:
        class_names = [str(n) for n in sample.labels.class_names.get(0)]
    elif "class_ids" in sample.labels:
        class_names = [str(int(c)) for c in sample.labels.class_ids.get(0).numpy()]
    else:
        class_names = ["?"] * len(instance_ids)

    box_colors = []
    for iid in instance_ids:
        torch.manual_seed(hash(int(iid)) % (2**32))
        box_colors.append(torch.rand(3).numpy())
    box_colors = np.array(box_colors)

    rr.log(
        f"{entity_root}/boxes/current",
        rr.Boxes3D(
            centers=centers,
            sizes=sizes,
            quaternions=quaternions,
            colors=box_colors,
            labels=[f"{name} · {iid}" for name, iid in zip(class_names, instance_ids)],
        ),
    )

    # Past trajectories are ego-motion compensated using each frame's ego pose,
    # which comes from the LiDAR meta; skip them when LiDAR is not loaded.
    if hasattr(sample, "points"):
        max_history_length = 20
        current_ego_pose = sample.points.meta.transforms.pose

        for i, iid in enumerate(instance_ids):
            iid_int = int(iid)
            trajectory_history.setdefault(iid_int, []).append(
                {"position": centers[i].copy(), "ego_pose": current_ego_pose}
            )
            if len(trajectory_history[iid_int]) > max_history_length:
                trajectory_history[iid_int] = trajectory_history[iid_int][
                    -max_history_length:
                ]

        past_trajectory_strips = []
        past_trajectory_colors = []
        for i, iid in enumerate(instance_ids):
            iid_int = int(iid)
            history = trajectory_history.get(iid_int, [])
            if len(history) > 1:
                compensated = []
                for hist_entry in history:
                    past_pos_homo = np.append(hist_entry["position"], 1.0)
                    position_world = (
                        hist_entry["ego_pose"].matrix.numpy() @ past_pos_homo
                    )
                    position_current_ego = (
                        current_ego_pose.inv.matrix.numpy() @ position_world
                    )
                    compensated.append(position_current_ego[:3])
                past_trajectory_strips.append(np.array(compensated))
                past_trajectory_colors.append(box_colors[i])

        if past_trajectory_strips:
            rr.log(
                f"{entity_root}/past_trajectories/paths",
                rr.LineStrips3D(
                    strips=past_trajectory_strips,
                    colors=past_trajectory_colors,
                    radii=0.15,
                ),
            )
        else:
            rr.log(f"{entity_root}/past_trajectories/paths", rr.Clear(recursive=False))
    else:
        rr.log(f"{entity_root}/past_trajectories/paths", rr.Clear(recursive=False))

    if "trajectories" not in sample.labels:
        return

    trajectories = sample.labels.trajectories
    traj_centers = trajectories.center.get(0).numpy()
    traj_valid = trajectories.valid.get(0).numpy()

    trajectory_strips = []
    trajectory_colors = []
    endpoint_positions = []
    endpoint_colors = []

    for i in range(traj_centers.shape[0]):
        n_valid = traj_valid[i].sum()
        if n_valid > 1:
            trajectory_points = traj_centers[i, :n_valid]
            trajectory_strips.append(trajectory_points)
            trajectory_colors.append(box_colors[i])
            endpoint_positions.append(trajectory_points[-1])
            endpoint_colors.append(box_colors[i])

    if trajectory_strips:
        rr.log(
            f"{entity_root}/future_trajectories/paths",
            rr.LineStrips3D(
                strips=trajectory_strips, colors=trajectory_colors, radii=0.15
            ),
        )
        rr.log(
            f"{entity_root}/future_trajectories/endpoints",
            rr.Points3D(
                positions=np.array(endpoint_positions),
                colors=np.array(endpoint_colors),
                radii=0.3,
            ),
        )
    else:
        rr.log(f"{entity_root}/future_trajectories/paths", rr.Clear(recursive=False))
        rr.log(
            f"{entity_root}/future_trajectories/endpoints", rr.Clear(recursive=False)
        )
