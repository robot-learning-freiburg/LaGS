# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render detection boxes and instance trajectories in viser.

Complements the occupancy voxels with the vector annotations that make a panoptic
figure legible: wireframe bounding boxes, future trajectory splines (from the
dataset's forecasting steps) and ego-motion-compensated past trajectories. Every
element is coloured by :func:`scripts.vis.viser.utils.occupancy.instance_color`, so
a track keeps one colour across voxels, boxes and trajectories.

Past trajectories need per-frame ego pose (``sample.points.meta.transforms.pose``),
which is only present when the dataset is built with lidar loading; without it,
:func:`render_past_trajectories` is a no-op.
"""

import numpy as np

from scripts.vis.viser.utils.occupancy import instance_color


def _yaw_to_quaternion(yaw_angles: np.ndarray) -> np.ndarray:
    """Yaw angles (rad) -> viser ``[w, x, y, z]`` quaternions about +z.

    The box convention is offset by -pi/2 relative to viser's, matching the
    original panoptic viewer.
    """
    adjusted = yaw_angles - np.pi / 2
    cos_half = np.cos(adjusted / 2)
    sin_half = np.sin(adjusted / 2)
    return np.stack(
        [cos_half, np.zeros_like(yaw_angles), np.zeros_like(yaw_angles), sin_half],
        axis=-1,
    )


def _rgb_uint8(instance_id) -> tuple:
    return tuple((instance_color(instance_id) * 255).astype(int).tolist())


def render_boxes(server, sample, *, scene_scale=0.1, name="annotations/boxes"):
    """Draw wireframe bounding boxes for the current frame, coloured per instance."""
    boxes = sample.labels.boxes.get(0)
    centers = boxes[:, :3].numpy() * scene_scale
    sizes = boxes[:, 3:6].numpy() * scene_scale
    quaternions = _yaw_to_quaternion(boxes[:, 6].numpy())
    instance_ids = sample.labels.instance_ids.get(0).numpy()

    for center, size, quat, iid in zip(centers, sizes, quaternions, instance_ids):
        server.scene.add_box(
            name=f"{name}/{int(iid)}",
            color=_rgb_uint8(iid),
            dimensions=tuple(size.tolist()),
            wxyz=tuple(quat.tolist()),
            position=tuple(center.tolist()),
            wireframe=True,
            opacity=0.8,
        )


def render_future_trajectories(
    server, sample, *, scene_scale=0.1, name="annotations/trajectories/future"
):
    """Draw forecast trajectory splines (box center -> future centers)."""
    if "trajectories" not in sample.labels:
        return
    boxes = sample.labels.boxes.get(0)
    starts = boxes[:, :3].numpy() * scene_scale
    instance_ids = sample.labels.instance_ids.get(0).numpy()
    centers = sample.labels.trajectories.center.get(0).numpy()  # [n, steps, 3]
    valid = sample.labels.trajectories.valid.get(0).numpy()  # [n, steps]

    for i, iid in enumerate(instance_ids):
        n_valid = int(valid[i].sum())
        if n_valid < 2:
            continue
        color = _rgb_uint8(iid)
        points = centers[i, :n_valid] * scene_scale
        full = np.vstack([starts[i].reshape(1, 3), points])
        server.scene.add_spline_catmull_rom(
            name=f"{name}/{int(iid)}",
            points=full,
            tension=0.0,
            line_width=3.0,
            color=color,
            segments=n_valid * 10,
        )
        server.scene.add_icosphere(
            name=f"{name}/endpoint/{int(iid)}",
            radius=0.15 * scene_scale,
            color=color,
            position=tuple(points[-1].tolist()),
        )


def _past_history(samples, current_idx, max_history):
    """Collect ``{instance_id: [(position, ego_pose), ...]}`` over past frames."""
    history: dict = {}
    start = max(0, current_idx - max_history + 1)
    for idx in range(start, current_idx + 1):
        sample = samples[idx]
        ego_pose = sample.points.meta.transforms.pose
        centers = sample.labels.boxes.get(0)[:, :3].numpy()
        instance_ids = sample.labels.instance_ids.get(0).numpy()
        for i, iid in enumerate(instance_ids):
            history.setdefault(int(iid), []).append((centers[i].copy(), ego_pose))
    return history


def render_past_trajectories(
    server,
    samples,
    current_idx,
    *,
    scene_scale=0.1,
    max_history=20,
    name="annotations/trajectories/past",
):
    """Draw ego-motion-compensated past trajectories for currently visible tracks.

    No-op when the samples lack lidar-derived ego pose. ``samples`` is the full
    per-frame list of the scene; ``current_idx`` selects the frame to render.
    """
    # pylint: disable=too-many-arguments,too-many-locals
    current = samples[current_idx]
    if not hasattr(current, "points") or current.points is None:
        return

    history = _past_history(samples, current_idx, max_history)
    current_pose = current.points.meta.transforms.pose
    visible_ids = current.labels.instance_ids.get(0).numpy()

    for iid in visible_ids:
        track = history.get(int(iid))
        if not track or len(track) < 2:
            continue
        positions = []
        for position, past_pose in track:
            world = past_pose.matrix.numpy() @ np.append(position, 1.0)
            positions.append((current_pose.inv.matrix.numpy() @ world)[:3])
        positions = np.array(positions) * scene_scale
        color = _rgb_uint8(iid)
        server.scene.add_spline_catmull_rom(
            name=f"{name}/{int(iid)}",
            points=positions,
            tension=0.0,
            line_width=10.0,
            color=color,
            segments=len(positions) * 5,
        )
        server.scene.add_icosphere(
            name=f"{name}/start/{int(iid)}",
            radius=0.1 * scene_scale,
            color=color,
            position=tuple(positions[0].tolist()),
        )
