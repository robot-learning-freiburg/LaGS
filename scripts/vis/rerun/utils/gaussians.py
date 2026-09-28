# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Loading + Rerun rendering of predicted gaussians.

The prediction writer stores all streams concatenated; ``load_gaussians`` splits
them back out via ``gaussian_stream_counts``. ``visualize_gaussians`` renders each
stream as ``rr.Ellipsoids3D`` (ellipsoid rendering adapted from
``test/visualize_gaussians_rerun.py``), coloured either by opacity or by semantic
class, into ``<entity_root>/<mode>/<stream>``.
"""

import numpy as np
import rerun as rr
import torch

from scripts.common.datasets import color_for_class


def load_gaussians(npz) -> dict[str, dict]:
    """Split flat gaussian arrays into per-stream tensor dicts (``{}`` if absent)."""
    if npz is None or "gaussian_center" not in npz.files:
        return {}

    centers = torch.from_numpy(npz["gaussian_center"]).float()
    scales = torch.from_numpy(npz["gaussian_scale"]).float()
    rotations = torch.from_numpy(npz["gaussian_rotation"]).float()  # wxyz
    opacities = torch.from_numpy(npz["gaussian_opacity"]).float()
    instance_ids = torch.from_numpy(npz["gaussian_instance_id"]).long()
    logits = (
        torch.from_numpy(npz["gaussian_logits"]).float()
        if "gaussian_logits" in npz.files
        else None
    )

    if "gaussian_streams" in npz.files:
        streams = [str(s) for s in npz["gaussian_streams"]]
        counts = [int(c) for c in npz["gaussian_stream_counts"]]
    else:
        streams, counts = ["default"], [len(centers)]

    out: dict[str, dict] = {}
    start = 0
    for name, n in zip(streams, counts):
        sl = slice(start, start + n)
        out[name] = {
            "centers": centers[sl],
            "scales": scales[sl],
            "rotations": rotations[sl],
            "opacities": opacities[sl],
            "instance_ids": instance_ids[sl],
            "logits": None if logits is None else logits[sl],
        }
        start += n
    return out


def _semantic_colors(logits, labels) -> np.ndarray:
    if logits is None:
        return None
    sem_labels = labels.semantic if "semantic" in labels else labels.all
    classes = torch.argmax(logits, dim=-1).tolist()
    return np.array([color_for_class(sem_labels[c]) for c in classes], dtype=np.float32)


def visualize_gaussians(
    gaussians_by_stream: dict[str, dict],
    labels,
    entity_root="prediction/gaussians",
    mode="semantic",
    opacity_threshold=0.1,
):
    """Render each stream's gaussians as opacity- or semantic-coloured ellipsoids."""
    # pylint: disable=too-many-locals
    for stream, g in gaussians_by_stream.items():
        entity = f"{entity_root}/{mode}/{stream}"

        opacities = g["opacities"]
        keep = opacities > opacity_threshold
        if keep.sum() == 0:
            rr.log(entity, rr.Clear(recursive=True))
            continue

        centers = g["centers"][keep]
        scales = g["scales"][keep]
        # Gaussians store [w, x, y, z]; rerun expects [x, y, z, w].
        quaternions = g["rotations"][keep][:, [1, 2, 3, 0]].numpy()

        if mode == "semantic":
            logits = None if g["logits"] is None else g["logits"][keep]
            colors = _semantic_colors(logits, labels)
            if colors is None:  # no class head in this run
                colors = np.tile([0.5, 0.5, 0.5], (len(centers), 1)).astype(np.float32)
        else:  # opacity: grayscale by opacity value in [0, 1]
            v = opacities[keep].clamp(0.0, 1.0).numpy()
            colors = np.stack([v, v, v], axis=1).astype(np.float32)

        rr.log(
            entity,
            rr.Ellipsoids3D(
                centers=centers.numpy(),
                half_sizes=scales.numpy(),
                quaternions=quaternions,
                colors=colors,
                fill_mode="solid",
            ),
        )
