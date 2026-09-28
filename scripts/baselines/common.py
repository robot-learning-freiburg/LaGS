# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared helpers for the occupancy tracking baselines.

Every baseline here starts from per-frame panoptic-occupancy predictions in the
unified layout -- ``<preds>/<sequence_id>/<sample_id>.npz`` with ``pano_sem`` /
``pano_inst`` in ``[Z, Y, X]`` and ``-1`` = no instance -- and turns them into
*tracked* predictions in the same layout, ready for ``scripts/evals``.

Frame grouping and temporal order come from the in-code dataset pipeline
(:mod:`scripts.common.datasets`), the same source the evaluator reads, so no
dataset-specific info files are needed. Instance ids stay in the stored
convention throughout -- ``-1`` = no instance, ids ``>= 0`` -- which is exactly
what the ground truth and the metric use, so no re-indexing is needed: valid
voxels are ``inst >= 0`` and tracked maps start at ``-1``.
"""

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

import scripts.common.datasets as od


def _jsonable(sample_id: object) -> object:
    """Coerce a ``sample_id`` to a JSON-native scalar (numpy int/str -> int/str).

    Some pipelines carry numpy scalar ``sample_id``s; they must survive
    ``json.dump`` into ``boxes.json`` / ``mapping.json`` and compare equal to the
    same id read back, so we normalise once at the streaming boundary. Zero-padded
    prediction filenames are unaffected (``int`` and ``np.integer`` pad alike).
    """
    if isinstance(sample_id, np.integer):
        return int(sample_id)
    if isinstance(sample_id, np.str_):
        return str(sample_id)
    return sample_id


def stream_predictions(
    ctx: od.OccupancyContext, preds_root: str | Path
) -> Iterator[Tuple[str, int, object, Path]]:
    """Yield ``(sequence_id, frame_index, sample_id, path)`` in scene order.

    ``frame_index`` is a contiguous 0-based counter within each sequence. Frames
    whose prediction npz is missing are skipped and do not advance the counter,
    so each sequence is streamed exactly over the frames that have a prediction,
    contiguously and in temporal order.
    """
    current = None
    frame_index = 0
    for frame in od.iter_frames(ctx):
        path = od.prediction_path(preds_root, frame.sequence_id, frame.sample_id)
        if not path.exists():
            continue
        if frame.sequence_id != current:
            current = frame.sequence_id
            frame_index = 0
        yield frame.sequence_id, frame_index, _jsonable(frame.sample_id), path
        frame_index += 1


def write_tracked(
    output_root: str | Path,
    sequence_id: str,
    sample_id: object,
    pano_sem: np.ndarray,
    tracked: np.ndarray,
) -> Path:
    """Write a tracked prediction npz in the unified nested layout.

    ``tracked`` is already in the stored convention (``-1`` = none, ids ``>= 0``).
    """
    path = od.prediction_path(output_root, sequence_id, sample_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pano_sem=pano_sem, pano_inst=tracked)
    return path
