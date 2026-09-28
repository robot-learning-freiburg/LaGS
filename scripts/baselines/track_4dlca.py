#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""IoU-based tracking for per-frame panoptic-occupancy predictions.

Associates instances across consecutive frames of a sequence by voxel IoU
(Hungarian assignment, same-class only); unmatched instances start new tracks.

Example:
    ./scripts/baselines/track_4dlca.py \\
        --dataset nuscenes --preds runs/.../predictions --output tracked/
"""
import json
import sys
from pathlib import Path

import click
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.utils import progress

# Make the top-level ``scripts`` package importable when run directly (see the
# eval scripts for the rationale): add the repo root -- the parent of ``scripts``.
_REPO_ROOT = next(
    p.parent for p in Path(__file__).resolve().parents if p.name == "scripts"
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# pylint: disable=wrong-import-position
import scripts.common.datasets as od  # noqa: E402
from scripts.baselines import common as bl  # noqa: E402


def compute_iou_matrix(instances_prev, instances_curr, semantics_prev, semantics_curr):
    """IoU matrix between previous/current instances (same semantic class only)."""
    # pylint: disable=too-many-locals

    prev_ids = np.unique(instances_prev[instances_prev >= 0])
    curr_ids = np.unique(instances_curr[instances_curr >= 0])

    if len(prev_ids) == 0 or len(curr_ids) == 0:
        return np.zeros((0, 0)), prev_ids, curr_ids

    iou_matrix = np.zeros((len(prev_ids), len(curr_ids)))

    for prev_idx, prev_id in enumerate(prev_ids):
        prev_mask = instances_prev == prev_id
        prev_sem = semantics_prev[prev_mask][0] if np.any(prev_mask) else -1

        for curr_idx, curr_id in enumerate(curr_ids):
            curr_mask = instances_curr == curr_id
            curr_sem = semantics_curr[curr_mask][0] if np.any(curr_mask) else -1

            # Only match instances of the same semantic class.
            if prev_sem != curr_sem or prev_sem == -1:
                continue

            intersection = np.logical_and(prev_mask, curr_mask).sum()
            union = np.logical_or(prev_mask, curr_mask).sum()
            if union > 0:
                iou_matrix[prev_idx, curr_idx] = intersection / union

    return iou_matrix, prev_ids, curr_ids


def match_instances(iou_matrix, prev_ids, curr_ids, iou_threshold=0.1):
    """Hungarian match current->previous instances above ``iou_threshold``."""
    matches = {}
    unmatched_curr = set(curr_ids)

    if len(prev_ids) == 0 or len(curr_ids) == 0:
        return matches, unmatched_curr

    # Negate IoU because linear_sum_assignment minimizes cost.
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    for prev_idx, curr_idx in zip(row_ind, col_ind):
        if iou_matrix[prev_idx, curr_idx] > iou_threshold:
            matches[curr_ids[curr_idx]] = prev_ids[prev_idx]
            unmatched_curr.discard(curr_ids[curr_idx])

    return matches, unmatched_curr


class _SequenceTracker:
    """Per-sequence IoU tracking state; reset at each sequence boundary."""

    # pylint: disable=too-few-public-methods

    def __init__(self, iou_threshold):
        self.iou_threshold = iou_threshold
        self.prev_instances = None
        self.prev_semantics = None
        self.track_id_mapping = {}  # current-frame instance id -> global track id
        self.next_track_id = 0

    def step(self, instances_curr, semantics_curr):
        """Assign track ids to the current frame; return the tracked map."""
        # pylint: disable=too-many-locals
        if self.prev_instances is None:
            mapping = {}
            for inst_id in np.unique(instances_curr[instances_curr >= 0]):
                mapping[inst_id] = self.next_track_id
                self.next_track_id += 1
        else:
            iou_matrix, prev_ids, curr_ids = compute_iou_matrix(
                self.prev_instances,
                instances_curr,
                self.prev_semantics,
                semantics_curr,
            )
            matches, unmatched_curr = match_instances(
                iou_matrix, prev_ids, curr_ids, self.iou_threshold
            )
            assert len(matches) + len(unmatched_curr) == len(curr_ids)

            mapping = {}
            for curr_id, prev_id in matches.items():
                if prev_id in self.track_id_mapping:
                    mapping[curr_id] = self.track_id_mapping[prev_id]
            for curr_id in unmatched_curr:
                mapping[curr_id] = self.next_track_id
                self.next_track_id += 1

        tracked = np.full_like(instances_curr, -1)
        for inst_id, track_id in mapping.items():
            tracked[instances_curr == inst_id] = track_id

        self.track_id_mapping = mapping
        self.prev_instances = instances_curr
        self.prev_semantics = semantics_curr
        return tracked


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(od.DATASETS, case_sensitive=False),
    default="nuscenes",
    show_default=True,
    help="Dataset whose pipeline provides frame grouping/order.",
)
@click.option(
    "--preds",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory of predictions (<sequence_id>/<sample_id>.npz).",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Output directory for tracked predictions.",
)
@click.option("--split", default=None, help="Override the default (val) split.")
@click.option(
    "--iou-threshold",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum IoU to associate instances across frames.",
)
@click.option("--verbose", is_flag=True, help="Print per-frame details.")
@click.option(
    "--save-stats",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write tracking statistics to this JSON file.",
)
def main(dataset, preds, output, split, iou_threshold, verbose, save_stats):
    """Track instances across each sequence by voxel IoU."""
    # pylint: disable=too-many-locals
    print(f"Predictions path: {preds}")
    print(f"Output directory: {output}")
    print(f"IoU threshold: {iou_threshold}")
    print()

    ctx = od.load_context(dataset.lower(), split=split)

    seq_tracks: dict[str, int] = {}
    total_frames = 0
    current = None
    tracker = None

    stream = bl.stream_predictions(ctx, preds)
    for seq_id, frame_index, sample_id, path in progress.track(
        stream, "Tracking frames...", total=od.frame_count(ctx)
    ):
        if seq_id != current:
            current = seq_id
            tracker = _SequenceTracker(iou_threshold)

        with np.load(path) as data:
            pano_sem = data["pano_sem"]
            instances = data["pano_inst"]

        tracked = tracker.step(instances, pano_sem)
        bl.write_tracked(output, seq_id, sample_id, pano_sem, tracked)

        seq_tracks[seq_id] = tracker.next_track_id
        total_frames += 1
        if verbose:
            n = len(np.unique(instances[instances >= 0]))
            print(f"  {seq_id}/{sample_id} (frame {frame_index}): {n} instances")

    total_tracks = sum(seq_tracks.values())
    n_seq = len(seq_tracks)
    print("\nTracking complete!")
    print(f"Total frames:   {total_frames}")
    print(f"Total tracks:   {total_tracks}")
    print(f"Sequences:      {n_seq}")
    if n_seq:
        print(f"Avg tracks/seq: {total_tracks / n_seq:.2f}")

    if save_stats:
        payload = {
            "total_frames": total_frames,
            "total_tracks": total_tracks,
            "total_sequences": n_seq,
            "sequences": seq_tracks,
        }
        Path(save_stats).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Statistics saved to: {save_stats}")


if __name__ == "__main__":
    main()
