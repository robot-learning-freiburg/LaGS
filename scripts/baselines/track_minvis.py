#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""MinVIS-style embedding tracking for per-frame panoptic-occupancy predictions.

Associates instances across consecutive frames by cosine similarity of their
per-object embeddings (Hungarian assignment); unmatched instances start new
tracks. Requires predictions saved with ``obj_idxes`` / ``obj_embeddings``
(the ids in ``obj_idxes`` are stored in the same convention as ``pano_inst``).

Example:
    ./scripts/baselines/track_minvis.py \\
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


def extract_instance_embeddings(obj_idxes, obj_embeddings):
    """Map instance id -> embedding vector.

    ``obj_idxes`` are stored in the same convention as ``pano_inst`` (ids ``>= 0``).
    """
    instance_embeddings = {}
    for i, inst_id in enumerate(obj_idxes):
        assert inst_id >= 0
        instance_embeddings[int(inst_id)] = obj_embeddings[i]
    return instance_embeddings


def match_from_embeddings(prev_embeddings, curr_embeddings, similarity_threshold=0.5):
    """Match current->previous instances by cosine similarity (MinVIS style)."""
    # pylint: disable=too-many-locals

    if not prev_embeddings or not curr_embeddings:
        return {}, set(curr_embeddings.keys())

    prev_ids = list(prev_embeddings.keys())
    curr_ids = list(curr_embeddings.keys())

    prev_embds = np.stack([prev_embeddings[pid] for pid in prev_ids])
    curr_embds = np.stack([curr_embeddings[cid] for cid in curr_ids])

    # L2-normalize, then cosine similarity.
    prev_embds = prev_embds / (np.linalg.norm(prev_embds, axis=1, keepdims=True) + 1e-8)
    curr_embds = curr_embds / (np.linalg.norm(curr_embds, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.matmul(curr_embds, prev_embds.T)

    # MinVIS uses cost = 1 - cosine similarity.
    row_ind, col_ind = linear_sum_assignment(1.0 - cos_sim)

    matches = {}
    unmatched_curr = set(curr_ids)
    for curr_idx, prev_idx in zip(row_ind, col_ind):
        if cos_sim[curr_idx, prev_idx] >= similarity_threshold:
            matches[curr_ids[curr_idx]] = prev_ids[prev_idx]
            unmatched_curr.discard(curr_ids[curr_idx])

    return matches, unmatched_curr


class _SequenceTracker:
    """Per-sequence embedding tracking state; reset at each sequence boundary."""

    # pylint: disable=too-few-public-methods

    def __init__(self, similarity_threshold):
        self.similarity_threshold = similarity_threshold
        self.prev_embeddings = None
        self.track_id_mapping = {}  # current-frame instance id -> global track id
        self.next_track_id = 0

    def step(self, instances_curr, curr_embeddings):
        """Assign track ids to the current frame; return the tracked map."""
        # pylint: disable=too-many-locals
        unique_curr = set(np.unique(instances_curr[instances_curr >= 0]).tolist())

        if self.prev_embeddings is None:
            mapping = {}
            for inst_id in unique_curr:
                mapping[inst_id] = self.next_track_id
                self.next_track_id += 1
        else:
            if curr_embeddings and self.prev_embeddings:
                matches, unmatched_curr = match_from_embeddings(
                    self.prev_embeddings, curr_embeddings, self.similarity_threshold
                )
                # Instances present in the map but lacking an embedding are new.
                unmatched_curr = unmatched_curr | (unique_curr - set(curr_embeddings))
            else:
                matches, unmatched_curr = {}, unique_curr

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
        self.prev_embeddings = curr_embeddings
        return tracked, len(unique_curr)


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
    "--similarity-threshold",
    type=float,
    default=0.85,
    show_default=True,
    help="Minimum cosine similarity to associate instances across frames.",
)
@click.option("--verbose", is_flag=True, help="Print per-frame details.")
@click.option(
    "--save-stats",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write tracking statistics to this JSON file.",
)
def main(dataset, preds, output, split, similarity_threshold, verbose, save_stats):
    """Track instances across each sequence by embedding similarity."""
    # pylint: disable=too-many-locals
    print(f"Predictions path: {preds}")
    print(f"Output directory: {output}")
    print(f"Similarity threshold: {similarity_threshold}")
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
            tracker = _SequenceTracker(similarity_threshold)

        with np.load(path) as data:
            pano_sem = data["pano_sem"]
            instances = data["pano_inst"]
            embeddings = extract_instance_embeddings(
                data["obj_idxes"], data["obj_embeddings"]
            )

        tracked, n_inst = tracker.step(instances, embeddings)
        bl.write_tracked(output, seq_id, sample_id, pano_sem, tracked)

        seq_tracks[seq_id] = tracker.next_track_id
        total_frames += 1
        if verbose:
            print(f"  {seq_id}/{sample_id} (frame {frame_index}): {n_inst} instances")

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
