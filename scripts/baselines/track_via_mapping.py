#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Apply AB3DMOT track ids to panoptic-occupancy predictions.

Final stage of the AB3DMOT box-tracking baseline. Takes the per-frame
predictions plus the instance->track mapping from ``ab3dmot_map_iids.py`` and
writes tracked predictions: instances matched by AB3DMOT get the shared track id
(compacted per sequence), unmatched instances get fresh track ids.

Example:
    ./scripts/baselines/track_via_mapping.py --dataset nuscenes \\
        --preds runs/.../predictions --mapping mapping.json --output tracked/
"""
import json
import sys
from pathlib import Path

import click
import numpy as np

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


def load_ab3dmot_mapping(mapping_path):
    """Load the mapping as ``{(sequence_id, sample_id, instance_id): track_id}``."""
    print(f"Loading AB3DMOT mapping from: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for entry in data["mappings"]:
        key = (str(entry["sequence_id"]), entry["sample_id"], entry["instance_id"])
        mapping[key] = entry["track_id"]

    print(f"Loaded {len(mapping)} instance-to-track mappings")
    return mapping


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
    "--mapping",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Instance-to-track mapping JSON (from ab3dmot_map_iids.py).",
)
@click.option(
    "--output",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Output directory for tracked predictions.",
)
@click.option("--split", default=None, help="Override the default (val) split.")
@click.option("--verbose", is_flag=True, help="Print per-frame details.")
@click.option(
    "--save-stats",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write tracking statistics to this JSON file.",
)
def main(dataset, preds, mapping, output, split, verbose, save_stats):
    """Remap instance ids to AB3DMOT track ids, per sequence."""
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    print(f"Predictions path: {preds}")
    print(f"Mapping JSON: {mapping}")
    print(f"Output directory: {output}")
    print()

    ctx = od.load_context(dataset.lower(), split=split)
    ab3dmot_mapping = load_ab3dmot_mapping(mapping)

    total_frames = 0
    total_matched = 0
    total_unmatched = 0
    n_seq = 0

    # Per-sequence AB3DMOT-track -> compact track id, reset at each boundary.
    current = None
    ab3dmot_to_new = {}
    next_track_id = 0

    stream = bl.stream_predictions(ctx, preds)
    for seq_id, frame_index, sample_id, path in progress.track(
        stream, "Applying mapping...", total=od.frame_count(ctx)
    ):
        if seq_id != current:
            current = seq_id
            ab3dmot_to_new = {}
            next_track_id = 0
            n_seq += 1

        with np.load(path) as data:
            pano_sem = data["pano_sem"]
            instances = data["pano_inst"]

        tracked = np.full_like(instances, -1)
        for inst_id in np.unique(instances[instances >= 0]):
            key = (seq_id, sample_id, int(inst_id))
            if key in ab3dmot_mapping:
                ab3dmot_track = ab3dmot_mapping[key]
                if ab3dmot_track not in ab3dmot_to_new:
                    ab3dmot_to_new[ab3dmot_track] = next_track_id
                    next_track_id += 1
                final_track_id = ab3dmot_to_new[ab3dmot_track]
                total_matched += 1
            else:
                # Instance not tracked by AB3DMOT: give it its own fresh track.
                final_track_id = next_track_id
                next_track_id += 1
                total_unmatched += 1
            tracked[instances == inst_id] = final_track_id

        bl.write_tracked(output, seq_id, sample_id, pano_sem, tracked)
        total_frames += 1
        if verbose:
            print(
                f"  {seq_id}/{sample_id} (frame {frame_index}): "
                f"max track {int(tracked.max())}"
            )

    print("\nTracking complete!")
    print(f"Total frames:      {total_frames}")
    print(f"Matched instances: {total_matched}")
    print(f"Fresh instances:   {total_unmatched}")
    total_inst = total_matched + total_unmatched
    if total_inst:
        print(f"Match rate:        {total_matched / total_inst * 100:.1f}%")

    if save_stats:
        payload = {
            "total_frames": total_frames,
            "total_sequences": n_seq,
            "total_matched_instances": total_matched,
            "total_unmatched_instances": total_unmatched,
        }
        Path(save_stats).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Statistics saved to: {save_stats}")


if __name__ == "__main__":
    main()
