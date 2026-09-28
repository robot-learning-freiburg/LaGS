#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Per-frame baseline: assign a fresh track id to every instance in every frame.

This is the no-tracking floor for the tracking metric. Within a sequence each
instance gets a globally unique id (``frame_index * 100000 + instance_id``), so
no instance is ever associated across frames.

Example:
    ./scripts/baselines/track_perframe.py \\
        --dataset nuscenes --preds runs/.../predictions --output tracked/
"""
import json
import sys
from collections import defaultdict
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
@click.option("--verbose", is_flag=True, help="Print per-frame details.")
@click.option(
    "--save-stats",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write tracking statistics to this JSON file.",
)
def main(dataset, preds, output, split, verbose, save_stats):
    """Assign a unique track id per instance per frame (no association)."""
    # pylint: disable=too-many-locals
    print(f"Predictions path: {preds}")
    print(f"Output directory: {output}")
    print("Tracking mode: per-frame baseline (no tracking)")
    print()

    ctx = od.load_context(dataset.lower(), split=split)

    stats = {"total_frames": 0, "total_tracks": 0, "sequences": defaultdict(int)}

    stream = bl.stream_predictions(ctx, preds)
    for seq_id, frame_index, sample_id, path in progress.track(
        stream, "Tracking frames...", total=od.frame_count(ctx)
    ):
        with np.load(path) as data:
            pano_sem = data["pano_sem"]
            instances = data["pano_inst"]

        tracked = np.full_like(instances, -1, dtype=np.int64)
        unique_ids = np.unique(instances[instances >= 0])
        for inst_id in unique_ids:
            # frame_index disambiguates ids across frames -> no cross-frame link.
            track_id = frame_index * 100000 + int(inst_id)
            tracked[instances == inst_id] = track_id

        bl.write_tracked(output, seq_id, sample_id, pano_sem, tracked)

        stats["total_frames"] += 1
        stats["sequences"][seq_id] += len(unique_ids)
        stats["total_tracks"] += len(unique_ids)
        if verbose:
            print(
                f"  {seq_id}/{sample_id} (frame {frame_index}): {len(unique_ids)} tracks"
            )

    print("\nProcessing complete!")
    print(f"Total frames:   {stats['total_frames']}")
    print(f"Total tracks:   {stats['total_tracks']}")
    print(f"Sequences:      {len(stats['sequences'])}")

    if save_stats:
        payload = {
            "total_frames": stats["total_frames"],
            "total_tracks": stats["total_tracks"],
            "total_sequences": len(stats["sequences"]),
            "sequences": dict(stats["sequences"]),
        }
        Path(save_stats).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Statistics saved to: {save_stats}")


if __name__ == "__main__":
    main()
