#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Viser figure of panoptic occupancy for one scene (ground truth or predictions).

High-quality, interactive figures (as opposed to the Rerun debugging viewer):
each frame's occupancy is drawn as batched voxel cubes -- instances coloured per
track, "stuff" per semantic class -- with lighting/environment controls, a frame
navigator and batch screenshot export.

Predictions are optional: with no ``--preds`` it shows ground truth; with
``--preds DIR`` (unified ``<sequence_id>/<sample_id>.npz`` layout) it shows the
predicted occupancy, gated by the same visual filter used elsewhere.

Examples:
    ./scripts/vis/viser/visualize_predictions.py --dataset nuscenes --scene-name scene-0013
    ./scripts/vis/viser/visualize_predictions.py \\
        --dataset nuscenes --preds runs/.../predictions --scene 0
"""

import logging
import sys
import time
from pathlib import Path

import click
import viser

from tracker import utils

# Make the top-level ``scripts`` package importable when run directly (see the
# eval scripts for the rationale): add the repo root -- the parent of ``scripts``.
_REPO_ROOT = next(
    p.parent for p in Path(__file__).resolve().parents if p.name == "scripts"
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# pylint: disable=wrong-import-position
import scripts.common.datasets as od  # noqa: E402
from scripts.vis.viser import utils as ocr  # noqa: E402

log = logging.getLogger(__name__)


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(od.DATASETS, case_sensitive=False),
    default="nuscenes",
    show_default=True,
    help="Dataset whose ground-truth pipeline to build.",
)
@click.option(
    "--preds",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=None,
    help="Prediction dir (<sequence_id>/<sample_id>.npz). Omit for ground truth.",
)
@click.option("--split", default=None, help="Override the default (val) split.")
@click.option("--scene", "scene", type=int, default=0, help="Scene ordinal.")
@click.option(
    "--scene-name",
    default=None,
    help="Select a scene by name (nuScenes 'scene-0013' / sequence id), "
    "overriding --scene.",
)
@click.option("--sample", "sample", type=int, default=0, help="Initial frame index.")
@click.option(
    "--visual-filter/--no-visual-filter",
    default=True,
    show_default=True,
    help="Apply the per-dataset visual filter to 'stuff' voxels.",
)
@click.option(
    "--mask",
    default="lidar",
    show_default=True,
    help="Base observation mask when --no-visual-filter ('none' = all).",
)
@click.option(
    "--scene-scale", type=float, default=0.1, show_default=True, help="World scale."
)
@click.option("--port", type=int, default=8080, show_default=True, help="Viser port.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="screenshots",
    show_default=True,
    help="Directory for screenshot export.",
)
def main(
    dataset,
    preds,
    split,
    scene,
    scene_name,
    sample,
    visual_filter,
    mask,
    scene_scale,
    port,
    output_dir,
):
    """Serve an interactive viser occupancy figure for one scene."""
    # pylint: disable=too-many-arguments,too-many-locals
    utils.log.initialize()
    dataset = dataset.lower()

    log.info("building %s pipeline (split=%s)", dataset, split or "default")
    ctx, sequence_id, frames = ocr.scenes.resolve_scene(
        dataset, split, scene, scene_name
    )
    if not frames:
        raise click.ClickException(f"scene {sequence_id} has no frames")
    log.info("scene %s: %d frames", sequence_id, len(frames))

    server = viser.ViserServer(host="0.0.0.0", port=port)
    log.info("viser server at http://localhost:%s", port)
    ocr.viewer.setup_lighting(server)

    def render_fn(frame_idx: int) -> None:
        frame = frames[frame_idx]
        semantics, instances = frame.semantics, frame.instance_ids
        if preds is not None:
            pred = od.load_prediction(preds, frame.sequence_id, frame.sample_id)
            if pred is not None:
                semantics, instances = pred.semantics, pred.instance_ids
            else:
                log.warning(
                    "no prediction for %s/%s; showing ground truth",
                    frame.sequence_id,
                    frame.sample_id,
                )
        ocr.occupancy.render_occupancy(
            server,
            semantics,
            instances,
            frame.masks,
            ctx.labels,
            dataset,
            voxel_size=ctx.voxel_size,
            voxel_range=ctx.voxel_range,
            scene_scale=scene_scale,
            visual_filter=visual_filter,
            base_mask=mask,
        )

    navigator = ocr.viewer.FrameNavigator(
        server, len(frames), render_fn, sequence_id, Path(output_dir)
    )
    ocr.viewer.setup_navigation_controls(server, navigator)
    navigator.goto_frame(sample if 0 <= sample < len(frames) else 0)

    print("Viser session is running. Press Ctrl+C to exit...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
