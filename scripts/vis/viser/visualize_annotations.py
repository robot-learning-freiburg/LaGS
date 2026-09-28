#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Viser figure of ground-truth annotations: occupancy + boxes + trajectories.

The annotation-rich counterpart to ``visualize_predictions.py``: on top of the
ground-truth occupancy voxels it overlays wireframe detection boxes, future
forecast splines and ego-motion-compensated past trajectories -- the elements
that make a qualitative panoptic figure legible. Built on the full-sensor
pipeline (:mod:`scripts.vis.common.sensors`), so it needs the dataset's boxes /
forecasting / (optionally) lidar rather than the occupancy-only pipeline.

Past trajectories require per-frame ego pose, only available with ``--lidar``.

Examples:
    ./scripts/vis/viser/visualize_annotations.py --dataset nuscenes --scene 0
    ./scripts/vis/viser/visualize_annotations.py \\
        --dataset waymo --scene-name <sequence> --lidar --mask lidar
"""

import logging
import sys
import time
from pathlib import Path

import click
import viser
from omegaconf import OmegaConf

from tracker import config, utils
from tracker.config import paths

# Make the top-level ``scripts`` package importable when run directly: add the
# repo root -- the parent of ``scripts``.
_REPO_ROOT = next(
    p.parent for p in Path(__file__).resolve().parents if p.name == "scripts"
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# pylint: disable=wrong-import-position
import scripts.common.datasets as od  # noqa: E402
from scripts.vis.common import sensors  # noqa: E402
from scripts.vis.viser import utils as ocr  # noqa: E402

log = logging.getLogger(__name__)


def _setup_visibility_controls(server, navigator, flags):
    """Per-category visibility checkboxes; toggling re-renders the current frame."""
    with server.gui.add_folder("Visibility Controls"):
        for key, label in (
            ("things", "Show Things"),
            ("stuff", "Show Stuff"),
            ("boxes", "Show Boxes"),
            ("future", "Show Future Trajectories"),
            ("past", "Show Past Trajectories"),
        ):
            checkbox = server.gui.add_checkbox(label, initial_value=flags[key])

            def _make_callback(flag_key, handle):
                def _callback(_):
                    flags[flag_key] = handle.value
                    navigator.goto_frame(navigator.current_frame_idx)

                return _callback

            checkbox.on_update(_make_callback(key, checkbox))


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(od.DATASETS, case_sensitive=False),
    default="nuscenes",
    show_default=True,
    help="Dataset whose ground-truth pipeline to build.",
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
    "--mask",
    type=click.Choice(["valid", "camera", "lidar", "any", "none"]),
    default="none",
    show_default=True,
    help="Hard-mask both things and stuff by an observation mask.",
)
@click.option(
    "--lidar/--no-lidar",
    default=False,
    show_default=True,
    help="Load lidar (needed for past trajectories; slower).",
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
    split,
    scene,
    scene_name,
    sample,
    mask,
    lidar,
    scene_scale,
    port,
    output_dir,
):
    """Serve an interactive viser panoptic figure (occupancy + boxes + tracks)."""
    # pylint: disable=too-many-arguments,too-many-locals
    utils.log.initialize()
    dataset = dataset.lower()

    cfg = OmegaConf.create()
    cfg.paths = OmegaConf.create()
    cfg.paths.root = paths.root
    cfg.paths.data = paths.root / "data"
    cfg.paths.cache = paths.root / "cache"
    config.set_global_config(cfg)

    log.info(
        "building %s pipeline (split=%s, lidar=%s)", dataset, split or "default", lidar
    )
    dataset_obj, occupancy_labels, voxel_size, voxel_range = sensors.build_dataset(
        dataset, split=split, load_lidar=lidar
    )

    start = sensors.resolve_start_sample(
        dataset, dataset_obj, scene, 0, None, scene_name=scene_name
    )
    sequence_id, samples = sensors.collect_scene_samples(dataset_obj, start)
    log.info("scene %s: %d frames", sequence_id, len(samples))
    if not lidar:
        log.info("past trajectories disabled (needs --lidar)")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    log.info("viser server at http://localhost:%s", port)
    ocr.viewer.setup_lighting(server)

    flags = {
        "things": True,
        "stuff": True,
        "boxes": True,
        "future": True,
        "past": lidar,
    }

    def render_fn(frame_idx: int) -> None:
        server.scene.reset()
        frame = samples[frame_idx]
        occ = frame.labels.occupancy
        ocr.occupancy.render_occupancy_gt(
            server,
            occ.semantics,
            occ.instance_ids,
            occ.masks,
            occupancy_labels,
            voxel_size=voxel_size,
            voxel_range=voxel_range,
            scene_scale=scene_scale,
            mask_type=mask,
            show_things=flags["things"],
            show_stuff=flags["stuff"],
        )
        if flags["boxes"]:
            ocr.annotations.render_boxes(server, frame, scene_scale=scene_scale)
        if flags["future"]:
            ocr.annotations.render_future_trajectories(
                server, frame, scene_scale=scene_scale
            )
        if flags["past"]:
            ocr.annotations.render_past_trajectories(
                server, samples, frame_idx, scene_scale=scene_scale
            )

    navigator = ocr.viewer.FrameNavigator(
        server, len(samples), render_fn, sequence_id, Path(output_dir)
    )
    _setup_visibility_controls(server, navigator, flags)
    ocr.viewer.setup_navigation_controls(server, navigator)
    navigator.goto_frame(sample if 0 <= sample < len(samples) else 0)

    print("Viser session is running. Press Ctrl+C to exit...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
