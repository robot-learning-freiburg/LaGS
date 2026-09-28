#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rerun demo bundle: GT (+ optional predictions + gaussians) for one occupancy scene.

Loads the full-sensor dataset pipeline for a scene and, per frame, logs the
cameras, LiDAR, ground-truth panoptic occupancy and boxes/trajectories, plus --
for each prediction method given -- the matching predicted occupancy
(``occupancy.PanopticOccupancy`` npz) and predicted gaussians, each in its own
toggle-able entity sub-tree. Save the whole thing to a single ``.rrd`` for a demo,
or serve it over gRPC + web viewer.

Predictions are optional: with no ``--pred`` the script visualizes ground truth
only. Pass ``--pred name=path`` once per method to overlay predictions; each is
logged under its own toggle-able ``prediction/<name>/`` sub-tree, the first
visible by default and the rest hidden.

Examples:
    # ground truth only
    ./scripts/vis/rerun/visualize_occupancy.py --dataset nuscenes --scene 0

    # with one / several prediction methods
    ./scripts/vis/rerun/visualize_occupancy.py --dataset nuscenes \\
        --pred lags=runs/.../predictions --scene 0 --save bundle.rrd
    ./scripts/vis/rerun/visualize_occupancy.py --dataset waymo \\
        --pred lags=runs/lags/preds --pred trackocc=runs/trackocc/preds --scene 3
"""

import functools
import logging
import re
import sys
import time
from pathlib import Path

import click
import rerun as rr
from omegaconf import OmegaConf

from tracker import config, utils
from tracker.config import paths

# Make the top-level ``scripts`` package importable when run directly (see the
# eval scripts for the rationale): add the repo root -- the parent of ``scripts``.
_REPO_ROOT = next(
    p.parent for p in Path(__file__).resolve().parents if p.name == "scripts"
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# pylint: disable=wrong-import-position
from scripts.vis.common import sensors  # noqa: E402
from scripts.vis.common.heuristics import visual_filter_mask  # noqa: E402
from scripts.vis.rerun import utils as ocr  # noqa: E402

log = logging.getLogger(__name__)


def parse_pred_specs(pred_specs):
    """Parse ``--pred name=path`` entries into an ordered ``[(name, Path)]`` list.

    Returns an empty list when no ``--pred`` is given (ground-truth-only mode).
    Names are used as Rerun entity-path components: whitespace/slashes are
    replaced with ``_``. Raises ``click.BadParameter`` on malformed entries,
    missing directories, or duplicate names.
    """
    methods: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for spec in pred_specs:
        if "=" not in spec:
            raise click.BadParameter(
                f"expected 'name=path', got {spec!r}", param_hint="--pred"
            )
        raw_name, raw_path = spec.split("=", 1)
        name = re.sub(r"[\s/]+", "_", raw_name.strip())
        if not name:
            raise click.BadParameter(
                f"empty method name in {spec!r}", param_hint="--pred"
            )
        if name in seen:
            raise click.BadParameter(
                f"duplicate method name {name!r}", param_hint="--pred"
            )

        path = Path(raw_path).expanduser()
        if not path.is_dir():
            raise click.BadParameter(
                f"prediction dir for '{name}' does not exist: {path}",
                param_hint="--pred",
            )

        seen.add(name)
        methods.append((name, path))
    return methods


def visualize_frame(
    sample,
    occupancy_labels,
    voxel_size,
    voxel_range,
    trajectory_history,
    *,
    dataset_type,
    methods,
    gt_mask,
    pred_mask,
    show_gaussians,
    gaussian_streams,
    gaussian_opacity_threshold,
    fps,
    frame_index=0,
):
    # pylint: disable=too-many-arguments,too-many-locals
    # frame_index is the global flat dataset index (not a per-sequence 0-based
    # one), so recordings of different sequences occupy disjoint timeline ranges
    # and can be merged (e.g. via the rerun CLI) without colliding. Sequence
    # timeline for frame-by-frame stepping, plus a real-time timeline so the viewer
    # plays back at the dataset's true capture rate.
    rr.set_time("frame", sequence=frame_index)
    rr.set_time("time", duration=frame_index / fps)

    # --- Ground truth ---------------------------------------------------------
    ocr.render.visualize_cameras(sample)
    ocr.render.visualize_pointcloud(sample)
    ocr.render.visualize_occupancy_bounds(
        voxel_range, entity="ground_truth/occupancy/bounds"
    )
    if hasattr(sample, "labels") and "occupancy" in sample.labels:
        ocr.render.visualize_occupancy(
            sample.labels.occupancy,
            occupancy_labels,
            voxel_size,
            voxel_range,
            entity_root="ground_truth/occupancy",
            mask_type=gt_mask,
        )
    if hasattr(sample, "labels"):
        ocr.render.visualize_boxes(
            sample, trajectory_history, entity_root="annotations"
        )

    # --- Predictions ----------------------------------------------------------
    # One toggle-able sub-tree per method under prediction/<name>/.
    vf = functools.partial(visual_filter_mask, dataset=dataset_type)
    for name, pred_root in methods:
        path = ocr.predictions.prediction_path(
            pred_root, sample.meta.sequence_id, sample.meta.sample_id
        )
        pred = ocr.predictions.load_prediction(path)
        if pred is None:
            log.warning(
                "no '%s' prediction for %s/%s; skipping this method",
                name,
                sample.meta.sequence_id,
                sample.meta.sample_id,
            )
            continue

        root = f"prediction/{name}"
        pred_occupancy = ocr.predictions.build_pred_occupancy(pred, sample)
        # Log both variants as separate toggle-able sub-trees: the visual-filtered
        # prediction (default) and the unfiltered one (raw, no observation mask).
        ocr.render.visualize_occupancy(
            pred_occupancy,
            occupancy_labels,
            voxel_size,
            voxel_range,
            entity_root=f"{root}/occupancy/filtered",
            mask_type=pred_mask,
            visual_filter=vf,
        )
        ocr.render.visualize_occupancy(
            pred_occupancy,
            occupancy_labels,
            voxel_size,
            voxel_range,
            entity_root=f"{root}/occupancy/unfiltered",
            mask_type="none",
        )

        if show_gaussians:
            by_stream = ocr.gaussians.load_gaussians(pred["npz"])
            if gaussian_streams is not None:
                by_stream = {
                    k: v for k, v in by_stream.items() if k in gaussian_streams
                }
                if not by_stream:
                    log.warning(
                        "no gaussian streams matched %s for '%s' (available: %s)",
                        sorted(gaussian_streams),
                        name,
                        list(ocr.gaussians.load_gaussians(pred["npz"]).keys()),
                    )
            for mode in ("semantic", "opacity"):
                ocr.gaussians.visualize_gaussians(
                    by_stream,
                    occupancy_labels,
                    entity_root=f"{root}/gaussians",
                    mode=mode,
                    opacity_threshold=gaussian_opacity_threshold,
                )


@click.command()
@click.option(
    "--dataset",
    "dataset_type",
    type=click.Choice(["nuscenes", "waymo"]),
    default="nuscenes",
)
@click.option(
    "--pred",
    "pred_specs",
    multiple=True,
    metavar="NAME=PATH",
    help="A method to visualize, as 'name=prediction_dir' (dir holds "
    "<sequence_id>/<sample_id>.npz). Optional: omit for ground-truth only. "
    "Repeat for multiple methods; the first is shown by default, the rest "
    "start hidden.",
)
@click.option("--split", default=None, help="Override the default (val) split.")
@click.option("--scene", "scene_index", type=int, default=0)
@click.option(
    "--scene-name",
    "scene_name",
    type=str,
    default=None,
    help="Select a scene by name (e.g. 'scene-0013' for nuScenes), "
    "overriding --scene.",
)
@click.option("--sample", "sample_index", type=int, default=0)
@click.option("--token", "sample_token", type=str, default=None)
@click.option(
    "--num-samples",
    type=int,
    default=1000,
    help="Number of consecutive samples to visualize.",
)
@click.option(
    "--lidar/--no-lidar",
    "load_lidar",
    default=False,
    help="Load and render the LiDAR point cloud. Off by default; the "
    "occupancy gating masks do not depend on it.",
)
@click.option(
    "--gt-mask",
    default="none",
    help="Observation mask gating ground-truth voxels ('none' = all).",
)
@click.option(
    "--pred-mask",
    default="lidar",
    help="Base observation mask gating predictions ('none' = all); gates the "
    "unfiltered prediction and forms the base the visual filter builds on.",
)
@click.option(
    "--gaussians/--no-gaussians",
    "show_gaussians",
    default=True,
    help="Render predicted gaussians (opacity + semantic).",
)
@click.option(
    "--gaussian-streams",
    default="fine",
    help="Comma-separated gaussian streams to render ('all' = every "
    "stream); defaults to the fine stream only.",
)
@click.option(
    "--gaussian-opacity-threshold",
    default=0.01,
    help="Hide gaussians whose opacity (occupancy/solidity) is below " "this value.",
)
@click.option(
    "--fps",
    "fps",
    type=float,
    default=None,
    help="Playback rate (Hz) for the real-time timeline; defaults to "
    "the dataset's keyframe rate (nuScenes/Waymo 2).",
)
@click.option(
    "--save",
    "save_path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write the recording to this .rrd instead of serving.",
)
@click.option("--grpc-port", default=9876, help="gRPC port when serving.")
@click.option("--web-port", default=9090, help="Web viewer port when serving.")
def main(
    dataset_type,
    pred_specs,
    split,
    scene_index,
    scene_name,
    sample_index,
    sample_token,
    num_samples,
    load_lidar,
    gt_mask,
    pred_mask,
    show_gaussians,
    gaussian_streams,
    gaussian_opacity_threshold,
    fps,
    save_path,
    grpc_port,
    web_port,
):
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    utils.log.initialize()

    methods = parse_pred_specs(pred_specs)

    # Application id is per-dataset so the viewer keys the active blueprint (with
    # its dataset-specific camera views) correctly.
    rr.init(f"occupancy_prediction_bundle_{dataset_type}")

    cfg = OmegaConf.create()
    cfg.paths = OmegaConf.create()
    cfg.paths.root = paths.root
    cfg.paths.data = paths.root / "data"
    cfg.paths.cache = paths.root / "cache"
    config.set_global_config(cfg)

    log.info("building %s pipeline (split=%s)", dataset_type, split or "default")
    dataset, occupancy_labels, voxel_size, voxel_range = sensors.build_dataset(
        dataset_type, split=split, load_lidar=load_lidar
    )

    sample_index = sensors.resolve_start_sample(
        dataset_type,
        dataset,
        scene_index,
        sample_index,
        sample_token,
        scene_name=scene_name,
    )

    blueprint = ocr.blueprint.build_blueprint(
        dataset_type, [name for name, _ in methods]
    )

    if save_path is not None:
        log.info("saving recording to '%s'", save_path)
        rr.save(save_path, default_blueprint=blueprint)
    else:
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        rr.serve_web_viewer(
            connect_to=server_uri, open_browser=False, web_port=web_port
        )

        rr.send_blueprint(blueprint)

    trajectory_history: dict = {}

    sample = dataset[sample_index]
    initial_sequence_id = sample.meta.sequence_id
    log.info("scene %s, starting at sample %s", initial_sequence_id, sample_token)

    if gaussian_streams.strip().lower() == "all":
        gaussian_streams_set = None
    else:
        gaussian_streams_set = {
            s.strip() for s in gaussian_streams.split(",") if s.strip()
        }

    render_kwargs = {
        "dataset_type": dataset_type,
        "methods": methods,
        "gt_mask": gt_mask,
        "pred_mask": pred_mask,
        "show_gaussians": show_gaussians,
        "gaussian_streams": gaussian_streams_set,
        "gaussian_opacity_threshold": gaussian_opacity_threshold,
        "fps": fps if fps is not None else sensors.dataset_fps(dataset_type),
    }

    visualize_frame(
        sample,
        occupancy_labels,
        voxel_size,
        voxel_range,
        trajectory_history,
        frame_index=sample_index,
        **render_kwargs,
    )

    for i in range(1, num_samples):
        if sample_index + i >= len(dataset.source):
            break
        sample = dataset[sample_index + i]
        if sample.meta.sequence_id != initial_sequence_id:
            break
        log.info("visualizing sample %d", sample_index + i)
        visualize_frame(
            sample,
            occupancy_labels,
            voxel_size,
            voxel_range,
            trajectory_history,
            frame_index=sample_index + i,
            **render_kwargs,
        )

    if save_path is not None:
        rr.disconnect()
        log.info("recording saved to '%s'", save_path)
        return

    log.info("Rerun bundle served; web viewer at http://localhost:%s", web_port)
    print("Rerun session is running. Press Ctrl+C to exit...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
