#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluate panoptic-occupancy predictions per scene against pipeline ground truth.

Per-scene variant of ``eval_occupancy.py``: instead of accumulating a single
metric over the whole split, it accumulates (and computes) a fresh metric for
each scene/sequence, plus one aggregate over everything. All per-scene metric
dicts are written to a JSON file; use ``print_scene_metrics.py`` to render a
sortable table from it.

Frames stream in scene order (the ``*Sequential`` source presents each scene's
frames contiguously per rank), so each scene's metric is finalised at its
boundary and reset for the next. Under DDP whole scenes are sharded across ranks
(``distribute_scenes``), so each rank owns a disjoint set of scenes; per-scene
metrics are computed locally (sync disabled) and their result dicts gathered to
the main rank, while the aggregate metric syncs across ranks as usual.

Example:
    ./scripts/evals/eval_occupancy_per_scene.py \\
        --dataset nuscenes --preds runs/.../predictions --output scenes.json
"""

import json
import logging
import os
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click
import rich.progress as rp
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchmetrics import Metric

from tracker import utils
from tracker.data.validation.occupancy import (
    SegmentationTrackingQuality,
    SemanticQuality,
)

# Make the top-level ``scripts`` package importable when this file is run directly
# (`./scripts/evals/eval_occupancy_per_scene.py`): running a file as a script puts only
# its own directory on sys.path, and ``scripts`` lives at the repo root (it is not part
# of the editable-installed ``tracker`` package). Add the repo root -- the parent of
# the ``scripts`` dir -- ourselves.
_REPO_ROOT = next(
    p.parent for p in Path(__file__).resolve().parents if p.name == "scripts"
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import scripts.common.datasets as od  # noqa: E402  pylint: disable=wrong-import-position

log = logging.getLogger(__name__)


def _track(iterable, total, enabled):
    """Yield from ``iterable``, showing an N-of-total progress bar when enabled."""
    if not enabled:
        yield from iterable
        return
    with rp.Progress(
        rp.TextColumn("[progress.description]{task.description}"),
        rp.BarColumn(),
        rp.MofNCompleteColumn(),
        rp.TaskProgressColumn(),
        rp.TimeRemainingColumn(),
        rp.TimeElapsedColumn(),
    ) as pbar:
        task = pbar.add_task("Evaluating", total=total)
        for item in iterable:
            yield item
            pbar.advance(task)


# Invalid GT instance ids (thing voxels without a box) are masked out rather than
# scored, so partial occupancy labelling doesn't penalise tracking quality.
_ALLOW_INVALID_INSTANCES = "ignore"


def _build_metric(metric, labels, mask):
    """Construct the requested occupancy metric.

    ``stq`` is Segmentation & Tracking Quality; ``stq1`` is its
    single-frame variant (track association scored within each frame rather than
    across a sequence). ``semantic`` reports semantic quality / mIoU only.
    """
    if metric in ("stq", "stq1"):
        return SegmentationTrackingQuality(
            labels=labels,
            mask=mask,
            per_frame=metric == "stq1",
            allow_invalid_instances=_ALLOW_INVALID_INSTANCES,
            prefix="",
        )
    if metric == "semantic":
        return SemanticQuality(labels=labels, mask=mask, prefix="")
    raise ValueError(f"unknown metric {metric!r}")


def _disable_sync(metric: Metric) -> Metric:
    """Disable DDP collective sync on a metric and its sub-metrics.

    Per-scene metrics are computed on the rank that owns the scene; each rank
    owns a different set of scenes, so syncing in ``compute()`` would wrongly
    fuse different scenes' states across ranks (and mismatch collective counts).
    We gather the finished result dicts across ranks ourselves instead.

    ``compute()`` gates syncing on the private ``_to_sync`` flag, which torchmetrics
    copies from ``sync_on_compute`` only at construction, so both must be cleared
    (the public one also covers the ``_to_sync`` reset paths inside ``forward``).
    """
    for module in metric.modules():
        if isinstance(module, Metric):
            module.sync_on_compute = False
            module._to_sync = False  # pylint: disable=protected-access
    return metric


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(od.DATASETS, case_sensitive=False),
    required=True,
    help="Dataset whose ground-truth pipeline to evaluate against.",
)
@click.option(
    "--preds",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory of predictions (<sequence_id>/<sample_id>.npz).",
)
@click.option("--split", default=None, help="Override the default (val) split.")
@click.option(
    "--metric",
    type=click.Choice(["stq", "stq1", "semantic"]),
    default="stq",
    help="Metric to compute (stq1 = single-frame STQ; semantic = mIoU only).",
)
@click.option(
    "--mask",
    default="valid",
    help="Occupancy mask used to select evaluated voxels.",
)
@click.option("--device", default="cpu", help="Device for metric accumulation.")
@click.option("--limit", type=int, default=None, help="Evaluate at most N frames.")
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    required=True,
    help="Write the per-scene metric dicts to this JSON file.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Parallel worker processes. Whole sequences are sharded across them.",
)
def main(dataset, preds, split, metric, mask, device, limit, output, workers):
    # pylint: disable=too-many-arguments
    utils.log.initialize()

    args = _WorkerArgs(
        dataset=dataset.lower(),
        preds=str(preds),
        split=split,
        metric=metric,
        mask=mask,
        device=device,
        limit=limit,
        output=output,
        world_size=workers,
    )

    if workers > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_free_port())
        mp.spawn(_worker, args=(args,), nprocs=workers, join=True)
    else:
        _worker(0, args)


@dataclass
class _WorkerArgs:
    # pylint: disable=too-many-instance-attributes
    dataset: str
    preds: str
    split: Optional[str]
    metric: str
    mask: str
    device: str
    limit: Optional[int]
    output: str
    world_size: int


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _worker(rank: int, args: _WorkerArgs) -> None:
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    is_main = rank == 0
    device = f"cuda:{rank}" if args.device == "cuda" else args.device

    if args.device == "cpu":
        # Avoid MKL oversubscription when many workers share the CPU.
        torch.set_num_threads(1)

    if args.world_size > 1:
        backend = "nccl" if args.device == "cuda" else "gloo"
        if args.device == "cuda":
            torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend, rank=rank, world_size=args.world_size)

    if is_main:
        utils.log.initialize()
        log.info("Building %s ground-truth pipeline", args.dataset)

    # The Sequential source shards whole scenes across ranks via the process
    # group (distribute_scenes), so each rank loads/evaluates a disjoint subset
    # while sequence-level tracking stays intact.
    ctx = od.load_context(args.dataset, split=args.split)

    # One aggregate metric over everything this rank sees (synced across ranks in
    # compute()), plus a per-scene metric rebuilt at each scene boundary.
    overall = _build_metric(args.metric, ctx.labels, args.mask).to(device)

    # Progress is shown on the main rank only; its total is this rank's shard
    # (frame_count is rank-aware), capped by --limit.
    total = od.frame_count(ctx)
    if total is not None and args.limit is not None:
        total = min(total, args.limit)

    scene_results: dict[str, dict] = {}
    scene_frames: dict[str, int] = {}
    current_scene: Optional[str] = None
    scene_metric: Optional[Metric] = None

    def finalize(scene: Optional[str], scene_metric: Optional[Metric]) -> None:
        if scene is None or scene_metric is None:
            return
        result = {k: _to_float(v) for k, v in scene_metric.compute().items()}
        result["num_frames"] = scene_frames.get(scene, 0)
        scene_results[scene] = result

    # Iterate the raw stream so the bar advances over every step (incl. padding,
    # which is still transformed in lockstep across ranks); skip padding here.
    n_eval = 0
    n_missing = 0
    for sample in _track(ctx.dataset, total=total, enabled=is_main):
        frame = od.frame_from_sample(sample)
        if frame is None:
            continue

        pred = od.load_prediction(args.preds, frame.sequence_id, frame.sample_id)
        if pred is None:
            n_missing += 1
            continue

        # Scene boundary: finalise the previous scene and start a fresh metric.
        if frame.sequence_id != current_scene:
            finalize(current_scene, scene_metric)
            if frame.sequence_id in scene_results:
                # Scenes stream contiguously per rank; a reappearance would mean
                # the earlier finalisation was premature.
                log.warning(
                    "scene %s reappeared after finalisation; results may be wrong",
                    frame.sequence_id,
                )
            current_scene = frame.sequence_id
            scene_frames[current_scene] = 0
            scene_metric = _disable_sync(
                _build_metric(args.metric, ctx.labels, args.mask).to(device)
            )

        sample, preds_md = od.build_metric_inputs(frame, pred, device=device)
        scene_metric.update(sample, preds_md)
        overall.update(sample, preds_md)

        scene_frames[current_scene] += 1
        n_eval += 1
        if args.limit is not None and n_eval >= args.limit:
            break

    finalize(current_scene, scene_metric)

    # Sum the per-rank frame counts to a global total (collective; all ranks
    # participate, like overall.compute() below).
    counts = torch.tensor([n_eval, n_missing], device=device)
    if args.world_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    n_eval_total, n_missing_total = (int(x) for x in counts.tolist())

    # overall.compute() is a collective under DDP (states reduced across ranks),
    # so every rank must call it; only the main rank reports/writes.
    overall_result = {k: _to_float(v) for k, v in overall.compute().items()}

    # Gather each rank's disjoint per-scene results to the main rank.
    if args.world_size > 1:
        gathered: list[Optional[dict]] = [None] * args.world_size
        dist.all_gather_object(gathered, scene_results)
        merged: dict[str, dict] = {}
        for part in gathered:
            merged.update(part or {})
        scene_results = merged

    if is_main:
        log.info(
            "evaluated %d frames across %d scene(s), %d worker(s) "
            "(%d without predictions)",
            n_eval_total,
            len(scene_results),
            args.world_size,
            n_missing_total,
        )

        payload = {
            "dataset": args.dataset,
            "split": args.split,
            "metric": args.metric,
            "mask": args.mask,
            "num_scenes": len(scene_results),
            "num_frames": n_eval_total,
            "num_missing": n_missing_total,
            "overall": overall_result,
            "scenes": dict(sorted(scene_results.items())),
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("wrote per-scene metrics to '%s'", args.output)

    if args.world_size > 1:
        dist.destroy_process_group()


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    return value


if __name__ == "__main__":
    main()
