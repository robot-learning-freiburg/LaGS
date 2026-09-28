#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Evaluate panoptic-occupancy predictions against pipeline ground truth.

Iterates the dataset's occupancy ground truth (built in code via
``occupancy_dataset``) and joins it with prediction npz files written by the
``occupancy.PanopticOccupancy`` writer (``<preds>/<sequence_id>/<sample_id>.npz``),
feeding both into the project's Segmentation & Tracking Quality (STQ) metric.

Example:
    ./scripts/evals/eval_occupancy.py --dataset nuscenes --preds runs/.../predictions
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

from tracker import utils
from tracker.data.validation.occupancy import (
    SegmentationTrackingQuality,
    SemanticQuality,
)

# Make the top-level ``scripts`` package importable when this file is run directly
# (`./scripts/evals/eval_occupancy.py`): running a file as a script puts only its own
# directory on sys.path, and ``scripts`` lives at the repo root (it is not part of the
# editable-installed ``tracker`` package). Add the repo root -- the parent of the
# ``scripts`` dir -- ourselves.
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
    "--format",
    "output_format",
    type=click.Choice(["table", "plain"]),
    default="table",
    help="Console output: formatted tables or a plain summary list.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write the full metric dict to this JSON file.",
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=1,
    show_default=True,
    help="Parallel worker processes. Whole sequences are sharded across them.",
)
def main(
    dataset, preds, split, metric, mask, device, limit, output_format, output, workers
):
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
        output_format=output_format,
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
    output_format: str
    output: Optional[str]
    world_size: int


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _worker(rank: int, args: _WorkerArgs) -> None:
    # pylint: disable=too-many-locals,too-many-branches
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
    metric_obj = _build_metric(args.metric, ctx.labels, args.mask).to(device)

    # Progress is shown on the main rank only; its total is this rank's shard
    # (frame_count is rank-aware), capped by --limit.
    total = od.frame_count(ctx)
    if total is not None and args.limit is not None:
        total = min(total, args.limit)

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

        sample, preds_md = od.build_metric_inputs(frame, pred, device=device)
        metric_obj.update(sample, preds_md)

        n_eval += 1
        if args.limit is not None and n_eval >= args.limit:
            break

    # Sum the per-rank frame counts to a global total (collective; all ranks
    # participate, like compute() below).
    counts = torch.tensor([n_eval, n_missing], device=device)
    if args.world_size > 1:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    n_eval_total, n_missing_total = (int(x) for x in counts.tolist())

    # compute() is a collective under DDP (states are reduced across ranks), so
    # every rank must call it; only the main rank reports.
    results = {k: _to_float(v) for k, v in metric_obj.compute().items()}

    if is_main:
        log.info(
            "evaluated %d frames across %d worker(s) (%d without predictions)",
            n_eval_total,
            args.world_size,
            n_missing_total,
        )
        if args.output_format == "table":
            _print_results(args.dataset, args.metric, results, ctx.labels)
        else:
            _print_plain(results)

        if args.output is not None:
            Path(args.output).write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
            log.info("wrote metrics to '%s'", args.output)

    if args.world_size > 1:
        dist.destroy_process_group()


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value.tolist()
    return value


def _print_plain(results) -> None:
    """Log the summary scalars as a plain aligned list."""
    for key in sorted(results):
        if key.startswith("summary/") and isinstance(results[key], (int, float)):
            log.info("  %-16s %.4f", key, results[key])


def _fmt(value) -> str:
    """Counts (large/integer magnitudes) as thousands-separated ints, else 4dp."""
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{int(round(value)):,}"
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _print_results(dataset, metric, results, labels) -> None:
    """Print a summary table and a per-class table using rich."""
    # pylint: disable=too-many-locals
    # pylint: disable=import-outside-toplevel
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Summary / non-per-class scalars (e.g. summary/*, subset/*, occupancy/*).
    summary = Table(title=f"{dataset} · {metric}", title_justify="left")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    for key, value in sorted(results.items()):
        if not key.startswith("class/") and isinstance(value, (int, float)):
            summary.add_row(key, _fmt(value))

    # Per-class metrics: pivot class/<name>/<col> into rows=class, cols=<col>.
    per_class: dict[str, dict[str, float]] = {}
    columns: list[str] = []
    for key, value in results.items():
        if not key.startswith("class/") or not isinstance(value, (int, float)):
            continue
        _, name, col = key.split("/", 2)
        per_class.setdefault(name, {})[col] = value
        if col not in columns:
            columns.append(col)

    console.print()
    console.print(summary)

    if per_class:
        table = Table(title="per class", title_justify="left")
        table.add_column("class")
        for col in columns:
            table.add_column(col, justify="right")
        # Keep the dataset's class order for readability.
        for name in (n for n in labels.all if n in per_class):
            cells = [_fmt(per_class[name].get(c, "")) for c in columns]
            table.add_row(name, *cells)
        console.print()
        console.print(table)


if __name__ == "__main__":
    main()
