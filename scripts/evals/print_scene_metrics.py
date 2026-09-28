#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render a sortable per-scene table from ``eval_occupancy_per_scene.py`` output.

Reads the JSON produced by the per-scene evaluator and prints one row per scene
with the headline metrics (STQ, AQ, mIoU, mIoU things, mIoU stuff, IoU), sorted
by a chosen key metric. The aggregate over all scenes is shown as a footer row.

Example:
    python scripts/evals/print_scene_metrics.py scenes.json --sort AQ
"""

import json
import math
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Displayed metric label -> key in each scene's metric dict. Order defines the
# column order; the first entry is the default sort key.
_COLUMNS = {
    "STQ": "summary/STQ",
    "AQ": "summary/AQ",
    "mIoU": "summary/mIoU",
    "mIoU things": "subset/thing/mIoU",
    "mIoU stuff": "subset/stuff/mIoU",
    "IoU": "occupancy/IoU",
}


def _get(metrics: dict, label: str) -> float:
    """Metric value for a displayed label, or NaN when absent."""
    value = metrics.get(_COLUMNS[label])
    if not isinstance(value, (int, float)):
        return float("nan")
    return float(value)


def _fmt(value: float) -> str:
    return "—" if math.isnan(value) else f"{value:.4f}"


def _sort_key(item, label: str, ascending: bool):
    """Sort scenes by the chosen metric, always pushing NaNs to the end."""
    value = _get(item[1], label)
    if math.isnan(value):
        # NaN sorts last regardless of direction.
        return (1, 0.0)
    return (0, value if ascending else -value)


@click.command()
@click.argument(
    "results",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--sort",
    "sort_by",
    type=click.Choice(list(_COLUMNS), case_sensitive=False),
    default="STQ",
    show_default=True,
    help="Key metric to sort scenes by.",
)
@click.option(
    "--ascending/--descending",
    default=False,
    show_default=True,
    help="Sort direction (default: best-first / descending).",
)
@click.option("--limit", type=int, default=None, help="Show at most N scenes.")
def main(results: Path, sort_by: str, ascending: bool, limit):
    payload = json.loads(results.read_text(encoding="utf-8"))
    scenes: dict[str, dict] = payload.get("scenes", {})

    # Normalise the requested sort label to its canonical (cased) form.
    sort_by = next(c for c in _COLUMNS if c.lower() == sort_by.lower())

    ordered = sorted(scenes.items(), key=lambda it: _sort_key(it, sort_by, ascending))
    if limit is not None:
        ordered = ordered[:limit]

    title = (
        f"{payload.get('dataset', '?')} · {payload.get('metric', '?')} · "
        f"{len(scenes)} scenes · sorted by {sort_by} "
        f"({'asc' if ascending else 'desc'})"
    )
    table = Table(title=title, title_justify="left")
    table.add_column("scene", overflow="fold")
    table.add_column("frames", justify="right")
    for label in _COLUMNS:
        header = f"{label} *" if label == sort_by else label
        table.add_column(header, justify="right")

    for scene, metrics in ordered:
        cells = [_fmt(_get(metrics, label)) for label in _COLUMNS]
        table.add_row(scene, str(metrics.get("num_frames", "")), *cells)

    overall = payload.get("overall")
    if overall is not None:
        cells = [_fmt(_get(overall, label)) for label in _COLUMNS]
        table.add_section()
        table.add_row(
            "[bold]overall[/bold]",
            str(payload.get("num_frames", "")),
            *cells,
        )

    Console().print(table)


if __name__ == "__main__":
    main()
