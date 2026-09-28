#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render the metrics table from an ``eval_occupancy.py`` JSON dump.

``eval_occupancy.py --output metrics.json`` writes the flat metric dict; this
reprints the same summary + per-class tables it shows with ``--format table``,
without re-running the evaluation. (For the per-*scene* JSON from
``eval_occupancy_per_scene.py``, use ``print_scene_metrics.py`` instead.)

Example:
    ./scripts/evals/print_metrics.py metrics.json --dataset nuscenes --metric stq
"""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table


def _fmt(value) -> str:
    """Counts (large/integer magnitudes) as thousands-separated ints, else 4dp."""
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{int(round(value)):,}"
        return f"{value:.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


@click.command()
@click.argument(
    "results",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--dataset", default=None, help="Dataset name, shown in the table title.")
@click.option("--metric", default=None, help="Metric name, shown in the table title.")
def main(results: Path, dataset, metric):
    # pylint: disable=too-many-locals
    data = json.loads(results.read_text(encoding="utf-8"))

    console = Console()

    # Summary / non-per-class scalars (e.g. summary/*, subset/*, occupancy/*).
    title = " · ".join(x for x in (dataset, metric) if x) or results.name
    summary = Table(title=title, title_justify="left")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    for key, value in sorted(data.items()):
        if not key.startswith("class/") and isinstance(value, (int, float)):
            summary.add_row(key, _fmt(value))

    # Per-class metrics: pivot class/<name>/<col> into rows=class, cols=<col>.
    per_class: dict[str, dict[str, float]] = {}
    columns: list[str] = []
    for key, value in data.items():
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
        # Classes in first-appearance order, which is the dataset class order the
        # evaluator emits them in.
        for name, cols in per_class.items():
            cells = [_fmt(cols.get(c, "")) for c in columns]
            table.add_row(name, *cells)
        console.print()
        console.print(table)


if __name__ == "__main__":
    main()
