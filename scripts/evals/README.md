# Occupancy evaluation (offline)

Score panoptic-occupancy predictions against the in-code ground-truth pipeline (the
same one our models train on). This is the **offline** path — it scores prediction
files already written to disk (from a checkpoint, or an external baseline converted via
[`../data/trackocc/`](../data/trackocc/)). The model code also validates **online**
during training; both use the **same** metrics, so the numbers are directly comparable.

Predictions are read in the shared [prediction format](../README.md#prediction-format);
`--dataset nuscenes|waymo` selects the GT pipeline.

| script | what it does |
|---|---|
| [`eval_occupancy.py`](eval_occupancy.py) | one metric over the whole split; prints a table, optionally dumps JSON |
| [`eval_occupancy_per_scene.py`](eval_occupancy_per_scene.py) | a fresh metric per scene + an aggregate, written to JSON |
| [`print_metrics.py`](print_metrics.py) | reprint the `eval_occupancy` table from its JSON (no re-eval) |
| [`print_scene_metrics.py`](print_scene_metrics.py) | sortable per-scene table from the per-scene JSON |

All four are executable — run by path from any directory.

## Evaluate

```bash
# whole split
./scripts/evals/eval_occupancy.py --dataset nuscenes --preds runs/.../predictions

# per scene -> JSON
./scripts/evals/eval_occupancy_per_scene.py \
    --dataset nuscenes --preds runs/.../predictions --output scenes.json
```

Common options: `--metric [stq|stq1|semantic]` (`stq1` = single-frame STQ; default
`stq`), `--mask valid`, `--split` (override the default `val`), `--limit N`,
`--workers N` (whole sequences sharded across processes), `--device cpu|cuda`.
`eval_occupancy` adds `--format [table|plain]` and an optional `--output` JSON dump;
the per-scene variant's `--output` is required. See `--help` for the rest.

## Print tables from a JSON dump

```bash
# reprint the eval_occupancy table, no re-eval
./scripts/evals/print_metrics.py metrics.json --dataset nuscenes --metric stq

# per-scene table, sorted best-first by a chosen metric
./scripts/evals/print_scene_metrics.py scenes.json --sort AQ
```

`print_scene_metrics` columns are STQ, AQ, mIoU, mIoU things, mIoU stuff, IoU; `--sort`
picks the key, `--ascending` flips direction, `--limit N` caps rows.
