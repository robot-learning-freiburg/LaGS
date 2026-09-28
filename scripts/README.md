# `scripts/` — offline tooling

Standalone command-line tools that sit *around* the model: preparing datasets,
scoring predictions, running tracking baselines, and visualizing results. None of
them are needed to train — the training/eval entrypoint is [`main.py`](../main.py)
at the repo root — but they cover everything that happens on either side of it.

Every script is **executable and path-runnable** from any working directory
(a shebang plus a `sys.path` bootstrap to the repo root), e.g.:

```bash
./scripts/evals/eval_occupancy.py --dataset nuscenes --preds runs/.../predictions
```

## Prediction format

These tools share one on-disk prediction layout — written by `main.py predict`
([writer](../src/tracker/data/prediction/occupancy.py)) and by the baselines and
converters, consumed by the evaluators and viewers:

```
<pred_root>/<sequence_id>/<sample_id>.npz     # pano_sem / pano_inst in [Z, Y, X], -1 = no instance
```

Frames are joined to ground truth on `sequence_id` / `sample_id`, so the same
commands work across `nuscenes` and `waymo` via a `--dataset` switch.

## Directory map

| directory | what's there | details |
|---|---|---|
| [`evals/`](evals/) | Offline occupancy scoring (STQ / mIoU) against the in-code GT pipeline, plus table reprinting. | [README](evals/README.md) |
| [`baselines/`](baselines/) | Turn per-frame predictions into **tracked** predictions (per-frame, MinVIS, 4D-LCA, AB3DMOT) for comparison. | [README](baselines/README.md) |
| [`vis/`](vis/) | Interactive viewers — **Rerun** (full-sequence) and **Viser** (nicer single-scene renders / image export). | [README](vis/README.md) |
| [`data/`](data/) | Dataset preparation & external-baseline conversion (below). | per-subdir |
| [`common/`](common/datasets.py) | Shared library (not an entrypoint): drives the dataset+transform pipeline so GT/labels/grid match the model, joining predictions on sequence/sample ids. Used by evals, baselines and vis. | — |
| [`utils/`](utils/) | Repo maintenance. `clean_checkpoints.py` strips training-only fields from Lightning checkpoints (~4× smaller) for inference/archival. | — |

### `data/` — dataset prep & conversion

| subdir | purpose | details |
|---|---|---|
| [`data/waymo/`](data/waymo/) | Offline box ↔ occupancy-instance matching (`create_instances.py`) — the one preprocessing step that needs TensorFlow (isolated to that script). | [README](data/waymo/README.md) |
| [`data/trackocc/`](data/trackocc/) | Bridge to the external **TrackOcc** baseline: prepare our datasets into its on-disk format, and convert its raw predictions back to the unified layout for scoring. | [README](data/trackocc/README.md) |

## Typical flows

```bash
# 1. produce predictions (each run gets its own predict/<n> subdir, n auto-increments)
./main.py predict -d runs/my-run                       # -> runs/my-run/predict/0/predictions/<seq>/<sample>.npz

# 2. score them
./scripts/evals/eval_occupancy.py --dataset nuscenes --preds runs/my-run/predict/0/predictions

# 3. look at them
./scripts/vis/viser/visualize_predictions.py --dataset nuscenes --preds runs/my-run/predict/0/predictions --scene 0
```

Dataset-specific one-offs (Waymo instance matching, TrackOcc round-trip) live
under [`data/`](data/) and are documented in their own READMEs.
