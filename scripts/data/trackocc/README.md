# TrackOcc data prep & prediction conversion

These scripts bridge this repo and the external **TrackOcc** occupancy-tracking
baseline, in both directions:

- **Forward (prepare):** turn our datasets into the on-disk format TrackOcc
  trains/tests on.
- **Backward (convert):** turn TrackOcc's raw predictions back into our unified
  occupancy format, so the repo's evaluator can score them exactly like our own
  models.

Both directions keep the data in parity with our own pipeline: the GT TrackOcc
trains/tests on, and the predictions we score, come from the same code our models
use.

## The round trip

```
  this repo                         TrackOcc env                 this repo
 ┌──────────────────┐   infos pkl  ┌──────────────┐  raw npz   ┌────────────────────┐
 │ prepare_nusc     │ ───────────► │ train / test │ ─────────► │ convert_predictions│
 │                  │  (+ GT/pts)  │  TrackOcc    │ per-frame  │  → unified format  │
 └──────────────────┘              └──────────────┘            └─────────┬──────────┘
                                                                         │  <seq>/<sample>.npz
                                                                         ▼
                                                               ┌────────────────────┐
                                                               │  eval_occupancy    │
                                                               │  (STQ / mIoU)      │
                                                               └────────────────────┘
```

| script | direction | datasets | what it emits |
|---|---|---|---|
| [`prepare_nusc.py`](prepare_nusc.py) | forward | `nuscenes` | infos pkl only (**references** the existing Occ3D / image / LiDAR files) |
| [`convert_predictions.py`](convert_predictions.py) | backward | `nuscenes`, `waymo` | unified `<seq>/<sample>.npz` predictions |

Waymo has **no prepare step** — TrackOcc and our eval read the *same* Waymo infos
pkl, so only the backward conversion is needed.

## Running

The scripts are executable — run them by path from any directory.

### 1. Prepare inputs

**nuScenes** — builds only the infos pkl (TrackOcc reads the Occ3D GT natively).
Here `<out>` is the **pkl file path**:

```bash
./scripts/data/trackocc/prepare_nusc.py <split> <out.pkl>
# e.g.
./scripts/data/trackocc/prepare_nusc.py v1.0-val data/TrackOcc-nusc/nuscenes_val_infos.pkl
```

### 2. Run TrackOcc (external)

In the TrackOcc env, point its config at the prepared infos pkl and train/test as
usual. It writes one npz per frame to `.../occupancy_pred/<sample_idx>.npz`.

### 3. Convert predictions back

`convert_predictions.py` remaps TrackOcc's raw output to our unified format: it keys
frames by `(sequence_id, sample_id)`, restores the pipeline-native `(Z, Y, X)` axis
order, and shifts instance ids back to the `-1 = none` convention.

```bash
# nuScenes (the default) — mapping is reconstructed from the DB
./scripts/data/trackocc/convert_predictions.py \
    --input  ../baselines/TrackOcc/test/.../occupancy_pred \
    --output output/trackocc_nusc_preds

# Waymo — read straight from the shared Waymo infos pkl
./scripts/data/trackocc/convert_predictions.py \
    --dataset waymo \
    --input  ../baselines/TrackOcc/test/.../occupancy_pred \
    --output output/trackocc_waymo_preds
```

| option | meaning |
|---|---|
| `--dataset` | `nuscenes` (default) or `waymo`. Selects how frame identity is recovered and whether the semantic remap applies. |
| `--input DIR` | TrackOcc prediction dir, searched recursively for `<sample_idx>.npz`. |
| `--output DIR` | Writes `<output>/<sequence_id>/<sample_id>.npz`. |
| `--nuscenes-root`, `--nuscenes-split` | nuScenes DB root / version-split to index (default `v1.0-val`). |
| `--waymo-root`, `--waymo-split` | Waymo infos root / split (default `validation`). |

Frames whose `sample_idx` is not in the index are skipped with a warning.

### 4. Evaluate

The converted directory plugs straight into the repo's evaluator as `--preds`:

```bash
./scripts/evals/eval_occupancy.py \
    --dataset nuscenes \
    --preds   output/trackocc_nusc_preds
```

This scores TrackOcc against the *same* ground-truth pipeline our own models are
evaluated on (STQ, per-frame STQ, or semantic mIoU via `--metric`).
