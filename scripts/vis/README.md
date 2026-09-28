# Visualization scripts

Interactive viewers for panoptic **occupancy** — ground truth and model
predictions — across the supported datasets (`nuscenes`, `waymo`).

There are two rendering backends:

| Backend | Directory | Notes |
|---------|-----------|-------|
| **Rerun** | [`rerun/`](rerun/) | Better for interactive rendering of full sequences: a real-time timeline, cameras, LiDAR, per-class occupancy, boxes/trajectories and predicted gaussians, with several methods side by side. |
| **Viser** | [`viser/`](viser/) | Nicer-looking renders of a single scene. Can also export each frame of a scene to image files, which the Rerun viewer does not currently support. |

All scripts are self-contained — run them directly, from any working directory:

```bash
./scripts/vis/rerun/visualize_occupancy.py  --dataset nuscenes --scene 0
./scripts/vis/viser/visualize_predictions.py --dataset nuscenes --scene-name scene-0013
```

## Predictions

Viewers read the `.npz` files written per frame by the prediction writer
([`src/tracker/data/prediction/occupancy.py`](../../src/tracker/data/prediction/occupancy.py))
under `<run>/predictions/`. Point a viewer at that directory with
`--pred`/`--preds` (see each script below); omit it to view ground truth only.
Frames without a matching prediction fall back to ground truth.

Gaussians (the Rerun `--gaussians` overlay) are only present in the `.npz` when
the writer was run with `store_gaussians` enabled — off by default. Set it via a
config override when generating predictions, for example:

```bash
./main.py predict -d runs/my-run prediction.store_gaussians=true
```

## Selecting what to view

The CLI options are shared across the scripts:

- **Dataset** — `--dataset {nuscenes,waymo}` (default `nuscenes`).
- **Scene** — `--scene N` picks the *N*-th scene (deterministic ordering), or
  `--scene-name NAME` selects by human-readable name (e.g. `scene-0013` for
  nuScenes).
- **Split** — `--split` overrides the default (val) split.
- **Frame** — `--sample` sets the starting/initial frame.
- **Masks** — occupancy can be gated by an observation mask
  (`camera` / `lidar` / `valid` / `any` / `none`), so you only see voxels the
  sensors actually observed.

Run any script with `--help` for the full, up-to-date option list.

## Scripts

### `rerun/visualize_occupancy.py`

Streams a scene to the Rerun web viewer with a dataset-specific blueprint
(camera layout, timeline). Renders ground truth and, via one or more
`--pred name=DIR`, any number of methods at once (the first shown by default,
the rest toggleable). Optional LiDAR (`--lidar`), predicted gaussians
(`--gaussians`) and per-source mask gating (`--gt-mask` / `--pred-mask`).
`--save out.rrd` writes a recording instead of serving.

```bash
# Ground truth + two methods
./scripts/vis/rerun/visualize_occupancy.py --dataset nuscenes --scene 0 \
    --pred baseline=runs/a/predictions --pred ours=runs/b/predictions
```

### `viser/visualize_predictions.py`

A single scene's panoptic occupancy as lit voxel cubes — instances coloured per
track, "stuff" per semantic class. With `--preds DIR` it renders predicted
occupancy (gated by the per-dataset visual filter, toggle with
`--no-visual-filter` + `--mask`); with no `--preds` it renders ground truth.

```bash
./scripts/vis/viser/visualize_predictions.py \
    --dataset nuscenes --preds runs/ours/predictions --scene 0
```

### `viser/visualize_annotations.py`

Ground-truth occupancy voxels plus wireframe detection boxes, future forecast
splines and ego-motion-compensated past trajectories. Everything belonging to
one track shares a colour. `--mask` hard-masks the voxels; per-category
visibility checkboxes let you compose what is shown. Past trajectories need
per-frame ego pose, so pass `--lidar`.

```bash
./scripts/vis/viser/visualize_annotations.py \
    --dataset nuscenes --scene 0 --lidar --mask lidar
```
