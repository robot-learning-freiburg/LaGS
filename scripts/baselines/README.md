# Tracking baselines (offline)

Turn per-frame panoptic-occupancy predictions into **tracked** predictions, for
comparison against the model's own tracking. Each baseline reads and writes
predictions in the shared [prediction format](../README.md#prediction-format) —
score the output with [`../evals/`](../evals/).

Frame grouping and temporal order come from the in-code dataset pipeline
([`../common/datasets.py`](../common/datasets.py)), so `--dataset nuscenes|waymo`
selects the same source the evaluator uses.

## Four methods

| method | script(s) | associates instances by |
|---|---|---|
| per-frame (no tracking) | [`track_perframe.py`](track_perframe.py) | nothing — a fresh id per instance per frame (the metric floor) |
| MinVIS embeddings | [`track_minvis.py`](track_minvis.py) | cosine similarity of `obj_embeddings` |
| IoU / 4D-LCA | [`track_4dlca.py`](track_4dlca.py) | voxel IoU across consecutive frames, same class only |
| AB3DMOT box tracker | 4 scripts + external AB3DMOT | fit boxes → external Kalman tracker → map back (below) |

```bash
# single-shot methods
./scripts/baselines/track_perframe.py --dataset nuscenes --preds preds/ --output tracked/
./scripts/baselines/track_4dlca.py    --dataset nuscenes --preds preds/ --output tracked/ --iou-threshold 0.1
./scripts/baselines/track_minvis.py   --dataset nuscenes --preds preds/ --output tracked/ --similarity-threshold 0.85
```

Common options: `--split` (override the default `val`), `--verbose`,
`--save-stats stats.json`. `track_minvis` needs predictions saved with
`obj_idxes` / `obj_embeddings`.

## AB3DMOT chain

A four-stage pipeline around the **external** [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)
tracker; the box `.txt` files it reads/writes keep the historic KITTI-camera
format, so only our own glue JSON changed (now keyed on `sequence_id` /
`sample_id`, with a per-sequence `scene_index` / `frame_index`).

```bash
# a. fit a 3D box per instance
./scripts/baselines/convert_occupancy_to_boxes.py --dataset nuscenes --preds preds/ --output boxes.json

# b. boxes.json -> AB3DMOT input .txt (scene-XXXX.txt per category)
./scripts/baselines/ab3dmot_boxes_to_input.py --boxes boxes.json --output-dir ab3dmot_in/ --dataset nuscenes

# c. *** run the external AB3DMOT tracker on ab3dmot_in/  ***
#     -> results/.../trk_withid_0/scene-XXXX/*.txt

# d. match detections to tracked boxes -> instance->track mapping
./scripts/baselines/ab3dmot_map_iids.py --boxes boxes.json --tracking-dir results/.../ --output mapping.json

# e. apply the mapping to the predictions
./scripts/baselines/track_via_mapping.py --dataset nuscenes --preds preds/ --mapping mapping.json --output tracked/
```

`scene_index` is the ordinal of a sequence in scene order; `scene-{scene_index:04d}`
is the AB3DMOT scene stem, and `boxes.json` carries the `scene_index → sequence_id`
table so `map_iids` can translate the tracker's output back. The box conversion
supports `nuscenes` / `waymo` only (dataset-specific class maps).

[`common.py`](common.py) holds the shared helpers: streaming predictions in
scene order and writing tracked npz.
