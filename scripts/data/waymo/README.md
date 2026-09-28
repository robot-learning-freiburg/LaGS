# Waymo data prep — box ↔ occupancy-instance matching

The Waymo occupancy loader ([`src/tracker/data/dataset/waymo_to.py`](../../../src/tracker/data/dataset/waymo_to.py))
reads a **`..._with_instances.pkl`** infos file in which every frame carries an
`instances` record — a 3D box (center/size/heading), class, velocity and Waymo
track id — *already matched to the occupancy ground-truth instance ids*. This
matching is what [`create_instances.py`](create_instances.py) produces.

Unlike nuScenes (all preprocessing is online + cached on first run), this one
step is **offline**, because the full box set lives only in the Waymo
**TFRecords**.

## What it does

For every frame that has occupancy GT (Waymo stores it every 5th frame), it:

1. reads the frame's full 360° `laser_labels` (3D boxes) from the TFRecord,
2. rasterizes each box to an axis-aligned voxel region and takes the **majority
   occupancy instance id** in that region,
3. resolves conflicts (two boxes → same instance) by keeping the box with the
   most matched voxels, and drops boxes with no occupancy match,
4. writes the resulting `{waymo_id: {bbox_3d, instance_id, bbox_label, velocity,
   original_waymo_id, num_lidar_pts}}` back onto each sample as `instances`.

Frames are addressed through the `sample_idx` encoding (`zfill(7)` → digits
`[1:4]` = scene id, `[4:7]` = frame number), so the occupancy file for a sample
is `<occ>/<scene_id>/<frame_number>_04.npz`.

## Inputs and output

```
 base infos pkl ┐
 (cam_sync only)│
                │        ┌──────────────────┐
 occ GT npz ────┼──────► │ create_instances │ ──► <infos>_with_instances.pkl
 (instances vol)│        │  (needs TF)      │      (what the loader reads)
                │        └──────────────────┘
 TFRecords ─────┘        boxes ← laser_labels
```

| flag | meaning | default |
|---|---|---|
| `--pkl` | base infos pkl, e.g. `waymo_infos_train_jpg.pkl` | *(required)* |
| `--occ` | occupancy GT dir, e.g. `pano_voxel04/training` | *(required)* |
| `--waymo` | original Waymo TFRecord dir (`segment-*_with_camera_labels.tfrecord`) | *(required)* |
| `--output` / `-o` | output pkl, e.g. `waymo_infos_train_jpg_with_instances.pkl` | *(required)* |
| `--voxel-size` | grid voxel size `x y z` | `0.4 0.4 0.4` |
| `--voxel-range` | grid extent `x_min y_min z_min x_max y_max z_max` | `-40 -40 -1 40 40 5.4` |

```bash
# Note: this will automatically set up a dedicated ephemeral environment via uv
./scripts/data/waymo/create_instances.py \
    --pkl    data/TrackOcc-waymo/kitti_format/waymo_infos_train_jpg.pkl \
    --occ    data/TrackOcc-waymo/pano_voxel04/training \
    --waymo  data/waymo_open_dataset_v_1_4_3/training \
    -o       data/TrackOcc-waymo/kitti_format/waymo_infos_train_jpg_with_instances.pkl
```

Run it once per split (`train`, `val`, and the `*_mini` variants). **Re-run it if
the voxel grid changes** — the matching is grid-dependent (box voxelization uses
`--voxel-size` / `--voxel-range`); the box extraction itself is not.
