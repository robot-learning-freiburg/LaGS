# Setup

## Environments

The project uses **two** environments:

- A **conda/mamba/micromamba** environment (`environment.yml`, name `uv-cu118`)
  that provides *only* the system-level CUDA 11.8 toolchain: `nvcc`, the CUDA
  `-dev` libraries, matching `gcc`/`g++`, plus `ccache` and `uv`. Activating it
  puts this toolchain on `PATH`.
- **[uv](https://docs.astral.sh/uv/)**, which manages *all* Python packages
  (including PyTorch) from `pyproject.toml` into a project-local `.venv`.

Several dependencies (`mmcv`, `gsplat`, and this project's own `tracker.ops.*`)
build CUDA/C++ extensions **from source at install time**, picking up `nvcc` and
the compilers from the activated conda environment. Reference versions:
Python `3.11`, CUDA `11.8`, PyTorch `2.7.1` (cu118 build).

## Installation

1. Create and activate the CUDA toolchain environment (`conda`/`mamba` work the
   same way as `micromamba`):

   ```sh
   micromamba create -f environment.yml
   micromamba activate uv-cu118
   ```

2. Install all Python dependencies and build the extensions:

   ```sh
   uv sync
   ```

   This creates `.venv`, installs the CUDA 11.8 PyTorch build and the rest, and
   compiles the CUDA extensions. Run commands with `uv run …` or activate the
   venv with `source .venv/bin/activate`. Development tools (`black`, `isort`,
   `pre-commit`, `pylint`) are installed by default.

   Optional extras (not needed to train/evaluate the model):

   ```sh
   uv sync --extra vis      # scripts/vis/* front-ends (viser, rerun, imageio)
   uv sync --extra aim      # Aim logger backend
   uv sync --extra mlflow   # MLflow logger backend
   uv sync --extra wandb    # Weights & Biases logger backend
   ```

   The default logger backend is TensorBoard (a core dependency);
   Aim/MLflow/WandB are only needed if selected via `config/trainer/logger/*`
   (see [Configuration.md](Configuration.md)).

3. (Development only) install the pre-commit hooks:

   ```sh
   uv run pre-commit install
   ```

Keep the conda environment **activated whenever you build or run**, so the CUDA
toolchain stays on `PATH`.

> [!NOTE]
> The `./main.py …` and `scripts/…` examples throughout the docs assume the project
> `.venv` is active (`source .venv/bin/activate`), or that you prefix them with
> `uv run` (e.g. `uv run ./main.py …`). The Waymo prep script is the one exception —
> it runs via `uv run --script` in its own isolated environment (see below).

### Building on a GPU machine

> [!IMPORTANT]
> The CUDA extensions compile at install time, so run `uv sync` on a machine with
> the **target NVIDIA GPU** present — PyTorch detects the GPU's compute capability
> and compiles for it.

To build elsewhere — a login/CPU node, or for a different GPU than the build host —
set `TORCH_CUDA_ARCH_LIST` to the target architecture(s) before syncing, e.g.:

```sh
TORCH_CUDA_ARCH_LIST="8.0" uv sync    # compile for e.g. A100 (sm_80)
```

`nvcc` must still be on `PATH` (from the conda environment) for the compile.

### Build caching

Compilation is cached, so rebuilds are cheap and need no setup:

- **ccache** is enabled automatically by `setup.py` when `ccache` is on `PATH`
  (it is, via `environment.yml`). It wraps `nvcc` so compiled objects are reused
  across rebuilds.
- uv only recompiles the editable `tracker` package when the extension **sources**
  change (see `[tool.uv] cache-keys` in `pyproject.toml`) — editing an unrelated
  dependency does not trigger a full recompile.

## Directories

Paths are configured in `config/paths/default.yaml`. The default layout:

```
. (project root)
│
├── cache/          # files generated during preprocessing (auto-created)
├── ckpts/          # model checkpoints (see below)
│   └── pretrained/ # image-backbone weights training initializes from
├── config/         # configuration files
├── data/           # datasets, one entry per dataset (see below)
├── runs/           # run logs, checkpoints and artifacts (auto-created)
└── ...
```

> [!TIP]
> `data/`, `runs/` and `cache/` grow large. On a SLURM cluster, place them on
> workspace/local storage and **symlink** them into the project root (each of these
> directories, or individual datasets under `data/`, may be a symlink).

Trained model checkpoints go in `ckpts/`, and the pre-trained image backbones that
the training configs initialize from go in `ckpts/pretrained/`. The provided
checkpoints and which experiment each corresponds to are listed in the
[Model Zoo](ModelZoo.md).

## Datasets

Each dataset is expected under `data/<name>/`, with the concrete paths declared in
`config/paths/default.yaml` (`paths.datasets.*`). Point them at your data by
editing that file or by symlinking into `data/`. The expected on-disk layout per
dataset:

**nuScenes** (`data/nuscenes/`) — standard nuScenes devkit layout:

```
data/nuscenes/
├── v1.0-trainval/      # (and v1.0-test/, v1.0-mini/) metadata
├── samples/            # keyframe camera + LiDAR data
├── sweeps/             # intermediate sweeps
└── ...
```

Download from [nuscenes.org/nuscenes#download](https://www.nuscenes.org/nuscenes#download)
and extract into `data/nuscenes/`. Training/evaluation on the full split needs
the **Trainval** package (keyframes + sweeps); the `v1.0-mini` package (~4 GB)
is enough for the quick-start mini path in the [README](../README.md).

**Occ3D-nuScenes** (`data/nuscenes-occ3d/`) — occupancy ground truth:

```
data/nuscenes-occ3d/
├── annotations.json
└── gts/
    └── scene-XXXX/...
```

Download the Occ3D-nuScenes GT from the
[Occ3D project page](https://tsinghua-mars-lab.github.io/Occ3D/), which links the
dataset mirrors, and extract it into `data/nuscenes-occ3d/`. It layers on top of the
nuScenes package above — you need both.

**Waymo (TrackOcc)** — split across two roots:

```
data/TrackOcc-waymo/
├── kitti_format/
│   ├── training/                 # image_0../image_4/, velodyne/
│   ├── validation/
│   └── waymo_infos_{train,val}[_mini]_jpg[_with_instances].pkl
└── pano_voxel04/                 # panoptic occupancy GT
    ├── training/<seq>/...
    └── validation/<seq>/...
```

Download the pre-processed TrackOcc Waymo release from Hugging Face:
[`zgchen33/TrackOcc_waymo`](https://huggingface.co/datasets/zgchen33/TrackOcc_waymo).
It provides the panoptic occupancy GT (`pano_voxel04.zip`), the KITTI-format
infos pkls (`kitti_format/waymo_infos_{train,val}_jpg.pkl`), the camera images
(`kitti_format/training/image_*.zip`) and the LiDAR clouds
(`kitti_format/training/velodyne.zip`, split into parts). Extract them into
`data/TrackOcc-waymo/` following the layout above. See the [TrackOcc
repository](https://github.com/Tsinghua-MARS-Lab/TrackOcc) for the
authoritative preparation instructions.

The loader needs `*_with_instances.pkl` infos files, which pair 3D boxes with
occupancy instances. The boxes ship only in the original **Waymo Open
Dataset**, so also download the **Perception Dataset v1.4.3** from
[waymo.com/open](https://waymo.com/open/download) — per-segment TFRecords
grouped by split. Only the `training/` and `validation/` splits are needed:

```
data/waymo_open_dataset_v_1_4_3/
├── training/       # segment-<context_name>_with_camera_labels.tfrecord (one per 20 s sequence)
└── validation/
```

With both in place, build the infos files with the offline preparation step
below. For each frame it reads the 3D boxes from the corresponding Waymo
TFRecord, matches each to the majority occupancy-instance id in the
`pano_voxel04` GT, and writes the augmented `*_with_instances.pkl`. Run it once
per split (shown here for training; for validation replace `train` with `val`
and `training` with `validation`):

```bash
uv run --script ./scripts/data/waymo/create_instances.py \
    --pkl    data/TrackOcc-waymo/kitti_format/waymo_infos_train_jpg.pkl \
    --occ    data/TrackOcc-waymo/pano_voxel04/training \
    --waymo  data/waymo_open_dataset_v_1_4_3/training \
    -o       data/TrackOcc-waymo/kitti_format/waymo_infos_train_jpg_with_instances.pkl
```
> [!IMPORTANT]
> This script cannot run in the main environment — it requires TensorFlow, which
> can't coexist with the project's torch stack. Run it via `uv run --script` (as
> above) so it manages its **own** dependencies from the PEP 723 inline metadata.

See [`scripts/data/waymo/create_instances.py`](../scripts/data/waymo/create_instances.py) for details.


## Data preparation

Most preprocessing happens online on first use and is cached to `cache/`; you can
warm the caches ahead of time with `./main.py preprocess` (see
[Submission.md](Submission.md)). The offline exceptions live under
[`scripts/data/`](../scripts/README.md) and are documented in their own READMEs as well as above.
