<h1 align="center">LaGS &amp; SGE</h1>

<p align="center">
  Official implementation of Latent Gaussian Splatting (LaGS) and Streaming Gaussian Encoding (SGE),<br>
  two methods for camera-based 4D panoptic occupancy tracking.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.23172"><img src="https://img.shields.io/badge/arXiv-LaGS%20(2602.23172)-b31b1b.svg" alt="LaGS arXiv"></a>
  <a href="https://arxiv.org/abs/2606.30754"><img src="https://img.shields.io/badge/arXiv-SGE%20(2606.30754)-b31b1b.svg" alt="SGE arXiv"></a>
  <a href="https://lags.cs.uni-freiburg.de/"><img src="https://img.shields.io/badge/Project-LaGS-1f6feb.svg" alt="LaGS project page"></a>
  <a href="https://sge.cs.uni-freiburg.de/"><img src="https://img.shields.io/badge/Project-SGE-1f6feb.svg" alt="SGE project page"></a>
  <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11">
</p>

<h3 align="center">Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking</h3>
<p align="center">
  <b>IEEE RA-L 2026</b><br>
  Maximilian Luz<sup>1</sup>, Rohit Mohan<sup>1</sup>, Thomas Nürnberg<sup>2</sup>, Yakov Miron<sup>2,3</sup>, Daniele Cattaneo<sup>1</sup>, Abhinav Valada<sup>1</sup><br>
  <a href="https://lags.cs.uni-freiburg.de/">Project Page</a> &nbsp;|&nbsp; <a href="https://arxiv.org/abs/2602.23172">arXiv</a>
</p>

<h3 align="center">Streaming Gaussian Encoding for 4D Panoptic Occupancy Tracking</h3>
<p align="center">
  <b>IEEE/RSJ IROS 2026</b><br>
  Maximilian Luz<sup>1</sup>, Thomas Nürnberg<sup>2</sup>, Yakov Miron<sup>2,3</sup>, Abhinav Valada<sup>1</sup><br>
  <a href="https://sge.cs.uni-freiburg.de/">Project Page</a> &nbsp;|&nbsp; <a href="https://arxiv.org/abs/2606.30754">arXiv</a>
</p>

<p align="center">
  <sup>1</sup> University of Freiburg &nbsp;&nbsp; <sup>2</sup> Bosch Research &nbsp;&nbsp; <sup>3</sup> University of Haifa
</p>

<p align="center">
  <img src="docs/assets/readme/scene_1071_gt_lags_sge.gif" width="95%" alt="GT, LaGS, and SGE qualitative result on nuScenes scene 1071">
  <br>
  <sub>Qualitative nuScenes sequence: ground truth, LaGS, and SGE.</sub>
</p>

## Overview

This repository provides a camera-based framework for **4D panoptic occupancy
tracking** — jointly predicting semantic occupancy and temporally consistent
instance identities from surround-view cameras. It implements two Gaussian-based
models that turn multi-view image observations into compact latent 3D scene
representations before decoding panoptic occupancy and tracks.

### LaGS in Brief

**Latent Gaussian Splatting (LaGS)** revisits 4D panoptic occupancy tracking
through a sparse latent Gaussian scene representation. Multi-view image features
are lifted into 3D, summarized as feature-bearing Gaussian keypoints, processed
with hierarchical point-based attention, and splatted back into a dense voxel
volume for panoptic occupancy decoding. This gives the model adaptive spatial
support and long-range 3D interactions without relying only on dense voxel
operators.

### SGE in Brief

**Streaming Gaussian Encoding (SGE)** extends the Gaussian representation from
LaGS into a persistent streaming scene memory. Instead of rebuilding the
volumetric representation independently at every frame, it propagates latent
Gaussian queries with ego-motion compensation, uses opacity and confidence to
retain well-supported scene structure, and refreshes weak slots with new
observations. This adds representation-level temporal coherence, especially
through occlusion, while staying compatible with the LaGS-style decoder.

Both methods include trained checkpoints and configs for Occ3D-nuScenes and Waymo;
see the [Model Zoo](docs/ModelZoo.md) for the complete checkpoint list and metrics.

## Quick Start

This runs a provided SGE checkpoint through evaluation, prediction, and visualization on
the small **nuScenes mini** split — a fast, single-GPU path to see the models run. See
[docs/Setup.md](docs/Setup.md) for the full setup guide.

**1. Install.** The CUDA toolchain (`nvcc` + compilers) is provided by a
conda/micromamba environment; all Python packages, including PyTorch, are managed by
[uv](https://docs.astral.sh/uv/) from `pyproject.toml`.

```sh
micromamba create -f environment.yml   # CUDA 11.8 toolchain + uv
micromamba activate uv-cu118
uv sync                                # .venv with PyTorch (cu118) + extensions
```

The CUDA extensions compile at install time, so run `uv sync` on a machine with the
target GPU.

Then get the data. The mini path needs only the nuScenes **mini** package (not the full
trainval set) plus the matching Occ3D-nuScenes occupancy ground truth:

- Download `v1.0-mini` from [nuscenes.org](https://www.nuscenes.org/nuscenes#download)
  and extract it into `data/nuscenes/`.
- Download the [Occ3D-nuScenes](https://github.com/Tsinghua-MARS-Lab/Occ3D) occupancy
  ground truth and extract it into `data/nuscenes-occ3d/`.

```
data/
├── nuscenes/            # nuScenes mini package
│   ├── v1.0-mini/       # metadata
│   ├── samples/         # keyframe camera + LiDAR data
│   └── sweeps/          # intermediate sweeps
└── nuscenes-occ3d/      # occupancy ground truth
    ├── annotations.json
    └── gts/
        └── scene-XXXX/...
```

To keep the data elsewhere, point `paths.datasets.*` in
[config/paths/default.yaml](config/paths/default.yaml) at it (or symlink into `data/`).

**2. Get a checkpoint.** Download the final VoVNet-99 SGE model and place it under
`ckpts/`:

```
ckpts/sge-v99-nusc-f3p5.ckpt
```

See the [Model Zoo](docs/ModelZoo.md) for the full checkpoint list and [Models](#models)
for the `ckpts/` layout. The image backbones under `ckpts/pretrained/` are only needed
for training.

> [!NOTE]
> The `./main.py …` and `scripts/…` examples below assume the project `.venv` is
> active (`source .venv/bin/activate`), or that you prefix them with `uv run`
> (e.g. `uv run ./main.py …`).

**3. Evaluate on the mini split** (computes metrics):

```sh
./main.py eval experiment=sge/nuscenes-v99-f3p5 -p ckpts/sge-v99-nusc-f3p5.ckpt \
    data.source.val.source.split=v1.0-mini_val \
    trainer.devices=1 trainer.strategy=auto \
    meta.run.name=sge-nusc-mini
```

`data.source.val.source.split` selects the mini split (omit it for the full `v1.0-val`,
see [Evaluation](#evaluation)); `trainer.devices` sets how many GPUs to use; and
`meta.run.name` names the output directory (`runs/sge-nusc-mini/<version>/`). For
multi-GPU and SLURM runs, see [docs/Submission.md](docs/Submission.md).
This should report an STQ of roughly 32.7 on the mini split.

**4. Write predictions on the mini split** (for visualization or offline scoring):

```sh
./main.py predict experiment=sge/nuscenes-v99-f3p5 -p ckpts/sge-v99-nusc-f3p5.ckpt \
    data.source.predict.source.split=v1.0-mini_val \
    trainer.devices=1 trainer.strategy=auto \
    meta.run.name=sge-nusc-mini
```

This writes per-frame `.npz` files under
`runs/sge-nusc-mini/<version>/predict/<n>/predictions/` (`<n>` auto-increments per
predict run).

**5. Visualize.** Point a viewer at that `predictions/` directory:

```sh
./scripts/vis/rerun/visualize_occupancy.py --dataset nuscenes --scene 0 \
    --pred sge=runs/sge-nusc-mini/<version>/predict/<n>/predictions
```

See [`scripts/vis/`](scripts/vis/README.md) for the full option list and the Viser
rendering backend.

You can also score the same files offline with
[`scripts/evals/`](scripts/evals/) (e.g. `eval_occupancy.py`).

## Models

We provide trained checkpoints and the image-backbone weights used to initialize
training. Download them and place them under `ckpts/`:

- trained model checkpoints (e.g. `lags_2s-v99-nusc-f3.ckpt`,
  `sge-v99-waymo-f3p5.ckpt`) → `ckpts/`
- pre-trained image backbones that the training configs initialize from →
  `ckpts/pretrained/`

The main models use the VoVNet-99 (`v99`) backbone; ResNet-50 (`r50`) variants are
also provided, for both nuScenes and Waymo. See the [Model Zoo](docs/ModelZoo.md) for
the full list of checkpoints and their experiment configs.

## Training

Both models train in **two stages** — a single-frame stage followed by a
multi-frame stage, where the second stage is initialized from the first via
`init.checkpoint`. The stages are `f1` → `f3` for LaGS and `f1p7` → `f3p5` for SGE:

```sh
# LaGS on nuScenes (VoVNet-99)
./main.py train experiment=lags/nuscenes-v99-2s-f1
./main.py train experiment=lags/nuscenes-v99-2s-f3 \
    init.checkpoint=runs/<f1-run>/<version>/last.ckpt

# SGE on nuScenes (VoVNet-99)
./main.py train experiment=sge/nuscenes-v99-f1p7
./main.py train experiment=sge/nuscenes-v99-f3p5 \
    init.checkpoint=runs/<f1p7-run>/<version>/last.ckpt
```

The first stage already initializes its image backbone from `ckpts/pretrained/`;
only the second stage needs `init.checkpoint` set explicitly, to the checkpoint the
first stage produced.

## Evaluation

Evaluate a checkpoint against an experiment config — either a provided checkpoint
(`-p`) or a run you trained (`-d`):

```sh
# a provided checkpoint (final, multi-frame model)
./main.py eval experiment=lags/nuscenes-v99-2s-f3 -p ckpts/lags_2s-v99-nusc-f3.ckpt
./main.py eval experiment=sge/nuscenes-v99-f3p5   -p ckpts/sge-v99-nusc-f3p5.ckpt

# a run you trained (its own config + last.ckpt)
./main.py eval -d runs/<name>/<version>
```

To **write predictions** to disk instead of computing metrics, use `predict` (same
`-d` / `-p` / `experiment=…` selectors). It writes per-frame `.npz` files under the
run's `predict/<n>/predictions/` directory, which the offline tools then consume:

```sh
./main.py predict -d runs/<name>/<version>
```

See [docs/Submission.md](docs/Submission.md) for all command options,
[`scripts/evals/`](scripts/evals/) for offline scoring, and
[`scripts/vis/`](scripts/vis/) for visualizing predictions.

## Documentation

- [Setup](docs/Setup.md) — environments, installation, directories, datasets, and data preparation
- [Configuration](docs/Configuration.md) — the config system and overrides
- [Runs](docs/Runs.md) — run directories, metadata, resuming
- [Submission](docs/Submission.md) — running and submitting jobs (SLURM and others)
- [Model Zoo](docs/ModelZoo.md) — provided checkpoints and their experiment configs
- [`scripts/`](scripts/README.md) — offline tooling (evaluation, baselines, visualization, data prep)

## Citation

If you use this code in your research or find it otherwise helpful, please consider citing the relevant paper(s):

```bibtex
@article{luz2026lags,
  title   = {Latent Gaussian Splatting for 4D Panoptic Occupancy Tracking},
  author  = {Luz, Maximilian and Mohan, Rohit and N{\"u}rnberg, Thomas and Miron, Yakov and Cattaneo, Daniele and Valada, Abhinav},
  journal = {IEEE Robotics and Automation Letters (RA-L)},
  year    = {2026},
  volume  = {11},
  number  = {8},
  doi     = {10.1109/LRA.2026.3703990}
}

@inproceedings{luz2026sge,
  title     = {Streaming Gaussian Encoding for 4D Panoptic Occupancy Tracking},
  author    = {Luz, Maximilian and N{\"u}rnberg, Thomas and Miron, Yakov and Valada, Abhinav},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2026}
}
```

## Acknowledgements

This codebase adapts code from several open-source projects, including COTR,
TrackOcc, MMDetection3D / MMCV, DETR3D, PETR, and OccFormer. We thank the
authors for releasing their work.

## License

This project is licensed under AGPL-3.0-or-later (see [`LICENSE`](LICENSE)).

The ST-Refiner component of PF-Track is not part of this repository, and
`use_st_reasoner: true` will therefore not run successfully because the required module
is missing.

The codebase is designed to operate without this component; keep
`use_st_reasoner: false` (or omit the flag) to maintain the AGPL-3.0-or-later
compatible execution path.
