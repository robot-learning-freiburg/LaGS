# Model Zoo

Trained checkpoints for the **LaGS** and **SGE** models. Download them and place
them under `ckpts/` (image-backbone weights under `ckpts/pretrained/`); see the
[README](../README.md) for setup and how to evaluate.

Each model is trained in two stages (see
[Training](../README.md#training)): a single-frame stage (`f1` / `f1p7`) followed by
the final multi-frame tracking model (`f3` / `f3p5`). The **`f3` / `f3p5`
checkpoints are the ones to evaluate**; the single-frame checkpoints are provided as
the stage-1 initialization.

## [LaGS (RA-L 2026)](https://arxiv.org/abs/2602.23172)

| Dataset | Backbone | Experiment | Checkpoint | STQ | AQ | mIoU | IoU |
|---|---|---|---|---|---|---|---|
| nuScenes | VoVNet-99 | [`lags/nuscenes-v99-2s-f3`](../config/experiment/lags/nuscenes-v99-2s-f3.yaml) | `lags_2s-v99-nusc-f3.ckpt` | 32.3 | 25.6 | 40.7 | 67.3 |
| nuScenes | ResNet-50 | [`lags/nuscenes-r50-2s-f3`](../config/experiment/lags/nuscenes-r50-2s-f3.yaml) | `lags_2s-r50-nusc-f3.ckpt` | 27.4 | 21.4 | 36.2 | 64.4 |
| Waymo | VoVNet-99 | [`lags/waymo-v99-2s-f3`](../config/experiment/lags/waymo-v99-2s-f3.yaml) | `lags_2s-v99-waymo-f3.ckpt` | 21.2 | 18.3 | 24.6 | 61.5 |
| Waymo | ResNet-50 | [`lags/waymo-r50-2s-f3`](../config/experiment/lags/waymo-r50-2s-f3.yaml) | `lags_2s-r50-waymo-f3.ckpt` | 18.4 | 15.3 | 22.0 | 59.8 |

Stage-1 (single-frame) initialization checkpoints:
`lags_2s-{v99,r50}-{nusc,waymo}-f1.ckpt` (experiments `…-2s-f1`).

## [SGE (IROS 2026)](https://arxiv.org/abs/2606.30754)

| Dataset | Backbone | Experiment | Checkpoint | STQ | AQ | mIoU | IoU |
|---|---|---|---|---|---|---|---|
| nuScenes | VoVNet-99 | [`sge/nuscenes-v99-f3p5`](../config/experiment/sge/nuscenes-v99-f3p5.yaml) | `sge-v99-nusc-f3p5.ckpt` | 34.4 | 27.2 | 43.5 | 71.7 |
| nuScenes | ResNet-50 | [`sge/nuscenes-r50-f3p5`](../config/experiment/sge/nuscenes-r50-f3p5.yaml) | `sge-r50-nusc-f3p5.ckpt` | 31.8 | 24.8 | 40.9 | 69.5 |
| Waymo | VoVNet-99 | [`sge/waymo-v99-f3p5`](../config/experiment/sge/waymo-v99-f3p5.yaml) | `sge-v99-waymo-f3p5.ckpt` | 21.9 | 19.0 | 25.3 | 61.9 |
| Waymo | ResNet-50 | [`sge/waymo-r50-f3p5`](../config/experiment/sge/waymo-r50-f3p5.yaml) | `sge-r50-waymo-f3p5.ckpt` | 20.0 | 16.8 | 23.7 | 60.3 |

Stage-1 checkpoints: `sge-{v99,r50}-{nusc,waymo}-f1p7.ckpt` (experiments `…-f1p7`).

## Pretrained backbones

`ckpts/pretrained/` holds the image-backbone / detection pre-training weights that
the stage-1 (`f1` / `f1p7`) configs initialize from via `init.checkpoint` (e.g.
`fcos3d_vovnet_imgbackbone.ckpt`,
`cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_….ckpt`). These are only needed to
**train** from scratch, not to evaluate the provided checkpoints.

## Evaluating a checkpoint

```sh
./main.py eval experiment=lags/nuscenes-v99-2s-f3 -p ckpts/lags_2s-v99-nusc-f3.ckpt
./main.py eval experiment=sge/nuscenes-v99-f3p5   -p ckpts/sge-v99-nusc-f3p5.ckpt
```

See [Submission.md](Submission.md) for all options.
