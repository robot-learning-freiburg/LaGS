# Model Zoo

Trained checkpoints for the **LaGS** and **SGE** models, provided as assets of
separate GitHub releases: [`lags/v1.0`](https://github.com/robot-learning-freiburg/lags/releases/tag/lags/v1.0) (LaGS),
[`sge/v1.0`](https://github.com/robot-learning-freiburg/lags/releases/tag/sge/v1.0) (SGE), and
[`pretrained/v1.0`](https://github.com/robot-learning-freiburg/lags/releases/tag/pretrained/v1.0) (initialization weights). Download
them and place them under `ckpts/` (initialization weights under `ckpts/pretrained/`);
see [Setup](Setup.md#checkpoints) for details and the [README](../README.md) for how
to evaluate.

<!-- TODO: update the release tags (lags/v1.0, sge/v1.0, pretrained/v1.0) in all download links if they differ. -->

Each model is trained in two stages (see
[Training](../README.md#training)): a single-frame stage (`f1` / `f1p7`) followed by
the final multi-frame tracking model (`f3` / `f3p5`). The **`f3` / `f3p5`
checkpoints are the ones to evaluate**; the single-frame checkpoints are provided as
the stage-1 initialization.

## License

The model weights are released under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) for
**non-commercial research use only**, independent of the code license
(AGPL-3.0-or-later). They were trained on data with non-commercial terms, including
nuScenes (CC BY-NC-SA 4.0) and DDAD (CC BY-NC-SA 4.0, via the pretrained VoVNet
backbone).

**Waymo models** (`*-waymo-*`) are additionally subject to the
[Waymo Dataset License Agreement for Non-Commercial Use](https://waymo.com/open/terms),
which applies to anyone using or redistributing them. They must not be used in or to
assist the operation of a vehicle, in production systems, or for any primarily
commercial purpose. *These models were made using the Waymo Open Dataset, provided by
Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use, available
at waymo.com/open/terms.*

The [pretrained backbones](#pretrained-backbones) are redistributed from their
original sources and remain subject to their original terms.

## [LaGS (RA-L 2026)](https://arxiv.org/abs/2602.23172)

| Dataset | Backbone | Experiment | Checkpoint | STQ | AQ | mIoU | IoU |
|---|---|---|---|---|---|---|---|
| nuScenes | VoVNet-99 | [`lags/nuscenes-v99-2s-f3`](../config/experiment/lags/nuscenes-v99-2s-f3.yaml) | [`lags_2s-v99-nusc-f3.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/lags/v1.0/lags_2s-v99-nusc-f3.ckpt) | 32.3 | 25.6 | 40.7 | 67.3 |
| nuScenes | ResNet-50 | [`lags/nuscenes-r50-2s-f3`](../config/experiment/lags/nuscenes-r50-2s-f3.yaml) | [`lags_2s-r50-nusc-f3.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/lags/v1.0/lags_2s-r50-nusc-f3.ckpt) | 27.4 | 21.4 | 36.2 | 64.4 |
| Waymo | VoVNet-99 | [`lags/waymo-v99-2s-f3`](../config/experiment/lags/waymo-v99-2s-f3.yaml) | [`lags_2s-v99-waymo-f3.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/lags/v1.0/lags_2s-v99-waymo-f3.ckpt) | 21.2 | 18.3 | 24.6 | 61.5 |
| Waymo | ResNet-50 | [`lags/waymo-r50-2s-f3`](../config/experiment/lags/waymo-r50-2s-f3.yaml) | [`lags_2s-r50-waymo-f3.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/lags/v1.0/lags_2s-r50-waymo-f3.ckpt) | 18.4 | 15.3 | 22.0 | 59.8 |

Stage-1 (single-frame) initialization checkpoints (experiments `…-2s-f1`):
[`lags_2s-v99-nusc-f1.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/lags/v1.0/lags_2s-v99-nusc-f1.ckpt), [`lags_2s-r50-nusc-f1.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/lags/v1.0/lags_2s-r50-nusc-f1.ckpt), [`lags_2s-v99-waymo-f1.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/lags/v1.0/lags_2s-v99-waymo-f1.ckpt), [`lags_2s-r50-waymo-f1.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/lags/v1.0/lags_2s-r50-waymo-f1.ckpt).

## [SGE (IROS 2026)](https://arxiv.org/abs/2606.30754)

| Dataset | Backbone | Experiment | Checkpoint | STQ | AQ | mIoU | IoU |
|---|---|---|---|---|---|---|---|
| nuScenes | VoVNet-99 | [`sge/nuscenes-v99-f3p5`](../config/experiment/sge/nuscenes-v99-f3p5.yaml) | [`sge-v99-nusc-f3p5.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/sge/v1.0/sge-v99-nusc-f3p5.ckpt) | 34.4 | 27.2 | 43.5 | 71.7 |
| nuScenes | ResNet-50 | [`sge/nuscenes-r50-f3p5`](../config/experiment/sge/nuscenes-r50-f3p5.yaml) | [`sge-r50-nusc-f3p5.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/sge/v1.0/sge-r50-nusc-f3p5.ckpt) | 31.8 | 24.8 | 40.9 | 69.5 |
| Waymo | VoVNet-99 | [`sge/waymo-v99-f3p5`](../config/experiment/sge/waymo-v99-f3p5.yaml) | [`sge-v99-waymo-f3p5.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/sge/v1.0/sge-v99-waymo-f3p5.ckpt) | 21.9 | 19.0 | 25.3 | 61.9 |
| Waymo | ResNet-50 | [`sge/waymo-r50-f3p5`](../config/experiment/sge/waymo-r50-f3p5.yaml) | [`sge-r50-waymo-f3p5.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/sge/v1.0/sge-r50-waymo-f3p5.ckpt) | 20.0 | 16.8 | 23.7 | 60.3 |

Stage-1 checkpoints (experiments `…-f1p7`):
[`sge-v99-nusc-f1p7.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/sge/v1.0/sge-v99-nusc-f1p7.ckpt), [`sge-r50-nusc-f1p7.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/sge/v1.0/sge-r50-nusc-f1p7.ckpt), [`sge-v99-waymo-f1p7.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/sge/v1.0/sge-v99-waymo-f1p7.ckpt), [`sge-r50-waymo-f1p7.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/sge/v1.0/sge-r50-waymo-f1p7.ckpt).

## Pretrained backbones

`ckpts/pretrained/` holds the image-backbone weights that the stage-1 (`f1` /
`f1p7`) configs initialize from via `init.checkpoint`. They are only needed to
**train**, not to evaluate the provided checkpoints. Both are third-party weights,
converted to this code base, and remain subject to their original terms:

| Backbone | Checkpoint | Used by | Origin | Terms |
|---|---|---|---|---|
| VoVNet-99 | [`fcos3d_vovnet_imgbackbone.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/pretrained/v1.0/fcos3d_vovnet_imgbackbone.ckpt) | all `v99` stage-1 configs | [DD3D](https://github.com/TRI-ML/dd3d) (pretrained on DDAD15M, trained on nuScenes), as distributed by [DETR3D](https://github.com/WangYueFt/detr3d) / [PETR](https://github.com/megvii-research/PETR) | non-commercial (DDAD and nuScenes: CC BY-NC-SA 4.0) |
| ResNet-50 | [`cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.ckpt`](https://github.com/robot-learning-freiburg/lags/releases/download/pretrained/v1.0/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.ckpt) | all `r50` stage-1 configs | [MMDetection3D nuImages model zoo](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/nuimages) (Cascade Mask R-CNN, COCO-pretrained, trained on nuImages) | non-commercial (nuImages: CC BY-NC-SA 4.0) |

## Evaluating a checkpoint

```sh
./main.py eval experiment=lags/nuscenes-v99-2s-f3 -p ckpts/lags_2s-v99-nusc-f3.ckpt
./main.py eval experiment=sge/nuscenes-v99-f3p5   -p ckpts/sge-v99-nusc-f3p5.ckpt
```

See [Submission.md](Submission.md) for all options.
