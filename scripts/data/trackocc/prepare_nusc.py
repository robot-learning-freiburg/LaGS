#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Data-prep for running TrackOcc on nuScenes (referencing the Occ3D ground truth).

Emits the per-frame info pkl TrackOcc reads. The split / voxel grid / label
vocabularies come from the shared configs the model trains on via
``scripts.common.datasets``.

    python scripts/data/trackocc/prepare_nusc.py <split> <out>
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

import click
import numpy as np
import torch

from tracker import data, t3d, utils
from tracker.config import paths

# pylint: disable=protected-access
from tracker.data.dataset.nuscenes import _get_ego_pose, _get_sensor_params
from tracker.utils.types import MetaArray, MetaDict, TensorArray

# Make the top-level ``scripts`` package importable when this file is run directly
# (`python scripts/data/trackocc/prepare_nusc.py`), not only via
# ``python -m scripts.data.trackocc.prepare_nusc``: running a file as a script puts
# only its own directory on sys.path, and ``scripts`` lives at the repo root (it is
# not part of the editable-installed ``tracker`` package). So add the repo root -- the
# parent of the ``scripts`` dir -- ourselves; under ``-m`` it is already there (no-op).
_REPO_ROOT = next(
    p.parent for p in Path(__file__).resolve().parents if p.name == "scripts"
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Shared dataset helpers (split / voxel grid / label configs / global-config setup).
from scripts.common.datasets import (  # noqa: E402  pylint: disable=wrong-import-position
    _DATA,
    _GRID,
    _label_conf,
    _set_global_config,
)

log = logging.getLogger(__name__)


def _denumpy(obj):
    """Recursively replace numpy arrays/scalars with plain Python lists/scalars.

    The infos pkl is consumed by the TrackOcc env (numpy 1.x / py3.8), but this
    script runs under numpy >= 2.0, whose pickled arrays/scalars reference
    ``numpy._core`` -- a module numpy 1.x cannot import (``ModuleNotFoundError:
    No module named 'numpy._core'``). Stripping numpy out entirely makes the pkl
    version-agnostic; TrackOcc wraps these values in ``torch.Tensor`` / ``np.array``
    at load time, so lists are accepted unchanged.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):  # numpy scalar (e.g. np.float32, np.int64)
        return obj.item()
    if isinstance(obj, dict):
        return {k: _denumpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_denumpy(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_denumpy(v) for v in obj)
    return obj


# Detection (box) and occupancy label vocabularies, loaded from the shared configs
# the model trains on (config/data/labels/{detection,occupancy}/nuscenes.yaml). The
# occupancy ``instance`` list there matches the detection classes, as before.
box_labels = _label_conf("detection", "nuscenes")
occupancy_labels = _label_conf("occupancy", "nuscenes")

# Model voxel grid (voxel_size [x,y,z], voxel_range [xmin,ymin,zmin, xmax,ymax,zmax]).
voxel_size = torch.tensor(_GRID["nuscenes"][0], dtype=torch.float32)
voxel_range = torch.tensor(_GRID["nuscenes"][1], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Pipeline transforms
# ---------------------------------------------------------------------------
#
# Why these are local classes and not the shared pipeline:
#
# The nuScenes prep does not need to *load* any data -- TrackOcc reads the Occ3D GT
# npz, the raw camera images, and the LiDAR sweep natively. It only needs an info pkl
# that *references* those files (plus the calibration to project them). So the
# transforms below are path-recording analogues of the standard pipeline nodes: they
# store the ``sample_data['filename']`` and pose/calibration into the sample instead
# of decoding pixels / points / voxels. That different *output* (references, not
# tensors) is exactly why they cannot reuse ``_nuscenes_pipeline`` from the shared
# module, which builds the real data-loading pipeline for eval/viz -- same inputs,
# different product. Each class notes the standard transform it parallels.
#
# Two nodes are not path-recording: ``BuildInstanceLabels`` runs the real
# ``occupancy.VoxelInstanceData`` (there is no precomputed instance file to point at)
# but stores the *path* of the cache it builds; and the ``Prepare*TxFused`` nodes are
# plain compute steps that fuse the per-camera transforms the pkl records.


class LoadImages(data.transform.Transform):
    """Record per-camera image paths + pose/extrinsic/intrinsic (no pixels loaded).

    Path-recording analogue of ``nuscenes.LoadImages``: stores each camera's
    ``sample_data['filename']`` and calibration on ``sample.images`` so the info pkl
    can reference the raw images (and project depth) without decoding them here.
    """

    def __init__(self, channels: list[str] = None):
        """
        Args:
            channels (List[str]): List of channels to load images for. Images \
                will be stacked in the specified channel order.
            mode ("rgb" | "bgr"): Color channel mode. By default, images will
                be loaded in RGB mode.
        """

        super().__init__()

        if channels is None:
            channels = [
                "CAM_FRONT",
                "CAM_FRONT_RIGHT",
                "CAM_FRONT_LEFT",
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
            ]

        self.channels = channels

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"nuscenes.{self.__class__.__name__}",
            "channels": list(self.channels),
        }

    def _load_image(self, source, sample_data):
        # image
        path = sample_data["filename"]

        # metadata
        pose = _get_ego_pose(source, sample_data)
        extrinsic, intrinsic = _get_sensor_params(source, sample_data)

        meta = {
            "timestamp": sample_data["timestamp"],
            "pose": pose,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        }

        return path, MetaDict(meta)

    def _load_multiview_images(self, source, sample, channels):
        tokens = [sample["data"][cam] for cam in channels]
        sd = [source.get("sample_data", token) for token in tokens]
        imgs, metas = zip(*[self._load_image(source, x) for x in sd])

        transforms = {
            "pose": TensorArray([m.pose for m in metas]),
            "extrinsic": TensorArray([m.extrinsic for m in metas]),
            "intrinsic": TensorArray([m.intrinsic for m in metas]),
        }

        norm = {
            "mean": torch.zeros(3),
            "std": torch.ones(3),
        }

        meta = {
            "channel": MetaArray(channels),
            "timestamp": torch.tensor([m.timestamp for m in metas], dtype=torch.int64),
            "transforms": MetaDict(transforms),
            "norm": MetaDict(norm),
        }

        images = {
            "data": imgs,
            "meta": MetaDict(meta),
        }

        return MetaDict(images)

    def apply(self, sample):
        sample.images = self._load_multiview_images(
            source=sample.source.data,
            sample=sample.source.sample,
            channels=self.channels,
        )

        return sample


class LoadLidar(data.transform.Transform):
    """Record the LiDAR sweep path + transforms (no points loaded).

    Path-recording analogue of ``nuscenes.LoadLidar``: stores
    ``sample_data['filename']`` and the sensor pose/extrinsic on ``sample.points``;
    TrackOcc reads the ``.pcd.bin`` itself.
    """

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"nuscenes.{self.__class__.__name__}",
        }

    def _load_lidar_points(self, source, sample, channel="LIDAR_TOP"):
        sample_data = source.get("sample_data", sample["data"][channel])

        # load transforms
        extrinsic, _ = _get_sensor_params(source, sample_data)

        transforms = {
            "pose": _get_ego_pose(source, sample_data),
            "extrinsic": extrinsic,
        }

        # load lidar points
        points = sample_data["filename"]

        # combine
        meta = {
            "channel": channel,
            "timestamp": sample_data["timestamp"],
            "transforms": MetaDict(transforms),
        }

        sample = {
            "data": points,
            "meta": MetaDict(meta),
        }

        return MetaDict(sample)

    def apply(self, sample):
        sample.points = self._load_lidar_points(
            sample.source.data, sample.source.sample
        )
        return sample


class LoadSemanticOccupancy(data.transform.Transform):
    """Record the Occ3D semantic-GT path (no voxels loaded).

    Path-recording analogue of ``nuscenes.LoadSemanticOccupancy``: resolves each
    sample's Occ3D ``gt_path`` and stores it as a string on
    ``sample.labels.occupancy.semantics`` for TrackOcc to read.
    """

    def __init__(
        self,
        root: Path | str,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.index = self._build_index(self.root / "annotations.json")

    def _build_index(self, annots: Path) -> dict[str, Any]:
        # Load the Occ3D index file.
        with open(annots, "rb") as fd:
            infos = json.load(fd)

        # Note: data["train_split"] and data["val_split"] match the original
        # nuScenes splits.

        # The index file is structured in scenes. Each scene has a dict mapping
        # sample token to sample info/data. Essentially, the sample info is a
        # mashed together version of the original nuScenes sample information,
        # which is split across multiple dicts in nuScenes. Since all of our
        # management is modular and we already take care of this, just re-index
        # everything into a sample-to-gt dict.
        infos = infos["scene_infos"]

        index = {}
        for _scene_id, scene in infos.items():
            for sample_id, sample in scene.items():
                assert sample_id not in index

                index[sample_id] = {
                    "path": sample["gt_path"],
                }

        return index

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"nuscenes.{self.__class__.__name__}",
            "root": str(self.root),
        }

    def apply(self, sample):
        token = sample.source.sample["token"]
        infos = self.index[token]

        path = infos["path"]

        if "labels" not in sample:
            sample.labels = MetaDict()

        if "occupancy" not in sample.labels:
            sample.labels.occupancy = MetaDict()

        sample.labels.occupancy.semantics = path

        return sample


class BuildInstanceLabels(data.transform.Transform):
    """Record the voxel-instance-id cache path (labels are built, not loaded here).

    Thin wrapper over the real ``occupancy.VoxelInstanceData`` -- there is no
    precomputed instance file to reference, so it builds/looks up the instance-id
    cache and stores that file's relative path on
    ``sample.labels.occupancy.instance_ids``. Mirrors the shared
    ``occupancy.BuildInstanceLabels`` (same ``VoxelInstanceData``), but records the
    path instead of attaching the loaded tensor.
    """

    def __init__(
        self,
        source: data.dataset.Dataset,
        voxel_size: tuple[float, float, float],
        voxel_range: tuple[float, float, float, float, float, float],
        voxel_labels: Mapping[int, str],
        box_labels: Mapping[int, str],
    ):
        # pylint: disable=redefined-outer-name
        self.data = data.transform.occupancy.VoxelInstanceData(
            source=source,
            voxel_size=voxel_size,
            voxel_offset=voxel_range[:3],
            voxel_labels=voxel_labels,
            box_labels=box_labels,
        )

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"occupancy.{self.__class__.__name__}",
            **self.data.cache_hparams,
        }

    def apply(self, sample):
        root = self.data.root
        info = self.data.index[sample.meta.sample_id]
        path = root / info["data"]
        path = path.relative_to(paths.root / "cache")

        if "labels" not in sample:
            sample.labels = MetaDict()

        if "occupancy" not in sample.labels:
            sample.labels.occupancy = MetaDict()

        sample.labels.occupancy.instance_ids = str(path)
        return sample


@np.vectorize(otypes="O", excluded={"tx_lidar_to_global"})
def _compute_lidar_to_camera_tx_fused(pose, extrinsic, tx_lidar_to_global):
    # transformation from global to camera frome at time of camera capture
    tx_global_to_camera = t3d.Sequential(
        # global frame to ego vehicle (at image timestamp)
        pose.inv,
        # ego vehicle to camera frame
        extrinsic.inv,
    )

    # full transformation from lidar to camera
    tx_lidar_to_image = t3d.Sequential(
        tx_lidar_to_global,
        tx_global_to_camera,
    )

    # fuse and store transformation
    return tx_lidar_to_image.fused().matrix.numpy()


class PrepareLidarToCameraTxFused(data.transform.Transform):
    """Compute + store the fused lidar->camera transform per camera (pkl ``lidar2cam``).

    Not path-recording: a plain compute step that folds pose/extrinsic into one
    lidar->camera matrix per camera (there is no standard-pipeline node for this; the
    training pipeline projects on the fly).
    """

    def __init__(self, dtype: torch.dtype | str = torch.float32):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)

        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def _compute_tx(self, points_tx, images_tx):
        tx_lidar_to_global = t3d.Sequential(
            # lidar sensor to ego vehicle frame (at lidar timestamp)
            points_tx.extrinsic,
            # ego vehicle to global frame
            points_tx.pose,
        )

        lidar_to_camera = _compute_lidar_to_camera_tx_fused(
            pose=images_tx.pose.data,
            extrinsic=images_tx.extrinsic.data,
            tx_lidar_to_global=tx_lidar_to_global,
        )

        lidar_to_camera = np.array(lidar_to_camera.tolist())
        lidar_to_camera = torch.from_numpy(lidar_to_camera)
        lidar_to_camera = lidar_to_camera.to(dtype=self.dtype)

        images_tx.lidar_to_camera = lidar_to_camera

    def apply(self, sample):
        self._compute_tx(
            points_tx=sample.points.meta.transforms,
            images_tx=sample.images.meta.transforms,
        )

        return sample


class PrepareLidarEgoToCameraTxFused(data.transform.Transform):
    """Compute + store the fused ego->camera transform per camera (pkl ``ego2cam``).

    Like :class:`PrepareLidarToCameraTxFused` but starts from the ego frame: the
    lidar->global chain drops the lidar extrinsic and uses the ego pose alone.
    """

    def __init__(self, dtype: torch.dtype | str = torch.float32):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)

        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def _compute_tx(self, points_tx, images_tx):
        tx_lidar_to_global = t3d.Sequential(
            # ego vehicle to global frame
            points_tx.pose,
        )

        lidar_ego_to_camera = _compute_lidar_to_camera_tx_fused(
            pose=images_tx.pose.data,
            extrinsic=images_tx.extrinsic.data,
            tx_lidar_to_global=tx_lidar_to_global,
        )

        lidar_ego_to_camera = np.array(lidar_ego_to_camera.tolist())
        lidar_ego_to_camera = torch.from_numpy(lidar_ego_to_camera)
        lidar_ego_to_camera = lidar_ego_to_camera.to(dtype=self.dtype)

        images_tx.ego_to_camera = lidar_ego_to_camera

    def apply(self, sample):
        self._compute_tx(
            points_tx=sample.points.meta.transforms,
            images_tx=sample.images.meta.transforms,
        )

        return sample


def build_source_dataset(dataset):
    transforms = [
        data.dataset.nuscenes.LoadAnnotations(
            labels=box_labels, ignore_empty_annots="none", ref_frame="ego"
        ),
        data.dataset.nuscenes.LoadSemanticOccupancy(
            root=_DATA / "nuscenes-occ3d",
            labels=occupancy_labels,
        ),
    ]

    return data.dataset.wrapper.DatasetWrapper(dataset, transforms)


def build_dataset(split: str):
    dataset = data.dataset.NuScenes(
        root=_DATA / "nuscenes",
        split=split,
    )

    transforms = [
        LoadImages(),
        LoadLidar(),
        LoadSemanticOccupancy(root=_DATA / "nuscenes-occ3d"),
        BuildInstanceLabels(
            source=build_source_dataset(dataset),
            voxel_size=voxel_size,
            voxel_range=voxel_range,
            voxel_labels={
                i: n
                for i, n in enumerate(occupancy_labels.all)
                if n in occupancy_labels.instance
            },
            box_labels=dict(enumerate(box_labels.order)),
        ),
        # done
        data.dataset.nuscenes.LoadTimestamp(),
        data.dataset.transform.lidar_base.PrepareLidarEgoToGlobalTxFused(),
        PrepareLidarToCameraTxFused(),
        PrepareLidarEgoToCameraTxFused(),
    ]

    return data.dataset.wrapper.DatasetWrapper(dataset, transforms)


def _get_sample_tokens_for_scene(nusc, scene):
    token = scene["first_sample_token"]

    samples = []
    while token:
        samples.append(token)

        sample = nusc.get("sample", token)
        token = sample["next"]

    assert samples[0] == scene["first_sample_token"]
    assert samples[-1] == scene["last_sample_token"]
    assert len(samples) == scene["nbr_samples"]

    return samples


def build_sample_index(dataset):
    source = dataset.source

    scenes = data.dataset.nuscenes._get_scenes_for_split(source.data, source.split)
    scenes = sorted(scenes, key=lambda s: s["name"])

    sample_indices = {}
    for scene in scenes:
        scene_index = int(scene["name"].split("-")[-1], 10)

        # get sample tokens for scene
        sample_tokens = _get_sample_tokens_for_scene(source.data, scene)

        for frame_index, token in enumerate(sample_tokens):
            if token in sample_indices:
                raise ValueError(f"Duplicate sample token found: {token}")

            assert 0 <= frame_index < 1000, "Frame index must be in [0, 1000)"

            sample_index = scene_index * 1000 + frame_index
            sample_indices[token] = sample_index

    return sample_indices


def build_index(dataset):
    infos = []

    sample_index = build_sample_index(dataset)

    indices = range(len(dataset))
    indices = utils.progress.track(indices, "Building dataset...")
    for i in indices:
        sample = dataset[i]

        # collect sample information
        info = {
            "sample_idx": sample_index[sample.meta.sample_id],
            "timestamp": sample.timestamp.item(),
            "ego2global": sample.points.meta.transforms.ego_to_global.numpy(),
            "lidar_points": {
                "lidar_path": str(sample.points.data),
            },
            "images": {},
            "occupancy": {
                "semantic": str(sample.labels.occupancy.semantics),
                "instance": str(sample.labels.occupancy.instance_ids),
            },
        }

        for img_index, channel in enumerate(sample.images.meta.channel.data):
            img_path = sample.images.data[img_index]
            cam2img = sample.images.meta.transforms.intrinsic[img_index].matrix.numpy()
            ego2cam = sample.images.meta.transforms.ego_to_camera[img_index].numpy()
            lidar2cam = sample.images.meta.transforms.lidar_to_camera[img_index].numpy()

            info["images"][channel] = {
                "img_path": str(img_path),
                "cam2img": cam2img,
                "ego2cam": ego2cam,
                "lidar2cam": lidar2cam,
            }

        # store sample information
        infos.append(info)

    # sort by sample_idx
    infos = sorted(infos, key=lambda x: x["sample_idx"])

    return infos


@click.command()
@click.argument("split", nargs=1)
@click.argument("out", nargs=1)
def main(split: str, out: str):
    utils.log.initialize()

    # Minimal global config so the pipeline can resolve its paths (e.g. sample cache).
    _set_global_config()

    # load dataset
    log.info("loading dataset for split '%s'", split)
    dataset = build_dataset(split)

    # build index
    log.info("building dataset index for split '%s'", split)
    infos = build_index(dataset)

    # save index. Strip numpy and use protocol 4 so the pkl loads in the TrackOcc
    # env (numpy 1.x / py3.8); see _denumpy.
    with open(out, "wb") as fd:
        pickle.dump(_denumpy(infos), fd, protocol=4)

    log.info("saved dataset to '%s'", out)


if __name__ == "__main__":
    main()
