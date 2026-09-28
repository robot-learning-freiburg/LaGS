# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Self,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import nuscenes.nuscenes as nsc
import nuscenes.utils.splits as nsc_splits
import PIL
import torch
import torchvision
from nuscenes.utils.data_classes import Box
from omegaconf import OmegaConf
from pyquaternion import Quaternion

from ... import config, t3d, utils
from ...config.registry import Registry, RegistryBaseType
from ...utils.types import (
    MetaArray,
    MetaDict,
    PackedArray,
    PackedTensor,
    RefCache,
    Sample,
    SampleMetadata,
    SampleSource,
    TensorArray,
)
from ..transform.collection import _build_transforms
from ..transform.registry import registry as transform
from ..transform.transform import Transform
from .dataset import Dataset, IterableDataset
from .registry import registry as dataset
from .utils import distributed as dist

log = utils.log.get_logger(__name__)

resamplers = Registry("data.nuscenes.resampler")


_nusc_obj_cache = RefCache()


def acquire(version: str, dataroot: str | Path, verbose: bool = False) -> nsc.NuScenes:
    # normalize path
    dataroot = Path(dataroot).resolve(strict=True)
    dataroot = str(dataroot)

    # try to acquire the object or create a new one
    key = (version, dataroot)
    data = _nusc_obj_cache.try_acquire(key)
    if data is None:
        log.info("loading nuscenes dataset (version: %s, path: %s)", version, dataroot)
        data = nsc.NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        _nusc_obj_cache.store(key, data)
    else:
        log.info(
            "using cached nuscenes dataset (version: %s, path: %s)", version, dataroot
        )

    return data


def release(data: nsc.NuScenes | None):
    if data is None:
        return

    _nusc_obj_cache.release((data.version, data.dataroot))


def _parse_version_and_split(version_and_split: str) -> Tuple[str, str]:
    version, split = {
        "v1.0-train": ("v1.0-trainval", "train"),
        "v1.0-val": ("v1.0-trainval", "val"),
        "v1.0-test": ("v1.0-test", "test"),
        "v1.0-mini_train": ("v1.0-mini", "mini_train"),
        "v1.0-mini_val": ("v1.0-mini", "mini_val"),
    }[version_and_split]

    return version, split


def _get_scenes_for_split(
    data: nsc.NuScenes, split: str, verbose: bool = False
) -> Set[Dict[str, Any]]:
    scenes = nsc_splits.create_splits_scenes(verbose=verbose)
    scenes = set(scenes[split])
    scenes = [s for s in data.scene if s["name"] in scenes]

    return scenes


@dataclass
class Labels:
    map: Dict[str, str]
    order: List[str]

    @classmethod
    def from_conf(cls, conf: OmegaConf) -> Self:
        labels = OmegaConf.to_container(conf, resolve=True, throw_on_missing=True)

        return Labels(labels["map"], labels["order"])

    def to_dict(self):
        return {
            "map": self.map,
            "order": self.order,
        }


def _get_empty_annot_check_fn(
    ignore_empty_annots: Literal["none", "radar", "lidar", "both", "any"],
) -> Callable[[Dict[str, Any]], bool]:
    if ignore_empty_annots == "none":
        return lambda rec: True

    if ignore_empty_annots == "radar":
        return lambda rec: rec["num_radar_pts"] > 0

    if ignore_empty_annots == "lidar":
        return lambda rec: rec["num_lidar_pts"] > 0

    if ignore_empty_annots == "both":
        return lambda rec: rec["num_lidar_pts"] > 0 or rec["num_radar_pts"] > 0

    if ignore_empty_annots == "any":
        return lambda rec: rec["num_lidar_pts"] > 0 and rec["num_radar_pts"] > 0

    raise ValueError(
        f"unknown value for 'ignore_empty_annots': '{ignore_empty_annots}'"
    )


class Resampler(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def hparams(self) -> Mapping[str, Any]:
        """
        Any hyperparameters that influence the data sampling performed by this
        sampler.
        """

    def __call__(
        self, data: nsc.NuScenes, samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return self.resample(data, samples)

    @abstractmethod
    def resample(
        self, data: nsc.NuScenes, samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        pass


@resamplers.register
# pylint: disable=too-few-public-methods
class ClassBalancedResampler(Resampler, RegistryBaseType):
    def __init__(
        self,
        labels: Any,
        num_per_class: int | Literal["centerpoint"] = "centerpoint",
        ignore_empty_annots: Literal["none", "radar", "lidar", "both", "any"] = "both",
    ):
        super().__init__()

        self._labels = Labels.from_conf(labels)
        self.num_per_class = num_per_class
        self.ignore_empty_annots = ignore_empty_annots

        self.sample_check_fn = _get_empty_annot_check_fn(ignore_empty_annots)

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "labels": self._labels.to_dict(),
            "num_per_class": self.num_per_class,
            "ignore_empty_annots": self.ignore_empty_annots,
        }

    def resample(
        self, data: nsc.NuScenes, samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        class_names = set(self._labels.map.values()) - {"ignore"}

        # build class-to-samples index
        log.info("building class-to-samples index")

        def category_name(annot_token):
            record = data.get("sample_annotation", annot_token)
            category = record["category_name"]

            # exclude boxes without any lidar or radar points
            if not self.sample_check_fn(record):
                return "ignore"

            return self._labels.map[category]

        class_samples = {name: [] for name in class_names}
        for sample in samples:
            annots = {category_name(token) for token in sample["anns"]}
            annots = annots & class_names

            for name in annots:
                class_samples[name].append(sample)

        # compute number of samples to choose per class
        n_per_class = self.num_per_class
        if n_per_class == "centerpoint":
            # Note: This has been simplified from the original CenterPoint
            # implementation and is equivalent to their code.
            n_per_class = sum(len(s) for s in class_samples.values())
            n_per_class = n_per_class // len(class_names)

        # randomly choose n samples from each class
        log.info("re-sample using %s samples per class", n_per_class)

        # Note: We create an ordered list of per-class sample lists to ensure
        # that things are deterministic if the seed/rng state is controlled.
        # Without this, the order of classes would depend on the order that the
        # dict set, which depends on unpredictable/uncontrollable runtime
        # state.
        class_samples = [class_samples[name] for name in sorted(class_names)]

        samples = []
        for cls_infos in class_samples:
            samples += np.random.choice(cls_infos, n_per_class).tolist()

        return samples


@dataset.register
class NuScenes(Dataset, RegistryBaseType):
    def __init__(
        self,
        root: Union[Path, str],
        split: str,
        resampler: Resampler | OmegaConf | None = None,
        verbose: bool = False,
    ):
        version, split = _parse_version_and_split(split)

        self.root = Path(root)
        self.split = split

        # build the resampler
        if resampler is not None and not isinstance(resampler, Resampler):
            resampler = resamplers.from_config(resampler)

        self.resampler = resampler

        # load the NuScenes sample database (or use a cached one)
        self.data = acquire(version=version, dataroot=root, verbose=verbose)

        # get the scene tokens that are in the selected split
        scenes = _get_scenes_for_split(self.data, split, verbose=verbose)
        scenes = {s["token"] for s in scenes}

        # filter the samples by split
        samples = [x for x in self.data.sample if x["scene_token"] in scenes]

        log.info("loaded nuscenes dataset")
        log.info("    version:       %s", self.data.version)
        log.info("    split:         %s", self.split)
        log.info("    #scenes:       %s", len(scenes))
        log.info("    #samples:      %s", len(samples))

        # resample the samples, if specified
        if resampler is not None:
            log.info("resampling nuscenes dataset")
            samples = resampler(self.data, samples)

            log.info("resampled nuscenes dataset")
            log.info("    #samples:      %s", len(samples))

        self.samples = samples

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "version": self.data.version,
            "split": self.split,
            "resampler": self.resampler.hparams if self.resampler else None,
        }

    @property
    def metadata(self) -> MetaDict:
        return MetaDict()

    def __del__(self):
        if hasattr(self, "data"):
            release(self.data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        return Sample.create(
            source=SampleSource.create(
                dataset_type="nuscenes",
                data=self.data,
                meta=self.metadata,
                sample=sample,
            ),
            meta=SampleMetadata.create(
                dataset_type="nuscenes",
                sample_id=sample["token"],
                sequence_id=sample["scene_token"],
            ),
        )


@dataset.register
class NuScenesSequential(IterableDataset, RegistryBaseType):
    # pylint: disable=too-many-instance-attributes,too-many-ancestors

    def __init__(
        self,
        root: Union[Path, str],
        batch_size: int,
        device_mode: Literal["duplicate", "parallel"] = "parallel",
        batch_mode: Literal["grouped", "parallel"] = "parallel",
        split: str = "train",
        verbose: bool = False,
    ):
        version, split = _parse_version_and_split(split)

        self.root = Path(root)
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.device_mode = device_mode
        self.split = split

        # load the NuScenes sample database (or use a cached one)
        self.data = acquire(version=version, dataroot=root, verbose=verbose)

        # get the scenes that are in the selected split
        scene_dicts = _get_scenes_for_split(self.data, split, verbose=verbose)

        # create a list of ordered samples for each scene
        self.scenes = [self._get_samples_for_scene(scene) for scene in scene_dicts]

        # count the number of samples
        num_samples = [len(scene) for scene in self.scenes]
        num_samples = sum(num_samples)

        log.info("loaded sequential nuscenes dataset")
        log.info("    version:       %s", self.data.version)
        log.info("    split:         %s", self.split)
        log.info("    #scenes:       %s", len(self.scenes))
        log.info("    #samples:      %s", num_samples)

    def _get_samples_for_scene(self, scene: Dict[str, Any]):
        token = scene["first_sample_token"]

        samples = []
        while token:
            sample = self.data.get("sample", token)
            token = sample["next"]

            samples.append(sample)

        assert samples[-1]["token"] == scene["last_sample_token"]
        assert len(samples) == scene["nbr_samples"]

        return samples

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "version": self.data.version,
            "split": self.split,
        }

    @property
    def metadata(self) -> MetaDict:
        return MetaDict()

    def __del__(self):
        if hasattr(self, "data"):
            release(self.data)

    def __iter__(self) -> Iterator[Sample]:
        # distribute the samples across processes/workers according to the scenes
        scene_sizes = [len(scene) for scene in self.scenes]
        sample_ids = dist.distribute_scenes(
            scenes=scene_sizes,
            batch_size=self.batch_size,
            device_mode=self.device_mode,
            batch_mode=self.batch_mode,
        )

        # yield the samples of this process/worker
        for sample_id in sample_ids:
            sample = self.scenes[sample_id.scene_index][sample_id.sample_index]

            yield Sample.create(
                source=SampleSource.create(
                    dataset_type="nuscenes",
                    data=self.data,
                    meta=self.metadata,
                    sample=sample,
                ),
                meta=SampleMetadata.create(
                    dataset_type="nuscenes",
                    sample_id=sample["token"],
                    sequence_id=sample["scene_token"],
                ),
                is_valid=not sample_id.is_padding,
            )


def _get_ego_pose(source, sample_data):
    pose = source.get("ego_pose", sample_data["ego_pose_token"])

    return t3d.RotateTranslate(
        rotate=t3d.Rotate(pose["rotation"], dtype=torch.float64),
        translate=t3d.Translate(pose["translation"], dtype=torch.float64),
    )


def _get_sensor_params(source, sample_data):
    p = source.get("calibrated_sensor", sample_data["calibrated_sensor_token"])

    extrinsic = t3d.RotateTranslate(
        rotate=t3d.Rotate(p["rotation"], dtype=torch.float64),
        translate=t3d.Translate(p["translation"], dtype=torch.float64),
    )

    if "camera_intrinsic" in p.keys() and p["camera_intrinsic"]:
        intrinsic = t3d.Intrinsic(p["camera_intrinsic"], dtype=torch.float64)
    else:
        intrinsic = None

    return extrinsic, intrinsic


@transform.register(namespace="nuscenes")
class LoadImages(Transform, RegistryBaseType):
    """
    Load multi-view images using the specified views/channels.

    Images are returned in RGB format (default) with shape [N, 3, H, W] and
    values in range [0, 1].
    """

    def __init__(self, channels: List[str] = None, mode: Literal["rgb", "bgr"] = "rgb"):
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

        assert mode in ["rgb", "bgr"]

        self.channels = channels
        self.mode = mode

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"nuscenes.{self.__class__.__name__}",
            "channels": list(self.channels),
            "mode": self.mode,
        }

    def _load_image(self, source, sample_data):
        root = Path(source.dataroot)

        # image
        img = PIL.Image.open(root / sample_data["filename"])
        img = img.convert("RGB")
        img = torchvision.transforms.functional.pil_to_tensor(img)

        assert img.shape == (3, sample_data["height"], sample_data["width"])

        if self.mode == "bgr":
            img = img[[2, 1, 0], :, :]

        # metadata
        pose = _get_ego_pose(source, sample_data)
        extrinsic, intrinsic = _get_sensor_params(source, sample_data)

        meta = {
            "timestamp": sample_data["timestamp"],
            "pose": pose,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        }

        return img, MetaDict(meta)

    def _load_multiview_images(self, source, sample, channels):
        tokens = [sample["data"][cam] for cam in channels]
        data = [source.get("sample_data", token) for token in tokens]
        imgs, metas = zip(*[self._load_image(source, x) for x in data])
        imgs = torch.stack(imgs)

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
            "shape": torch.tensor(imgs.shape),
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


@transform.register(namespace="nuscenes")
class LoadLidar(Transform, RegistryBaseType):
    def __init__(
        self, sweeps=1, ref_frame: Literal["sensor", "ego"] = "sensor"
    ) -> None:
        super().__init__()

        self.sweeps = sweeps
        self.ref_frame = ref_frame

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"nuscenes.{self.__class__.__name__}",
            "sweeps": self.sweeps,
            "ref_frame": self.ref_frame,
        }

    def _load_lidar_points(self, source, sample, channel="LIDAR_TOP", sweeps=1):
        sample_data = source.get("sample_data", sample["data"][channel])

        # load transforms
        extrinsic, _ = _get_sensor_params(source, sample_data)

        transforms = {
            "pose": _get_ego_pose(source, sample_data),
            "extrinsic": extrinsic if self.ref_frame == "sensor" else t3d.Identity(),
        }

        # load lidar points (x, y, z, intensity, relative time)
        pcd, timestamps = nsc.LidarPointCloud.from_file_multisweep(
            source, sample, channel, channel, sweeps
        )

        points = torch.tensor(pcd.points.T, dtype=torch.float32)
        coords, intensity = points[:, 0:3], points[:, 3:4]
        timestamps = torch.tensor(timestamps.T, dtype=torch.float32)

        # optionally transform points to ego vehicle frame
        if self.ref_frame == "ego":
            extrinsic = extrinsic.to(dtype=points.dtype)
            coords = extrinsic.transform_points(coords)

        points = torch.cat((coords, intensity, timestamps), dim=1)
        points = PackedTensor(points)

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
            sample.source.data, sample.source.sample, sweeps=self.sweeps
        )
        return sample


@transform.register(namespace="nuscenes")
class LoadLidarTx(Transform, RegistryBaseType):
    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"nuscenes.{self.__class__.__name__}",
        }

    def _load_lidar_points(self, source, sample, channel="LIDAR_TOP"):
        sample_data = source.get("sample_data", sample["data"][channel])

        # metadata
        extrinsic, _ = _get_sensor_params(source, sample_data)

        transforms = {
            "pose": _get_ego_pose(source, sample_data),
            "extrinsic": extrinsic,
        }

        meta = {
            "channel": channel,
            "timestamp": sample_data["timestamp"],
            "transforms": MetaDict(transforms),
        }

        sample = {
            "meta": MetaDict(meta),
        }

        return MetaDict(sample)

    def apply(self, sample):
        sample.points = self._load_lidar_points(
            sample.source.data, sample.source.sample
        )
        return sample


@transform.register(namespace="nuscenes")
class LoadAnnotations(
    Transform, RegistryBaseType
):  # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        labels: Any,
        ref_channel="LIDAR_TOP",
        ref_frame: Literal["sensor", "ego"] = "sensor",
        ignore_empty_annots: Literal["none", "radar", "lidar", "both", "any"] = "both",
        forecasting_steps: int = 0,
    ) -> None:
        super().__init__()

        labels = OmegaConf.to_container(labels, resolve=True)
        self._label_to_cls: Dict[str, str] = labels["map"]
        self._cls_to_id: Dict[str, int] = {k: i for i, k in enumerate(labels["order"])}

        self.ref_channel = ref_channel
        self.ref_frame = ref_frame
        self.ignore_empty_annots = ignore_empty_annots
        self.forecasting_steps = forecasting_steps

        self.sample_check_fn = _get_empty_annot_check_fn(ignore_empty_annots)
        self.dtype = torch.float32

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"nuscenes.{self.__class__.__name__}",
            "ref_channel": self.ref_channel,
            "ref_frame": self.ref_frame,
            "ignore_empty_annots": self.ignore_empty_annots,
        }

    def _load_box(self, source, token, pose, extrinsic) -> Box | None:
        record = source.get("sample_annotation", token)

        # exclude boxes that we set to "ignore"
        name = self._label_to_cls[record["category_name"]]
        if name == "ignore":
            return None

        # exclude boxes without any lidar or radar points
        if not self.sample_check_fn(record):
            return None

        # construct box object
        box = Box(
            record["translation"],
            record["size"],
            Quaternion(record["rotation"]),
            name=name,
            token=record["token"],
        )

        # transform box and add velocity
        box.velocity = source.box_velocity(box.token)

        # transform: global to ego vehicle
        box.translate(pose.inv.translate.vector.numpy())
        box.rotate(pose.inv.rotate.quaternion)

        # transform: ego vehicle to sensor
        if extrinsic is not None:
            box.translate(extrinsic.inv.translate.vector.numpy())
            box.rotate(extrinsic.inv.rotate.quaternion)

        # get instance token
        box.instance_id = source.getind("instance", record["instance_token"])

        return box

    def _load_boxes(self, source, sample) -> List[Box]:
        sd_token = sample["data"][self.ref_channel]
        sample_data = source.get("sample_data", sd_token)

        # load vehicle and sensor transforms
        pose = _get_ego_pose(source, sample_data)

        if self.ref_frame == "sensor":
            extrinsic, _ = _get_sensor_params(source, sample_data)
        else:
            extrinsic = None

        # load boxes and convert to reference frame
        boxes = [self._load_box(source, t, pose, extrinsic) for t in sample["anns"]]
        boxes = [b for b in boxes if b is not None]

        return boxes

    def _load_trajectory(
        self,
        source: nsc.NuScenes,
        pose: t3d.Transform,
        extrinsic: t3d.Transform,
        token: str,
        n: int,
    ) -> Iterator[Box]:
        for _ in range(n):
            record = source.get("sample_annotation", token)

            # construct box object
            box = Box(
                record["translation"],
                record["size"],
                Quaternion(record["rotation"]),
                name=record["category_name"],
                token=record["token"],
            )

            # add velocity
            box.velocity = source.box_velocity(box.token)

            # transform: global to ego vehicle
            box.translate(pose.inv.translate.vector.numpy())
            box.rotate(pose.inv.rotate.quaternion)

            # transform: ego vehicle to sensor
            if extrinsic is not None:
                box.translate(extrinsic.inv.translate.vector.numpy())
                box.rotate(extrinsic.inv.rotate.quaternion)

            # load timestamp
            sample = source.get("sample", record["sample_token"])
            box.timestamp = sample["timestamp"]

            yield box

            token = record["next"]
            if token == "":
                break

    def _load_forecasting_trajectories(
        self,
        source: nsc.NuScenes,
        sample: Sample,
        tokens: Sequence[str],
        trajectory_len: int,
    ) -> MetaDict:
        # load ego pose and extrinsic for current frame
        sample_data = source.get("sample_data", sample["data"][self.ref_channel])
        pose = _get_ego_pose(source, sample_data)

        if self.ref_frame == "sensor":
            extrinsic, _ = _get_sensor_params(source, sample_data)
        else:
            extrinsic = None

        center = np.empty((len(tokens), trajectory_len, 3), dtype=np.float32)
        valid = np.empty((len(tokens), trajectory_len), dtype=bool)

        for i, t in enumerate(tokens):
            # load boxes
            boxes = self._load_trajectory(source, pose, extrinsic, t, trajectory_len)
            boxes = list(boxes)

            # collect box data and pad it to the specified number of frames
            n_valid = len(boxes)

            center[i, :n_valid, :] = np.stack([b.center for b in boxes])
            center[i, n_valid:, :] = boxes[-1].center

            valid[i, :n_valid] = True
            valid[i, n_valid:] = False

        trajectories = {
            "center": PackedTensor(torch.tensor(center, dtype=self.dtype)),
            "valid": PackedTensor(torch.tensor(valid, dtype=torch.bool)),
        }

        return MetaDict(trajectories)

    def _load_labels(self, source: nsc.NuScenes, sample: Any) -> MetaDict:
        # get bounding boxes
        boxes = self._load_boxes(source, sample)
        boxes = np.array(boxes)

        # collect box tokens
        tokens = np.array([b.token for b in boxes])

        # flatten bounding boxes into a single vector
        center = np.array([b.center for b in boxes]).reshape(-1, 3)
        dimension = np.array([b.wlh for b in boxes]).reshape(-1, 3)

        velocity = np.array([b.velocity for b in boxes]).reshape(-1, 3)
        velocity = velocity[:, :2]

        # the velocity field might be missing on some boxes, set that to zero
        velocity = np.nan_to_num(velocity, nan=0.0)

        # Note: This only works in the lidar or global frame...
        rotation = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes])
        rotation = rotation.reshape(-1, 1)

        data = np.concatenate((center, dimension, rotation, velocity), axis=1)

        # collect instance IDs
        instance_ids = np.array([b.instance_id for b in boxes])

        # collect class names and map label names to common class names
        class_names = np.array([b.name for b in boxes])
        class_ids = [self._cls_to_id[name] for name in class_names]

        labels = {
            "boxes": PackedTensor(torch.tensor(data, dtype=self.dtype)),
            "instance_ids": PackedTensor(torch.tensor(instance_ids, dtype=torch.long)),
            "class_ids": PackedTensor(torch.tensor(class_ids, dtype=torch.long)),
            "class_names": PackedArray(class_names),
        }
        labels = MetaDict(labels)

        # load future trajectories for predictions
        if self.forecasting_steps > 0:
            labels.trajectories = self._load_forecasting_trajectories(
                source, sample, tokens, self.forecasting_steps + 1
            )

        return labels

    def apply(self, sample):
        if "labels" not in sample:
            sample.labels = MetaDict()

        sample.labels |= self._load_labels(sample.source.data, sample.source.sample)

        return sample


@transform.register(namespace="nuscenes")
class LoadTimestamp(Transform, RegistryBaseType):
    def __init__(
        self, factor: float | int = 1.0e-6, dtype: torch.dtype | str | None = None
    ):
        """
        Create a new loader transformation for the sample timestamp.

        Args:
            factor (float/number): The factor to multiply the timestamp with.
            dtype (optional, torch.dtype or str): The dtype to use for storing \
               the timestamp.

        Note: The nuscenes timestamp is the unix time stored in microseconds.
        The default factor normalizes it to seconds.

        If dtype is None, it is chosen based on the factor. If factor is an
        integer, dtype will be set as torch.uint64 (reflecting the unix time in
        microseconds for a factor of 1). Otherwise, the dtype will be
        torch.float64.
        """
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, str)
            assert isinstance(dtype, torch.dtype)

        if dtype is None:
            dtype = torch.uint64 if isinstance(factor, int) else torch.float64

        self.factor = factor
        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"nuscenes.{self.__class__.__name__}",
            "factor": self.factor,
            "dtype": str(self.dtype),
        }

    def apply(self, sample):
        sample.timestamp = torch.as_tensor(
            sample.source.sample["timestamp"] * self.factor, dtype=self.dtype
        )
        return sample


@transform.register(namespace="nuscenes")
class LoadMultiFrameData(Transform, RegistryBaseType):
    def __init__(
        self, frames: Sequence[int] | int, transforms: List[Transform]
    ) -> None:
        if isinstance(frames, Sequence):
            frames = tuple(frames)
            assert len(frames) > 0
        else:
            assert frames > 0

        self.transforms = transforms
        self.frames = frames

    @classmethod
    # pylint: disable=arguments-differ
    def from_config(cls, conf: OmegaConf | Mapping[str, Any], *args, **kwargs) -> Self:
        conf_kwargs = config.utils.get_kwargs(conf)

        transforms = _build_transforms(conf_kwargs.pop("transforms"))
        return cls(*args, transforms=transforms, **conf_kwargs, **kwargs)

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "frames": self.frames,
            "transforms": [t.hparams for t in self.transforms],
        }

    def _get_frame_indices(self) -> Sequence[int]:
        if not isinstance(self.frames, Sequence):
            return tuple(range(1 - self.frames, 1))

        return self.frames

    def _next_sample(self, source, sample, field):
        token = sample[field]

        if not token:
            return sample

        return source.get("sample", token)

    def _get_sample_sequence(self, source, sample, field, n):
        samples = []

        for _ in range(n):
            sample = self._next_sample(source, sample, field)
            samples.append(sample)

        return samples

    def _get_sample_frames(self, source, sample, frames):
        max_past, max_future = -min(*frames, 0), max(*frames, 0)
        past = self._get_sample_sequence(source, sample, "prev", max_past)
        future = self._get_sample_sequence(source, sample, "next", max_future)

        samples = []
        for frame_offset in frames:
            if frame_offset < 0:
                samples.append(past[-1 - frame_offset])
            elif frame_offset > 0:
                samples.append(future[frame_offset - 1])
            else:
                samples.append(sample)

        return samples

    def _transform_sample(self, orig: Sample, new: Dict[str, Any]) -> Sample:
        sample = Sample.create(
            source=SampleSource.create(
                dataset_type="nuscenes",
                data=orig.source.data,
                meta=orig.source.meta,
                sample=new,
            ),
            meta=SampleMetadata.create(
                dataset_type="nuscenes",
                sample_id=new["token"],
                sequence_id=new["scene_token"],
                epoch=orig.meta.epoch,
            ),
        )

        assert sample.meta.sequence_id == orig.meta.sequence_id

        for tx in self.transforms:
            sample = tx(sample)

        return sample

    def apply(self, sample: Sample) -> Sample:
        # get the frame indices
        indices = self._get_frame_indices()

        # get the base samples
        frames = self._get_sample_frames(
            source=sample.source.data,
            sample=sample.source.sample,
            frames=indices,
        )

        # run the samples through the individual/per-sample transforms
        frames = [self._transform_sample(sample, s) for s in frames]

        # if any individual sample is None afterwards, abort and return None
        if any(frame is None for frame in frames):
            return None

        # transform the layout from list of structs to struct of lists
        for key in frames[0].keys() - sample.keys():
            sample[key] = [frame[key] for frame in frames]

        # add multi-frame sample metadata
        frames = {
            "indices": torch.tensor(indices, dtype=torch.int32),
            "sample_ids": MetaArray([frame.meta.sample_id for frame in frames]),
        }
        sample.frames = MetaDict(frames)

        return sample


@transform.register(namespace="nuscenes")
class LoadMultiFrameDataRandomized(LoadMultiFrameData):
    def __init__(
        self,
        frames: int,
        transforms: List[Transform],
        skip_probability: float,
        max_skip: int,
    ) -> None:
        assert not isinstance(frames, Sequence), "frames must be an integer"

        super().__init__(frames, transforms)
        self.skip_probability = skip_probability
        self.max_skip = max_skip

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "frames": self.frames,
            "transforms": [t.hparams for t in self.transforms],
            "skip_probability": self.skip_probability,
            "max_skip": self.max_skip,
        }

    def _get_frame_indices(self) -> Sequence[int]:
        indices = [0]

        frame = 0
        for _ in range(1, self.frames):
            skip = 0
            while np.random.rand() < self.skip_probability and skip < self.max_skip:
                skip += 1

            frame = frame + skip + 1
            indices.append(-frame)

        return list(reversed(indices))


@transform.register(namespace="nuscenes")
class LoadSemanticOccupancy(Transform, RegistryBaseType):
    """
    Load the semantic occupancy grid for the current sample from the
    (unofficial) Occ3D nuScenes extension.

    Note: Labels are given in the ego-vehicle frame. The ego vehicle pose (and
    timestamp) is equivalent to the LIDAR_TOP sensor pose.

    Note: The occupancy grid is a 3D voxel grid with the following
    properties:

    - shape [X=200, Y=200, Z=16] (will be loaded as [Z, Y, X]),
    - voxel size: [0.4m, 0.4m, 0.4m],
    - range: [-40m, -40m, -1m, 40m, 40m, 5.4m].

    Label IDs follow the nuScenes lidarseg label IDs, with
    additional ID 17 (free). See [1] for more information. The full list of
    classes is:

    - 0: void/ignore
    - 1: barrier
    - 2: bicycle
    - 3: bus
    - 4: car
    - 5: construction_vehicle
    - 6: motorcycle
    - 7: pedestrian
    - 8: traffic_cone
    - 9: trailer
    - 10: truck
    - 11: driveable_surface
    - 12: other_flat
    - 13: sidewalk
    - 14: terrain
    - 15: manmade
    - 16: vegetation
    - 17: free

    [1]: https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/lidarseg/README.md#classes

    We remap the label IDs according to the provided label config (from
    labels.map to the order induced by labels.all).
    """  # pylint: disable=line-too-long

    def __init__(
        self,
        root: Path | str,
        labels: OmegaConf,
        valid_mask: str | Sequence[str] | None = ("camera", "lidar"),
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.labels = labels

        self.label_map = self._build_label_map(labels)

        if isinstance(valid_mask, str):
            valid_mask = [valid_mask]

        self.valid_mask = valid_mask

        self.index = self._build_index(self.root / "annotations.json")

    def _build_label_map(self, labels: OmegaConf):
        # name to mapped index
        indices = OmegaConf.to_container(labels.all, resolve=True)
        indices = {name: index for index, name in enumerate(indices)}
        indices["ignore"] = labels.ignore_index

        label_map = OmegaConf.to_container(labels.map, resolve=True)
        label_map = {int(i): indices[name] for i, name in label_map.items()}

        return np.vectorize(label_map.get, otypes=[np.int64])

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
        labels = OmegaConf.to_container(self.labels, resolve=True)
        labels["map"] = {str(k): v for k, v in labels["map"].items()}

        return {
            "type": f"nuscenes.{self.__class__.__name__}",
            "root": str(self.root),
            "labels": labels,
        }

    def apply(self, sample):
        token = sample.source.sample["token"]
        infos = self.index[token]

        path = self.root / infos["path"]
        with open(path, "rb") as fd:
            data = np.load(fd)

            semantics = data["semantics"]
            semantics = self.label_map(semantics)
            semantics = torch.as_tensor(semantics, dtype=torch.long)
            semantics = semantics.permute(2, 1, 0)  # [x, y, z] -> [z, y, x]
            semantics = semantics.contiguous()

            mask_camera = data["mask_camera"]
            mask_camera = torch.as_tensor(mask_camera, dtype=torch.bool)
            mask_camera = mask_camera.permute(2, 1, 0)  # [x, y, z] -> [z, y, x]
            mask_camera = mask_camera.contiguous()

            mask_lidar = data["mask_lidar"]
            mask_lidar = torch.as_tensor(mask_lidar, dtype=torch.bool)
            mask_lidar = mask_lidar.permute(2, 1, 0)  # [x, y, z] -> [z, y, x]
            mask_lidar = mask_lidar.contiguous()

        masks = {
            "camera": mask_camera,
            "lidar": mask_lidar,
        }

        if self.valid_mask is not None:
            valid = [masks[m] for m in self.valid_mask]
            valid = torch.stack(valid, dim=0).all(dim=0)
            masks["valid"] = valid

        occupancy = {
            "semantics": semantics,
            "masks": MetaDict(masks),
        }

        if "labels" not in sample:
            sample.labels = MetaDict()

        if "occupancy" not in sample.labels:
            sample.labels.occupancy = MetaDict()

        sample.labels.occupancy |= MetaDict(occupancy)

        return sample
