# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import io
import pickle
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
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
    Union,
)

import numpy as np
import PIL
import torch
import torchvision
from nuscenes.utils.data_classes import Box
from omegaconf import OmegaConf
from pyquaternion import Quaternion

# Waymo Open Dataset
from ... import config, t3d, utils
from ...config.registry import RegistryBaseType
from ...utils.types import (
    MetaArray,
    MetaDict,
    PackedArray,
    PackedTensor,
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


def dict_to_namespace(d: Any) -> Union[SimpleNamespace, List[Any], Any]:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    return d


def _get_pkl_data(root: Union[str, Path], split: str) -> List[str]:
    root = Path(root)
    pkl_filename = {
        "training": "waymo_infos_train_jpg_with_instances.pkl",
        "validation": "waymo_infos_val_jpg_with_instances.pkl",
        "training_mini": "waymo_infos_train_mini_jpg_with_instances.pkl",
        "validation_mini": "waymo_infos_val_mini_jpg_with_instances.pkl",
    }[split]
    split_pkl_path = root / pkl_filename
    if not split_pkl_path:
        log.error("No pkl file found for split %s", split)
    with open(split_pkl_path, "rb") as f:
        data = pickle.load(f)
    data_list = data["data_list"]
    filtered_data_list = []
    new_ind = 0
    for item in data_list:
        sample_idx = item["sample_idx"]
        if sample_idx % 5 == 0:
            item["index"] = new_ind
            item["root"] = root
            item["split"] = split.removesuffix("_mini")
            filtered_data_list.append(item)
            new_ind += 1

    data["data_list"] = filtered_data_list
    return dict_to_namespace(data)


def _get_scenes_for_split(data_list: List[Any]) -> List[Dict[str, Any]]:
    scenes = []
    grouped = defaultdict(list)

    # Group frames by scene_id (1:4 of sample_idx)
    for frame in data_list:
        scene_id = int(str(frame.sample_idx).zfill(7)[1:4])
        grouped[scene_id].append(frame)

    # Build scene info dict for each scene
    for scene_id, frames in grouped.items():
        frames = sorted(frames, key=lambda f: int(str(frame.sample_idx).zfill(7)))
        first_frame, last_frame = frames[0], frames[-1]

        scene = {
            "token": first_frame.context_name,  # unique scene identifier
            "log_token": first_frame.context_name[:16],  # optional shortened version
            "nbr_samples": len(frames),
            "first_sample_token": first_frame.sample_idx,
            "first_sample_idx": first_frame.index,
            "last_sample_token": last_frame.sample_idx,
            "last_sample_idx": last_frame.index,
            "name": f"scene-{scene_id:04d}",
            "description": f"Scene {scene_id} with {len(frames)} frames",
        }
        scenes.append(scene)

    return scenes


def _get_empty_annot_check_fn(
    ignore_empty_annots: Literal["none", "radar", "lidar", "both", "any"],
) -> Callable[[Dict[str, Any]], bool]:
    if ignore_empty_annots == "none":
        return lambda rec: True

    if ignore_empty_annots == "radar":
        return lambda rec: rec.num_radar_pts > 0

    if ignore_empty_annots == "lidar":
        return lambda rec: rec.num_lidar_pts > 0

    if ignore_empty_annots == "both":
        return lambda rec: rec.num_lidar_pts > 0 or rec.num_radar_pts > 0

    if ignore_empty_annots == "any":
        return lambda rec: rec.num_lidar_pts > 0 and rec.num_radar_pts > 0

    raise ValueError(
        f"unknown value for 'ignore_empty_annots': '{ignore_empty_annots}'"
    )


def release(data: "SimpleNamespace | None") -> None:
    """
    Recursively clear attributes in a SimpleNamespace to free memory.

    Args:
        data (SimpleNamespace | None): Waymo dataset object or None.
    """
    if data is None:
        return

    for key, value in vars(data).items():
        if isinstance(value, type(data)):  # another nested SimpleNamespace
            release(value)
        elif isinstance(value, list):
            # Clear nested structures
            for item in value:
                if isinstance(item, type(data)):
                    release(item)
            value.clear()
        elif isinstance(value, dict):
            for subval in value.values():
                if isinstance(subval, type(data)):
                    release(subval)
            value.clear()

        setattr(data, key, None)


# ---------------------------
# Core dataset classes
# ---------------------------


@dataset.register
class WaymoTO(Dataset, RegistryBaseType):
    """
    Minimal Waymo dataset wrapper to match the project interface used by nuscenes.py.

    root: directory that contains training, validation, testing subfolders with .tfrecord files
    split: one of train, val, test
    labels: OmegaConf with fields map and order like in nuscenes.py
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
    ):
        self.root = Path(root)
        self.split = split

        self.data = _get_pkl_data(self.root, split)
        scenes = _get_scenes_for_split(self.data.data_list)
        scenes = {s["token"] for s in scenes}
        samples = self.data.data_list

        log.info("loaded waymo dataset")
        log.info("    version:       %s", self.data.metainfo.version)
        log.info("    split:         %s", self.split)
        log.info("    #scenes:       %s", len(scenes))
        log.info("    #samples:      %s", len(samples))

        self.samples = samples

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "version": self.data.metainfo.version,
            "split": self.split,
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

        # Convert SimpleNamespace to dict for compatibility with transforms
        sample_dict = {
            "token": sample.sample_idx,
            "scene_token": sample.context_name,
            "index": sample.index,
            "timestamp": sample.timestamp,
        }

        return Sample.create(
            source=SampleSource.create(
                dataset_type="waymo",
                data=self.data.data_list,
                meta=self.metadata,
                sample=sample_dict,
            ),
            meta=SampleMetadata.create(
                dataset_type="waymo",
                sample_id=sample.sample_idx,
                sequence_id=sample.context_name,
            ),
        )


@dataset.register
class WaymoTOSequential(IterableDataset, RegistryBaseType):
    """
    Iterable version that yields consecutive frames from a single TFRecord at a time.
    """

    # pylint: disable=too-many-instance-attributes,too-many-ancestors

    def __init__(
        self,
        root: Union[Path, str],
        batch_size: int,
        device_mode: Literal["duplicate", "parallel"] = "parallel",
        batch_mode: Literal["grouped", "parallel"] = "parallel",
        split: str = "train",
    ):
        self.root = Path(root)
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.device_mode = device_mode
        self.split = split
        self.data = _get_pkl_data(self.root, split)
        scene_dicts = _get_scenes_for_split(self.data.data_list)

        # build simple file sized scene list for distribution helper
        self.scenes = [self._get_samples_for_scene(scene) for scene in scene_dicts]

        # count the number of samples
        num_samples = [len(scene) for scene in self.scenes]
        num_samples = sum(num_samples)

        log.info("loaded sequential waymo dataset")
        log.info("    version:       %s", self.data.metainfo.version)
        log.info("    split:         %s", self.split)
        log.info("    #scenes:       %s", len(self.scenes))
        log.info("    #samples:      %s", num_samples)

    def _get_samples_for_scene(self, scene: Dict[str, Any]):
        index = scene["first_sample_idx"]
        prev_index = None
        samples = []
        while 1:
            sample = self.data.data_list[index]
            sample = {
                "token": int(str(sample.sample_idx).zfill(7)),
                "index": index,
                "timestamp": sample.timestamp,
                "prev": prev_index,
                "next": index + 1 if index < scene["last_sample_idx"] else None,
                "scene_token": sample.context_name,
            }
            samples.append(sample)
            prev_index = index
            index += 1
            if index == scene["last_sample_idx"] + 1:
                break

        assert samples[-1]["token"] == scene["last_sample_token"]
        assert len(samples) == scene["nbr_samples"]

        return samples

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "version": self.data.metainfo.version,
            "split": self.split,
        }

    @property
    def metadata(self) -> MetaDict:
        return MetaDict()

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
                    dataset_type="waymo",
                    data=self.data.data_list,
                    meta=self.metadata,
                    sample=sample,
                ),
                meta=SampleMetadata.create(
                    dataset_type="waymo",
                    sample_id=sample["token"],
                    sequence_id=sample["scene_token"],
                ),
                is_valid=not sample_id.is_padding,
            )


# ---------------------------
# Utility transforms
# ---------------------------
def _get_ego_pose(ego2global):
    ego2global = torch.as_tensor(ego2global, dtype=torch.float64)

    return t3d.Fused(
        matrix=ego2global, dtype=ego2global.dtype, device=ego2global.device
    )


def _get_sensor_params(lidar2cam, cam2img=None):
    if cam2img is None:
        cam2img = np.eye(4)

    lidar2cam = torch.as_tensor(lidar2cam, dtype=torch.float64)

    # Intrinsic matrix (K)
    intrinsic = t3d.Intrinsic(cam2img[:3, :3], dtype=torch.float64)

    # Extrinsic (camera -> lidar | ego)
    extrinsic = t3d.Fused(
        matrix=lidar2cam, dtype=lidar2cam.dtype, device=lidar2cam.device
    ).inv

    return extrinsic, intrinsic


@transform.register(namespace="waymo")
class LoadImages(Transform, RegistryBaseType):
    """
    Load multi-view images using the specified views/channels.

    Images are returned in RGB format (default) with shape [N, 3, H, W] and
    values in range [0, 1].
    """

    def __init__(
        self,
        channels: List[str] = None,
        mode: Literal["rgb", "bgr"] = "rgb",
        mid_path: str = "training",
        height: int = 1280,
        width: int = 1920,
    ):
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
                "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT",
                "CAM_SIDE_LEFT",
                "CAM_SIDE_RIGHT",
            ]
        assert mode in ["rgb", "bgr"]
        self.channels_map = {
            "CAM_FRONT": "image_0",
            "CAM_FRONT_LEFT": "image_1",
            "CAM_FRONT_RIGHT": "image_2",
            "CAM_SIDE_LEFT": "image_3",
            "CAM_SIDE_RIGHT": "image_4",
        }

        self.channels = channels
        self.mode = mode
        self.mid_path = mid_path
        self.height = height
        self.width = width

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"waymo.{self.__class__.__name__}",
            "channels": list(self.channels),
            "mode": self.mode,
        }

    def _decode_image(self, image_proto) -> torch.Tensor:
        img = PIL.Image.open(io.BytesIO(image_proto.image))
        img = img.convert("RGB")
        img = torchvision.transforms.functional.pil_to_tensor(img)
        if self.mode == "bgr":
            img = img[[2, 1, 0], :, :]
        return img

    def _load_image(self, dir_path, sample_data):
        root = Path(dir_path)

        # image
        img = PIL.Image.open(root / sample_data.img_path)
        img = img.convert("RGB")
        img = torchvision.transforms.functional.pil_to_tensor(img)

        assert img.shape == (3, sample_data.height, sample_data.width)

        # Check if image needs resizing
        original_height = img.shape[1]
        original_width = img.shape[2]
        scale_h = self.height / original_height
        scale_w = self.width / original_width

        if original_height != self.height or original_width != self.width:
            img = torchvision.transforms.functional.resize(
                img,
                size=[self.height, self.width],
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )

        if self.mode == "bgr":
            img = img[[2, 1, 0], :, :]

        # metadata
        # pose = _get_ego_pose(source, sample_data)
        # Scale intrinsic matrix based on image resize
        cam2img = torch.as_tensor(sample_data.cam2img, dtype=torch.float32).clone()
        cam2img[0, :] *= scale_w  # Scale x coordinates
        cam2img[1, :] *= scale_h  # Scale y coordinates

        extrinsic, intrinsic = _get_sensor_params(sample_data.lidar2cam, cam2img)

        meta = {
            # "timestamp": sample_data["timestamp"],
            "extrinsic": extrinsic.to(dtype=torch.float32),
            "intrinsic": intrinsic.to(dtype=torch.float32),
        }

        return img, MetaDict(meta)

    def _load_multiview_images(self, source, sample, channels):
        # pylint: disable=too-many-locals

        token = sample["index"]
        root = Path(source[token].root)
        cams_data = [getattr(source[token].images, cam) for cam in channels]
        cam_dirs = [root / self.mid_path / self.channels_map[cam] for cam in channels]
        imgs, metas = zip(
            *[self._load_image(cam_dir, x) for cam_dir, x in zip(cam_dirs, cams_data)]
        )
        imgs = torch.stack(imgs)
        ego2global = _get_ego_pose(source[token].ego2global)
        ego2global = ego2global.to(dtype=torch.float32)
        timestamp = source[token].timestamp
        for m in metas:
            m["pose"] = ego2global
        for m in metas:
            m["timestamp"] = timestamp

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


@transform.register(namespace="waymo")
class LoadLidar(Transform, RegistryBaseType):
    def __init__(
        self,
        path: str = "training/velodyne",
    ):
        super().__init__()

        self.path = path

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {"type": f"waymo.{self.__class__.__name__}", "path": self.path}

    def _load_lidar_points(self, source, sample):
        token = sample["index"]
        sample_data = source[token]

        transforms = {
            "pose": _get_ego_pose(sample_data.ego2global).to(dtype=torch.float32),
            "extrinsic": t3d.Identity(),
        }

        # load lidar points (x, y, z, intensity, relative time)
        lidar_path = sample_data.root / self.path / sample_data.lidar_points.lidar_path
        num_pts_feats = sample_data.lidar_points.num_pts_feats
        np_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, num_pts_feats)

        # Assuming channel 4 is the intensity.
        points = torch.tensor(np_points, dtype=torch.float32)
        coords, intensity = points[:, 0:3], points[:, 4:5]
        timestamps = torch.tensor(np_points[:, 5:6], dtype=torch.float32)

        points = torch.cat((coords, intensity, timestamps), dim=1)
        points = PackedTensor(points)

        # combine
        meta = {
            "timestamp": sample_data.timestamp,
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


@transform.register(namespace="waymo")
class LoadAnnotations(Transform, RegistryBaseType):
    def __init__(
        self,
        labels: Any,
        ignore_empty_annots: Literal["none", "radar", "lidar", "both", "any"] = "lidar",
        forecasting_steps: int = 0,
    ):
        super().__init__()

        labels = OmegaConf.to_container(labels, resolve=True)
        self._label_to_cls: Dict[int, str] = labels["map"]
        self._cls_to_id: Dict[str, int] = {k: i for i, k in enumerate(labels["order"])}

        self.dtype = torch.float32
        self.sample_check_fn = _get_empty_annot_check_fn(ignore_empty_annots)
        self.ignore_empty_annots = ignore_empty_annots
        self.forecasting_steps = forecasting_steps

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {"type": f"waymo.{self.__class__.__name__}"}

    def _load_box(self, token, extrinsic) -> Box | None:
        record = token
        # exclude boxes that we set to "ignore"
        name = self._label_to_cls[token.bbox_label]
        if name == "ignore":
            return None

        # exclude boxes without any lidar or radar points
        if not self.sample_check_fn(record):
            return None

        x, y, z, w, l, h, yaw = token.bbox_3d
        # construct box object
        box = Box(
            center=[x, y, z],
            size=[w, l, h],  # Box expects [width, length, height]
            orientation=Quaternion(axis=[0, 0, 1], angle=yaw),
            name=name,
            token=record.original_waymo_id,
            velocity=np.array(record.velocity[:3]),
        )

        # transform: ego vehicle to sensor
        if extrinsic is not None:
            box.translate(extrinsic.inv.translate.vector.numpy())
            box.rotate(extrinsic.inv.rotate.quaternion)

        # get instance token
        box.instance_id = record.instance_id

        return box

    def _load_boxes(self, source, sample) -> List[Box]:
        sd_token = sample["index"]
        sample_data = source[sd_token]

        # load boxes and convert to reference frame
        boxes = [
            self._load_box(t, extrinsic=None)
            for i, t in vars(sample_data.instances).items()
        ]
        boxes = [b for b in boxes if b is not None]

        return boxes

    def _load_trajectory(
        self,
        source: List,
        index: int,
        token: str,
        n: int,
    ) -> Iterator[Box]:
        # pylint: disable=too-many-locals

        scene_id = source[index].context_name
        global2ego_base = _get_ego_pose(source[index].ego2global).inv
        for _ in range(n):
            record = source[index]
            bbox_token = (
                getattr(record.instances, token)
                if hasattr(record.instances, token)
                else None
            )
            if bbox_token is not None:
                ego2global_current = _get_ego_pose(record.ego2global)
                x, y, z, w, l, h, yaw = bbox_token.bbox_3d
                # construct box object
                box = Box(
                    center=[x, y, z],
                    size=[w, l, h],  # Box expects [width, length, height]
                    orientation=Quaternion(axis=[0, 0, 1], angle=yaw),
                    name=self._label_to_cls[bbox_token.bbox_label],
                    token=bbox_token.original_waymo_id,
                    velocity=np.array(bbox_token.velocity[:3]),
                )

                current_to_base_tx = t3d.Sequential(ego2global_current, global2ego_base)

                box.timestamp = record.timestamp
                matrix_np = current_to_base_tx.matrix.cpu().numpy()

                r_noisy = matrix_np[:3, :3]
                t = matrix_np[:3, 3]

                # Orthogonalize the rotation matrix
                u, _, vh = np.linalg.svd(r_noisy)
                r = u @ vh
                if np.linalg.det(r) < 0:
                    u[:, -1] *= -1
                    r = u @ vh

                # Quaternion from clean rotation matrix using 4x4 matrix
                q = Quaternion(matrix=r)

                box.rotate(q)
                box.translate(t)

                yield box

            index += 1
            if index >= len(source) or source[index].context_name != scene_id:
                break

    def _load_forecasting_trajectories(
        self,
        source: List,
        sample: Sample,
        tokens: Sequence[str],
        trajectory_len: int,
    ) -> MetaDict:
        # load ego pose and extrinsic for current frame
        sd_token = sample["index"]
        _ = source[sd_token]  # sample_data unused but kept for future use

        center = np.empty((len(tokens), trajectory_len, 3), dtype=np.float32)
        valid = np.empty((len(tokens), trajectory_len), dtype=bool)

        for i, t in enumerate(tokens):
            # load boxes
            boxes = self._load_trajectory(source, sd_token, t, trajectory_len)
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

    def _load_labels(self, source: List, sample: Any) -> MetaDict:
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


@transform.register(namespace="waymo")
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
        apply_fov_mask: bool = True,
    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.labels = labels
        self.apply_fov_mask = apply_fov_mask

        self.label_map = self._build_label_map(labels)

        if isinstance(valid_mask, str):
            valid_mask = [valid_mask]

        self.valid_mask = valid_mask

        # self.index = self._build_index(self.root / "annotations.json")

    def _build_label_map(self, labels: OmegaConf):
        # name to mapped index
        indices = OmegaConf.to_container(labels.all, resolve=True)
        indices = {name: index for index, name in enumerate(indices)}
        indices["ignore"] = labels.ignore_index

        label_map = OmegaConf.to_container(labels.map, resolve=True)
        label_map = {int(i): indices[name] for i, name in label_map.items()}

        return np.vectorize(label_map.get, otypes=[np.int64])

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
        # pylint: disable=too-many-locals

        token = str(sample.source.sample["token"]).zfill(7)
        index = sample.source.sample["index"]
        scene_id = token[1:4]
        frame_id = token[4:] + "_04.npz"
        mid_path = sample.source.data[index].split
        path = self.root / mid_path / scene_id / frame_id

        with open(path, "rb") as fd:
            data = np.load(fd)

            semantics = data["semantics"]
            semantics = self.label_map(semantics)
            semantics = torch.as_tensor(semantics, dtype=torch.long)
            semantics = semantics.permute(2, 1, 0)  # [x, y, z] -> [z, y, x]
            semantics = semantics.contiguous()

            mask_infov = data["infov"]
            mask_infov = torch.as_tensor(mask_infov, dtype=torch.bool)
            mask_infov = mask_infov.permute(2, 1, 0)  # [x, y, z] -> [z, y, x]
            mask_infov = mask_infov.contiguous()

            mask_camera = data["mask_camera"]
            mask_camera = torch.as_tensor(mask_camera, dtype=torch.bool)
            mask_camera = mask_camera.permute(2, 1, 0)  # [x, y, z] -> [z, y, x]
            mask_camera = mask_camera.contiguous()

            mask_lidar = data["mask_lidar"]
            mask_lidar = torch.as_tensor(mask_lidar, dtype=torch.bool)
            mask_lidar = mask_lidar.permute(2, 1, 0)  # [x, y, z] -> [z, y, x]
            mask_lidar = mask_lidar.contiguous()

            instances = data["instances"]
            instances = torch.as_tensor(instances, dtype=torch.long)
            instances = instances.permute(2, 1, 0)  # [x, y, z] -> [z, y, x]
            instances = torch.where(instances == 0, -1, instances)  # set bg to -1
            instances = instances.contiguous()

        if self.apply_fov_mask:
            mask_camera = mask_camera & mask_infov

        masks = {
            "camera": mask_camera,
            "lidar": mask_lidar,
            "in_fov": mask_infov,
        }

        if self.valid_mask is not None:
            valid = [masks[m] for m in self.valid_mask]
            valid = torch.stack(valid, dim=0).all(dim=0)
            masks["valid"] = valid

        occupancy = {
            "semantics": semantics,
            "instance_ids": instances,
            "masks": MetaDict(masks),
        }

        if "labels" not in sample:
            sample.labels = MetaDict()

        if "occupancy" not in sample.labels:
            sample.labels.occupancy = MetaDict()

        sample.labels.occupancy |= MetaDict(occupancy)

        return sample


@transform.register(namespace="waymo")
class LoadTimestamp(Transform, RegistryBaseType):
    def __init__(
        self, factor: float | int = 1.0e-6, dtype: torch.dtype | str | None = None
    ):
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype is None:
            dtype = torch.uint64 if isinstance(factor, int) else torch.float64
        self.factor = factor
        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"waymo.{self.__class__.__name__}",
            "factor": self.factor,
            "dtype": str(self.dtype),
        }

    def apply(self, sample: Sample) -> Sample:
        sample.timestamp = torch.as_tensor(
            sample.source.sample["timestamp"] * self.factor, dtype=self.dtype
        )
        return sample


@transform.register(namespace="waymo")
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
        if field == "prev":
            index = sample["index"]
            scene_token = sample["scene_token"]
            prev_index = max(0, index - 1)
            prev_scene_token = source[prev_index].context_name
            if prev_scene_token != scene_token or prev_index == index:
                return sample
            prev_scene_sample = source[prev_index]
            sample_dict = {
                "token": prev_scene_sample.sample_idx,
                "scene_token": prev_scene_sample.context_name,
                "index": prev_scene_sample.index,
                "timestamp": prev_scene_sample.timestamp,
            }

        elif field == "next":
            index = sample["index"]
            scene_token = sample["scene_token"]
            next_index = min(len(source) - 1, index + 1)
            next_scene_token = source[next_index].context_name
            if next_scene_token != scene_token or next_index == index:
                return sample
            next_scene_sample = source[next_index]
            sample_dict = {
                "token": next_scene_sample.sample_idx,
                "scene_token": next_scene_sample.context_name,
                "index": next_scene_sample.index,
                "timestamp": next_scene_sample.timestamp,
            }

        else:
            raise ValueError(f"Unknown field '{field}' for sample navigation.")

        return sample_dict

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
                dataset_type="waymo",
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
