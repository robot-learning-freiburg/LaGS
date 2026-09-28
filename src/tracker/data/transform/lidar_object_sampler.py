# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Hashable, Iterator, List, Mapping, Sequence, Set, Tuple

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from ... import config, utils
from ...config.registry import RegistryBaseType
from ...utils.types import PackedArray, PackedTensor, Sample
from .. import dataset
from ..dataset.dataset import Dataset
from .registry import registry as transform
from .transform import Transform

log = utils.log.get_logger(__name__)


@dataclass
class ObjectInstance:
    sample_index: int
    object_index: int

    # box parameters (center, dimensions, rotation, velocity)
    box: torch.Tensor

    # class name (mapped name via the dataset)
    class_name: str

    # number of points inside this box
    num_points: int

    @property
    def points_path(self) -> str:
        """
        Relative path to the file storing the points inside the box.
        """
        return f"points_{self.sample_index:08d}_{self.object_index:08d}.pt"


@dataclass
class ObjectIndex:
    objects: Dict[str, List[ObjectInstance]]


torch.serialization.add_safe_globals([ObjectIndex, ObjectInstance])


@transform.register(namespace="augment")
# pylint: disable-next=too-few-public-methods, too-many-instance-attributes
class SampleObjects(Transform, RegistryBaseType):
    """
    Sample random objects as LiDAR points and bounding boxes by class and
    integrate them into the current sample.
    """

    def __init__(
        self,
        data: Dataset,
        labels: Any,
        index_classes: Sequence[str] | None = None,
        index_min_points: Mapping[str, int] | int = 1,
        sample_classes: Sequence[str] | None = None,
        sample_rate: Mapping[str, float] | float = 1.0,
        sample_max_obj: Mapping[str, int] | int = 5,
    ):
        """
        Args:
            data: The dataset to sample objects from.
            labels: Detection label config with map and order fields.
            index_classes: The classes (by name) to index objects for. \
                If None, classes are determined by the index_min_points mapping \
                (if it is in fact a mapping), else all classes in labels are indexed.
            index_min_points: The minimum number of points per class to index. \
                If a single integer is given, this is used for all classes.
            sample_classes: The classes to sample objects from. \
                If None, all indexed classes are used.
            sample_rate: The rate of objects to sample per class. \
                If a single float is given, this is used for all classes.
            sample_max_obj: The maximum number of objects to sample per class. \
                This inludes objects already present in the sample. \
                If a single integer is given, this is used for all classes.
        """
        # pylint: disable=too-many-branches

        labels = OmegaConf.to_container(labels, resolve=True)
        self._id_to_cls: List[str] = labels["order"]
        self._cls_to_id: Dict[str, int] = {k: i for i, k in enumerate(labels["order"])}

        self.data = data

        if index_classes is not None:
            self.index_classes: List[str] = list(index_classes)
        elif isinstance(index_min_points, Mapping):
            self.index_classes: List[str] = list(index_min_points.keys())
        else:
            self.index_classes: List[str] = list(self._id_to_cls)

        if isinstance(index_min_points, Mapping):
            self.index_min_points: Dict[str, int] = dict(index_min_points)
        else:
            self.index_min_points: Dict[str, int] = {
                name: int(index_min_points) for name in self.index_classes
            }

        if sample_classes is not None:
            self.sample_classes: List[str] = list(sample_classes)
        elif isinstance(sample_max_obj, Mapping):
            self.sample_classes: List[str] = sorted(sample_max_obj.keys())
        elif isinstance(sample_rate, Mapping):
            self.sample_classes: List[str] = sorted(sample_rate.keys())
        else:
            self.sample_classes: List[str] = sorted(self.index_classes)

        if isinstance(sample_max_obj, Mapping):
            self.sample_max_obj: Dict[str, int] = dict(sample_max_obj)
        else:
            self.sample_max_obj: Dict[str, int] = {
                name: int(sample_max_obj) for name in self.sample_classes
            }

        if isinstance(sample_rate, Mapping):
            self.sample_rate: Dict[str, float] = dict(sample_rate)
        else:
            self.sample_rate: Dict[str, float] = {
                name: float(sample_rate) for name in self.sample_classes
            }

        dataset_classes: Set[str] = set(self._cls_to_id.keys())
        assert set(self.index_classes) <= dataset_classes
        assert set(self.index_classes) == set(self.index_min_points.keys())
        assert set(self.sample_classes) == set(self.sample_max_obj.keys())
        assert set(self.sample_classes) == set(self.sample_rate.keys())

        self.index, self.index_root = self._load_index()

        self.samplers = {
            name: ObjectSampler(objs) for name, objs, in self.index.objects.items()
        }

    @classmethod
    def from_config(cls, conf: OmegaConf | Mapping[str, Any], *args, **kwargs):
        conf_kwargs = config.utils.get_kwargs(conf)

        log.info("building object sampler")
        data = dataset.build(conf_kwargs.pop("data"))
        labels = conf_kwargs.pop("labels")

        index_classes = None
        index_min_points = 1
        sample_rate = 1.0
        sample_classes = None
        sample_max_obj = 5

        if "index" in conf_kwargs:
            index = conf_kwargs.pop("index")

            index_classes = index.pop("classes", default=index_classes)
            if index_classes is not None and OmegaConf.is_config(index_classes):
                index_classes = OmegaConf.to_container(index_classes)

            index_min_points = index.pop("min_points", default=index_min_points)
            if index_min_points is not None and OmegaConf.is_config(index_min_points):
                index_min_points = OmegaConf.to_container(index_min_points)

        if "sample" in conf_kwargs:
            sample = conf_kwargs.pop("sample")

            sample_rate = sample.pop("rate", default=sample_rate)
            if OmegaConf.is_config(sample_rate):
                sample_rate = OmegaConf.to_container(sample_rate)

            sample_classes = sample.pop("classes", default=sample_classes)
            if sample_classes is not None and OmegaConf.is_config(sample_classes):
                sample_classes = OmegaConf.to_container(sample_classes)

            sample_max_obj = sample.pop("max_objects", default=sample_max_obj)
            if sample_max_obj is not None and OmegaConf.is_config(sample_max_obj):
                sample_max_obj = OmegaConf.to_container(sample_max_obj)

        return cls(
            *args,
            data=data,
            labels=labels,
            index_classes=index_classes,
            index_min_points=index_min_points,
            sample_classes=sample_classes,
            sample_rate=sample_rate,
            sample_max_obj=sample_max_obj,
            **conf_kwargs,
            **kwargs,
        )

    @property
    def hparams(self) -> Mapping[str, Hashable]:
        return {
            "type": f"augment.{self.__class__.__name__}",
            "data": self.data.hparams,
            "index": {
                "classes": self.index_classes,
                "min_points": self.index_min_points,
            },
            "sample": {
                "rate": self.sample_rate,
                "classes": self.sample_classes,
                "max_obj": self.sample_max_obj,
            },
        }

    def _load_index(self) -> Tuple[ObjectIndex, Path]:
        cache_hparams = {
            "data": self.data.hparams,
            "index": {
                "classes": self.index_classes,
                "min_points": self.index_min_points,
            },
        }

        cache = utils.cache.get(self.__class__, cache_hparams)
        index_path = cache.storage_path / "index.pkl"

        if cache.available():
            log.info("cache found, loading index from '%s'", index_path)
            with open(index_path, "rb") as fd:
                index = torch.load(fd, weights_only=True, map_location="cpu")

        else:
            log.info("no cache found, indexing data...")
            index = self._build_index(cache, index_path)

        return index, cache.storage_path

    def _build_index(
        self, cache: utils.cache.Artifact, index_path: Path
    ) -> ObjectIndex:
        # pylint: disable=too-many-locals

        # If we are running in distributed mode, only compute things on rank 0
        # and sync them over later.
        use_dist = dist.is_available() and dist.is_initialized()
        rank, world_size = (
            (dist.get_rank(), dist.get_world_size()) if use_dist else (0, 1)
        )

        # Note: We need this barrier to ensure that the cache is only seen as
        # "available" by other processes if it really has been created. So we
        # need to make sure that all processes have entered this function
        # before we can actually call cache.initialize() below.
        if use_dist:
            dist.barrier()

        if rank == 0:
            cache.prepare()

        # distribute samples across processes
        num_samples = len(self.data)
        chunk_size = math.ceil(num_samples / world_size)
        index_start = rank * chunk_size
        index_end = min(index_start + chunk_size, num_samples)
        indices = range(index_start, index_end)

        # perform actual indexing
        index = self._build_index_part(indices, cache.storage_path)

        # synchronize and merge
        if use_dist:
            # synchronize
            synced = [[] for _ in range(world_size)]
            with warnings.catch_warnings(action="ignore", category=FutureWarning):
                # Note: This will emit a warning that pickle is unsafe... we take
                # not of that and ignore it.
                dist.all_gather_object(synced, index)

            # merge results
            index = {name: [] for name in self.index_classes}
            for partial in synced:
                for key, vals in partial.items():
                    index[key] += vals

        # sort to keep things deterministic
        def sort_key(obj: ObjectInstance):
            return obj.sample_index, obj.object_index

        index = {k: sorted(v, key=sort_key) for k, v in index.items()}
        index = ObjectIndex(index)

        # save index
        if rank == 0:
            log.info("dumping index to '%s'", index_path)
            with open(index_path, "wb") as fd:
                torch.save(index, fd)

            cache.initialize()

        return index

    def _build_index_part(
        self, indices: Sequence[int], storage_root: Path
    ) -> Dict[str, List[ObjectInstance]]:
        # pylint: disable=too-many-locals

        # compile filter for classes minimum number of points
        min_points = {name: np.iinfo(np.int64).max for name in self._id_to_cls}
        min_points |= self.index_min_points
        min_points = np.vectorize(min_points.get)

        objects = {name: [] for name in self.index_classes}

        indices = utils.progress.track(indices, description="Indexing dataset...")
        for sample_index in indices:
            sample = self.data[sample_index]

            # we're preprocessing... the data really shouldn't be batched here
            assert sample.batch_size == 1

            # get sample data
            points: torch.Tensor = sample.points.data.get(0)
            boxes: torch.Tensor = sample.labels.boxes.get(0)
            class_names: np.ndarray = sample.labels.class_names.get(0)

            if boxes.shape[0] == 0:
                continue

            # compute which points are in which boxes
            contained = utils.math.intersect_points_box(
                points[:, :3], boxes[:, 0:3], boxes[:, 3:6], boxes[:, 6]
            )

            # filter by classes and minimum number of points per class
            num_points = contained.sum(1)
            mask = num_points >= torch.from_numpy(min_points(class_names))

            boxes = boxes[mask, :]
            class_names = class_names[mask.numpy()]
            contained = contained[mask, :]

            # store boxes and points inside boxes
            for i, (box, cname, mask) in enumerate(zip(boxes, class_names, contained)):
                # clone to avoid storing tensor views
                box_points = points[mask, :].clone()

                # construct object
                obj = ObjectInstance(
                    sample_index=sample_index,
                    object_index=i,
                    box=box.clone(),
                    class_name=cname.item(),
                    num_points=box_points.shape[0],
                )

                # store points as separate file
                with open(storage_root / obj.points_path, "wb") as fd:
                    torch.save(box_points, fd)

                # store object instance
                objects[cname].append(obj)

        return objects

    def _sample_class(self, class_name, num, coll_boxes):
        objects = self.samplers[class_name].sample(num)
        indices = torch.arange(len(objects), dtype=torch.int64)

        # collect all boxes
        boxes = torch.stack([obj.box for obj in objects], dim=0)

        # filter by collisions against collision-boxes in BEV space
        mask = utils.math.interesect_box_box(
            boxes[:, 0:2],
            boxes[:, 3:5],
            boxes[:, 6],
            coll_boxes[:, 0:2],
            coll_boxes[:, 3:5],
            coll_boxes[:, 6],
        )
        mask = ~(mask.any(1))

        boxes = boxes[mask, :]
        indices = indices[mask]

        # filter by self-collisions in BEV space
        mask = utils.math.interesect_box_box(
            boxes[:, 0:2],
            boxes[:, 3:5],
            boxes[:, 6],
            boxes[:, 0:2],
            boxes[:, 3:5],
            boxes[:, 6],
        )
        mask = ~(torch.triu(mask, diagonal=1).any(1))

        boxes = boxes[mask, :]
        indices = indices[mask]

        # load point-cloud points
        #
        # Note: Adding a torch.empty() here prevents torch.cat() from
        # complaining later when the list of points is empty.
        points = [torch.empty(0)]
        for i in indices:
            obj = objects[i.item()]

            with open(self.index_root / obj.points_path, "rb") as fd:
                obj_pts = torch.load(fd, map_location="cpu", weights_only=True)

            points.append(obj_pts)

        return boxes, torch.cat(points, dim=0)

    def apply(self, sample: Sample) -> Sample:
        # pylint: disable=too-many-locals
        assert sample.batch_size == 1

        labels = sample.labels

        # prepare the local class-name-to-id map
        cls_to_id = {name: self._cls_to_id[name] for name in self.sample_classes}

        # initialize collision boxes to avoid when sampling
        coll_boxes = labels.boxes.get(0)

        # main sampling loop over classes
        sampled_boxes = []
        sampled_points = []
        sampled_instance_ids = []
        sampled_cls_names = []
        sampled_cls_ids = []

        for name in self.sample_classes:
            # count the number of objects per class in the sample
            n_present = torch.sum(labels.class_ids.get(0) == cls_to_id[name]).item()

            # compute the (maximum) number of objects that we should sample
            n_objs = self.sample_max_obj[name] - n_present
            n_objs = round(self.sample_rate[name] * n_objs)

            if n_objs <= 0:
                continue

            # sample at most n_objs objects of specified class
            obj_boxes, obj_points = self._sample_class(name, n_objs, coll_boxes)
            obj_cls_names = np.array([name]).repeat(obj_boxes.shape[0])
            obj_cls_ids = torch.tensor([cls_to_id[name]]).expand(obj_boxes.shape[0])

            # FIXME: use actual instance IDs?
            instance_ids = torch.tensor(-1).expand(obj_boxes.shape[0])

            # collect results
            sampled_points.append(obj_points)
            sampled_boxes.append(obj_boxes)
            sampled_instance_ids.append(instance_ids)
            sampled_cls_names.append(obj_cls_names)
            sampled_cls_ids.append(obj_cls_ids)

            # update collision boxes
            coll_boxes = torch.cat((coll_boxes, obj_boxes), dim=0)

        # concatenate everything and add back to the sample
        points = torch.cat((sample.points.data.data, *sampled_points), dim=0)
        boxes = torch.cat((labels.boxes.data, *sampled_boxes), dim=0)
        class_names = np.concatenate((labels.class_names.data, *sampled_cls_names))
        class_ids = torch.cat((labels.class_ids.data, *sampled_cls_ids))
        instance_ids = torch.cat((labels.instance_ids.data, *sampled_instance_ids))

        sample.points.data = PackedTensor(points)
        sample.labels.boxes = PackedTensor(boxes)
        sample.labels.instance_ids = PackedTensor(instance_ids)
        sample.labels.class_names = PackedArray(class_names)
        sample.labels.class_ids = PackedTensor(class_ids)

        return sample


# pylint: disable-next=too-few-public-methods
class ObjectSampler:
    def __init__(self, objects: List[ObjectInstance]):
        self.objects = objects
        self.indices = self._indices(len(objects))

    def _indices(self, n) -> Iterator[int]:
        while True:
            yield from torch.randperm(n).tolist()

    def sample(self, num) -> List[ObjectInstance]:
        return [self.objects[next(self.indices)] for _ in range(num)]
