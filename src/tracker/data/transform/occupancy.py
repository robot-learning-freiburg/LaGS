# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import math
import warnings
from collections.abc import Hashable
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.ndimage
import scipy.spatial
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from ... import config, utils
from ...config.registry import RegistryBaseType
from ...ops.voxelize import voxelize_trace
from ...utils.types import MetaDict, PackedArray, PackedTensor, Sample
from .. import dataset
from ..dataset.dataset import Dataset
from .registry import registry as transform
from .transform import Transform

log = utils.log.get_logger(__name__)


def generate_voxel_instance_labels(
    box_center: torch.Tensor,
    box_dim: torch.Tensor,
    box_rot: torch.Tensor,
    box_cls: torch.Tensor,
    box_iid: torch.Tensor,
    voxel_cls: torch.Tensor,
    voxel_size: torch.Tensor,
    voxel_offset: torch.Tensor,
    labels_voxel: Mapping[int, str],
    labels_box: Mapping[int, str],
    max_distance: float = 2.0,
) -> torch.Tensor:
    # pylint: disable=too-many-locals

    # map voxel semantic classes to box classes
    label_map = {name: i for i, name in labels_box.items()}
    label_map = {i: label_map[name] for i, name in labels_voxel.items()}

    # generate instance labels for each class
    instances = [
        _generate_voxel_instance_labels_for_class(
            box_center=box_center,
            box_dim=box_dim,
            box_rot=box_rot,
            box_cls=box_cls,
            box_iid=box_iid,
            voxel_cls=voxel_cls,
            voxel_size=voxel_size,
            voxel_offset=voxel_offset,
            cls_voxel=cls_voxel,
            cls_box=cls_box,
            max_distance=max_distance,
        )
        for cls_voxel, cls_box in label_map.items()
    ]

    idx, iid = zip(*instances)
    idx = torch.cat(idx, dim=0)
    iid = torch.cat(iid, dim=0)

    # map everything back to a grid
    grid = torch.full(voxel_cls.shape, -1, dtype=iid.dtype)
    grid[idx[:, 2], idx[:, 1], idx[:, 0]] = iid

    return grid


def _generate_voxel_instance_labels_for_class(
    box_center: torch.Tensor,
    box_dim: torch.Tensor,
    box_rot: torch.Tensor,
    box_cls: torch.Tensor,
    box_iid: torch.Tensor,
    voxel_cls: torch.Tensor,
    voxel_size: torch.Tensor,
    voxel_offset: torch.Tensor,
    cls_voxel: int,
    cls_box: int,
    max_distance: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # pylint: disable=too-many-locals

    # get the voxel indices for the current class
    voxel_indices = torch.nonzero(voxel_cls == cls_voxel, as_tuple=False)
    voxel_indices = voxel_indices[:, [2, 1, 0]]  # (z, y, x) -> (x, y, z)

    # skip if no voxels are present
    if voxel_indices.shape[0] == 0:
        return voxel_indices, torch.empty(0, dtype=box_iid.dtype)

    # get the boxes for the current class
    box_mask = box_cls == cls_box
    box_center = box_center[box_mask]
    box_dim = box_dim[box_mask]
    box_rot = box_rot[box_mask]
    box_iid = box_iid[box_mask]

    # skip if no boxes are present; voxels remain at -1 (no instance)
    if box_center.shape[0] == 0:
        return torch.empty((0, 3), dtype=torch.int64), torch.empty(
            0, dtype=box_iid.dtype
        )

    # compute the intersection of boxes and voxels [n_boxes, n_voxels]
    intersect = utils.math.intersect_box_voxel(
        box_center=box_center,
        box_dim=box_dim,
        box_rot=box_rot,
        voxel_indices=voxel_indices,
        voxel_size=voxel_size,
        voxel_offset=voxel_offset,
    )

    # get the number of boxes intersecting each voxel
    count = intersect.sum(dim=0)

    # prepare the voxel-to-box mapping
    instances = torch.full((voxel_indices.shape[0],), -1, dtype=box_iid.dtype)

    # we can directly assign voxels with a single intersection
    instances[count == 1] = torch.argmax(intersect[:, count == 1].int(), dim=0)

    # for voxels with no intersection, assign to the nearest box within max_distance
    if torch.any(count == 0):
        # NOTE: maybe box-to-box distance would be better here, but that is
        # quite a bit more complex
        distance = utils.math.distance_points_box(
            points=utils.math.voxel_centers(
                voxel_indices=voxel_indices[count == 0],
                voxel_size=voxel_size,
                voxel_offset=voxel_offset,
            ),
            box_center=box_center,
            box_dim=box_dim,
            box_rot=box_rot,
        )

        nearest = torch.argmin(distance, dim=0)
        in_range = distance.min(dim=0).values <= max_distance
        no_box_indices = torch.where(count == 0)[0]
        instances[no_box_indices[in_range]] = nearest[in_range]

    # for voxels with multiple intersections, compute the axis-relative distance
    # to the box center and assign the voxel to the closest box
    if torch.any(count > 1):
        # NOTE: this could be improved by computing the box-voxel overlap
        # volume instead, but this is a good approximation for now
        distance = utils.math.distance_points_box_axis_relative(
            points=utils.math.voxel_centers(
                voxel_indices=voxel_indices[count > 1],
                voxel_size=voxel_size,
                voxel_offset=voxel_offset,
            ),
            box_center=box_center,
            box_dim=box_dim,
            box_rot=box_rot,
        )
        # pylint: disable-next=not-callable
        distance = torch.linalg.vector_norm(distance, dim=-1)

        instances[count > 1] = torch.argmin(distance, dim=0)

    # map the local box indices to the global instance IDs
    #
    # NOTE: voxels with no covering box and none within max_distance (case
    # `count == 0` above) remain unassigned (local index -1). They must stay
    # unlabeled (-1); only remap the assigned entries so we don't index box_iid
    # with -1 (which would silently pick the last box).
    assigned = instances >= 0
    out = torch.full_like(instances, -1)
    out[assigned] = box_iid[instances[assigned]]

    return voxel_indices, out


class VoxelInstanceData:
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        source: Dataset,
        voxel_size: torch.Tensor | tuple[float, float, float],
        voxel_offset: torch.Tensor | tuple[float, float, float],
        voxel_labels: Mapping[int, str],
        box_labels: Mapping[int, str],
        max_distance: float = 2.0,
    ):
        self.source = source

        self.voxel_size = torch.as_tensor(voxel_size)
        self.voxel_offset = torch.as_tensor(voxel_offset)
        self.voxel_labels = voxel_labels
        self.box_labels = box_labels
        self.max_distance = max_distance

        # build/load the index
        index, root = self._load()
        self.index = index
        self.root = root

    @property
    def cache_hparams(self) -> Mapping[str, Hashable]:
        return {
            "data": self.source.hparams,
            "voxel_size": self.voxel_size.tolist(),
            "voxel_offset": self.voxel_offset.tolist(),
            "voxel_labels": {str(k): v for k, v in self.voxel_labels.items()},
            "box_labels": {str(k): v for k, v in self.box_labels.items()},
            "max_distance": self.max_distance,
        }

    def _load(self):
        cache = utils.cache.get(
            "data.transform.occupancy.VoxelInstanceData", self.cache_hparams
        )
        index_path = cache.storage_path / "index.json"

        if cache.available():
            log.info("cache found, loading index from '%s'", index_path)
            with open(index_path, "r", encoding="utf-8") as fd:
                index = json.load(fd)
        else:
            log.info("no cache found, generating occupancy instance labels...")
            index = self._build(cache, index_path)

        return index, cache.storage_path

    def _build(self, cache: utils.cache.Artifact, index_path: Path):
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
        num_samples = len(self.source)
        chunk_size = math.ceil(num_samples / world_size)
        index_start = rank * chunk_size
        index_end = min(index_start + chunk_size, num_samples)
        indices = range(index_start, index_end)

        # perform actual indexing
        index = self._build_partial(indices, cache.storage_path)

        # synchronize and merge
        if use_dist:
            # synchronize
            synced = [[] for _ in range(world_size)]
            with warnings.catch_warnings(action="ignore", category=FutureWarning):
                # Note: This will emit a warning that pickle is unsafe... we take
                # not of that and ignore it.
                dist.all_gather_object(synced, index)

            # merge results
            index = {k: v for part in synced for k, v in part.items()}

        # save the index
        if rank == 0:
            log.info("dumping index to '%s'", index_path)
            with open(index_path, "w", encoding="utf-8") as fd:
                json.dump(index, fd)

            cache.initialize()

        return index

    def _build_partial(self, indices: Sequence[int], storage_root: Path):
        box_labels = self.box_labels

        data = {}
        indices = utils.progress.track(indices, "Generating voxel instance labels...")
        for sample_index in indices:
            sample = self.source[sample_index]
            sample_id = sample.meta.sample_id

            # we're preprocessing... the data really shouldn't be batched here
            assert sample.batch_size == 1

            # build the instance labels
            box_data = sample.labels.boxes.get(0)
            box_cls = sample.labels.class_ids.get(0)
            box_iid = sample.labels.instance_ids.get(0)
            voxel_cls = sample.labels.occupancy.semantics

            instance_ids = generate_voxel_instance_labels(
                box_center=box_data[:, :3],
                box_dim=box_data[:, 3:6],
                box_rot=box_data[:, 6],
                box_cls=box_cls,
                box_iid=box_iid,
                voxel_cls=voxel_cls,
                voxel_size=self.voxel_size,
                voxel_offset=self.voxel_offset,
                labels_voxel=self.voxel_labels,
                labels_box=box_labels,
                max_distance=self.max_distance,
            )

            # store the instance labels
            # NOTE: We're storing the instance IDs as a compressed numpy array
            # to save space. This seems to be quite a bit smaller than saving
            # pytorch tensors directly.
            path = f"voxel_iid_{sample_id}.npz"
            np.savez_compressed(storage_root / path, instance_ids=instance_ids.numpy())

            data[sample_id] = {"data": str(path)}

        return data

    def __getitem__(self, sample_id):
        info = self.index[sample_id]
        path = self.root / info["data"]

        data = np.load(path)
        data = torch.as_tensor(data["instance_ids"], dtype=torch.int64)

        return data


@transform.register(namespace="occupancy")
class BuildInstanceLabels(Transform, RegistryBaseType):
    def __init__(
        self,
        source: Dataset,
        voxel_size: tuple[float, float, float],
        voxel_range: tuple[float, float, float, float, float, float],
        voxel_labels: Mapping[int, str],
        box_labels: Mapping[int, str],
        max_distance: float = 2.0,
    ):
        self.data = VoxelInstanceData(
            source=source,
            voxel_size=voxel_size,
            voxel_offset=voxel_range[:3],
            voxel_labels=voxel_labels,
            box_labels=box_labels,
            max_distance=max_distance,
        )

    @classmethod
    def from_config(cls, conf: OmegaConf | Mapping[str, Any], *args, **kwargs):
        conf_kwargs = config.utils.get_kwargs(conf)

        log.info("building voxel instance ID index")
        source = dataset.build(conf_kwargs.pop("data"))

        voxel_labels_conf = conf_kwargs.pop("labels")
        voxel_labels = {
            i: n
            for i, n in enumerate(voxel_labels_conf.all)
            if n in voxel_labels_conf.instance
        }

        box_labels_conf = conf_kwargs.pop("box_labels")
        box_labels = dict(enumerate(box_labels_conf.order))

        return cls(
            *args,
            source=source,
            voxel_labels=voxel_labels,
            box_labels=box_labels,
            **conf_kwargs,
            **kwargs,
        )

    @property
    def hparams(self) -> Mapping[str, Hashable]:
        return {
            "type": f"occupancy.{self.__class__.__name__}",
            **self.data.cache_hparams,
        }

    def apply(self, sample: Sample) -> Sample:
        if "labels" not in sample:
            sample.labels = MetaDict()

        if "occupancy" not in sample.labels:
            sample.labels.occupancy = MetaDict()

        sample.labels.occupancy.instance_ids = self.data[sample.meta.sample_id]
        return sample


# Grid boundary faces for the pseudo-endpoint shell. ``z-`` is the bottom face;
# tracing to it can leak free space through holes in the (occupied) ground where
# labels are missing, so it is excluded by default.
BOUNDARY_FACES = ("x-", "x+", "y-", "y+", "z-", "z+")
_BOUNDARY_SPECS: Mapping[str, tuple[str, ...]] = {
    "none": (),
    "all": BOUNDARY_FACES,
    "sides": ("x-", "x+", "y-", "y+"),
    "sides+top": ("x-", "x+", "y-", "y+", "z+"),
    "top": ("z+",),
}


def boundary_faces_from_spec(spec: str | Sequence[str]) -> tuple[str, ...]:
    """Resolve a boundary spec (a named preset or explicit face list)."""
    if isinstance(spec, str):
        if spec not in _BOUNDARY_SPECS:
            raise ValueError(
                f"unknown boundary spec {spec!r}; "
                f"expected one of {', '.join(_BOUNDARY_SPECS)} or a face list"
            )
        return _BOUNDARY_SPECS[spec]
    return tuple(spec)


def _boundary_points(
    voxel_range: torch.Tensor, voxel_size: torch.Tensor, faces: Sequence[str]
) -> torch.Tensor:
    """Pseudo ray-endpoints just outside the requested grid faces.

    Tracing to these (with ``stop_at_occupied``) carves free along every outward
    direction until the first occupied voxel or the grid edge, so open space (no
    occupied voxel behind it) is still marked observed. They lie just outside the
    grid, so they are not themselves marked occupied (out-of-bounds end voxels are
    skipped) and rays clamp at the grid boundary. ``faces`` selects which of
    :data:`BOUNDARY_FACES` to include.
    """
    # pylint: disable=too-many-locals

    vmin, vmax = voxel_range[:3], voxel_range[3:]
    n = ((vmax - vmin) / voxel_size).round().to(torch.int64)

    def centres(axis: int) -> torch.Tensor:
        return (
            vmin[axis]
            + (torch.arange(n[axis], dtype=torch.float64) + 0.5) * voxel_size[axis]
        )

    cx, cy, cz = centres(0), centres(1), centres(2)
    lo = vmin - voxel_size  # just outside the low faces
    hi = vmax + voxel_size  # just outside the high faces
    faces = set(faces)

    out = []
    if {"x-", "x+"} & faces:
        yy, zz = torch.meshgrid(cy, cz, indexing="ij")
        if "x-" in faces:
            out.append(
                torch.stack([torch.full_like(yy, lo[0]), yy, zz], -1).reshape(-1, 3)
            )
        if "x+" in faces:
            out.append(
                torch.stack([torch.full_like(yy, hi[0]), yy, zz], -1).reshape(-1, 3)
            )
    if {"y-", "y+"} & faces:
        xx, zz = torch.meshgrid(cx, cz, indexing="ij")
        if "y-" in faces:
            out.append(
                torch.stack([xx, torch.full_like(xx, lo[1]), zz], -1).reshape(-1, 3)
            )
        if "y+" in faces:
            out.append(
                torch.stack([xx, torch.full_like(xx, hi[1]), zz], -1).reshape(-1, 3)
            )
    if {"z-", "z+"} & faces:
        xx, yy = torch.meshgrid(cx, cy, indexing="ij")
        if "z-" in faces:
            out.append(
                torch.stack([xx, yy, torch.full_like(xx, lo[2])], -1).reshape(-1, 3)
            )
        if "z+" in faces:
            out.append(
                torch.stack([xx, yy, torch.full_like(xx, hi[2])], -1).reshape(-1, 3)
            )

    return torch.cat(out, dim=0) if out else torch.zeros((0, 3), dtype=torch.float64)


def retrace_visibility(
    semantics: torch.Tensor,
    origins: torch.Tensor,
    voxel_size,
    voxel_range,
    free_index: int,
    ignore_index: int,
    *,
    stop_at_occupied: bool = True,
    boundary_faces: Sequence[str] = (),
    hide_occluded_occupied: bool = False,
    device: str | None = None,
) -> torch.Tensor:
    """Occlusion-aware observation mask for one occupancy frame.

    Traces rays from each sensor ``origin`` to the occupied voxel centres (the
    scene geometry); with ``stop_at_occupied`` the rays stop at the first occupied
    voxel, so free space behind a surface stays unobserved. ``boundary_faces``
    additionally traces to pseudo-endpoints just outside the listed grid faces
    (see :func:`boundary_faces_from_spec`), recovering free space in open
    directions; excluding the bottom (``z-``) avoids leaking free through gaps in
    the ground where occupied labels are missing.

    By default every occupied voxel is observed (occluded surfaces included). With
    ``hide_occluded_occupied`` only occupied voxels actually reached by a ray
    (those bordering carved free space) are observed, so surfaces in shadow are
    left unobserved too.

    Args:
        semantics: ``[Z, Y, X]`` class indices.
        origins: ``[S, 3]`` sensor origins (metric, same frame as the grid).
        voxel_size: ``[x, y, z]`` metres; voxel_range: ``[xmin..zmax]``.

    Returns:
        ``[Z, Y, X]`` bool observed mask.
    """
    # pylint: disable=too-many-locals
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vs = torch.as_tensor(voxel_size, dtype=torch.float64)
    vr = torch.as_tensor(voxel_range, dtype=torch.float64)
    vmin = vr[:3]

    occupied = (semantics != free_index) & (semantics != ignore_index)
    zyx = occupied.nonzero(as_tuple=False).to(torch.float64)  # [N, 3] (z, y, x)
    if zyx.numel() == 0 and not boundary_faces:
        return torch.zeros_like(semantics, dtype=torch.bool)

    points = torch.stack(
        [
            vmin[0] + (zyx[:, 2] + 0.5) * vs[0],
            vmin[1] + (zyx[:, 1] + 0.5) * vs[1],
            vmin[2] + (zyx[:, 0] + 0.5) * vs[2],
        ],
        dim=1,
    )
    if boundary_faces:
        points = torch.cat([points, _boundary_points(vr, vs, boundary_faces)], dim=0)
    points = points.to(device=device, dtype=torch.float32)

    points_range = torch.stack([vr[:3], vr[3:]], dim=1).to(torch.float32)
    voxel_size_t = vs.to(torch.float32)

    # Union of carved free space over all sensor origins ([X, Y, Z]).
    free = None
    for origin in origins:
        grid = voxelize_trace(
            points,
            None,
            origin.to(torch.float32),
            points_range,
            voxel_size_t,
            stop_at_occupied=stop_at_occupied,
        )
        free_i = grid[0] < 0.0  # value_free defaults to -1
        free = free_i if free is None else (free | free_i)

    free = free.permute(2, 1, 0).contiguous().cpu()  # -> [Z, Y, X]

    if hide_occluded_occupied:
        # Only occupied voxels reached by a ray (bordering carved free) are
        # observed; surfaces in shadow are dropped.
        reachable = scipy.ndimage.binary_dilation(
            free.numpy(), structure=np.ones((3, 3, 3), dtype=bool)
        )
        observed_occupied = occupied & torch.from_numpy(reachable)
    else:
        observed_occupied = occupied

    return free | observed_occupied


class VisibilityData:
    """Per-sample occlusion-aware observation masks, cached on disk.

    Mirrors :class:`VoxelInstanceData`: builds an hparams-keyed cache once (one
    compressed npz per sample + an ``index.json``) and serves masks by
    ``sample_id``. Each mask is recomputed by tracing rays from the frame's LiDAR
    origins to the occupied voxel centres with ``stop_at_occupied=True`` so that
    voxels occluded behind a surface are *not* marked observed.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        source: Dataset,
        voxel_size: tuple[float, float, float],
        voxel_range: tuple[float, ...],
        free_index: int,
        ignore_index: int,
        stop_at_occupied: bool = True,
        boundary_faces: Sequence[str] = (),
        hide_occluded_occupied: bool = False,
        device: str = "cpu",
    ):
        # pylint: disable=too-many-arguments
        self.source = source
        self.voxel_size = torch.as_tensor(voxel_size, dtype=torch.float64)  # [x,y,z]
        self.voxel_range = torch.as_tensor(voxel_range, dtype=torch.float64)  # [6]
        self.free_index = int(free_index)
        self.ignore_index = int(ignore_index)
        self.stop_at_occupied = bool(stop_at_occupied)
        self.boundary_faces = boundary_faces_from_spec(boundary_faces)
        self.hide_occluded_occupied = bool(hide_occluded_occupied)
        # The cache is built inside the data pipeline; default to CPU so the
        # dataloader (and its forked workers) never touches the CUDA context.
        # Under DDP the build is sharded across ranks, so CPU stays tractable.
        self.device = device

        index, root = self._load()
        self.index = index
        self.root = root

    @property
    def cache_hparams(self) -> Mapping[str, Hashable]:
        return {
            "data": self.source.hparams,
            "voxel_size": self.voxel_size.tolist(),
            "voxel_range": self.voxel_range.tolist(),
            "free_index": self.free_index,
            "ignore_index": self.ignore_index,
            "stop_at_occupied": self.stop_at_occupied,
            "boundary_faces": list(self.boundary_faces),
            "hide_occluded_occupied": self.hide_occluded_occupied,
        }

    def _load(self):
        cache = utils.cache.get(
            "data.transform.occupancy.VisibilityData", self.cache_hparams
        )
        index_path = cache.storage_path / "index.json"

        if cache.available():
            log.info("cache found, loading index from '%s'", index_path)
            with open(index_path, "r", encoding="utf-8") as fd:
                index = json.load(fd)
        else:
            log.info("no cache found, re-tracing occupancy visibility masks...")
            index = self._build(cache, index_path)

        return index, cache.storage_path

    def _build(self, cache: utils.cache.Artifact, index_path: Path):
        use_dist = dist.is_available() and dist.is_initialized()
        rank, world_size = (
            (dist.get_rank(), dist.get_world_size()) if use_dist else (0, 1)
        )

        if use_dist:
            dist.barrier()

        if rank == 0:
            cache.prepare()

        num_samples = len(self.source)
        chunk_size = math.ceil(num_samples / world_size)
        index_start = rank * chunk_size
        index_end = min(index_start + chunk_size, num_samples)
        indices = range(index_start, index_end)

        index = self._build_partial(indices, cache.storage_path)

        if use_dist:
            synced = [[] for _ in range(world_size)]
            with warnings.catch_warnings(action="ignore", category=FutureWarning):
                dist.all_gather_object(synced, index)
            index = {k: v for part in synced for k, v in part.items()}

        if rank == 0:
            log.info("dumping index to '%s'", index_path)
            with open(index_path, "w", encoding="utf-8") as fd:
                json.dump(index, fd)
            cache.initialize()

        return index

    def _build_partial(self, indices: Sequence[int], storage_root: Path):
        data = {}
        indices = utils.progress.track(indices, "Re-tracing visibility masks...")
        for sample_index in indices:
            sample = self.source[sample_index]
            sample_id = sample.meta.sample_id

            assert sample.batch_size == 1

            mask = self._trace(sample.labels.occupancy.semantics, sample.lidar_origins)

            path = f"visibility_{sample_id}.npz"
            np.savez_compressed(storage_root / path, mask=mask.numpy())
            data[sample_id] = {"data": str(path)}

        return data

    def _trace(self, semantics: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
        return retrace_visibility(
            semantics,
            origins,
            self.voxel_size,
            self.voxel_range,
            self.free_index,
            self.ignore_index,
            stop_at_occupied=self.stop_at_occupied,
            boundary_faces=self.boundary_faces,
            hide_occluded_occupied=self.hide_occluded_occupied,
            device=self.device,
        )

    def __getitem__(self, sample_id):
        info = self.index[sample_id]
        data = np.load(self.root / info["data"])
        return torch.as_tensor(data["mask"], dtype=torch.bool)


@transform.register(namespace="occupancy")
class TraceVisibility(Transform, RegistryBaseType):
    """
    Replace the occupancy observation mask with an occlusion-aware one.

    Per frame, visibility is re-traced from the LiDAR origins
    (``sample.lidar_origins``, populated by a sensor-origins loader) against the
    GT occupied voxels with rays that stop at the first occupied voxel, so voxels
    occluded behind a surface are no longer marked observed-free. Results are
    cached on disk (keyed by hparams) like ``BuildInstanceLabels``; ``apply`` is a
    cache lookup that overwrites the named masks (``valid``/``lidar``).
    """

    def __init__(
        self,
        source: Dataset,
        voxel_size: tuple[float, float, float],
        voxel_range: tuple[float, ...],
        free_index: int,
        ignore_index: int,
        masks: Sequence[str] = ("valid", "lidar"),
        stop_at_occupied: bool = True,
        boundary_faces: Sequence[str] = (),
        hide_occluded_occupied: bool = False,
        device: str = "cpu",
    ):
        # pylint: disable=too-many-arguments
        self.masks = tuple(masks)
        self.data = VisibilityData(
            source=source,
            voxel_size=voxel_size,
            voxel_range=voxel_range,
            free_index=free_index,
            ignore_index=ignore_index,
            stop_at_occupied=stop_at_occupied,
            boundary_faces=boundary_faces,
            hide_occluded_occupied=hide_occluded_occupied,
            device=device,
        )

    @classmethod
    def from_config(cls, conf: OmegaConf | Mapping[str, Any], *args, **kwargs):
        conf_kwargs = config.utils.get_kwargs(conf)

        log.info("building occupancy visibility index")
        source = dataset.build(conf_kwargs.pop("data"))

        labels = conf_kwargs.pop("labels")
        free_index = list(labels.all).index("free")
        ignore_index = labels.ignore_index

        return cls(
            *args,
            source=source,
            free_index=free_index,
            ignore_index=ignore_index,
            **conf_kwargs,
            **kwargs,
        )

    @property
    def hparams(self) -> Mapping[str, Hashable]:
        return {
            "type": f"occupancy.{self.__class__.__name__}",
            "masks": list(self.masks),
            **self.data.cache_hparams,
        }

    def apply(self, sample: Sample) -> Sample:
        mask = self.data[sample.meta.sample_id]
        mask = mask.to(sample.labels.occupancy.semantics.device)
        for name in self.masks:
            sample.labels.occupancy.masks[name] = mask
        return sample


@transform.register(namespace="occupancy")
class ComputeSemanticMasks(Transform):
    def __init__(self, mask: str | None = "valid", ignore_index: int = -1):
        super().__init__()

        self.mask = mask if mask != "none" else None
        self.ignore_index = ignore_index

    @property
    def hparams(self) -> Mapping[str, Hashable]:
        return {
            "type": f"occupancy.{self.__class__.__name__}",
            "mask": self.mask,
            "ignore_index": self.ignore_index,
        }

    def apply(self, sample: Sample) -> Sample:
        semantics = sample.labels.occupancy.semantics

        # build the mask for valid semantic voxels
        mask = semantics != self.ignore_index
        if self.mask is not None:
            mask = mask & sample.labels.occupancy.masks[self.mask]

        # get the unique semantic classes for the masked area
        class_ids = semantics[mask]
        class_ids = torch.unique(class_ids, sorted=True)

        # compute the masks for each class
        class_masks = class_ids[:, None, None, None] == semantics[None, :, :, :]

        # store
        sample.labels.occupancy.semantic_ids = PackedTensor(class_ids)
        sample.labels.occupancy.semantic_masks = PackedTensor(class_masks)

        return sample


@transform.register(namespace="occupancy")
class ComputeInstanceMasks(Transform):
    @property
    def hparams(self) -> Mapping[str, Hashable]:
        return {
            "type": f"occupancy.{self.__class__.__name__}",
        }

    def apply(self, sample: Sample) -> Sample:
        iids = sample.labels.instance_ids.get(0)
        iids_occ = sample.labels.occupancy.instance_ids

        m = iids[:, None, None, None] == iids_occ[None, :, :, :]

        sample.labels.occupancy.instance_masks = PackedTensor(m)
        return sample


@transform.register(namespace="occupancy")
class DropLabelsWithoutOccupancy(Transform):
    def __init__(self, min_occupancy: int = 1, mask: str | None = "valid"):
        super().__init__()

        self.min_occupancy = min_occupancy
        self.mask = mask if mask != "none" else None

    @property
    def hparams(self) -> Mapping[str, Hashable]:
        return {
            "type": f"occupancy.{self.__class__.__name__}",
            "min_occupancy": self.min_occupancy,
            "mask": self.mask,
        }

    def apply(self, sample: Sample) -> Sample:
        labels = sample.labels

        assert "instance_ids" in labels
        assert "occupancy" in labels
        assert "instance_ids" in labels.occupancy

        iids = labels.instance_ids.data
        iids_occ = labels.occupancy.instance_ids

        if self.mask is not None:
            mask = labels.occupancy.masks[self.mask]
            mask = mask[None, :, :, :]

        # compute mask for valid instances
        valid = iids[:, None, None, None] == iids_occ[None, :, :, :]
        valid = valid & mask
        valid = valid.flatten(start_dim=1).sum(dim=1)
        valid = valid >= self.min_occupancy

        # filter labels
        if "boxes" in labels:
            labels.boxes = PackedTensor(labels.boxes.data[valid])

        if "instance_ids" in labels:
            labels.instance_ids = PackedTensor(labels.instance_ids.data[valid])

        if "class_ids" in labels:
            labels.class_ids = PackedTensor(labels.class_ids.data[valid])

        if "class_names" in labels:
            labels.class_names = PackedArray(labels.class_names.data[valid.numpy()])

        if "trajectories" in labels:
            labels.trajectories.center = PackedTensor(
                labels.trajectories.center.data[valid]
            )
            labels.trajectories.valid = PackedTensor(
                labels.trajectories.valid.data[valid]
            )

        if "instance_masks" in labels.occupancy:
            labels.occupancy.instance_masks = PackedTensor(
                labels.occupancy.instance_masks.data[valid]
            )

        return sample


@transform.register(namespace="occupancy")
class ComputeDistanceField(Transform):
    """
    Compute distance field for occupancy data, creating vectors pointing from
    each voxel to the nearest occupied voxel.

    Supports two methods:
    - 'gradient': Fast approximation using distance transform gradients
    - 'nearest': Accurate computation using nearest neighbor search

    Args:
        labels: Occupancy label configuration
        mask: Optional validation mask name (e.g., 'camera', 'valid')
        method: Distance computation method ('gradient' or 'nearest')
    """

    def __init__(
        self,
        labels: Any,
        mask: str | None = "valid",
        method: str = "gradient",
    ):
        super().__init__()

        self.labels = labels
        self.mask = mask if mask != "none" else None
        self.method = method

        if self.method not in ["gradient", "nearest"]:
            raise ValueError(f"Unknown method: {self.method}")

        self._build_class_mappings()

    def _build_class_mappings(self):
        """Build mappings from label config to identify free/occupied class IDs."""
        # Get the final remapped indices (what we see in semantics tensor)
        all_classes = OmegaConf.to_container(self.labels.all, resolve=True)
        class_to_id = {name: index for index, name in enumerate(all_classes)}

        # Get free space class IDs (in the remapped space)
        free_classes = OmegaConf.to_container(self.labels.free, resolve=True)
        self.free_class_ids = [
            class_to_id[cls] for cls in free_classes if cls in class_to_id
        ]

        # Get occupied class IDs (everything except free)
        all_class_names = set(all_classes)
        occupied_classes = all_class_names - set(free_classes)
        self.occupied_class_ids = [
            class_to_id[cls] for cls in occupied_classes if cls in class_to_id
        ]

    @property
    def hparams(self) -> Mapping[str, Hashable]:
        return {
            "type": f"occupancy.{self.__class__.__name__}",
            "mask": self.mask,
            "method": self.method,
        }

    def apply(self, sample: Sample) -> Sample:
        """Apply distance field computation to the sample."""
        semantics = sample.labels.occupancy.semantics  # Already remapped data

        # Apply validation mask
        if self.mask is not None:
            valid_mask = sample.labels.occupancy.masks[self.mask]
        else:
            valid_mask = torch.ones_like(semantics, dtype=torch.bool)

        # Create target mask (occupied voxels that pass validation)
        occupied_mask = torch.isin(
            semantics, torch.tensor(self.occupied_class_ids, dtype=semantics.dtype)
        )
        target_mask = occupied_mask & valid_mask

        # Compute distance transform
        distance_grid = ~target_mask.numpy()
        distances = scipy.ndimage.distance_transform_edt(distance_grid)

        # Compute direction vectors
        if self.method == "gradient":
            vectors = self._compute_gradient_vectors(distances)
        elif self.method == "nearest":
            vectors = self._compute_nearest_vectors(distances, target_mask.numpy())
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Store results
        if "distance_field" not in sample.labels.occupancy:
            sample.labels.occupancy.distance_field = MetaDict()

        sample.labels.occupancy.distance_field.distances = torch.tensor(
            distances, dtype=torch.float32
        )
        sample.labels.occupancy.distance_field.vectors = torch.tensor(
            vectors, dtype=torch.float32
        )

        return sample

    def _compute_gradient_vectors(self, distances, epsilon=1e-9):
        """
        Compute direction vectors using gradient approximation.
        Fast but approximate - points toward steepest descent.
        """
        gradients = np.gradient(distances)  # (grad_z, grad_y, grad_x)
        vectors = np.stack(
            [gradients[2], gradients[1], gradients[0]], axis=-1
        )  # (x,y,z)

        # Negate gradients to point toward nearest occupied voxel (decreasing distance)
        vectors = -vectors

        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        norms = np.where(norms > epsilon, norms, 1.0)
        vectors = vectors / norms

        return vectors

    def _compute_nearest_vectors(self, distances, target_mask, epsilon=1e-9):
        """
        Compute direction vectors pointing to actual nearest occupied voxel.
        Slower but mathematically correct for surface regression training.
        """
        target_coords = np.stack(np.where(target_mask), axis=1)
        if len(target_coords) == 0:
            return np.zeros((*distances.shape, 3))

        non_target_coords = np.stack(np.where(distances > 0), axis=1)
        if len(non_target_coords) == 0:
            return np.zeros((*distances.shape, 3))

        kdtree = scipy.spatial.cKDTree(target_coords)
        _, nearest_indices = kdtree.query(non_target_coords)

        nearest_coords = target_coords[nearest_indices]

        # Compute directions in voxel index space
        directions = nearest_coords - non_target_coords

        # Convert to (x, y, z) order: (z, y, x) -> (x, y, z)
        directions = directions[:, [2, 1, 0]]

        # Normalize direction vectors
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms > epsilon, norms, 1.0)
        directions = directions / norms

        vectors = np.zeros((*distances.shape, 3))
        vectors[tuple(non_target_coords.T)] = directions

        return vectors


def _as_xyz_triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    """Normalize a scalar or [X, Y, Z] sequence to a tuple of three ints."""
    if isinstance(value, int):
        return (value, value, value)
    v = [int(x) for x in value]
    assert len(v) == 3, f"expected a scalar or [X, Y, Z] triple, got {value!r}"
    return (v[0], v[1], v[2])


@transform.register(namespace="occupancy")
class Pool(Transform, RegistryBaseType):
    """
    Downsample occupancy labels by integer block-pooling.

    Pools ``sample.labels.occupancy.semantics`` and every entry of
    ``sample.labels.occupancy.masks`` by an integer ``factor`` per axis, reducing
    the grid resolution while leaving ``transforms.pose`` / ``timestamp``
    untouched. Operates purely on the generic occupancy tensors (``[Z, Y, X]``),
    so it works for any dataset.

    Semantics use a foreground-preserving majority by default: within each block
    the ``ignore`` voxels are dropped, and if any non-``free`` class is present
    the most frequent non-``free`` class wins (so small/thin objects are not
    erased by surrounding free space), otherwise ``free``, otherwise ``ignore``.
    ``"majority"`` instead takes the plain most-frequent non-``ignore`` class.

    Masks use logical OR by default (a coarse voxel is observed if any of its
    fine voxels was), or ``"majority"`` (> 50% of fine voxels observed).

    Note: this is a pure resolution change; the configured ``voxel_size`` is
    expected to be the source ``voxel_size`` times ``factor`` (same range), and
    must be kept consistent with the model grid by the config.

    Args:
        labels: Occupancy label config (provides ``ignore_index`` and the class
            order ``all``, from which the ``free`` index is taken).
        factor: Integer pooling factor, scalar or ``[X, Y, Z]`` (default ``2``).
        semantics_mode: ``"foreground_majority"`` (default) or ``"majority"``.
        mask_mode: ``"any"`` (default) or ``"majority"``.
    """

    def __init__(
        self,
        labels: OmegaConf,
        factor: int | Sequence[int] = 2,
        semantics_mode: str = "foreground_majority",
        mask_mode: str = "any",
    ):
        super().__init__()

        assert semantics_mode in ("foreground_majority", "majority")
        assert mask_mode in ("any", "majority")

        conf = OmegaConf.to_container(labels, resolve=True, throw_on_missing=True)
        self.ignore_index: int = conf["ignore_index"]
        self.all_classes = list(conf["all"])
        self.free_index: int = self.all_classes.index("free")
        self.num_classes = len(self.all_classes)

        self.factor = _as_xyz_triple(factor)
        self.semantics_mode = semantics_mode
        self.mask_mode = mask_mode

    @property
    def hparams(self) -> Mapping[str, Hashable]:
        return {
            "type": f"occupancy.{self.__class__.__name__}",
            "factor": list(self.factor),
            "semantics_mode": self.semantics_mode,
            "mask_mode": self.mask_mode,
            "ignore_index": self.ignore_index,
            "free_index": self.free_index,
        }

    def _blocks(self, grid: torch.Tensor) -> torch.Tensor:
        """Reshape a [Z, Y, X] grid into [Zt, Yt, Xt, K] blocks (K = fz*fy*fx)."""
        fx, fy, fz = self.factor  # factor is [X, Y, Z]
        z, y, x = grid.shape
        assert (
            z % fz == 0 and y % fy == 0 and x % fx == 0
        ), f"grid {tuple(grid.shape)} not divisible by factor (z,y,x)=({fz},{fy},{fx})"
        zt, yt, xt = z // fz, y // fy, x // fx
        return (
            grid.reshape(zt, fz, yt, fy, xt, fx)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(zt, yt, xt, fz * fy * fx)
        )

    def _pool_semantics(self, semantics: torch.Tensor) -> torch.Tensor:
        # pylint: disable=too-many-locals

        blocks = self._blocks(semantics)  # [Zt, Yt, Xt, K] int64
        zt, yt, xt, _ = blocks.shape
        flat = blocks.reshape(-1, blocks.shape[-1])  # [N, K]
        n, _ = flat.shape
        c = self.num_classes

        # Count class occurrences per coarse voxel, excluding ignore / out-of-range.
        valid = (flat >= 0) & (flat < c)
        idx = torch.where(valid, flat, torch.zeros_like(flat))
        counts = torch.zeros(n, c, dtype=torch.int32)
        counts.scatter_add_(1, idx, valid.to(torch.int32))

        total = counts.sum(dim=1)
        ignore = torch.full((n,), self.ignore_index, dtype=torch.long)

        if self.semantics_mode == "majority":
            out = torch.where(total > 0, counts.argmax(dim=1).long(), ignore)
            return out.reshape(zt, yt, xt)

        # foreground_majority: non-free > free > ignore
        nonfree = counts.clone()
        nonfree[:, self.free_index] = 0
        has_free = counts[:, self.free_index] > 0
        has_nonfree = nonfree.sum(dim=1) > 0

        out = ignore
        out = torch.where(has_free, torch.full_like(out, self.free_index), out)
        out = torch.where(has_nonfree, nonfree.argmax(dim=1).long(), out)
        return out.reshape(zt, yt, xt)

    def _pool_mask(self, mask: torch.Tensor) -> torch.Tensor:
        blocks = self._blocks(mask)  # [Zt, Yt, Xt, K] bool
        if self.mask_mode == "any":
            return blocks.any(dim=-1)
        return blocks.to(torch.float32).mean(dim=-1) > 0.5

    def apply(self, sample: Sample) -> Sample:
        if "labels" not in sample or "occupancy" not in sample.labels:
            return sample

        occ = sample.labels.occupancy
        occ.semantics = self._pool_semantics(occ.semantics)
        if "masks" in occ:
            occ.masks = MetaDict({k: self._pool_mask(v) for k, v in occ.masks.items()})

        return sample


@transform.register(namespace="occupancy")
class Crop(Transform, RegistryBaseType):
    """
    Crop occupancy labels to a fixed grid size.

    Slices ``sample.labels.occupancy.semantics`` and every entry of
    ``sample.labels.occupancy.masks`` to ``size`` voxels per axis, leaving
    ``transforms.pose`` / ``timestamp`` untouched. Operates on the generic
    ``[Z, Y, X]`` occupancy tensors (e.g. a symmetric centre crop of a
    factor-2-pooled grid down to ``[200, 200, 16]``).

    Note: cropping shifts the grid origin, so the configured ``voxel_range`` must
    be set to the cropped extent to stay consistent with the model grid.

    Args:
        size: Output size in voxels as ``[X, Y, Z]`` (default ``[200, 200, 16]``).
        offset: Low-side voxels to drop as ``[X, Y, Z]``. ``None`` (default)
            centre-crops, which requires each axis margin to split evenly.
    """

    def __init__(
        self,
        size: Sequence[int] = (200, 200, 16),
        offset: Sequence[int] | None = None,
    ):
        super().__init__()

        self.size = _as_xyz_triple(size)
        self.offset = None if offset is None else _as_xyz_triple(offset)

    @property
    def hparams(self) -> Mapping[str, Hashable]:
        return {
            "type": f"occupancy.{self.__class__.__name__}",
            "size": list(self.size),
            "offset": None if self.offset is None else list(self.offset),
        }

    def _slices(self, shape_zyx: tuple[int, int, int]) -> tuple[slice, slice, slice]:
        # size / offset are [X, Y, Z]; tensor dims are [Z, Y, X].
        sizes_zyx = (self.size[2], self.size[1], self.size[0])
        if self.offset is None:
            offs_zyx = []
            for dim, size in zip(shape_zyx, sizes_zyx):
                margin = dim - size
                assert margin >= 0, f"crop size {size} exceeds grid dim {dim}"
                assert margin % 2 == 0, (
                    f"centre crop of dim {dim} to {size} is not symmetric; "
                    f"pass an explicit offset"
                )
                offs_zyx.append(margin // 2)
        else:
            offs_zyx = [self.offset[2], self.offset[1], self.offset[0]]

        slices = []
        for off, size, dim in zip(offs_zyx, sizes_zyx, shape_zyx):
            assert (
                0 <= off and off + size <= dim
            ), f"crop [{off}, {off + size}) out of bounds for dim {dim}"
            slices.append(slice(off, off + size))
        return tuple(slices)

    def apply(self, sample: Sample) -> Sample:
        if "labels" not in sample or "occupancy" not in sample.labels:
            return sample

        occ = sample.labels.occupancy
        sl = self._slices(tuple(occ.semantics.shape))
        occ.semantics = occ.semantics[sl]
        if "masks" in occ:
            occ.masks = MetaDict({k: v[sl] for k, v in occ.masks.items()})

        return sample
