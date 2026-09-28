# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import math
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.distributed as dist
from scipy.spatial import cKDTree

from ... import config, t3d, utils
from ...config.registry import RegistryBaseType
from ...utils.types import PackedTensor, Sample
from .. import dataset
from ..dataset.dataset import Dataset
from .registry import registry as transform
from .transform import Transform

log = utils.log.get_logger(__name__)


@transform.register
class PrepareLidarToGlobalTxFused(Transform):
    def __init__(self, dtype: torch.dtype | str = torch.float32):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)

        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": self.__class__.__name__,
        }

    def apply(self, sample: Sample) -> Sample:
        tx_lidar_to_global = t3d.Sequential(
            # lidar sensor to ego vehicle frame (at lidar timestamp)
            sample.points.meta.transforms.extrinsic,
            # ego vehicle to global frame
            sample.points.meta.transforms.pose,
        )

        tx_lidar_to_global = tx_lidar_to_global.matrix
        tx_lidar_to_global = tx_lidar_to_global.to(dtype=self.dtype)

        sample.points.meta.transforms.lidar_to_global = tx_lidar_to_global
        return sample


@transform.register
class PrepareLidarEgoToGlobalTxFused(Transform):
    def __init__(self, dtype: torch.dtype | str = torch.float32):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)

        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": self.__class__.__name__,
        }

    def apply(self, sample: Sample) -> Sample:
        # ego vehicle (at lidar timestamp) to global frame
        tx_ego_to_global = sample.points.meta.transforms.pose
        tx_ego_to_global = tx_ego_to_global.matrix
        tx_ego_to_global = tx_ego_to_global.to(dtype=self.dtype)

        sample.points.meta.transforms.ego_to_global = tx_ego_to_global
        return sample


@transform.register
class FilterPointsByRange(Transform):
    """
    Filter LiDAR points by range.
    """

    # pylint: disable-next=redefined-builtin
    def __init__(self, range: tuple[float]):
        super().__init__()

        range = torch.as_tensor(range)

        # break up range
        self.ndim = range.shape[0] // 2
        self.aabb_min = range[: self.ndim]
        self.aabb_max = range[self.ndim :]

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": self.__class__.__name__,
            "range": self.aabb_min.tolist() + self.aabb_max.tolist(),
        }

    def apply(self, sample: Sample) -> Sample:
        points: PackedTensor = sample.points.data

        assert points.batch_size == 1

        coords = points.data[:, 0 : 0 + self.ndim]
        mask = (coords >= self.aabb_min[None, :]) & (coords <= self.aabb_max[None, :])
        mask = mask.all(1)

        points = PackedTensor(points.data[mask, :])

        sample.points.data = points
        return sample


def statistical_outlier_keep_mask(
    xyz: torch.Tensor, k: int, std_ratio: float, range_scale: float
) -> torch.Tensor:
    """Boolean [N] keep mask via range-adaptive statistical outlier removal.

    For each point the mean distance to its ``k`` nearest neighbours is compared
    against a global threshold ``mean + std_ratio * std``; points above it are
    dropped (``False``). Isolated sensor-noise points have few/distant neighbours
    and are removed, while points on real surfaces are densely supported and kept.

    LiDAR point density falls off with range, so with ``range_scale > 0`` the
    threshold is scaled by ``max(1, range_scale * r)`` (DSOR), where ``r`` is the
    distance from the sensor origin, granting far points proportionally more slack
    so genuine distant returns are not over-pruned. ``0`` recovers plain
    statistical outlier removal. Purely geometric: only the xyz coordinates are
    used (intensity is ignored).
    """
    n = xyz.shape[0]
    # Need at least k+1 points (each point's nearest neighbour is itself).
    if n <= k:
        return torch.ones(n, dtype=torch.bool)

    pts = xyz.detach().cpu().numpy().astype(np.float64)
    tree = cKDTree(pts)
    # k+1 neighbours: the first is the point itself (distance 0), dropped below.
    dists, _ = tree.query(pts, k=k + 1, workers=-1)
    mean_knn = dists[:, 1:].mean(axis=1)  # [N]

    threshold = mean_knn.mean() + std_ratio * mean_knn.std()
    if range_scale > 0.0:
        rng = np.linalg.norm(pts, axis=1)
        threshold = threshold * np.maximum(1.0, range_scale * rng)

    return torch.from_numpy(mean_knn <= threshold)


def _mask_key(sample: Sample) -> str:
    """Filesystem-safe per-frame key.

    Includes the sequence id because a frame token (sample id) is only unique
    within its sequence, so keying on the token alone would collide across
    overlapping sequences.
    """
    return f"{sample.meta.sequence_id}__{sample.meta.sample_id}"


class LidarOutlierMaskData:
    """Eagerly built, on-disk cache of per-frame LiDAR outlier masks.

    Mirrors :class:`occupancy.VoxelInstanceData`: at construction every frame of
    ``source`` is loaded, its merged cloud passed through
    :func:`statistical_outlier_keep_mask`, and the removed-point indices stored
    under a hashed ``utils.cache`` directory. The cache is keyed by the source
    hparams plus the filter params, so any change rebuilds it.
    """

    def __init__(self, source: Dataset, k: int, std_ratio: float, range_scale: float):
        self.source = source
        self.k = k
        self.std_ratio = std_ratio
        self.range_scale = range_scale

        index, root = self._load()
        self.index = index
        self.root = root

    @property
    def cache_hparams(self) -> Mapping[str, Any]:
        return {
            "data": self.source.hparams,
            "k": self.k,
            "std_ratio": self.std_ratio,
            "range_scale": self.range_scale,
        }

    def _load(self):
        artifact = utils.cache.get(
            "data.transform.lidar_base.LidarOutlierMask", self.cache_hparams
        )
        index_path = artifact.storage_path / "index.json"

        if artifact.available():
            log.info("cache found, loading lidar outlier index from '%s'", index_path)
            with open(index_path, "r", encoding="utf-8") as fd:
                index = json.load(fd)
        else:
            log.info("no cache found, generating lidar outlier masks...")
            index = self._build(artifact, index_path)

        return index, artifact.storage_path

    def _build(self, artifact: utils.cache.Artifact, index_path: Path):
        use_dist = dist.is_available() and dist.is_initialized()
        rank, world_size = (
            (dist.get_rank(), dist.get_world_size()) if use_dist else (0, 1)
        )

        # Barrier so the cache is only advertised as "available" once it has
        # actually been written (see occupancy.VoxelInstanceData._build).
        if use_dist:
            dist.barrier()

        if rank == 0:
            artifact.prepare()

        num_samples = len(self.source)
        chunk_size = math.ceil(num_samples / world_size)
        index_start = rank * chunk_size
        index_end = min(index_start + chunk_size, num_samples)
        index = self._build_partial(
            range(index_start, index_end), artifact.storage_path
        )

        if use_dist:
            synced = [[] for _ in range(world_size)]
            with warnings.catch_warnings(action="ignore", category=FutureWarning):
                dist.all_gather_object(synced, index)
            index = {k: v for part in synced for k, v in part.items()}

        if rank == 0:
            log.info("dumping lidar outlier index to '%s'", index_path)
            with open(index_path, "w", encoding="utf-8") as fd:
                json.dump(index, fd)
            artifact.initialize()

        return index

    def _build_partial(self, indices: Sequence[int], storage_root: Path):
        data = {}
        indices = utils.progress.track(indices, "Generating lidar outlier masks...")
        for sample_index in indices:
            sample = self.source[sample_index]
            assert sample.batch_size == 1

            key = _mask_key(sample)
            xyz = sample.points.data.get(0)[:, :3]
            keep = statistical_outlier_keep_mask(
                xyz, self.k, self.std_ratio, self.range_scale
            )
            removed = torch.nonzero(~keep, as_tuple=False).squeeze(1).to(torch.int32)

            path = f"lidar_mask_{key}.npz"
            np.savez_compressed(
                storage_root / path, removed=removed.numpy(), n=np.int64(len(keep))
            )
            data[key] = {"data": path}

        return data

    def __getitem__(self, key: str) -> tuple[torch.Tensor, int]:
        info = self.index[key]
        data = np.load(self.root / info["data"])
        removed = torch.as_tensor(data["removed"], dtype=torch.long)
        n = int(data["n"])
        return removed, n


@transform.register
class FilterLidarOutliers(Transform, RegistryBaseType):
    """
    Remove LiDAR outliers using cached statistical-outlier-removal masks.

    The outlier decision (range-adaptive statistical outlier removal; see
    :func:`statistical_outlier_keep_mask`) costs a per-frame kNN search, so it is
    computed once over ``source`` and cached (mirroring
    :class:`occupancy.BuildInstanceLabels`). At apply time the cached mask for the
    frame is loaded and the merged cloud — plus the parallel ``sensor_id`` /
    ``ring`` tensors — filtered in place.

    ``source`` MUST produce the same merged cloud as the pipeline this transform
    runs in: identical ``LoadLidar`` point selection (sensors / ``min_range`` /
    ``angle_filter`` / ``ring_filter``), since the mask stores point indices into
    that cloud. ``ref_frame`` may differ — it is a rigid transform and changes
    neither the point order nor the outlier decision. A length check guards
    against drift. Configure ``source`` to reuse the exact ``LoadLidar`` config
    node (e.g. via a YAML anchor).

    Args:
        source: Dataset yielding the merged cloud (``sample.points``) per frame.
        k, std_ratio, range_scale: :func:`statistical_outlier_keep_mask` params.
    """

    def __init__(
        self,
        source: Dataset,
        k: int = 8,
        std_ratio: float = 2.0,
        range_scale: float = 0.0,
    ):
        super().__init__()
        self.data = LidarOutlierMaskData(
            source=source, k=k, std_ratio=std_ratio, range_scale=range_scale
        )

    @classmethod
    def from_config(cls, conf, *args, **kwargs):
        conf_kwargs = config.utils.get_kwargs(conf)
        source = dataset.build(conf_kwargs.pop("data"))
        return cls(*args, source=source, **conf_kwargs, **kwargs)

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": self.__class__.__name__,
            **self.data.cache_hparams,
        }

    def apply(self, sample: Sample) -> Sample:
        points: PackedTensor = sample.points.data

        assert points.batch_size == 1

        removed, n = self.data[_mask_key(sample)]

        actual = points.data.shape[0]
        if actual != n:
            raise RuntimeError(
                f"cached LiDAR outlier mask length {n} != current cloud {actual} "
                f"for {sample.meta.sequence_id}/{sample.meta.sample_id}: the "
                f"LoadLidar point selection in this pipeline differs from the mask "
                f"source. Align them (or clear the cache) and rebuild."
            )

        # Nothing to drop: skip reindexing (also avoids PackedTensor's empty-cloud
        # masking edge case).
        if removed.numel() == 0:
            return sample

        keep_mask = torch.ones(n, dtype=torch.bool)
        keep_mask[removed] = False
        keep = PackedTensor(keep_mask, points.offsets)

        # Filter the cloud and every parallel per-point tensor with one mask so
        # they stay aligned (PackedTensor.__getitem__ recomputes offsets).
        sample.points.data = points[keep]
        for field in ("sensor_id", "ring"):
            if field in sample.points:
                sample.points[field] = sample.points[field][keep]

        return sample
