# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared helpers for occupancy visualization/eval on the standard dataset pipeline.

Drives the dataset + transform pipeline via a composed config, so ground truth,
labels and the voxel grid come from exactly the code the model trains on. Works
for any dataset the pipeline supports (NuScenes, Waymo) without per-dataset
conversions, joining predictions to samples purely on
``meta.sequence_id`` / ``meta.sample_id``.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from tracker import config
from tracker.data import dataset as dataset_mod
from tracker.data.dataset.utils.distributed import distribute_scenes
from tracker.utils.types import MetaDict, Sample

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

# Single master class-name -> RGB palette, merged across NuScenes / Waymo. Class
# names that mean the same thing across datasets (car, pedestrian, ...)
# intentionally share a colour. Names absent here fall back to a deterministic
# hash colour, so the palette degrades gracefully for any future class.
PALETTE: dict[str, tuple[int, int, int]] = {
    # free / background-ish
    "others": (5, 5, 5),
    "background": (5, 5, 5),
    "background_dynamic": (40, 40, 40),
    # vehicles
    "car": (255, 158, 0),
    "van": (255, 175, 60),
    "truck": (255, 99, 71),
    "bus": (255, 127, 80),
    "trailer": (255, 140, 0),
    "construction_vehicle": (233, 150, 70),
    "large_vehicle": (200, 90, 40),
    "rideable_vehicle": (255, 110, 130),
    "vehicle": (255, 158, 0),
    # vulnerable road users
    "pedestrian": (0, 0, 230),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "rider": (180, 20, 120),
    "cyclist": (220, 20, 60),
    "animal": (150, 75, 0),
    # static objects
    "traffic_cone": (47, 79, 79),
    "construction_cone": (47, 79, 79),
    "barrier": (112, 128, 144),
    "traffic_sign": (160, 160, 40),
    "sign": (233, 150, 70),
    "traffic_light": (255, 127, 80),
    "pole": (112, 128, 144),
    "lost_cargo": (200, 40, 200),
    # surfaces / stuff
    "driveable_surface": (0, 207, 191),
    "road": (0, 207, 191),
    "other_flat": (175, 0, 75),
    "sidewalk": (75, 0, 75),
    "walkable": (75, 0, 75),
    "terrain": (112, 180, 60),
    "manmade": (222, 184, 135),
    "building": (222, 184, 135),
    "vegetation": (0, 175, 0),
    "tree_trunk": (175, 175, 0),
}


def color_for_class(name: str) -> tuple[float, float, float]:
    """RGB in [0, 1] for a class name, with a deterministic hash fallback."""
    rgb = PALETTE.get(name)
    if rgb is None:
        digest = hashlib.md5(name.encode("utf-8")).digest()
        rgb = (digest[0], digest[1], digest[2])
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)


# ---------------------------------------------------------------------------
# Config / dataset construction
# ---------------------------------------------------------------------------


# Supported datasets. The occupancy pipeline for each is defined entirely in code
# below (no external/experiment config files), using the fixed grids/splits the
# project uses for occupancy.
DATASETS = ("nuscenes", "waymo")

# Model voxel grid per dataset: (voxel_size [x,y,z], voxel_range [xmin..zmax]).
_GRID: dict[str, tuple[list, list]] = {
    "nuscenes": ([0.4, 0.4, 0.4], [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]),
    "waymo": ([0.4, 0.4, 0.4], [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]),
}

# Native (pre-downsample) grid; nuscenes/waymo are already loaded at the model
# grid, so this mirrors the model grid.
_NATIVE_GRID: dict[str, tuple[list, list]] = {
    "nuscenes": _GRID["nuscenes"],
    "waymo": _GRID["waymo"],
}

# Default (validation) split name per dataset.
_DEFAULT_SPLIT = {
    "nuscenes": "v1.0-val",
    "waymo": "validation",
}

# Hard-coded dataset roots (matching config/paths/default.yaml).
_DATA = config.paths.root / "data"
_LABELS = config.paths.config / "data" / "labels"


def _label_conf(kind: str, name: str):
    """Load a shared label definition (``occupancy``/``detection``)."""
    return OmegaConf.load(_LABELS / kind / f"{name}.yaml")


def _set_global_config() -> None:
    """Set a minimal global config providing the paths the pipeline reads.

    Some datasets read ``config.get().paths`` (e.g. for the sample cache), which
    is unset outside a full run.
    """
    config.set_global_config(
        OmegaConf.create(
            {
                "paths": {
                    "root": str(config.paths.root),
                    "cache": str(config.paths.root / "cache"),
                }
            }
        )
    )


# Each builder returns ``(types, source_kwargs, occ_loaders, post_transforms)``:
# ``types`` is ``(random_access_type, sequential_type)``; ``occ_loaders`` are the
# transforms producing semantic occupancy + masks at the model grid (its first
# entry alone is the raw, pre-downsample loader, used for native mode);
# ``post_transforms`` are the transforms applied after them (e.g. instance
# labels; empty when the loader needs nothing more). Inner ``data`` sources are
# always the random-access variant.


def _nuscenes_pipeline(split, occ, det, vs, vr):
    root = str(_DATA / "nuscenes")
    occ3d = str(_DATA / "nuscenes-occ3d")
    src_kwargs = {"root": root, "split": split}
    occ_loaders = [
        {
            "type": "nuscenes.LoadSemanticOccupancy",
            "root": occ3d,
            "labels": occ,
            "valid_mask": ["camera"],
        },
    ]
    instance_data = {
        "source": {"type": "NuScenes", **src_kwargs},
        "transforms": [
            {
                "type": "nuscenes.LoadAnnotations",
                "labels": det,
                "ignore_empty_annots": "none",
                "ref_frame": "ego",
            },
            {"type": "nuscenes.LoadSemanticOccupancy", "root": occ3d, "labels": occ},
        ],
    }
    instance_transform = {
        "type": "occupancy.BuildInstanceLabels",
        "data": instance_data,
        "labels": occ,
        "box_labels": det,
        "voxel_size": vs,
        "voxel_range": vr,
    }
    return (
        ("NuScenes", "NuScenesSequential"),
        src_kwargs,
        occ_loaders,
        [instance_transform],
    )


def _waymo_pipeline(split, occ, det, vs, vr):
    # Waymo's LoadSemanticOccupancy provides instance_ids natively, so no
    # BuildInstanceLabels (hence ``det``/``vs``/``vr`` unused) is needed.
    # pylint: disable=unused-argument
    root = str(_DATA / "TrackOcc-waymo" / "kitti_format")
    occ3d = str(_DATA / "TrackOcc-waymo" / "pano_voxel04")
    src_kwargs = {"root": root, "split": split}
    occ_loaders = [
        {
            "type": "waymo.LoadSemanticOccupancy",
            "root": occ3d,
            "labels": occ,
            "valid_mask": ["camera"],
        },
    ]
    return ("WaymoTO", "WaymoTOSequential"), src_kwargs, occ_loaders, []


_PIPELINE = {
    "nuscenes": _nuscenes_pipeline,
    "waymo": _waymo_pipeline,
}

# Per-dataset accessors for grouping a random-access source's raw records by
# scene and ordering frames within a scene (records are dicts for nuscenes,
# light objects for waymo).
_SEQUENCE_KEY = {
    "nuscenes": lambda r: r["scene_token"],
    "waymo": lambda r: r.context_name,
}
_ORDER_KEY = {
    "nuscenes": lambda r: r["timestamp"],
    "waymo": lambda r: r.sample_idx,
}


@dataclass
class OccupancyContext:
    """Everything a viewer/eval needs that is shared across all frames."""

    dataset_name: str  # one of DATASETS
    dataset: Any  # built (wrapped) dataset for the chosen split
    labels: Any  # occupancy label config
    voxel_size: np.ndarray  # (3,) float, [x, y, z] metres
    voxel_range: np.ndarray  # (6,) float, [xmin, ymin, zmin, xmax, ymax, zmax]

    @property
    def free_index(self) -> int:
        return list(self.labels.all).index("free")


def load_context(
    dataset: str,
    split: Optional[str] = None,
    overrides: Optional[List[str]] = None,
    sequential: bool = True,
    native: bool = False,
) -> OccupancyContext:
    """Build the occupancy ground-truth pipeline for ``dataset``.

    ``dataset`` is one of :data:`DATASETS`; the matching single-frame,
    occupancy-only pipeline (fast: no image/lidar/depth) is assembled in code, so
    callers only pick the dataset (and optionally a non-default ``split``).
    ``overrides`` is an OmegaConf dotlist applied to the assembled config for
    occasional tweaks (e.g. ``data.source.source.split=...``).

    ``sequential`` selects the source variant: the streaming ``*Sequential``
    dataset (default; supports DDP scene sharding) or the random-access dataset
    (supports ``__getitem__``, so :func:`collect_scene` can jump straight to a
    scene without streaming through the ones before it).

    ``native`` loads the raw, pre-downsample occupancy (no pooling/cropping, and
    no instance labels) on the native grid — for debugging the ground truth as
    stored. For nuscenes/waymo (already at the model grid) it just drops the
    instance labels.
    """
    # pylint: disable=too-many-locals

    if dataset not in _PIPELINE:
        raise ValueError(
            f"unknown dataset {dataset!r}; expected one of {', '.join(DATASETS)}"
        )

    _set_global_config()

    split = split or _DEFAULT_SPLIT[dataset]
    occ = _label_conf("occupancy", dataset)
    det = _label_conf("detection", dataset)

    # The builder embeds the model grid into BuildInstanceLabels; the context grid
    # depends on whether we want the native or model resolution.
    (random_type, seq_type), src_kwargs, occ_loaders, post = _PIPELINE[dataset](
        split, occ, det, *_GRID[dataset]
    )

    if native:
        vs, vr = _NATIVE_GRID[dataset]
        transforms = [occ_loaders[0]]  # raw loader only, before pool/crop
    else:
        vs, vr = _GRID[dataset]
        transforms = [*occ_loaders, *post]

    if sequential:
        source_node = {
            "type": seq_type,
            "batch_size": 1,
            "batch_mode": "parallel",
            "device_mode": "parallel",
            **src_kwargs,
        }
    else:
        source_node = {"type": random_type, **src_kwargs}

    conf = OmegaConf.create(
        {"data": {"source": {"source": source_node, "transforms": transforms}}}
    )
    if overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(overrides))

    return OccupancyContext(
        dataset_name=dataset,
        dataset=dataset_mod.build(conf.data.source),
        labels=occ,
        voxel_size=np.asarray(vs, dtype=np.float64),
        voxel_range=np.asarray(vr, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Sample iteration
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """A single transformed ground-truth frame, on CPU, in ``[Z, Y, X]`` order."""

    dataset_type: str
    sequence_id: str
    sample_id: Any
    semantics: torch.Tensor  # [Z, Y, X] long
    instance_ids: Optional[torch.Tensor]  # [Z, Y, X] long, -1 = no instance
    masks: dict[str, torch.Tensor]  # name -> [Z, Y, X] bool


def _is_valid(sample: Sample) -> bool:
    iv = sample.is_valid
    if isinstance(iv, torch.Tensor):
        return bool(iv.all())
    return bool(iv)


def frame_from_sample(sample: Sample) -> Optional[Frame]:
    """Extract a :class:`Frame` from a transformed sample, or ``None`` if padding.

    Under DDP the source pads each rank's stream to a common (max) length; those
    padding samples are still transformed but carry ``is_valid=False``.
    """
    if not _is_valid(sample):
        return None

    occ = sample.labels.occupancy
    meta = sample.meta

    instance_ids = None
    if "instance_ids" in occ:
        instance_ids = occ.instance_ids.cpu().long()

    masks = {k: v.cpu().bool() for k, v in occ.masks.items()}

    return Frame(
        dataset_type=getattr(meta, "dataset_type", ""),
        sequence_id=str(meta.sequence_id),
        sample_id=meta.sample_id,
        semantics=occ.semantics.cpu().long(),
        instance_ids=instance_ids,
        masks=masks,
    )


def iter_frames(ctx: OccupancyContext) -> Iterator[Frame]:
    """Stream transformed ground-truth frames in dataset (scene) order.

    Skips padding samples; use :func:`frame_from_sample` directly if you also want
    to account for padding (e.g. to drive a progress bar over the raw stream).
    """
    for sample in ctx.dataset:
        frame = frame_from_sample(sample)
        if frame is not None:
            yield frame


def frame_count(ctx: OccupancyContext) -> Optional[int]:
    """Number of samples the current process iterates, or ``None`` if unknown.

    Mirrors the source's own scene distribution (``distribute_scenes``), which is
    rank-aware: with no process group this is the full split; under DDP it is the
    rank's padded shard. Padding samples are included because the source streams
    (and transforms) them in lockstep across ranks, so this is the true number of
    iteration steps — and what a progress bar over the raw stream advances through.
    """
    source = getattr(ctx.dataset, "source", None)
    scenes = getattr(source, "sequences", None) or getattr(source, "scenes", None)
    if scenes is None:
        return None

    sample_ids = distribute_scenes(
        [len(s) for s in scenes],
        batch_size=getattr(source, "batch_size", 1),
        device_mode=getattr(source, "device_mode", "parallel"),
        batch_mode=getattr(source, "batch_mode", "parallel"),
    )
    return len(sample_ids)


def scene_index(ctx: OccupancyContext) -> "dict[str, List[int]]":
    """Map each scene id to its random-access dataset indices, in frame order.

    Requires a random-access context (``load_context(..., sequential=False)``);
    reads the source's raw records (cheap, no transforms) to group by scene and
    order frames within each scene.
    """
    source = getattr(ctx.dataset, "source", None)
    samples = getattr(source, "samples", None)
    if samples is None:
        raise TypeError(
            "scene_index requires a random-access context "
            "(load_context(..., sequential=False))"
        )

    seq_key = _SEQUENCE_KEY[ctx.dataset_name]
    order_key = _ORDER_KEY[ctx.dataset_name]

    groups: dict[str, List[int]] = {}
    for idx, record in enumerate(samples):
        groups.setdefault(seq_key(record), []).append(idx)

    for _, indices in groups.items():
        indices.sort(key=lambda i: order_key(samples[i]))

    return groups


def collect_scene(ctx: OccupancyContext, scene: int) -> List[Frame]:
    """Collect all frames of the ``scene``-th sequence (ordinal index).

    With a random-access context this jumps straight to the scene's frames and
    transforms only those; with a sequential context it streams and stops once
    the target scene has been read.
    """
    if getattr(getattr(ctx.dataset, "source", None), "samples", None) is not None:
        index = scene_index(ctx)
        seq_ids = list(index)
        if scene < 0 or scene >= len(seq_ids):
            raise IndexError(
                f"scene index {scene} out of range ({len(seq_ids)} scenes)"
            )
        frames = [frame_from_sample(ctx.dataset[i]) for i in index[seq_ids[scene]]]
        return [f for f in frames if f is not None]

    # Sequential fallback: stream until the target scene has been read.
    seen: List[str] = []
    target: Optional[str] = None
    frames = []
    for frame in iter_frames(ctx):
        if frame.sequence_id not in seen:
            seen.append(frame.sequence_id)
            if len(seen) - 1 == scene:
                target = frame.sequence_id

        if target is None:
            continue
        if frame.sequence_id == target:
            frames.append(frame)
        elif frames:
            break  # moved past the target scene

    if not frames:
        raise IndexError(f"scene index {scene} not found (saw {len(seen)} scenes)")

    return frames


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def _sample_id_stem(sample_id: Any, pad: int = 7) -> str:
    """Path-safe filename stem for a ``sample_id`` (ints zero-padded to ``pad``)."""
    if isinstance(sample_id, (int, np.integer)):
        return f"{int(sample_id):0{pad}d}"
    return str(sample_id)


@dataclass
class Prediction:
    semantics: torch.Tensor  # [Z, Y, X] long
    instance_ids: torch.Tensor  # [Z, Y, X] long, -1 = no instance


def prediction_path(
    pred_root: str | Path, sequence_id: str, sample_id: Any, pad: int = 7
) -> Path:
    return Path(pred_root) / str(sequence_id) / f"{_sample_id_stem(sample_id, pad)}.npz"


def load_prediction(
    pred_root: str | Path, sequence_id: str, sample_id: Any, pad: int = 7
) -> Optional[Prediction]:
    """Load a unified-format prediction npz, or ``None`` if it does not exist.

    Predictions are stored ``[Z, Y, X]`` with ``-1`` = no instance, identical to
    the ground-truth convention, so no transpose or re-indexing is applied here.
    """
    path = prediction_path(pred_root, sequence_id, sample_id, pad)
    if not path.exists():
        return None

    data = np.load(path)
    return Prediction(
        semantics=torch.from_numpy(data["pano_sem"]).long(),
        instance_ids=torch.from_numpy(data["pano_inst"]).long(),
    )


# ---------------------------------------------------------------------------
# Batched-of-one Sample / preds, for the SegmentationTrackingQuality metric
# ---------------------------------------------------------------------------


def build_metric_inputs(
    frame: Frame, pred: Prediction, device: torch.device | str = "cpu"
) -> tuple[Sample, MetaDict]:
    """Wrap a GT frame + prediction as the ``(sample, preds)`` the metric expects.

    The metric's ``update`` indexes a batch dimension and reads
    ``sample.is_valid``/``sample.meta``/``sample.labels.occupancy.*`` and
    ``preds.occupancy.*``; we provide a batch of one.
    """
    meta = MetaDict()
    meta.dataset_type = frame.dataset_type
    meta.sample_id = frame.sample_id
    meta.sequence_id = frame.sequence_id

    occ = MetaDict()
    occ.semantics = frame.semantics.unsqueeze(0).to(device)
    occ.instance_ids = frame.instance_ids.unsqueeze(0).to(device)
    occ.masks = MetaDict({k: v.unsqueeze(0).to(device) for k, v in frame.masks.items()})

    labels = MetaDict()
    labels.occupancy = occ

    sample = Sample()
    sample.is_valid = torch.ones(1, dtype=torch.bool, device=device)
    sample.meta = [meta]
    sample.labels = labels

    preds = MetaDict()
    preds.occupancy = MetaDict()
    preds.occupancy.semantics = pred.semantics.unsqueeze(0).to(device)
    preds.occupancy.instance_ids = pred.instance_ids.unsqueeze(0).to(device)

    return Sample(sample), preds
