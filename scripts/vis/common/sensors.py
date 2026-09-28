# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Full-sensor dataset builders for the occupancy viewers.

Shared by the Rerun and viser occupancy viewers. Each builder returns
``(dataset, occupancy_labels, voxel_size, voxel_range)`` and loads the full
sensor pipeline (images + optional lidar + annotations + occupancy), so the
resulting samples carry everything the viewers need (cameras, boxes,
trajectories, per-frame ego pose). ``resolve_start_sample`` maps a
scene/scene-name/token/sample selection to a flat dataset index.
"""

import torch
from omegaconf import OmegaConf

from tracker import data
from tracker.config import paths

# Per-dataset default splits. The GT-only script used ``v1.0-mini_val`` for
# nuscenes; for prediction bundles the default is the full val split so it lines
# up with typical prediction runs. Callers may override via ``split``.
DEFAULT_SPLITS = {
    "nuscenes": "v1.0-val",
    "waymo": "validation",
}

# Per-dataset keyframe capture rate (Hz), used to play the Rerun recording back at
# real time. nuScenes/Waymo occupancy keyframes are annotated at 2 Hz.
DATASET_FPS = {
    "nuscenes": 2.0,
    "waymo": 2.0,
}


def dataset_fps(dataset_type: str) -> float:
    """Keyframe rate (Hz) for the dataset's occupancy frames."""
    try:
        return DATASET_FPS[dataset_type]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset type: {dataset_type}") from exc


def build_dataset_nuscenes(split: str | None = None, load_lidar: bool = False):
    split = split or DEFAULT_SPLITS["nuscenes"]

    box_labels = "config/data/labels/detection/nuscenes.yaml"
    box_labels = OmegaConf.load(paths.root / box_labels)

    occupancy_labels = "config/data/labels/occupancy/nuscenes.yaml"
    occupancy_labels = OmegaConf.load(paths.root / occupancy_labels)

    voxel_size = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float32)
    voxel_range = torch.tensor(
        [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], dtype=torch.float32
    )

    dataset = data.dataset.NuScenes(
        root=paths.root / "data" / "nuscenes",
        split=split,
    )

    instance_transforms = [
        data.dataset.nuscenes.LoadAnnotations(
            labels=box_labels,
            ignore_empty_annots="none",
            ref_frame="ego",
        ),
        data.dataset.nuscenes.LoadSemanticOccupancy(
            root=paths.root / "data" / "nuscenes-occ3d",
            labels=occupancy_labels,
        ),
    ]

    transforms = [
        data.dataset.nuscenes.LoadImages(),
    ]
    if load_lidar:
        transforms.append(data.dataset.nuscenes.LoadLidar(sweeps=1, ref_frame="ego"))
    transforms += [
        data.dataset.nuscenes.LoadAnnotations(
            labels=box_labels,
            ignore_empty_annots="none",
            ref_frame="ego",
            forecasting_steps=12,
        ),
        data.dataset.nuscenes.LoadSemanticOccupancy(
            root=paths.root / "data" / "nuscenes-occ3d",
            labels=occupancy_labels,
        ),
        data.dataset.transform.occupancy.BuildInstanceLabels(
            source=data.dataset.wrapper.DatasetWrapper(dataset, instance_transforms),
            voxel_size=voxel_size.tolist(),
            voxel_range=voxel_range.tolist(),
            voxel_labels={
                i: n
                for i, n in enumerate(occupancy_labels.all)
                if n in occupancy_labels.instance
            },
            box_labels=dict(enumerate(box_labels.order)),
        ),
        data.dataset.transform.occupancy.DropLabelsWithoutOccupancy(),
    ]

    dataset = data.dataset.wrapper.DatasetWrapper(dataset, transforms)

    return dataset, occupancy_labels, voxel_size, voxel_range


def build_dataset_waymo(split: str | None = None, load_lidar: bool = False):
    split = split or DEFAULT_SPLITS["waymo"]

    box_labels = "config/data/labels/detection/waymo.yaml"
    box_labels = OmegaConf.load(paths.root / box_labels)

    occupancy_labels = "config/data/labels/occupancy/waymo.yaml"
    occupancy_labels = OmegaConf.load(paths.root / occupancy_labels)

    voxel_size = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float32)
    voxel_range = torch.tensor(
        [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], dtype=torch.float32
    )

    dataset = data.dataset.WaymoTO(
        root=paths.root / "data" / "TrackOcc-waymo/kitti_format",
        split=split,
    )

    transforms = [
        data.dataset.waymo_to.LoadImages(),
    ]
    if load_lidar:
        transforms.append(data.dataset.waymo_to.LoadLidar())
    transforms += [
        data.dataset.waymo_to.LoadAnnotations(
            labels=box_labels,
            ignore_empty_annots="none",
            forecasting_steps=12,
        ),
        data.dataset.waymo_to.LoadSemanticOccupancy(
            root=paths.root / "data" / "TrackOcc-waymo/pano_voxel04",
            labels=occupancy_labels,
        ),
    ]

    dataset = data.dataset.wrapper.DatasetWrapper(dataset, transforms)

    return dataset, occupancy_labels, voxel_size, voxel_range


BUILDERS = {
    "nuscenes": build_dataset_nuscenes,
    "waymo": build_dataset_waymo,
}


def build_dataset(
    dataset_type: str, split: str | None = None, load_lidar: bool = False
):
    """Dispatch to the per-dataset builder."""
    try:
        builder = BUILDERS[dataset_type]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset type: {dataset_type}") from exc
    return builder(split=split, load_lidar=load_lidar)


def _nusc_get_samples_for_scene(nusc, scene):
    token = scene["first_sample_token"]

    samples = []
    while token:
        sample = nusc.get("sample", token)
        token = sample["next"]

        samples.append(sample)

    assert samples[-1]["token"] == scene["last_sample_token"]
    assert len(samples) == scene["nbr_samples"]

    return samples


def _pick_scene(scenes: list, scene_index: int, scene_name: str | None):
    """Select a scene by its ``name`` field, or fall back to the ordinal index."""
    if scene_name is None:
        return scenes[scene_index]
    for scene in scenes:
        if scene["name"] == scene_name:
            return scene
    available = ", ".join(sorted(s["name"] for s in scenes))
    raise ValueError(
        f"scene '{scene_name}' not in the selected split. Available: {available}"
    )


def resolve_start_sample(
    dataset_type: str,
    dataset,
    scene_index: int,
    sample_index: int,
    sample_token: str | None,
    scene_name: str | None = None,
) -> int:
    """Resolve the flat dataset index of the starting frame.

    A ``--token`` pins an exact frame; a ``--scene-name`` selects a scene by its
    human-readable name (e.g. ``scene-0013``); otherwise ``--scene``/``--sample``
    select the Nth frame of the Nth scene (scenes ordered deterministically per
    dataset).
    """
    # pylint: disable=too-many-locals
    if dataset_type == "nuscenes":
        source = dataset.source
        if sample_token is None:
            # pylint: disable-next=protected-access
            scenes = data.dataset.nuscenes._get_scenes_for_split(
                source.data, source.split
            )
            scenes = sorted(scenes, key=lambda s: s["token"])
            scene = _pick_scene(scenes, scene_index, scene_name)
            samples = _nusc_get_samples_for_scene(source.data, scene)
            sample_token = samples[sample_index]["token"]

        indices = {s["token"]: i for i, s in enumerate(source.samples)}
        return indices[sample_token]

    if dataset_type == "waymo":
        if sample_token is None:
            # pylint: disable-next=protected-access
            scenes = data.dataset.waymo_to._get_scenes_for_split(
                dataset.source.data.data_list
            )
            scenes = sorted(scenes, key=lambda s: s["token"])
            scene = _pick_scene(scenes, scene_index, scene_name)
            sample_token = int(scene["first_sample_idx"]) + sample_index

        return int(sample_token)

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def collect_scene_samples(dataset, start_index: int) -> tuple[str, list]:
    """Collect all consecutive samples of the scene that ``start_index`` begins.

    ``start_index`` should be the first frame of a scene (as returned by
    :func:`resolve_start_sample` with ``sample_index=0``). Walks forward while the
    sequence id stays constant and returns ``(sequence_id, [samples])``.
    """
    first = dataset[start_index]
    sequence_id = first.meta.sequence_id
    samples = [first]
    total = len(dataset.source)
    index = start_index + 1
    while index < total:
        sample = dataset[index]
        if sample.meta.sequence_id != sequence_id:
            break
        samples.append(sample)
        index += 1
    return sequence_id, samples
