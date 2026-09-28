# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import itertools
from dataclasses import dataclass
from typing import List, Literal, TypeVar

import numpy as np
import torch
import torch.distributed as dist

T = TypeVar("T")


@dataclass(frozen=True)
class ElementId:
    index: int
    is_padding: bool


@dataclass(frozen=True)
class SampleId:
    scene_index: int
    sample_index: int
    is_padding: bool


def pad_elements(indices: List[int], n: int) -> List[ElementId]:
    # if the list of sequences is completely empty, just fill it with index 0
    if not indices:
        return [ElementId(index=0, is_padding=True) for _ in range(n)]

    # otherwise: try to get the last index and repeat it
    pad_index = indices[-1]
    n_pad = n - len(indices)

    indices = [ElementId(index=i, is_padding=False) for i in indices]
    padding = [ElementId(index=pad_index, is_padding=True) for _ in range(n_pad)]

    return indices + padding


def distribute_indices(num_elements, num_devices) -> List[List[ElementId]]:
    """
    Distribute element indices across the specified number of devices.

    Padding is applied to ensure that the number of indices per device is equal
    for each device.
    """

    num_per_device = (num_elements - 1) // num_devices + 1

    splits = [list(range(i, num_elements, num_devices)) for i in range(num_devices)]
    splits = [pad_elements(seqs, num_per_device) for seqs in splits]

    return splits


def _collect_samples_for_proc(
    scenes: List[int], scene_len: List[int], splits: List[List[ElementId]], proc: int
) -> List[SampleId]:
    samples = []
    for scene, n_samples in zip(splits[proc], scene_len):
        sample_indices = list(range(scenes[scene.index]))
        sample_indices = pad_elements(sample_indices, n_samples)

        samples += [
            SampleId(
                scene_index=scene.index,
                sample_index=i.index,
                is_padding=scene.is_padding or i.is_padding,
            )
            for i in sample_indices
        ]

    return samples


def distribute_scenes_to_procs(
    scenes: List[int], batch_size: int = 0, num_procs: int = 1
) -> List[List[SampleId]]:
    """
    Distribute scenes across different processes (e.g., in a DDP setting).

    This function will shard scenes across different processes/devices such
    that the scenes are kept intact. Meaning, scenes are not split across
    processes. Furthermore, padding is applied such that
    - Scenes are completed in parallel. E.g., new scenes start at the same time
      across each device. Shorter scenes will be padded to the longest
      concurrent scene.
    - The number of samples per scene is a multiple of the batch size. This
      means, when appropriate batching is performed, each batch only contains
      samples from the same scene (or padding).

    Args:
        scenes: The number of samples for each scene.
        batch_size: The batch size. Used for padding.
        num_procs: The total number of processes.

    Returns:
        A list over processes of lists of (ordered) sample IDs, meaning
        `ret[proc]` contains all samples for process `proc`.
    """

    # distribute scenes across devices
    splits = distribute_indices(len(scenes), num_procs)

    # calcualte the maximum number of samples per parallel scene for padding
    scene_len = [[scenes[s.index] for s in split] for split in splits]
    scene_len = np.max(scene_len, axis=0)

    # pad the samples to multiples of the batch size, so that we never have
    # different scenes in the same batch
    scene_pad = batch_size - (scene_len % batch_size)
    scene_len = scene_len + scene_pad % batch_size

    # collect the samples for each scene on over all procoesses and make sure
    # that all scenes running in parallel have the same number of samples
    return [
        _collect_samples_for_proc(scenes, scene_len, splits, proc)
        for proc in range(num_procs)
    ]


def distribute_samples_to_workers(
    samples: List[T],
    batch_size: int,
) -> List[T]:
    """
    Distribute a sequence of samples to multiple dataloader worker processes.

    This function distributes an ordered list of samples across dataloader
    worker processes such that the order is kept. For example for samples [0,
    1, 2, 3] and batch size 2, the batches will be [0, 1], [2, 3].

    Args:
        samples: The list of samples.
        batch_size: The batch size.
        process_rank: The DDP/distributed process.

    Returns:
        The list of (ordered) samples for the specified worker ID.
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
        return samples

    start = worker_info.id * batch_size
    step = worker_info.num_workers * batch_size

    sample_indices = []
    for batch_start in range(start, len(samples), step):
        batch_stop = min(batch_start + batch_size, len(samples))
        sample_indices += list(range(batch_start, batch_stop))

    return [samples[i] for i in sample_indices]


def _roundrobin(iterables):
    yield from map(next, itertools.cycle(map(iter, iterables)))


def distribute_scenes(
    scenes: List[int],
    batch_size: int = 1,
    device_mode: Literal["duplicate", "parallel"] = "parallel",
    batch_mode: Literal["grouped", "parallel"] = "parallel",
) -> List[SampleId]:
    """
    Distribute scenes across different DDP/distributed and dataloader processes.

    Args:
        scenes: The number of samples for each scene.
        batch_size: The batch size. Used for padding.
        device_mode: The way in which scenes are distributed across devices.
          - `duplicate` means all scenes are duplicated to all devices.
            Duplicated scenes are marked as invalid samples/pading.
          - `parallel` means scenes are distributed across devices. A single
             scene is bound to a specific device and will only be processed by
             that device.
        batch_mode: The way in which scenes are distributed across batches.
          - `grouped` means a batch contains only a single scene.
          - `parallel` means a scene is bound to a specific batch index and a
            batch contains exactly `batch_size` scenes processed in parallel.

    Returns:
        The list of (ordered) sample IDs for the current dataloader process.
    """
    if device_mode not in ["duplicate", "parallel"]:
        raise ValueError(f"invalid device_mode '{device_mode}'")

    if batch_mode not in ["grouped", "parallel"]:
        raise ValueError(f"invalid batch_mode '{batch_mode}'")

    # get rank and world size to handle sample distribution
    rank, world_size = 0, 1
    if dist.is_available() and dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()

    # if we duplicate across all devices, pretend we just have one device... we
    # then mark the other devices as paddiong later
    proc, num_procs = 0, 1
    if device_mode != "duplicate":
        proc, num_procs = rank, world_size

    # shard scenes to DDP/distributed processes and potentially batch-indices
    if batch_mode == "grouped":
        samples = distribute_scenes_to_procs(scenes, batch_size, num_procs)
        samples = samples[proc]

    elif batch_mode == "parallel":
        samples = distribute_scenes_to_procs(scenes, 1, num_procs * batch_size)

        # collect all different "lanes" that we process on this device/DDP process
        samples = [samples[proc * batch_size + i] for i in range(batch_size)]

        # join them in round-robin fashion
        samples = list(_roundrobin(samples))

    # if we duplicate across devices, mark all non-zero ranks as padding
    if device_mode == "duplicate" and rank != 0:
        samples = [SampleId(s.scene_index, s.sample_index, True) for s in samples]

    return distribute_samples_to_workers(samples, batch_size)
