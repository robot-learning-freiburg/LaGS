# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Functions for creating voxels from point clouds.
"""

import unittest
from typing import List, Tuple

import torch

from . import voxelize_ext as _ext

_voxelize_trace = torch.ops.voxelize.voxelize_trace
_voxelize_assign = torch.ops.voxelize.voxelize_assign
_aggregate_mean = torch.ops.voxelize.aggregate_mean


def voxelize_trace(
    points,
    batches,
    origin,
    points_range,
    voxel_size,
    value_free=-1.0,
    value_occupied=1.0,
    value_unknown=0.0,
    dtype: torch.dtype | None = None,
    stop_at_occupied: bool = False,
) -> torch.Tensor:
    """
    Voxelize the given point-cloud yielding occupied, known-free, and unknown
    voxels.

    Returns a (full) 3D voxel grid catgorizing each voxel into possible
    classes: free, occupied, and unknown. The returned tensor will be of shape
    [B, X, Y, Z], where B is the batch size, and [X, Y, Z] the dimension of the
    voxel grid as determined by `points_range` (for the extents in units of the
    point cloud data) and `voxel_size` (for the size of the individual voxels).

    Voxel values/classes are determined as follows:

    - Occupied voxels are determined as voxels containing at least one point of
      the given point cloud.

    - Free voxels are determined by tracing a ray from the sensor origin to
      each point. Each voxel hit by the ray is marked as free, unless the voxel
      contains a point of the pointcloud, in which case it is marked as
      occupied.

    - Remaining voxels that have neither been marked as free nor occupied are
      designated as "unknown".

    By default a ray passes through occupied voxels (only skipping them), so
    free space can be carved behind an occupied surface. With
    ``stop_at_occupied=True`` a ray instead stops at the first occupied voxel it
    hits, so voxels occluded behind an occupied surface remain "unknown" rather
    than being marked free.

    Points are assigned to a batch element based on the `batches` array. Given
    a batch size of `B`, this array of size `B+1` indicates the start and end
    of each element of the batch in the `points` array. Specifically, element
    `i` of the batch contains `points[batches[i] : batches[i+1]]`. If `batches`
    is set to `None`, it will be treated as a batch size of 1, i.e., as if
    `batches = [0, N]` had been specified, where `N` is the number of points.

    The actual implementation (CPU or CUDA) will be determined depending on the
    device on which `points` resides. `points` may either reside in CPU or CUDA
    memory, all other data needs to reside in CPU memory.

    Args:
        points (array_like [N, 3]):
            point cloud as array of points
        batches (array_like [B+1], optional):
            array indicating how to split points into batches or None, which
            will then be treated as having a batch size of 1 and a `batches`
            array of [0, N]
        origin (array_like [3]):
            sensor origin used for tracing and determining free voxels
        points_range (array_like [3, 2]):
            range of the resulting grid, in units of the point cloud data
        voxel_size (array_like [3]):
            size of the resulting voxels, in units of the point cloud data
        value_free (scalar):
            value to use for free voxels
        value_occupied (scalar):
            value to use for occupied voxels
        value_unknown (scalar):
            value to use for unknown voxels
        dtype (torch.dtype, optional):
            scalar type of the returned voxel grid
        stop_at_occupied (bool):
            if True, rays stop at the first occupied voxel (occlusion-aware), so
            voxels behind an occupied surface stay "unknown" instead of free

    Returns:
        a 4D tensor [B, X, Y, Z] representing the voxel grid
    """

    if batches is None:
        batches = [0, points.shape[0]]

    points = torch.as_tensor(points)
    batches = torch.as_tensor(batches, dtype=torch.int64)
    origin = torch.as_tensor(origin, dtype=points.dtype)
    points_range = torch.as_tensor(points_range, dtype=points.dtype)
    voxel_size = torch.as_tensor(voxel_size, dtype=points.dtype)

    if dtype is None:
        dtype = points.dtype

    values = torch.tensor((value_free, value_occupied, value_unknown), dtype=dtype)

    range_min = points_range[:, 0].to(dtype=points.dtype)
    range_max = points_range[:, 1].to(dtype=points.dtype)
    grid_size = (range_max - range_min) / voxel_size.to(dtype=points.dtype)
    grid_size = grid_size.to(dtype=torch.int64).tolist()

    return _voxelize_trace(
        points,
        batches,
        origin,
        points_range,
        grid_size,
        voxel_size,
        values,
        dtype,
        stop_at_occupied,
    )


def voxelize_assign(
    points,
    batches,
    points_range,
    voxel_size,
    dtype: torch.dtype = torch.int32,
    ktype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Voxelize the given point-cloud, assigning each point in the input to its
    respective voxel.

    Performs dynamic voxelization, i.e., assigns all points in the given tensor
    to their respecitve voxels as long as they lay inside the defined voxel
    grid extends given by `points_range`.

    Points are assigned to a batch element based on the `batches` array. Given
    a batch size of `B`, this array of size `B+1` indicates the start and end
    of each element of the batch in the `points` array. Specifically, element
    `i` of the batch contains `points[batches[i] : batches[i+1]]`. If `batches`
    is set to `None`, it will be treated as a batch size of 1, i.e., as if
    `batches = [0, N]` had been specified, where `N` is the number of points.

    Args:
        points (array_like [N, 3]):
            point cloud as array of points
        batches (array_like [B+1], optional):
            array indicating how to split points into batches or None, which
            will then be treated as having a batch size of 1 and a `batches`
            array of [0, N]
        points_range (array_like [3, 2]):
            range of the resulting grid, in units of the point cloud data
        voxel_size (array_like [3]):
            size of the resulting voxels, in units of the point cloud data
        dtype (torch.dtype):
            scalar type of the returned index tensors
        ktype (torch.dtype, optional):
            key type for internal processing; smaller types may yield better
            performance, but must be large enough to hold a linearized
            representation of voxel indices, i.e., B*X*Y*Z for a batch size of
            B and voxel grid of dimensions [X, Y, Z]; allowed values are
            torch.int32 or torch.int64.

    Returns:
        A tuple `(indices, offsets, coords, batches)` representing the result of
        the point-voxel assignment.

        indices (torch.Tensor [N]):
            index tensor containing indices into the original `points` input,
            grouped by voxels
        offsets (torch.Tensor [V+1]):
            offsets into the `indices` tensor for each voxel
        coords (torch.Tensor [V, 4]):
            coordinates of the voxels as (batch_id, x, y, z)
        batches (torch.Tensor [B+1]):
            array indicating how to split voxels into batches

        In the above, N is the number of points, V the number of voxels, and B
        the number of batches.

        Given a voxel with index `v`, points falling into it can be obtained by
        ```
        points[indices[offsets[v] : offsets[v+1]], :]
        ```
        It may be useful to create a list of points grouped by voxels before
        accessing individual voxels, which can be done by
        ```
        points_grouped = points[indices, :]
        points_in_voxel_v = points_grouped[offsets[v] : offsets[v+1], :]
        ```
        The respective voxel coordinate is given by `coords[v]`.

        The `batches` tensor behave similar to its input namesake, however,
        indexes the `offsets` and `coords` tensor instead of `points`. I.e.,
        all voxel indices belonging to batch `i` are given by the range
        `[batches[i] : batches[i+1]]`.
    """

    if batches is None:
        batches = [0, points.shape[0]]

    points = torch.as_tensor(points)
    batches = torch.as_tensor(batches, dtype=torch.int64)
    points_range = torch.as_tensor(points_range, dtype=points.dtype)
    voxel_size = torch.as_tensor(voxel_size, dtype=points.dtype)

    return _voxelize_assign(points, batches, points_range, voxel_size, dtype, ktype)


def aggregate_mean(features, offsets) -> torch.Tensor:
    """
    Compute the mean of the given point-wise features inside each voxel.

    Takes features associated with points from a point cloud and for each
    occupied voxel computes the mean of features associated with the points
    falling into that voxel.

    In case `features` is not grouped by voxels, the `indices` array returned
    by :func:`voxelize_assign` may be used to group them before calling
    `aggregate_mean()`, i.e.
    ```
    aggregate_mean(features[indices], offsets)

    Args:
        features (array_like [N, D]):
            input features grouped by voxels
        offsets (array_like [V+1]):
            voxel offsets (e.g., as returned by :func:`voxelize_assign`)

    Returns:
        the mean of the given features inside each voxel as torch.Tensor of
        shape [V, D], where V is the number of voxels and D the number of
        feature dimensions
    """

    features = torch.as_tensor(features)
    offsets = torch.as_tensor(offsets)

    return _aggregate_mean(features, offsets)


@torch.library.register_fake("voxelize::voxelize_trace")
def _(
    points: torch.Tensor,
    batches: torch.Tensor,
    sensor_origin: torch.Tensor,
    range: torch.Tensor,
    grid_size: List[int],
    voxel_size: torch.Tensor,
    values: torch.Tensor,
    dtype: torch.dtype,
    stop_at_occupied: bool = False,
) -> torch.Tensor:
    # pylint: disable=unused-argument,redefined-builtin

    num_batches = batches.shape[0] - 1
    sx, sy, sz = grid_size

    return torch.empty((num_batches, sx, sy, sz), dtype=dtype, device=points.device)


@torch.library.register_fake("voxelize::voxelize_assign")
def _(
    points: torch.Tensor,
    batches: torch.Tensor,
    range: torch.Tensor,
    voxel_size: torch.Tensor,
    dtype: torch.dtype,
    ktype: torch.dtype | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # pylint: disable=unused-argument,redefined-builtin

    ctx = torch.library.get_ctx()
    num_points = ctx.new_dynamic_size()
    num_voxels = ctx.new_dynamic_size()

    indices = torch.empty((num_points,), dtype=dtype, device=points.device)
    offsets = torch.empty((num_voxels + 1,), dtype=dtype, device=points.device)
    coords = torch.empty((num_voxels, 4), dtype=dtype, device=points.device)
    batch_offsets = torch.empty((batches.shape[0],), dtype=dtype, device=points.device)

    return indices, offsets, coords, batch_offsets


@torch.library.register_fake("voxelize::aggregate_mean")
def _(features: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    num_voxels = offsets.shape[0] - 1
    num_channels = features.shape[1]

    return torch.empty(
        (num_voxels, num_channels), dtype=features.dtype, device=features.device
    )


class TestOps(unittest.TestCase):
    @staticmethod
    def _create_random_offsets(n_max: int, max_step: int, device: torch.device | str):
        offsets = torch.cumsum(torch.randint(1, max_step, (n_max,)), dim=0)
        offsets = offsets[offsets < n_max]
        offsets = torch.cat((torch.tensor([0]), offsets, torch.tensor([n_max])))
        return offsets.to(device=device)

    def test_voxelize_trace(self):
        devices = ("cpu", "cuda")

        # Note: Floating point arithmetic can cause the outputs of CPU and CUDa
        # to be slightly different. The problem is that the outputs are
        # discrete values (free, occupied, unknown), so the error magnitude can
        # be quite large. Due to that, the opcheck can fail. To prevent it from
        # failing we just do very basic and deterministic inputs here that we
        # expect to lead to the exact same result, regardless of device.

        # base arguments
        points_dtype = torch.float32
        dtype = torch.float32

        n_points = 1024

        sensor_origin = torch.tensor([0.0, 0.0, 0.0])
        voxel_size = torch.tensor([0.075, 0.075, 0.2])
        points_range = torch.tensor([(-54.0, 54.0), (-54.0, 54.0), (-5.0, 3.0)])
        values = torch.tensor([-1.0, 1.0, 0.0])

        # compute grid size
        range_min = points_range[:, 0].to(dtype=points_dtype)
        range_max = points_range[:, 1].to(dtype=points_dtype)
        grid_size = (range_max - range_min) / voxel_size.to(dtype=points_dtype)
        grid_size = grid_size.to(dtype=torch.int64).tolist()

        batches = torch.tensor([0, n_points // 3, n_points])

        def args(device):
            points = torch.zeros((n_points, 3), dtype=points_dtype, device=device)
            points = points + 1e-4

            return (
                points,
                batches,
                sensor_origin,
                points_range,
                grid_size,
                voxel_size,
                values,
                dtype,
            )

        for device in devices:
            torch.library.opcheck(_voxelize_trace, args(device))

    def test_voxelize_trace_stop_at_occupied(self):
        # 1-D scene along x: origin near x=0, occupied voxels at x=10 and x=20.
        # Without stop_at_occupied the ray to x=20 carves the voxels behind x=10
        # free; with it, those voxels (11..19) stay unknown.
        voxel_size = torch.tensor([1.0, 1.0, 1.0])
        points_range = torch.tensor([(0.0, 30.0), (0.0, 1.0), (0.0, 1.0)])
        free, occ, unk = -1.0, 1.0, 0.0
        # slight y/z slope (kept within the single y/z voxel) to avoid a
        # degenerate axis-aligned ray
        origin = torch.tensor([0.5, 0.2, 0.2])
        points = torch.tensor([[10.5, 0.5, 0.5], [20.5, 0.5, 0.5]])

        for device in ("cpu", "cuda"):

            def trace(stop, dev):
                grid = voxelize_trace(
                    points.to(device=dev),
                    None,
                    origin,
                    points_range,
                    voxel_size,
                    value_free=free,
                    value_occupied=occ,
                    value_unknown=unk,
                    stop_at_occupied=stop,
                )
                return grid[0, :, 0, 0].cpu()  # [B, X, Y, Z] -> x profile

            # occupied voxels present in both modes; voxels in front are free
            for g in (trace(False, device), trace(True, device)):
                self.assertEqual(g[10].item(), occ)
                self.assertEqual(g[20].item(), occ)
                self.assertTrue((g[1:10] == free).all())

            # the difference is behind the first occupied voxel (11..19)
            self.assertTrue((trace(False, device)[11:20] == free).all())
            self.assertTrue((trace(True, device)[11:20] == unk).all())

    def test_voxelize_assign(self):
        devices = ("cpu", "cuda")

        dtype = torch.int32
        ktype = None
        voxel_size = torch.tensor([0.075, 0.075, 0.2])
        points_range = torch.tensor([(-54.0, 54.0), (-54.0, 54.0), (-5.0, 3.0)])

        def args(device, n_points=1024, c=4):
            points = torch.randn((n_points, c), dtype=torch.float32, device=device)
            batches = TestOps._create_random_offsets(
                n_points, n_points // 3, device="cpu"
            )

            return points, batches, points_range, voxel_size, dtype, ktype

        for device in devices:
            torch.library.opcheck(_voxelize_assign, args(device))

    def test_aggregate_mean(self):
        devices = ("cpu", "cuda")

        def args(device, n=64, c=4):
            offsets = TestOps._create_random_offsets(n, 10, device=device)
            features = torch.rand((n, c), device=device)

            return features, offsets

        for device in devices:
            torch.library.opcheck(_aggregate_mean, args(device))
