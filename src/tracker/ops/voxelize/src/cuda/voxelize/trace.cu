// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>

#include "cuda/utils/batchmap.cuh"
#include "cuda/utils/loop.cuh"

#include "math/vec3.hpp"

#include "utils/checks.hpp"
#include "utils/dispatch.hpp"

namespace vxops::cuda {

namespace detail::kernel {

/**
 * Trace a ray through a voxel grid.
 * @extents: The voxel grid size.
 * @start: Starting point of the ray.
 * @stop: Final point of the ray.
 * @visit: Visitor function with signature `(vec3<T> const& voxel, bool terminal) -> bool`.
 *   Called for each voxel the ray traverses; `terminal` is true at the final
 *   (stop) voxel. Returning true stops the ray (no further voxels are visited).
 *
 * Note:
 * - This function assumes a voxel size of (1, 1, 1). Scale start and stop
 *   points by the inverse voxel size beforehand size if necessary.
 * - This function assumes that the voxel grid is of range `(0, 0, 0) -- extents`.
 * - This function assumes that start lies inside of the grid.
 */
template <typename T, typename F>
__device__ __forceinline__ void trace(math::vec3<T> const &extents, math::vec3<T> const &start,
                                      math::vec3<T> const &stop, F visit)
{
    using math::vec3;

    // ray direction
    auto const ray = stop - start;

    // voxel step direction of ray
    auto const step = math::sign(ray);

    // compute where the ray intersects with the grid bounds. use small epsilon
    // (0.1) instead of actual bounds to ensure we actually stay inside
    auto const t_bounds = (vec3<T>(0.1) + vec3<T>(step > 0) * (extents - T(0.2)) - start) / ray;

    // we stop the ray either at the grid bounds or at the terminating voxel
    // (t=1.0), whatever comes first
    auto const t_stop = fmin(T(1), math::min(t_bounds));

    // delta for incrementing t to move exactly one voxel
    auto const t_delta = step / ray;

    // the current voxel
    auto voxel = math::floor(start);

    // ray distance at which the ray crosses the next x/y/z boundary (resp.)
    auto t_max = (voxel + math::max(step, T(0)) - start) / ray;

    // the current t value along the ray (current_pos = start + t * ray)
    auto t = T(0);

    // trace ray
    while (t <= t_stop)
    {
        // compute the t value along the ray where we enter the next voxel and
        // the axis on which we have to step to get to it
        t = min(t_max);

        // compute mask for step direction
        auto const mask = vec3<T>(math::mask_first(t == t_max));

        // visit the current voxel; stop the ray if the visitor requests it
        if (visit(voxel, t > T(1))) return;

        // step to the next voxel
        t_max = t_max + mask * t_delta;
        voxel = voxel + mask * step;
    }
}

template <typename points_t, typename grid_t, typename batch_t>
__global__ void trace_mark_free(
    at::PackedTensorAccessor64<points_t, 2, at::RestrictPtrTraits> const points,
    at::PackedTensorAccessor64<batch_t, 1, at::RestrictPtrTraits> const batch_index,
    at::PackedTensorAccessor64<grid_t, 4, at::RestrictPtrTraits> grid,
    math::vec3<points_t> const origin,
    math::vec3<points_t> const offset,
    math::vec3<points_t> const voxel_size,
    grid_t const value_free,
    bool const stop_at_occupied)
{
    using math::vec3;

    auto const extents = vec3<points_t>(grid.size(1), grid.size(2), grid.size(3));

    // trace rays to find free voxels and mark them as free
    CUDA_1D_KERNEL_LOOP(i, points.size(0))
    {
        auto const point_c = vec3<points_t>(points[i][0], points[i][1], points[i][2]);
        auto const point = (point_c - offset) / voxel_size;

        auto const batch = batch_index[i];

        // the visitor: what we do for each traversed voxel. Returns true to stop
        // the ray.
        auto visitor = [&](vec3<points_t> const &voxel, bool terminal) -> bool
        {
            const int x = voxel.x();
            const int y = voxel.y();
            const int z = voxel.z();

            // occupied voxel (marked separately): stop the ray (occlusion) or
            // just skip it and carry on
            if (grid[batch][x][y][z] > 0)
                return stop_at_occupied;

            // the end voxel is handled by the occupied pass
            if (terminal)
                return false;

            // mark this point as free
            grid[batch][x][y][z] = value_free;
            return false;
        };

        trace(extents, origin, point, visitor);
    }
}

template <typename points_t, typename grid_t, typename batch_t>
__global__ void mark_occupied(
    at::PackedTensorAccessor64<points_t, 2, at::RestrictPtrTraits> const points,
    at::PackedTensorAccessor64<batch_t, 1, at::RestrictPtrTraits> const batch_index,
    at::PackedTensorAccessor64<grid_t, 4, at::RestrictPtrTraits> grid,
    math::vec3<points_t> const origin,
    math::vec3<points_t> const offset,
    math::vec3<points_t> const voxel_size,
    grid_t const value_occupied)
{
    using math::vec3;

    auto const extents = vec3<int64_t>(grid.size(1), grid.size(2), grid.size(3));

    // compute voxel coordinates of points and mark voxels as occupied
    CUDA_1D_KERNEL_LOOP(i, points.size(0))
    {
        auto const point_c = vec3<points_t>(points[i][0], points[i][1], points[i][2]);
        auto const point = (point_c - offset) / voxel_size;

        auto const index = vec3<int64_t>(floor(point));

        auto const batch = batch_index[i];

        if (all(index >= vec3<int64_t>(0) && index < extents))
        {
            grid[batch][index.x()][index.y()][index.z()] = value_occupied;
        }
    }
}

} /* namespace detail::kernel */

at::Tensor voxelize_trace(at::Tensor const points,
                          at::Tensor const batches,
                          at::Tensor const sensor_origin,
                          at::Tensor const range,
                          at::IntArrayRef const grid_size,
                          at::Tensor const voxel_size,
                          at::Tensor const values,
                          at::ScalarType dtype,
                          bool const stop_at_occupied)
{
    using math::vec3;
    using batch_t = int16_t;

    CHECK_ARRAY_SIZE(grid_size, 3);

    CHECK_SHAPE(sensor_origin, 3);
    CHECK_SHAPE(range, 3, 2);
    CHECK_SHAPE(voxel_size, 3);
    CHECK_SHAPE(values, 3);

    CHECK_CUDA(points);
    CHECK_CPU(batches);
    CHECK_CPU(sensor_origin);
    CHECK_CPU(range);
    CHECK_CPU(voxel_size);
    CHECK_CPU(values);

    at::cuda::OptionalCUDAGuard const guard(device_of(points));
    auto const stream = at::cuda::getCurrentCUDAStream();

    // batch index map
    auto const num_batches = batches.size(0) - 1;
    auto const batch_index = detail::compute_batch_index_map<batch_t>(
        batches, points.size(0), points.device(), stream
    );

    // dispatch across point cloud (input) types
    return VXOPS_DISPATCH_POINT_TYPES(points.scalar_type(), "voxelize_trace", ([&] {
        using points_t = scalar_t;

        // accessors
        auto const points_ = points.packed_accessor64<points_t, 2, at::RestrictPtrTraits>();
        auto const batch_index_ = batch_index.packed_accessor64<batch_t, 1, at::RestrictPtrTraits>();
        auto const sensor_origin_ = sensor_origin.accessor<points_t, 1>();
        auto const range_ = range.accessor<points_t, 2>();
        auto const voxel_size_ = voxel_size.accessor<points_t, 1>();

        // voxel size
        auto const vsize = vec3<points_t>(voxel_size_[0], voxel_size_[1], voxel_size_[2]);

        // point cloud range
        auto const range_min = vec3<points_t>(range_[0][0], range_[1][0], range_[2][0]);

        // voxel grid size
        auto const extents = vec3<int64_t>(grid_size[0], grid_size[1], grid_size[2]);

        // sensor origin
        auto const origin_point = vec3<points_t>(sensor_origin_[0], sensor_origin_[1], sensor_origin_[2]);
        auto const origin = (origin_point - range_min) / vsize;

        // dispatch across grid (output) types
        return VXOPS_DISPATCH_GRID_TYPES(dtype, "voxelize_trace_internal", ([&] {
            using grid_t = scalar_t;

            auto const values_ = values.accessor<grid_t, 1>();
            auto const v_free = values_[0];
            auto const v_occ = values_[1];
            auto const v_unk = values_[2];

            // grid
            auto const grid_opts = c10::TensorOptions()
                    .device(points.device())
                    .dtype(dtype);

            auto grid = at::full({num_batches, extents.x(), extents.y(), extents.z()}, v_unk, grid_opts);
            auto grid_ = grid.packed_accessor64<grid_t, 4, at::RestrictPtrTraits>();

            // perform voxelization
            // Note: We do this as a two-step process. First, we mark all occupied
            // voxels, then we trace rays and mark all free voxels. This is split
            // across 2 kernels to ensure that we only ever write the same value to the
            // same voxels. Hence, we don't need to care much about synchronizing
            // access to the same voxel from multiple threads, since it will only ever
            // write the same values. Which (although arguably lazy) is well-defined
            // behavior.

            auto const threads = 1024;
            auto const blocks = (points.size(0) + threads - 1) / threads;

            // step 1: mark all occupied voxels
            detail::kernel::mark_occupied<<<blocks, threads, 0, stream>>>(
                points_, batch_index_, grid_, origin, range_min, vsize, v_occ);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            // step 2: mark all free voxels
            detail::kernel::trace_mark_free<<<blocks, threads, 0, stream>>>(
                points_, batch_index_, grid_, origin, range_min, vsize, v_free, stop_at_occupied);
            C10_CUDA_KERNEL_LAUNCH_CHECK();

            return grid;
        }));
    }));
}

} /* namespace vxops::cuda */
