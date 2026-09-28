// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <torch/python.h>

#include "math/vec3.hpp"
#include "utils/checks.hpp"
#include "utils/dispatch.hpp"

namespace vxops::cpu {

namespace detail {

/**
 * Trace a ray through a voxel grid.
 * @extents: The voxel grid size.
 * @start: Starting point of the ray.
 * @stop: Final point of the ray.
 * @visit: Visitor function with signature `(vec3<T> const& voxel, bool terminal) -> bool`.
 *   Called for each voxel the ray traverses; `terminal` is true at the final
 *   (stop) voxel. Returning true stops the ray (no further voxels are visited).
 *
 * Notes:
 * - This function assumes a voxel size of (1, 1, 1). Scale start and stop
 *   points by the inverse voxel size beforehand size if necessary.
 * - This function assumes that the voxel grid is of range `(0, 0, 0) -- extents`.
 * - This function assumes that start lies inside of the grid.
 */
template <typename T, typename F>
inline void trace(math::vec3<T> const &extents, math::vec3<T> const &start, math::vec3<T> const &stop, F visit)
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
        if (t_max.x() <= t_max.y() && t_max.x() <= t_max.z()) {
            t = t_max.x();
            if (visit(voxel, t > T(1))) return;
            t_max.x() = t_max.x() + t_delta.x();
            voxel.x() = voxel.x() + step.x();
        } else if (t_max.x() > t_max.y() && t_max.y() <= t_max.z()) {
            t = t_max.y();
            if (visit(voxel, t > T(1))) return;
            t_max.y() = t_max.y() + t_delta.y();
            voxel.y() = voxel.y() + step.y();
        } else {
            t = t_max.z();
            if (visit(voxel, t > T(1))) return;
            t_max.z() = t_max.z() + t_delta.z();
            voxel.z() = voxel.z() + step.z();
        }
    }
}

} /* namespace detail */

at::Tensor voxelize_trace(at::Tensor const points,          // point cloud points
                          at::Tensor const batches,         // batch splits for points
                          at::Tensor const sensor_origin,   // sensor/ray origin
                          at::Tensor const range,           // grid extents (in point cloud coordinates)
                          at::IntArrayRef const grid_size,  // grid size (in cells)
                          at::Tensor const voxel_size,      // size of the voxels
                          at::Tensor const values,          // values for free, occupied, and unknown voxels (in this order)
                          at::ScalarType dtype,             // output scalar type
                          bool const stop_at_occupied)      // stop rays at the first occupied voxel
{
    using math::vec3;

    CHECK_ARRAY_SIZE(grid_size, 3);

    CHECK_SHAPE(sensor_origin, 3);
    CHECK_SHAPE(range, 3, 2);
    CHECK_SHAPE(voxel_size, 3);
    CHECK_SHAPE(values, 3);

    CHECK_CPU(points);
    CHECK_CPU(batches);
    CHECK_CPU(sensor_origin);
    CHECK_CPU(range);
    CHECK_CPU(voxel_size);
    CHECK_CPU(values);

    // dispatch across point cloud (input) types
    return VXOPS_DISPATCH_POINT_TYPES(points.scalar_type(), "voxelize_trace", ([&] {
        using points_t = scalar_t;

        // accessors
        auto const points_ = points.accessor<points_t, 2>();
        auto const batches_ = batches.accessor<int64_t, 1>();
        auto const sensor_origin_ = sensor_origin.accessor<points_t, 1>();
        auto const range_ = range.accessor<points_t, 2>();
        auto const voxel_size_ = voxel_size.accessor<points_t, 1>();

        // voxel size
        auto const vsize = vec3<points_t>(voxel_size_[0], voxel_size_[1], voxel_size_[2]);

        // point cloud range
        auto const range_min = vec3<points_t>(range_[0][0], range_[1][0], range_[2][0]);

        // voxel grid size
        auto const extents = vec3<points_t>(grid_size[0], grid_size[1], grid_size[2]);

        // sensor origin
        auto const origin_point = vec3<points_t>(sensor_origin_[0], sensor_origin_[1], sensor_origin_[2]);
        auto const origin = (origin_point - range_min) / vsize;

        // dispatch across grid (output) types
        return VXOPS_DISPATCH_GRID_TYPES(dtype, "voxelize_trace_internal", ([&] {
            using grid_t = scalar_t;

            auto const num_batches = batches.size(0) - 1;

            auto const values_ = values.accessor<grid_t, 1>();
            auto const v_free = values_[0];
            auto const v_occ = values_[1];
            auto const v_unk = values_[2];

            auto const grid_opts = at::dtype(dtype);
            auto grid = at::full({num_batches, grid_size[0], grid_size[1], grid_size[2]}, v_unk, grid_opts);
            auto grid_ = grid.template accessor<grid_t, 4>();

            // perform voxelization
            //
            // Two-pass (mirrors the CUDA path): first mark all occupied voxels,
            // then trace rays and mark free voxels. Marking occupied up front
            // makes the result order-independent and lets free-carving see the
            // full occupied set. When `stop_at_occupied` is set, a ray terminates
            // at the first occupied voxel it hits, so voxels occluded behind an
            // occupied surface stay unknown instead of being carved free.
            for (int b = 0; b < num_batches; ++b) {
                auto const batch_start = batches_[b];
                auto const batch_end = batches_[b+1];

                // pass 1: mark all occupied voxels (the points' end voxels)
                for (int64_t i = batch_start; i < batch_end; ++i)
                {
                    auto const point_c = vec3<points_t>(points_[i][0], points_[i][1], points_[i][2]);
                    auto const idx = math::floor((point_c - range_min) / vsize);
                    const int x = idx.x();
                    const int y = idx.y();
                    const int z = idx.z();

                    if (x >= 0 && x < extents.x() && y >= 0 && y < extents.y()
                        && z >= 0 && z < extents.z())
                        grid_[b][x][y][z] = v_occ;
                }

                // the visitor: what we do for each traversed voxel. Returns true
                // to stop the ray.
                auto visitor = [&](vec3<points_t> const &voxel, bool terminal) -> bool
                {
                    const int x = voxel.x();
                    const int y = voxel.y();
                    const int z = voxel.z();

                    // occupied voxel: stop the ray (occlusion) or just skip it
                    if (grid_[b][x][y][z] > 0)
                        return stop_at_occupied;

                    // the end voxel is handled by the occupied pass above
                    if (terminal)
                        return false;

                    // free voxel
                    grid_[b][x][y][z] = v_free;
                    return false;
                };

                // pass 2: for all points trace from origin to point
                for (int64_t i = batch_start; i < batch_end; ++i)
                {
                    auto const point_c = vec3<points_t>(points_[i][0], points_[i][1], points_[i][2]);
                    auto const point = (point_c - range_min) / vsize;

                    detail::trace(extents, origin, point, visitor);
                }
            }

            return grid;
        }));
    }));
}

} /* namespace vxops::cpu */
