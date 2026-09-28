// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <torch/python.h>

#include "math/vec3.hpp"
#include "math/vec4.hpp"
#include "utils/checks.hpp"
#include "utils/dispatch.hpp"

namespace vxops::cpu {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
voxelize_assign(at::Tensor const points,                // point cloud points
                at::Tensor const batches,               // batch splits for points
                at::Tensor const range,                 // grid extents
                at::Tensor const voxel_size,            // size of the voxels
                at::ScalarType dtype,                   // output index type
                std::optional<at::ScalarType> ktype)    // internal voxel key type
{
    using math::vec3;
    using math::vec4;

    CHECK_SHAPE(range, 3, 2);
    CHECK_SHAPE(voxel_size, 3);

    CHECK_CPU(points);
    CHECK_CPU(batches);
    CHECK_CPU(range);
    CHECK_CPU(voxel_size);

    auto const num_batches = batches.size(0) - 1;

    return VXOPS_DISPATCH_POINT_TYPES(points.scalar_type(), "voxelize_assign", ([&] {
        using points_t = scalar_t;

        // accessor
        auto const points_ = points.accessor<points_t, 2>();
        auto const batches_ = batches.accessor<int64_t, 1>();
        auto const range_ = range.accessor<points_t, 2>();
        auto const voxel_size_ = voxel_size.accessor<points_t, 1>();

        // voxel size
        auto const vsize = vec3<points_t>(voxel_size_[0], voxel_size_[1], voxel_size_[2]);

        // point cloud range
        auto const range_min = vec3<points_t>(range_[0][0], range_[1][0], range_[2][0]);
        auto const range_max = vec3<points_t>(range_[0][1], range_[1][1], range_[2][1]);

        // voxel grid size
        auto const grid_size_ = vec3<int64_t>((range_max - range_min) / vsize);

        // determine key type
        auto const ktype_ = [&] {
            if (ktype) {
                return *ktype;
            }

            if (num_batches * prod(grid_size_) > std::numeric_limits<int32_t>::max()) {
                return c10::ScalarType::Long;
            } else {
                return c10::ScalarType::Int;
            }
        }();

        // dispatch across index (output) types
        return VXOPS_DISPATCH_INDEX_TYPES(ktype_, "voxelize_assign_internal", ([&] {
            using key_t = scalar_t;

            // linear index strides
            auto const grid_size = vec3<key_t>(grid_size_);
            auto const strides = vec4<key_t>(
                grid_size.y() * grid_size.z(),
                grid_size.z(),
                1,
                grid_size.x() * grid_size.y() * grid_size.z()
            );

            return VXOPS_DISPATCH_INDEX_TYPES(dtype, "voxelize_assign_internal2", ([&] {
                using index_t = scalar_t;

                // compute and associate voxel ID with original point index
                auto ids = std::vector<std::pair<key_t, index_t>>(points_.size(0));
                for (int64_t b = 0; b < num_batches; ++b) {
                    for (int64_t i = batches_[b]; i < batches_[b+1]; ++i) {
                        // transform to voxel grid coordinates
                        auto const point_c = vec3<points_t>(points_[i][0], points_[i][1], points_[i][2]);
                        auto const point = (point_c - range_min) / vsize;
                        auto const coord = vec3<key_t>(point);

                        // check if point is inside grid
                        if (all((coord >= 0) && (coord < grid_size))) {
                            // compute linear voxel grid index
                            auto const linear_index = strides.dot(vec4<key_t>(coord, b));

                            // store linear index together with original point array index
                            ids[i] = std::make_pair(linear_index, i);
                        } else {
                            // store invalid index
                            ids[i] = std::make_pair(static_cast<key_t>(-1), i);
                        }
                    }
                }

                // sort by voxel ID
                std::sort(ids.begin(), ids.end(), [](auto const &a, auto const &b) {
                    return a.first < b.first;
                });

                // trim the points that are out of bounds
                auto const ids_begin = std::find_if(ids.begin(), ids.end(), [](auto const &x) {
                    return x.first != static_cast<key_t>(-1);
                });

                // count the number of voxels
                index_t num_voxels = 0;
                {
                    auto id = ids_begin;

                    for (; id < ids.end();) {
                        num_voxels += 1;

                        // find the start of the next voxel
                        id = std::find_if(id, ids.end(), [=](auto const &x) {
                            return x.first != id->first;
                        });
                    }
                }

                // copy point map, compute point offsets for each voxel (plus
                // num points in last element), compute voxel offsets for each
                // batch, and compute coordinates
                auto const tensor_opts = at::dtype(dtype);

                auto indices = at::empty({ids.end() - ids_begin}, tensor_opts);
                auto offsets = at::empty({num_voxels + 1}, tensor_opts);
                auto coords = at::empty({num_voxels, 4}, tensor_opts);

                auto batch_offsets = at::zeros({num_batches + 1}, tensor_opts);

                {
                    auto indices_ = indices.template accessor<index_t, 1>();
                    auto offsets_ = offsets.template accessor<index_t, 1>();
                    auto batch_offsets_ = batch_offsets.template accessor<index_t, 1>();
                    auto coords_ = coords.template accessor<index_t, 2>();

                    index_t voxel = 0;
                    index_t batch = -1;
                    auto id = ids_begin;

                    for (; id < ids.end();) {
                        auto const current_id = id->first;

                        // set the voxel offset
                        offsets_[voxel] = id - ids_begin;

                        // compute the batch index and set the batch offset if we changed the batch
                        auto const current_batch = static_cast<index_t>(current_id / strides.w());
                        if (batch != current_batch) {
                            batch = current_batch;
                            batch_offsets_[batch] = voxel;
                        }

                        // compute the coordinate
                        auto const i = id->second;
                        auto const point_c = vec3<points_t>(points_[i][0], points_[i][1], points_[i][2]);
                        auto const point = (point_c - range_min) / vsize;
                        auto const coord = vec3<index_t>(point);

                        coords_[voxel][0] = current_batch;
                        coords_[voxel][1] = coord.x();
                        coords_[voxel][2] = coord.y();
                        coords_[voxel][3] = coord.z();

                        // find the next start of a voxel
                        for (; id < ids.end() && id->first == current_id; ++id) {
                            indices_[id - ids_begin] = id->second;
                        }

                        voxel += 1;
                    }
                    offsets[num_voxels] = id - ids_begin;
                    batch_offsets[num_batches] = voxel;
                }

                return std::make_tuple(
                    std::move(indices),
                    std::move(offsets),
                    std::move(coords),
                    std::move(batch_offsets));
            }));
        }));
    }));
}

} /* namespace vxops::cpu */
