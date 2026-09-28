// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>
#include <limits>
#include <utility>

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/python.h>

#include "cuda/utils/batchmap.cuh"
#include "cuda/utils/cub.cuh"
#include "cuda/utils/loop.cuh"
#include "cuda/utils/mem.hpp"

#include "math/vec3.hpp"
#include "math/vec4.hpp"

#include "utils/checks.hpp"
#include "utils/dispatch.hpp"

namespace vxops::cuda {

namespace detail {

namespace kernel {

template <typename points_t, typename batch_t, typename key_t, typename index_t>
__global__ void init_idmap(
    at::PackedTensorAccessor64<points_t, 2, at::RestrictPtrTraits> const points,
    at::PackedTensorAccessor64<batch_t, 1, at::RestrictPtrTraits> const batches,
    at::PackedTensorAccessor64<key_t, 1, at::RestrictPtrTraits> keys,
    at::PackedTensorAccessor64<index_t, 1, at::RestrictPtrTraits> indices,
    math::vec3<points_t> const offset,
    math::vec3<points_t> const voxel_size,
    math::vec3<key_t> const extents,
    math::vec4<key_t> const strides)
{
    using math::vec3;
    using math::vec4;

    CUDA_1D_KERNEL_LOOP(i, points.size(0)) {
        auto const point_c = vec3<points_t>(points[i][0], points[i][1], points[i][2]);
        auto const batch = batches[i];

        // transform to voxel grid coordinates
        auto const point = (point_c - offset) / voxel_size;
        auto const coord = vec3<key_t>(point);

        // store point index
        indices[i] = static_cast<index_t>(i);

        // check if point is inside grid
        if (all((coord >= 0) && (coord < extents))) {
            // compute and store linear voxel grid index
            keys[i] = strides.dot(vec4<key_t>(coord, batch));
        } else {
            // store invalid/oob key
            keys[i] = static_cast<key_t>(-1);
        }
    }
}

template <typename key_t, typename index_t>
__global__ void compute_batch_indices(
    at::PackedTensorAccessor64<index_t, 1, at::RestrictPtrTraits> const offsets,
    at::PackedTensorAccessor64<key_t, 1, at::RestrictPtrTraits> const keys,
    at::PackedTensorAccessor64<index_t, 1, at::RestrictPtrTraits> batches,
    key_t const batch_stride)
{
    auto const n_voxels = offsets.size(0) - 1;

    CUDA_1D_KERNEL_LOOP(i, n_voxels) {
        // get key for the first point in this voxel
        auto const k = keys[offsets[i]];

        // compute the batch index
        batches[i] = k / batch_stride;
    }
}

template <typename points_t, typename index_t>
__global__ void compute_coords(
    at::PackedTensorAccessor64<points_t, 2, at::RestrictPtrTraits> const points,
    at::PackedTensorAccessor64<index_t, 1, at::RestrictPtrTraits> const indices,
    at::PackedTensorAccessor64<index_t, 1, at::RestrictPtrTraits> const offsets,
    at::PackedTensorAccessor64<index_t, 1, at::RestrictPtrTraits> const batch_ids,
    at::PackedTensorAccessor64<index_t, 2, at::RestrictPtrTraits> coords,
    math::vec3<points_t> const offset,
    math::vec3<points_t> const voxel_size)
{
    using math::vec3;

    CUDA_1D_KERNEL_LOOP(i, coords.size(0)) {
        // get point index for the first point in this voxel
        auto const k = indices[offsets[i]];

        // transform to voxel grid coordinates
        auto const point_c = vec3<points_t>(points[k][0], points[k][1], points[k][2]);
        auto const point = (point_c - offset) / voxel_size;
        auto const coord = vec3<index_t>(point);

        // store coordinate
        coords[i][0] = batch_ids[i];
        coords[i][1] = coord.x();
        coords[i][2] = coord.y();
        coords[i][3] = coord.z();
    }
}

} /* namepsace kernel */

template <typename points_t, typename batch_t, typename key_t, typename index_t>
inline std::tuple<at::Tensor, at::Tensor>
compute_idmap(
    at::Tensor const& points,
    at::Tensor const& batches,
    math::vec3<points_t> const& offset,
    math::vec3<points_t> const& voxel_size,
    math::vec3<key_t> const& grid_size,
    math::vec4<key_t> const& strides,
    c10::cuda::CUDAStream const& stream)
{
    using math::vec3;
    using math::vec4;

    auto const n_points = points.size(0);

    // allocate voxel ID/key to index mapping
    auto const key_opts = c10::TensorOptions()
            .device(points.device())
            .dtype(c10::CppTypeToScalarType<key_t>());

    auto const index_opts = c10::TensorOptions()
            .device(points.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto id_keys = at::empty({n_points}, key_opts);
    auto id_indices = at::empty({n_points}, index_opts);

    // fill voxel ID/key to index mapping
    {
        auto const threads = 1024;
        auto const blocks = (n_points + threads - 1) / threads;

        kernel::init_idmap<<<blocks, threads, 0, stream>>>(
            points.template packed_accessor64<points_t, 2, at::RestrictPtrTraits>(),
            batches.template packed_accessor64<batch_t, 1, at::RestrictPtrTraits>(),
            id_keys.template packed_accessor64<key_t, 1, at::RestrictPtrTraits>(),
            id_indices.template packed_accessor64<index_t, 1, at::RestrictPtrTraits>(),
            offset,
            voxel_size,
            grid_size,
            strides);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    // sort by voxel ID/key
    auto id_keys_out = at::empty_like(id_keys);
    auto id_indices_out = at::empty_like(id_indices);

    cub::radix_sort_pairs(
        id_keys.template const_data_ptr<key_t>(),
        id_keys_out.template mutable_data_ptr<key_t>(),
        id_indices.template const_data_ptr<index_t>(),
        id_indices_out.template mutable_data_ptr<index_t>(),
        n_points,
        0, sizeof(key_t) * CHAR_BIT,
        stream);

    return std::make_tuple(
        std::move(id_keys_out),
        std::move(id_indices_out));
}

template <typename key_t, typename index_t>
inline at::Tensor
compute_batch_indices(
    at::Tensor const& offsets,
    at::Tensor const& keys,
    key_t const batch_stride,
    c10::cuda::CUDAStream const& stream)
{
    auto const n_voxels = offsets.size(0) - 1;

    auto const opts = c10::TensorOptions()
            .device(offsets.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto batches = at::empty({n_voxels}, opts);

    auto const threads = 1024;
    auto const blocks = (n_voxels + threads - 1) / threads;

    kernel::compute_batch_indices<<<blocks, threads, 0, stream>>>(
        offsets.template packed_accessor64<index_t, 1, at::RestrictPtrTraits>(),
        keys.template packed_accessor64<key_t, 1, at::RestrictPtrTraits>(),
        batches.template packed_accessor64<index_t, 1, at::RestrictPtrTraits>(),
        batch_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return batches;
}

template <typename key_t, typename index_t>
inline std::tuple<at::Tensor, index_t>
compute_counts(
    at::Tensor const& keys,
    c10::cuda::CUDAStream const& stream)
{
    auto const n_points = keys.size(0);

    auto const opts = c10::TensorOptions()
            .device(keys.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto unique = at::empty_like(keys);
    auto counts = at::empty({n_points}, opts);
    auto num_runs = at::empty({1}, opts);

    cub::run_length_encode(
        keys.template const_data_ptr<key_t>(),
        unique.template mutable_data_ptr<key_t>(),
        counts.template mutable_data_ptr<index_t>(),
        num_runs.template mutable_data_ptr<index_t>(),
        n_points,
        stream);

    auto const first_key = utils::mem::load(unique.template const_data_ptr<key_t>(), stream);
    auto const first_count = utils::mem::load(counts.template const_data_ptr<index_t>(), stream);
    auto const num_buckets = utils::mem::load(num_runs.template const_data_ptr<index_t>(), stream);

    stream.synchronize();

    // the first bucket might contain all out-of-bounds points, handle that
    auto const points_oob = first_key == static_cast<key_t>(-1) ? first_count : 0;
    auto const num_voxels = num_buckets - (points_oob != 0);

    auto const counts_out = counts.narrow(0, points_oob != 0, num_voxels);

    return std::make_tuple(std::move(counts_out), points_oob);
}

template <typename index_t>
inline at::Tensor compute_offsets(
    at::Tensor const& counts,
    c10::cuda::CUDAStream const& stream)
{
    auto const n_voxels = counts.size(0);

    auto const opts = c10::TensorOptions()
            .device(counts.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto offsets = at::empty({n_voxels + 1}, opts);

    // the first offset is always zero
    utils::mem::set_zero(offsets.template mutable_data_ptr<index_t>(), 1, stream);

    // compute and set the remaining offsets
    cub::inclusive_sum(
        counts.template const_data_ptr<index_t>(),
        offsets.template mutable_data_ptr<index_t>() + 1,
        n_voxels,
        stream);

    return offsets;
}

template <typename index_t>
inline at::Tensor
compute_batch_offsets(
    at::Tensor const& batch_indices,
    index_t n_batches,
    c10::cuda::CUDAStream const& stream)
{
    auto const n_voxels = batch_indices.size(0);

    // step 1: count the number of voxels inside each batch
    auto const opts = at::TensorOptions()
            .device(batch_indices.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto unique_buf = at::empty({n_voxels}, opts);
    auto counts_buf = at::empty({n_voxels}, opts);
    auto num_runs = at::empty({1}, opts);

    cub::run_length_encode(
        batch_indices.template const_data_ptr<index_t>(),
        unique_buf.template mutable_data_ptr<index_t>(),
        counts_buf.template mutable_data_ptr<index_t>(),
        num_runs.template mutable_data_ptr<index_t>(),
        n_voxels,
        stream);

    auto const num_buckets = utils::mem::load(num_runs.template const_data_ptr<index_t>(), stream);

    stream.synchronize();

    auto const unique = unique_buf.narrow(0, 0, num_buckets);
    auto const counts = counts_buf.narrow(0, 0, num_buckets);

    // step 2: compute offsets from potentially sparse counts
    auto offsets = at::zeros({n_batches + 1}, opts);

    // the first offset is always zero, create a slice that skips this
    auto offsets_incl = offsets.narrow(0, 1, n_batches);

    // scatter the potentially sparse batch indices
    offsets_incl.index_put_({unique}, counts);

    // compute and set the (remaining) offsets in place
    cub::inclusive_sum(
        offsets_incl.template const_data_ptr<index_t>(),
        offsets_incl.template mutable_data_ptr<index_t>(),
        n_batches,
        stream);

    return offsets;
}

template <typename points_t, typename index_t>
inline at::Tensor compute_coords(
    at::Tensor const& points,
    at::Tensor const& indices,
    at::Tensor const& offsets,
    at::Tensor const& batch_ids,
    math::vec3<points_t> const& range_min,
    math::vec3<points_t> const& voxel_size,
    c10::cuda::CUDAStream const& stream)
{
    auto const n_voxels = offsets.size(0) - 1;

    auto const opts = c10::TensorOptions()
            .device(indices.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto coords = at::empty({n_voxels, 4}, opts);

    auto const threads = 1024;
    auto const blocks = (n_voxels + threads - 1) / threads;

    kernel::compute_coords<<<blocks, threads, 0, stream>>>(
        points.template packed_accessor64<points_t, 2, at::RestrictPtrTraits>(),
        indices.template packed_accessor64<index_t, 1, at::RestrictPtrTraits>(),
        offsets.template packed_accessor64<index_t, 1, at::RestrictPtrTraits>(),
        batch_ids.template packed_accessor64<index_t, 1, at::RestrictPtrTraits>(),
        coords.template packed_accessor64<index_t, 2, at::RestrictPtrTraits>(),
        range_min,
        voxel_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return coords;
}

} /* namepsace detail */

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
    using batch_t = int16_t;

    CHECK_SHAPE(range, 3, 2);
    CHECK_SHAPE(voxel_size, 3);

    CHECK_CUDA(points);
    CHECK_CPU(range);
    CHECK_CPU(voxel_size);

    at::cuda::OptionalCUDAGuard const guard(device_of(points));
    auto const stream = at::cuda::getCurrentCUDAStream();

    // set up batch index map
    auto const num_batches = batches.size(0) - 1;
    auto const batch_indices = detail::compute_batch_index_map<batch_t>(
        batches, points.size(0), points.device(), stream
    );

    return VXOPS_DISPATCH_POINT_TYPES(points.scalar_type(), "voxelize_assign", ([&] {
        using points_t = scalar_t;

        // accessor
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

        return VXOPS_DISPATCH_INDEX_TYPES(dtype, "voxelize_assign_internal", ([&] {
            using index_t = scalar_t;

            auto const [indices, offsets, batch_ids, batch_offsets]
                    = VXOPS_DISPATCH_INDEX_TYPES(ktype_, "voxelize_assign_internal2", ([&]
            {
                using key_t = scalar_t;

                // linear index strides
                auto const grid_size = vec3<key_t>(grid_size_);
                auto const strides = vec4<key_t>(
                    grid_size.y() * grid_size.z(),                  // x
                    grid_size.z(),                                  // y
                    1,                                              // z
                    grid_size.x() * grid_size.y() * grid_size.z()   // batch
                );

                // build the (sorted) index-key mapping
                auto const [id_keys, id_indices]
                    = detail::compute_idmap<points_t, batch_t, key_t, index_t>(
                        points, batch_indices, range_min, vsize, grid_size, strides, stream);

                // compute the number of voxels and points inside each voxel
                auto const [counts, points_oob]
                    = detail::compute_counts<key_t, index_t>(id_keys, stream);

                // copy indices excluding out-of-bounds points
                auto const indices = id_indices.slice(0, points_oob).clone();

                // compute voxel offsets from counts
                auto const offsets = detail::compute_offsets<index_t>(counts, stream);

                // compute batch index for each voxel
                auto const id_batches
                    = detail::compute_batch_indices<key_t, index_t>(
                        offsets, id_keys.slice(0, points_oob), strides.w(), stream);

                // compute voxel offset for each batch
                auto const batch_offsets
                    = detail::compute_batch_offsets<index_t>(id_batches, num_batches, stream);

                return std::make_tuple(
                    std::move(indices),
                    std::move(offsets),
                    std::move(id_batches),
                    std::move(batch_offsets));
            }));

            // compute coordinates
            auto const coords = detail::compute_coords<points_t, index_t>(
                points, indices, offsets, batch_ids, range_min, vsize, stream);

            return std::make_tuple(
                std::move(indices),
                std::move(offsets),
                std::move(coords),
                std::move(batch_offsets));
        }));
    }));
}

} /* namespace vxops::cuda */
