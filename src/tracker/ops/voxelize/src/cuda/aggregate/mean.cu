// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <algorithm>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "cuda/utils/loop.cuh"

#include "utils/checks.hpp"
#include "utils/dispatch.hpp"

namespace vxops::cuda {

namespace detail::kernel {

template <typename feat_t, typename index_t>
__global__ void aggregate_mean(
    at::PackedTensorAccessor64<feat_t, 2, at::RestrictPtrTraits> const features,
    at::PackedTensorAccessor64<index_t, 1, at::RestrictPtrTraits> const offsets,
    at::PackedTensorAccessor64<feat_t, 2, at::RestrictPtrTraits> mean)
{
    auto const num_voxels = offsets.size(0) - 1;
    auto const num_dims = features.size(1);

    CUDA_2D_KERNEL_LOOP(voxel, num_voxels, dim, num_dims) {
        auto const i_begin = offsets[voxel];
        auto const i_end = offsets[voxel+1];
        auto const n = static_cast<feat_t>(i_end - i_begin);

        feat_t sum = 0;

        for (auto i = i_begin; i < i_end; i++) {
            sum += features[i][dim];
        }

        mean[voxel][dim] = sum / n;
    }
}

} /* namespace detail::kernel */

at::Tensor aggregate_mean(at::Tensor const features,
                          at::Tensor const offsets)
{
    CHECK_CUDA(features);
    CHECK_CUDA(offsets);

    at::cuda::OptionalCUDAGuard const guard(device_of(features));
    auto const stream = at::cuda::getCurrentCUDAStream();

    auto const num_voxels = offsets.size(0) - 1;
    auto const num_dims = features.size(1);

    auto const opts = c10::TensorOptions()
            .device(features.device())
            .dtype(features.scalar_type());

    auto mean = at::empty({num_voxels, num_dims}, opts);

    VXOPS_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "aggregate_mean", ([&] {
        using index_t = scalar_t;

        VXOPS_DISPATCH_FLOAT_TYPES(features.scalar_type(), "aggregate_mean_internal", ([&] {
            using feat_t = scalar_t;

            // try to compute the optimal thread and block size
            auto const threads_max = static_cast<int64_t>(1024);
            auto const warp_size = static_cast<int64_t>(32);

            auto const threads_y = std::min(num_dims, threads_max);
            auto const threads_x = ((threads_max / threads_y) / warp_size) * warp_size;
            auto const threads = dim3(threads_x, threads_y);

            auto const blocks_y = (num_dims + threads_y - 1) / threads_y;
            auto const blocks_x = (num_voxels + threads_x - 1) / threads_x;
            auto const blocks = dim3(blocks_x, blocks_y);

            // aggregate mean
            detail::kernel::aggregate_mean<<<blocks, threads, 0, stream>>>(
                features.packed_accessor64<feat_t, 2, at::RestrictPtrTraits>(),
                offsets.packed_accessor64<index_t, 1, at::RestrictPtrTraits>(),
                mean.packed_accessor64<feat_t, 2, at::RestrictPtrTraits>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));
    }));

    return mean;
}

} /* namespace vxops::cuda */
