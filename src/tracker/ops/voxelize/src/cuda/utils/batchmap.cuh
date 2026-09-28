// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <limits>

#include <ATen/Tensor.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include "cuda/utils/loop.cuh"

namespace vxops::cuda::detail {

namespace kernel {

template <typename value_t>
__global__ void fill_slice(value_t *dst, int64_t n, value_t const value)
{
    CUDA_1D_KERNEL_LOOP(i, n) {
        dst[i] = value;
    }
}

} /* namespace kernel */

template <typename index_t>
inline at::Tensor compute_batch_index_map(
    at::Tensor const& batches,
    int64_t n_points,
    c10::optional<c10::Device> device,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_batches = batches.size(0) - 1;
    TORCH_CHECK(num_batches <= std::numeric_limits<index_t>::max(),
                "Number of batches cannot be larger than %lld",
                std::numeric_limits<index_t>::max());

    auto const opts = c10::TensorOptions()
            .device(device)
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto batch_index = at::empty({n_points}, opts);
    auto batch_index_ptr = batch_index.template mutable_data_ptr<index_t>();

    auto const batches_ = batches.accessor<int64_t, 1>();

    for (index_t i = 0; i < num_batches; i++) {
        // get batch range
        auto batch_start = batches_[i];
        auto batch_end = batches_[i + 1];
        auto batch_n = batch_end - batch_start;

        // launch fill kernel
        auto const threads = 1024;
        auto const blocks = (batch_n + threads - 1) / threads;

        kernel::fill_slice<<<blocks, threads, 0, stream>>>(batch_index_ptr + batch_start, batch_n, i);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    return batch_index;
}

} /* namespace vxops::cuda::detail */
