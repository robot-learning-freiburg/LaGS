// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <climits>

#include <cub/cub.cuh>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace vsplat3d {
namespace cuda {
namespace cub {

namespace detail {

template <typename fn_t, typename... args_t>
inline void wrapper(fn_t fn, args_t... args) {
    cudaError_t status;
    size_t n_bytes = 0;

    // get temporary storage requirements
    status = fn(nullptr, n_bytes, args...);
    C10_CUDA_CHECK(status);

    // allocate temporary storage
    auto& caching_allocator = *::c10::cuda::CUDACachingAllocator::get();
    auto buffer = caching_allocator.allocate(n_bytes);

    // run the actual function
    status = fn(buffer.get(), n_bytes, args...);
    C10_CUDA_CHECK(status);
}

#define VXOPS_CUB_RESOLVE_OVERLOAD(fn)                          \
    [] (auto&&... args) -> decltype (auto)                      \
    {                                                           \
        return fn(std::forward <decltype (args)> (args)...);    \
    }

} /* namespace detail */

template<typename KeyT, typename ValueT, typename NumItemsT>
inline void radix_sort_pairs(
    const KeyT *d_keys_in, KeyT *d_keys_out,
    const ValueT *d_values_in, ValueT *d_values_out,
    NumItemsT num_items,
    int begin_bit = 0, int end_bit = sizeof(KeyT) * CHAR_BIT,
    const c10::cuda::CUDAStream stream=at::cuda::getCurrentCUDAStream())
{
    detail::wrapper(
        VXOPS_CUB_RESOLVE_OVERLOAD(::cub::DeviceRadixSort::SortPairs),
        d_keys_in, d_keys_out,
        d_values_in, d_values_out,
        num_items,
        begin_bit, end_bit,
        stream);
}

template<typename InputIteratorT, typename OutputIteratorT>
inline void inclusive_sum(
    InputIteratorT d_in, OutputIteratorT d_out,
    int num_items,
    const c10::cuda::CUDAStream stream=at::cuda::getCurrentCUDAStream())
{
    detail::wrapper(
        VXOPS_CUB_RESOLVE_OVERLOAD(::cub::DeviceScan::InclusiveSum),
        d_in, d_out,
        num_items,
        stream);
}

} /* namespace cub */
} /* namespace cuda */
} /* namepsace vsplat3d */
