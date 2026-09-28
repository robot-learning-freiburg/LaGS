// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace vxops::cuda {
namespace utils::mem {

template <typename T>
inline void copy(
    T* dst, const T* src, size_t count,
    cudaMemcpyKind kind,
    const c10::cuda::CUDAStream stream=at::cuda::getCurrentCUDAStream())
{
    C10_CUDA_CHECK(cudaMemcpyAsync(dst, src, count * sizeof(T), kind, stream));
}

template <typename T>
inline T load(const T* src, const c10::cuda::CUDAStream stream=at::cuda::getCurrentCUDAStream())
{
    T value;

    copy(&value, src, 1, cudaMemcpyDeviceToHost, stream);
    return value;
}

template <typename T>
inline void set_zero(
    T* dst, size_t count,
    const c10::cuda::CUDAStream stream=at::cuda::getCurrentCUDAStream())
{
    C10_CUDA_CHECK(cudaMemsetAsync(dst, 0, count * sizeof(T), stream));
}

} /* namespace utils::mem */
} /* namespace vxops::cuda */
