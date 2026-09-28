// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This source code is derived from:
// * BEVDet (https://github.com/HuangJunJie2017/BEVDet), Copyright (c) Phigent Robotics, licensed under Apache-2.0.
// See the LICENSES/ directory for full license texts.

// BEV pooling operations.

#include <cmath>

#include <ATen/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/python.h>

#include "utils/checks.hpp"

namespace bev_pool_v2::cuda {

namespace detail::kernel {

// TODO: add support for generic floating-point types

/**
 * Function: pillar pooling
 * Args:
 *   c                : number of channels
 *   n_intervals      : number of unique points
 *   depth            : input depth, FloatTensor[b,n,d,h,w]
 *   feat             : input feat, FloatTensor[b,n,h,w,c]
 *   ranks_depth      : input index of depth, IntTensor[n]
 *   ranks_feat       : input index of feat, IntTensor[n]
 *   ranks_bev        : output index, IntTensor[n]
 *   interval_lengths : starting position for pooled point, IntTensor[n_intervals]
 *   interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
 *   out              : output features, FloatTensor[b, d, h, w, c]
 */
__global__ void pool_forward(int c, int n_intervals,
                             const float *__restrict__ depth,
                             const float *__restrict__ feat,
                             const int *__restrict__ ranks_depth,
                             const int *__restrict__ ranks_feat,
                             const int *__restrict__ ranks_bev,
                             const int *__restrict__ interval_starts,
                             const int *__restrict__ interval_lengths,
                             float* __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;

    if (index >= n_intervals)
        return;

    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];

    float psum = 0;
    const float* cur_depth;
    const float* cur_feat;

    for (int i = 0; i < interval_length; i++) {
        cur_depth = depth + ranks_depth[interval_start+i];
        cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c;
        psum += *cur_feat * *cur_depth;
    }

    const int* cur_rank = ranks_bev + interval_start;
    float* cur_out = out + *cur_rank * c + cur_c;
    *cur_out = psum;
}


/**
 * Function: pillar pooling backward
 * Args:
 *   c                : number of channels
 *   n_intervals      : number of unique points
 *   out_grad         : gradient of the BEV fmap from top, FloatTensor[b, d, h, w, c]
 *   depth            : input depth, FloatTensor[b,n,d,h,w]
 *   feat             : input feat, FloatTensor[b,n,h,w,c]
 *   ranks_depth      : input index of depth, IntTensor[n]
 *   ranks_feat       : input index of feat, IntTensor[n]
 *   ranks_bev        : output index, IntTensor[n]
 *   interval_lengths : starting position for pooled point, IntTensor[n_intervals]
 *   interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
 *   depth_grad       : gradient of the depth fmap, FloatTensor
 *   feat_grad        : gradient of the feature fmap, FloatTensor
 */
__global__ void pool_backward(int c, int n_intervals,
                              const float *__restrict__ out_grad,
                              const float *__restrict__ depth,
                              const float *__restrict__ feat,
                              const int *__restrict__ ranks_depth,
                              const int *__restrict__ ranks_feat,
                              const int *__restrict__ ranks_bev,
                              const int *__restrict__ interval_starts,
                              const int *__restrict__ interval_lengths,
                              float* __restrict__ depth_grad,
                              float* __restrict__ feat_grad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_intervals)
        return;

    int interval_start = interval_starts[idx];
    int interval_length = interval_lengths[idx];

    const int* cur_rank;
    const float* cur_out_grad;
    const float* cur_out_grad_start;

    const float* cur_feat;
    const float* cur_feat_start;
    float* cur_depth_grad;
    float grad_sum;

    for (int i = 0; i < interval_length; i++) {
        cur_rank = ranks_bev + interval_start + i;
        cur_out_grad_start = out_grad +  * cur_rank * c;
        cur_feat_start = feat + ranks_feat[interval_start+i] * c;

        grad_sum = 0;
        for (int cur_c = 0; cur_c < c; cur_c++) {
            cur_out_grad = cur_out_grad_start + cur_c;
            cur_feat = cur_feat_start + cur_c;
            grad_sum += *cur_out_grad * *cur_feat;
        }

        cur_depth_grad = depth_grad + ranks_depth[interval_start+i];
        *cur_depth_grad = grad_sum;
    }

    float* cur_feat_grad;
    const float* cur_depth;

    for (int cur_c = 0; cur_c < c; cur_c++) {
        grad_sum = 0;
        for (int i = 0; i < interval_length; i++) {
          cur_rank = ranks_bev + interval_start + i;
          cur_out_grad = out_grad + *cur_rank * c + cur_c;

          cur_depth = depth + ranks_depth[interval_start+i];
          grad_sum += *cur_out_grad * *cur_depth;
        }
        cur_feat_grad = feat_grad + ranks_feat[interval_start] * c + cur_c ;
        *cur_feat_grad = grad_sum;
    }
}

} /* namespace detail::kernel */

/*
  Function: pillar pooling (forward, cuda)
  Args:
    depth            : input depth, FloatTensor[n, d, h, w]
    feat             : input features, FloatTensor[n, h, w, c]
    out              : output features, FloatTensor[b, c, h_out, w_out]
    ranks_depth      : depth index of points, IntTensor[n_points]
    ranks_feat       : feat index of points, IntTensor[n_points]
    ranks_bev        : output index of points, IntTensor[n_points]
    interval_lengths : starting position for pooled point, IntTensor[n_intervals]
    interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
  Return:
*/
void pool_forward(at::Tensor const depth,
                  at::Tensor const feat,
                  at::Tensor out,
                  at::Tensor const ranks_depth,
                  at::Tensor const ranks_feat,
                  at::Tensor const ranks_bev,
                  at::Tensor const interval_lengths,
                  at::Tensor const interval_starts)
{
    CHECK_CUDA(depth);
    CHECK_CUDA(feat);
    CHECK_CUDA(out);
    CHECK_CUDA(ranks_depth);
    CHECK_CUDA(ranks_feat);
    CHECK_CUDA(ranks_bev);
    CHECK_CUDA(interval_lengths);
    CHECK_CUDA(interval_starts);

    CHECK_CONTIGUOUS(depth);
    CHECK_CONTIGUOUS(feat);
    CHECK_CONTIGUOUS(out);
    CHECK_CONTIGUOUS(ranks_depth);
    CHECK_CONTIGUOUS(ranks_feat);
    CHECK_CONTIGUOUS(ranks_bev);
    CHECK_CONTIGUOUS(interval_lengths);
    CHECK_CONTIGUOUS(interval_starts);

    const at::cuda::OptionalCUDAGuard guard(device_of(depth));

    auto const c = feat.size(4);
    auto const n_intervals = interval_lengths.size(0);

    auto const threads = static_cast<int64_t>(256);
    auto const blocks = static_cast<int64_t>(
        std::ceil(static_cast<double>(n_intervals * c) / static_cast<double>(threads))
    );

    detail::kernel::pool_forward<<<blocks, threads>>>(
        c, n_intervals,
        depth.data_ptr<float>(),
        feat.data_ptr<float>(),
        ranks_depth.data_ptr<int>(),
        ranks_feat.data_ptr<int>(),
        ranks_bev.data_ptr<int>(),
        interval_starts.data_ptr<int>(),
        interval_lengths.data_ptr<int>(),
        out.data_ptr<float>());
}

/**
 * Function: pillar pooling (backward, cuda)
 *
 * Args:
 *   out_grad         : grad of output bev feature, FloatTensor[b, c, h_out, w_out]
 *   depth_grad       : grad of input depth, FloatTensor[n, d, h, w]
 *   feat_grad        : grad of input feature, FloatTensor[n, h, w, c]
 *   depth            : input depth, FloatTensor[n, d, h, w]
 *   feat             : input features, FloatTensor[n, h, w, c]
 *   ranks_depth      : depth index of points, IntTensor[n_points]
 *   ranks_feat       : feat index of points, IntTensor[n_points]
 *   ranks_bev        : output index of points, IntTensor[n_points]
 *   interval_lengths : starting position for pooled point, IntTensor[n_intervals]
 *   interval_starts  : how many points in each pooled point, IntTensor[n_intervals]
 */
void pool_backward(at::Tensor const out_grad,
                   at::Tensor depth_grad,
                   at::Tensor feat_grad,
                   at::Tensor const depth,
                   at::Tensor const feat,
                   at::Tensor const ranks_depth,
                   at::Tensor const ranks_feat,
                   at::Tensor const ranks_bev,
                   at::Tensor const interval_lengths,
                   at::Tensor const interval_starts)
{
    CHECK_CUDA(out_grad);
    CHECK_CUDA(depth_grad);
    CHECK_CUDA(feat_grad);
    CHECK_CUDA(depth);
    CHECK_CUDA(feat);
    CHECK_CUDA(ranks_depth);
    CHECK_CUDA(ranks_feat);
    CHECK_CUDA(ranks_bev);
    CHECK_CUDA(interval_lengths);
    CHECK_CUDA(interval_starts);

    CHECK_CONTIGUOUS(out_grad);
    CHECK_CONTIGUOUS(depth_grad);
    CHECK_CONTIGUOUS(feat_grad);
    CHECK_CONTIGUOUS(depth);
    CHECK_CONTIGUOUS(feat);
    CHECK_CONTIGUOUS(ranks_depth);
    CHECK_CONTIGUOUS(ranks_feat);
    CHECK_CONTIGUOUS(ranks_bev);
    CHECK_CONTIGUOUS(interval_lengths);
    CHECK_CONTIGUOUS(interval_starts);

    at::cuda::OptionalCUDAGuard const guard(device_of(out_grad));

    auto const c = out_grad.size(4);
    auto const n_intervals = interval_lengths.size(0);

    auto const threads = static_cast<int64_t>(256);
    auto const blocks = static_cast<int64_t>(
        std::ceil(static_cast<double>(n_intervals * c) / static_cast<double>(threads))
    );

    detail::kernel::pool_backward<<<blocks, threads>>>(
        c, n_intervals,
        out_grad.data_ptr<float>(),
        depth.data_ptr<float>(),
        feat.data_ptr<float>(),
        ranks_depth.data_ptr<int>(),
        ranks_feat.data_ptr<int>(),
        ranks_bev.data_ptr<int>(),
        interval_starts.data_ptr<int>(),
        interval_lengths.data_ptr<int>(),
        depth_grad.data_ptr<float>(),
        feat_grad.data_ptr<float>());
}

} /* namespace bev_pool_v2::cuda */
