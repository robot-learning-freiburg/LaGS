// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This source code is derived from:
// * BEVDet (https://github.com/HuangJunJie2017/BEVDet), Copyright (c) Phigent Robotics, licensed under Apache-2.0.
// See the LICENSES/ directory for full license texts.

// BEV pooling operations.

#include <cmath>

#include <ATen/Tensor.h>
#include <torch/python.h>

#include "utils/checks.hpp"

namespace bev_pool_v2::cpu {

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
    CHECK_CPU(depth);
    CHECK_CPU(feat);
    CHECK_CPU(out);
    CHECK_CPU(ranks_depth);
    CHECK_CPU(ranks_feat);
    CHECK_CPU(ranks_bev);
    CHECK_CPU(interval_lengths);
    CHECK_CPU(interval_starts);

    // TODO: Implement the forward pass for CPU
    TORCH_CHECK(false, "bev_pool_v2::cpu::pool_forward not implemented");
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
    CHECK_CPU(out_grad);
    CHECK_CPU(depth_grad);
    CHECK_CPU(feat_grad);
    CHECK_CPU(depth);
    CHECK_CPU(feat);
    CHECK_CPU(ranks_depth);
    CHECK_CPU(ranks_feat);
    CHECK_CPU(ranks_bev);
    CHECK_CPU(interval_lengths);
    CHECK_CPU(interval_starts);

    // TODO: Implement the backward pass for CPU
    TORCH_CHECK(false, "bev_pool_v2::cpu::pool_backward not implemented");
}

} /* namespace bev_pool_v2::cpu */
