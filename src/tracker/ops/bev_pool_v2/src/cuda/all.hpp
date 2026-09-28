// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <ATen/Tensor.h>

namespace bev_pool_v2::cuda {

void pool_forward(at::Tensor const depth,
                  at::Tensor const feat,
                  at::Tensor out,
                  at::Tensor const ranks_depth,
                  at::Tensor const ranks_feat,
                  at::Tensor const ranks_bev,
                  at::Tensor const interval_lengths,
                  at::Tensor const interval_starts);

void pool_backward(at::Tensor const out_grad,
                   at::Tensor depth_grad,
                   at::Tensor feat_grad,
                   at::Tensor const depth,
                   at::Tensor const feat,
                   at::Tensor const ranks_depth,
                   at::Tensor const ranks_feat,
                   at::Tensor const ranks_bev,
                   at::Tensor const interval_lengths,
                   at::Tensor const interval_starts);

} /* namespace bev_pool_v2::cuda */
