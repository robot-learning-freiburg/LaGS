// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <torch/python.h>

#include "cpu/all.hpp"
#include "cuda/all.hpp"

namespace bev_pool_v2 {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(bev_pool_v2, m)
{
    m.set_python_module("tracker.ops.bev_pool_v2");
    m.def("pool_forward(Tensor depth, Tensor feat, Tensor out, Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev, Tensor interval_lengths, Tensor interval_starts) -> ()");
    m.def("pool_backward(Tensor out_grad, Tensor depth_grad, Tensor feat_grad, Tensor depth, Tensor feat, Tensor ranks_depth, Tensor ranks_feat, Tensor ranks_bev, Tensor interval_lengths, Tensor interval_starts) -> ()");
}

TORCH_LIBRARY_IMPL(bev_pool_v2, CPU, m)
{
    m.impl("pool_forward", &cpu::pool_forward);
    m.impl("pool_backward", &cpu::pool_backward);
}

TORCH_LIBRARY_IMPL(bev_pool_v2, CUDA, m)
{
    m.impl("pool_forward", &cuda::pool_forward);
    m.impl("pool_backward", &cuda::pool_backward);
}

} /* namespace bev_pool_v2 */
