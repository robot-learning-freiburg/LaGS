// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <torch/python.h>

#include "cpu/all.hpp"
#include "cuda/all.hpp"

namespace vxops {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(voxelize, m)
{
    m.set_python_module("tracker.ops.voxelize");
    m.def("aggregate_mean(Tensor features, Tensor offsets) -> Tensor");
    m.def("voxelize_trace(Tensor points, Tensor batches, Tensor sensor_origin, Tensor range, int[] grid_size, Tensor voxel_size, Tensor values, ScalarType dtype, bool stop_at_occupied=False) -> Tensor");
    m.def("voxelize_assign(Tensor points, Tensor batches, Tensor range, Tensor voxel_size, ScalarType dtype, ScalarType? ktype) -> (Tensor indices, Tensor offsets, Tensor coords, Tensor batch_offsets)");
}

TORCH_LIBRARY_IMPL(voxelize, CPU, m)
{
    m.impl("aggregate_mean", &cpu::aggregate_mean);
    m.impl("voxelize_trace", &cpu::voxelize_trace);
    m.impl("voxelize_assign", &cpu::voxelize_assign);
}

TORCH_LIBRARY_IMPL(voxelize, CUDA, m)
{
    m.impl("aggregate_mean", &cuda::aggregate_mean);
    m.impl("voxelize_trace", &cuda::voxelize_trace);
    m.impl("voxelize_assign", &cuda::voxelize_assign);
}

} /* namespace vxops */
