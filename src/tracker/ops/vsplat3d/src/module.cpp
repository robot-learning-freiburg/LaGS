// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <torch/python.h>
#include "cuda/aggregate.hpp"

namespace vsplat3d {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(vsplat3d, m)
{
    m.set_python_module("tracker.ops.vsplat3d");
    m.def("aggregate_forward(Tensor sample_points, Tensor gaussian_means, Tensor gaussian_icovs, Tensor gaussian_idets, Tensor gaussian_opacities, Tensor gaussian_semantics, Tensor sample_points_int, Tensor gaussian_means_int, Tensor gaussian_radii, int h, int w, int d, float default_value) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def("aggregate_backward(Tensor voxel_offsets, Tensor voxel_indices, Tensor sample_points_int, Tensor sample_points, Tensor gaussian_means, Tensor gaussian_icovs, Tensor gaussian_idets, Tensor gaussian_opacities, Tensor gaussian_semantics, Tensor out_semantics, Tensor out_occupancy, Tensor out_probability, Tensor out_semantics_grad, Tensor out_occupancy_grad, Tensor out_density_grad, int h, int w, int d) -> (Tensor, Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(vsplat3d, CUDA, m)
{
    m.impl("aggregate_forward", &cuda::aggregate_forward);
    m.impl("aggregate_backward", &cuda::agregate_backward);
}

} /* namespace vsplat3d */
