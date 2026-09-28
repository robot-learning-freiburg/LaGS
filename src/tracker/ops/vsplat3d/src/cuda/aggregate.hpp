// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <tuple>
#include <torch/extension.h>

namespace vsplat3d
{
    namespace cuda
    {
        std::tuple<
            torch::Tensor,
            torch::Tensor,
            torch::Tensor,
            torch::Tensor,
            torch::Tensor,
            torch::Tensor>
        aggregate_forward(
            const torch::Tensor &sample_points,
            const torch::Tensor &gaussian_means,
            const torch::Tensor &gaussian_icovs,
            const torch::Tensor &gaussian_idets,
            const torch::Tensor &gaussian_opacities,
            const torch::Tensor &gaussian_semantics,
            const torch::Tensor &sample_points_int,
            const torch::Tensor &gaussian_means_int,
            const torch::Tensor &gaussian_radii,
            const int64_t h,
            const int64_t w,
            const int64_t d,
            const double default_value);

        std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
        agregate_backward(
            const torch::Tensor &voxel_offsets,
            const torch::Tensor &voxel_indices,
            const torch::Tensor &sample_points_int,
            const torch::Tensor &sample_points,
            const torch::Tensor &gaussian_means,
            const torch::Tensor &gaussian_icovs,
            const torch::Tensor &gaussian_idets,
            const torch::Tensor &gaussian_opacities,
            const torch::Tensor &gaussian_semantics,
            const torch::Tensor &out_semantics,
            const torch::Tensor &out_occupancy,
            const torch::Tensor &out_probability,
            const torch::Tensor &out_semantics_grad,
            const torch::Tensor &out_occupancy_grad,
            const torch::Tensor &out_density_grad,
            const int64_t h,
            const int64_t w,
            const int64_t d);
    } /* namesace cuda */
} /* namespace vsplat3d */
