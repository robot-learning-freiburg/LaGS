// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <ATen/Tensor.h>

namespace vxops::cpu {

at::Tensor aggregate_mean(at::Tensor const features,
                          at::Tensor const offsets);

at::Tensor voxelize_trace(at::Tensor const points,
                          at::Tensor const batches,
                          at::Tensor const sensor_origin,
                          at::Tensor const range,
                          at::IntArrayRef const grid_size,
                          at::Tensor const voxel_size,
                          at::Tensor const values,
                          at::ScalarType dtype,
                          bool const stop_at_occupied);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
voxelize_assign(at::Tensor const points,
                at::Tensor const batches,
                at::Tensor const range,
                at::Tensor const voxel_size,
                at::ScalarType dtype,
                std::optional<at::ScalarType> ktype);

} /* namespace vxops::cpu */
