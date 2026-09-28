// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include "utils/decls.hpp"

namespace vsplat3d {
namespace math {

template <typename T>
__forceinline__ __host__ __device__ T clamp(T const& value, T const& min_value, T const& max_value)
{
    return (value < min_value) ? min_value : ((value > max_value) ? max_value : value);
}

} /* namespace math */
} /* namespace vsplat3d */
