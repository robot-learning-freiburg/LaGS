// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include "utils/decls.hpp"
#include "math/vec3.hpp"

namespace vsplat3d {
namespace math {

template <typename T>
struct rect3 {
    vec3<T> min;
    vec3<T> max;

    __forceinline__ __host__ __device__ bool empty() const
    {
        return (min.x >= max.x) || (min.y >= max.y) || (min.z >= max.z);
    }

    __forceinline__ __host__ __device__ vec3<T> dim() const
    {
        auto const d = max - min;

        return vec3<T>{
            d.x >= T(0) ? d.x : T(0),
            d.y >= T(0) ? d.y : T(0),
            d.z >= T(0) ? d.z : T(0),
        };
    }

    __forceinline__ __host__ __device__ T size() const
    {
        return prod(this->dim());
    }
};

} /* namespace math */
} /* namespace vsplat3d */
