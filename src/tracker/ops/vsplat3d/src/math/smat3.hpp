// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include "utils/decls.hpp"
#include "vec3.hpp"

namespace vsplat3d {
namespace math {

template <typename T>
struct smat3 {
    T a11, a22, a33, a12, a23, a13;

    static __host__ __device__ __forceinline__ smat3<T> zero()
    {
        return smat3<T>{T(0), T(0), T(0), T(0), T(0), T(0)};
    }

    static __host__ __device__ __forceinline__ smat3<T> identity()
    {
        return smat3<T>{T(1), T(0), T(0), T(1), T(0), T(1)};
    }

    static __host__ __device__ __forceinline__ smat3<T> load(T const *ptr)
    {
        return smat3<T>{
            .a11 = ptr[0],
            .a22 = ptr[1],
            .a33 = ptr[2],
            .a12 = ptr[3],
            .a23 = ptr[4],
            .a13 = ptr[5],
        };
    }

    __host__ __device__ __forceinline__ T neg_half_quadratic(vec3<T> d) const
    {
        auto const a = d.x * d.x * a11 + d.y * d.y * a22 + d.z * d.z * a33;
        auto const b = d.x * d.y * a12 + d.y * d.z * a23 + d.z * d.x * a13;

        return -static_cast<T>(0.5) * a - b;
    }

    __host__ __device__ __forceinline__ T det() const
    {
        return a11 * (a22 * a33 - a23 * a23)
             - a12 * (a12 * a33 - a23 * a13)
             + a13 * (a12 * a23 - a22 * a13);
    }

    // d(det)/d(component), with the off-diagonals carrying a factor of two because
    // each stored value occupies two positions of the symmetric matrix. Matches the
    // convention of neg_half_quadratic's gradient.
    __host__ __device__ __forceinline__ smat3<T> det_grad() const
    {
        return smat3<T>{
            .a11 = a22 * a33 - a23 * a23,
            .a22 = a11 * a33 - a13 * a13,
            .a33 = a11 * a22 - a12 * a12,
            .a12 = static_cast<T>(2) * (a23 * a13 - a33 * a12),
            .a23 = static_cast<T>(2) * (a12 * a13 - a11 * a23),
            .a13 = static_cast<T>(2) * (a12 * a23 - a22 * a13),
        };
    }
};

} /* namespace math */
} /* namespace vsplat3d */
