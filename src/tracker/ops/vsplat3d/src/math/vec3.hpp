// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include "utils/decls.hpp"
#include "scalar.hpp"

namespace vsplat3d {
namespace math {

template <typename T>
struct vec3
{
    T x, y, z;

    // scalar constructor, acting as default zero-init constructor
    __host__ __device__ __forceinline__ vec3(T const &s = T(0))
        : x{s}, y{s}, z{s}
    {
    }

    // value constructor
    __host__ __device__ __forceinline__ vec3(T const x, T const y, T const z)
        : x{x}, y{y}, z{z}
    {
    }

    // casting constructor
    template <typename V>
    __host__ __device__ __forceinline__ explicit vec3(vec3<V> const &v)
        : x{static_cast<T>(v.x)}, y{static_cast<T>(v.y)}, z{static_cast<T>(v.z)}
    {
    }

    static __host__ __device__ __forceinline__ vec3<T> zero()
    {
        return vec3<T>(T(0));
    }

    static __host__ __device__ __forceinline__ vec3<T> load(T const *ptr)
    {
        return vec3<T>(ptr[0], ptr[1], ptr[2]);
    }

    __host__ __device__ __forceinline__ T dot(vec3<T> const &other) const
    {
        return (this->x * other.x)
             + (this->y * other.y)
             + (this->z * other.z);
    }

    // unary negate
    __host__ __device__ __forceinline__ friend vec3<T> operator-(vec3<T> const &rhs)
    {
        return {
            -rhs.x,
            -rhs.y,
            -rhs.z,
        };
    }

    // vector add
    __host__ __device__ __forceinline__ friend vec3<T> operator+(vec3<T> const &lhs, vec3<T> const &rhs)
    {
        return {
            lhs.x + rhs.x,
            lhs.y + rhs.y,
            lhs.z + rhs.z,
        };
    }

    // vector subtract
    __host__ __device__ __forceinline__ friend vec3<T> operator-(vec3<T> const &lhs, vec3<T> const &rhs)
    {
        return {
            lhs.x - rhs.x,
            lhs.y - rhs.y,
            lhs.z - rhs.z,
        };
    }

    // scalar multiply
    __host__ __device__ __forceinline__ friend vec3<T> operator*(vec3<T> const &lhs, T const rhs)
    {
        return {
            lhs.x * rhs,
            lhs.y * rhs,
            lhs.z * rhs,
        };
    }

    // scalar multiply
    __host__ __device__ __forceinline__ friend vec3<T> operator*(T const lhs, vec3<T> const &rhs)
    {
        return vec3<T>{
            lhs * rhs.x,
            lhs * rhs.y,
            lhs * rhs.z,
        };
    }

    // vector component-wise multiply
    __host__ __device__ __forceinline__ friend vec3<T> operator*(vec3<T> const &lhs, vec3<T> const &rhs)
    {
        return vec3<T>{
            lhs.x * rhs.x,
            lhs.y * rhs.y,
            lhs.z * rhs.z,
        };
    }

    // scalar divide
    __host__ __device__ __forceinline__ friend vec3<T> operator/(vec3<T> const &lhs, T const rhs)
    {
        return vec3<T>{
            lhs.x / rhs,
            lhs.y / rhs,
            lhs.z / rhs,
        };
    }

    // scalar divide
    __host__ __device__ __forceinline__ friend vec3<T> operator/(T const lhs, vec3<T> const &rhs)
    {
        return vec3<T>{
            lhs / rhs.x,
            lhs / rhs.y,
            lhs / rhs.z,
        };
    }

    // component-wise divide
    __host__ __device__ __forceinline__ friend vec3<T> operator/(vec3<T> const &lhs, vec3<T> const &rhs)
    {
        return vec3<T>{
            lhs.x / rhs.x,
            lhs.y / rhs.y,
            lhs.z / rhs.z,
        };
    }

    // vector add assign
    __host__ __device__ __forceinline__ friend vec3<T> &operator+=(vec3<T> &lhs, vec3<T> const &rhs)
    {
        lhs.x += rhs.x;
        lhs.y += rhs.y;
        lhs.z += rhs.z;

        return lhs;
    }

    // vector sub assign
    __host__ __device__ __forceinline__ friend vec3<T> &operator-=(vec3<T> &lhs, vec3<T> const &rhs)
    {
        lhs.x -= rhs.x;
        lhs.y -= rhs.y;
        lhs.z -= rhs.z;

        return lhs;
    }

    // scalar mutliply assign
    __host__ __device__ __forceinline__ friend vec3<T> &operator*=(vec3<T> &lhs, T const rhs)
    {
        lhs.x *= rhs;
        lhs.y *= rhs;
        lhs.z *= rhs;

        return lhs;
    }

    // component-wise multiply assign
    __host__ __device__ __forceinline__ friend vec3<T> &operator*=(vec3<T> &lhs, vec3<T> const &rhs)
    {
        lhs.x *= rhs.x;
        lhs.y *= rhs.y;
        lhs.z *= rhs.z;

        return lhs;
    }

    // scalar divide assign
    __host__ __device__ __forceinline__ friend vec3<T> &operator/=(vec3<T> &lhs, T const rhs)
    {
        lhs.x /= rhs;
        lhs.y /= rhs;
        lhs.z /= rhs;

        return lhs;
    }

    // component-wise divide assign
    __host__ __device__ __forceinline__ friend vec3<T> &operator/=(vec3<T> &lhs, vec3<T> const &rhs)
    {
        lhs.x /= rhs.x;
        lhs.y /= rhs.y;
        lhs.z /= rhs.z;

        return lhs;
    }

    // scalar less
    __host__ __device__ __forceinline__ friend vec3<bool> operator<(vec3<T> const &lhs, T const rhs)
    {
        return {
            lhs.x < rhs,
            lhs.y < rhs,
            lhs.z < rhs,
        };
    }

    // scalar less
    __host__ __device__ __forceinline__ friend vec3<bool> operator<(T const lhs, vec3<T> const &rhs)
    {
        return {
            lhs < rhs.x,
            lhs < rhs.y,
            lhs < rhs.z,
        };
    }

    // scalar less-equal
    __host__ __device__ __forceinline__ friend vec3<bool> operator<=(vec3<T> const &lhs, T const rhs)
    {
        return {
            lhs.x <= rhs,
            lhs.y <= rhs,
            lhs.z <= rhs,
        };
    }

    // scalar less-equal
    __host__ __device__ __forceinline__ friend vec3<bool> operator<=(T const lhs, vec3<T> const &rhs)
    {
        return {
            lhs <= rhs.x,
            lhs <= rhs.y,
            lhs <= rhs.z,
        };
    }

    // scalar greater
    __host__ __device__ __forceinline__ friend vec3<bool> operator>(vec3<T> const &lhs, T const rhs)
    {
        return {
            lhs.x > rhs,
            lhs.y > rhs,
            lhs.z > rhs,
        };
    }

    // scalar greater
    __host__ __device__ __forceinline__ friend vec3<bool> operator>(T const lhs, vec3<T> const &rhs)
    {
        return {
            lhs > rhs.x,
            lhs > rhs.y,
            lhs > rhs.z,
        };
    }

    // scalar greater-equal
    __host__ __device__ __forceinline__ friend vec3<bool> operator>=(vec3<T> const &lhs, T const rhs)
    {
        return {
            lhs.x >= rhs,
            lhs.y >= rhs,
            lhs.z >= rhs,
        };
    }

    // scalar greater-equal
    __host__ __device__ __forceinline__ friend vec3<bool> operator>=(T const lhs, vec3<T> const &rhs)
    {
        return {
            lhs >= rhs.x,
            lhs >= rhs.y,
            lhs >= rhs.z,
        };
    }

    // component-wise less
    __host__ __device__ __forceinline__ friend vec3<bool> operator<(vec3<T> const &lhs, vec3<T> const &rhs)
    {
        return {
            lhs.x < rhs.x,
            lhs.y < rhs.y,
            lhs.z < rhs.z,
        };
    }

    // component-wise less-equal
    __host__ __device__ __forceinline__ friend vec3<bool> operator<=(vec3<T> const &lhs, vec3<T> const &rhs)
    {
        return {
            lhs.x <= rhs.x,
            lhs.y <= rhs.y,
            lhs.z <= rhs.z,
        };
    }

    // component-wise greater
    __host__ __device__ __forceinline__ friend vec3<bool> operator>(vec3<T> const &lhs, vec3<T> const &rhs)
    {
        return {
            lhs.x > rhs.x,
            lhs.y > rhs.y,
            lhs.z > rhs.z,
        };
    }

    // component-wise greater-equal
    __host__ __device__ __forceinline__ friend vec3<bool> operator>=(vec3<T> const &lhs, vec3<T> const &rhs)
    {
        return {
            lhs.x >= rhs.x,
            lhs.y >= rhs.y,
            lhs.z >= rhs.z,
        };
    }

    // component-wise equal
    __host__ __device__ __forceinline__ friend vec3<bool> operator==(vec3<T> const &lhs, vec3<T> const &rhs)
    {
        return {
            lhs.x == rhs.x,
            lhs.y == rhs.y,
            lhs.z == rhs.z,
        };
    }

    // scalar equal
    __host__ __device__ __forceinline__ friend vec3<bool> operator==(vec3<T> const &lhs, T const rhs)
    {
        return {
            lhs.x == rhs,
            lhs.y == rhs,
            lhs.z == rhs,
        };
    }

    // scalar equal
    __host__ __device__ __forceinline__ friend vec3<bool> operator==(T const lhs, vec3<T> const &rhs)
    {
        return {
            lhs == rhs.x,
            lhs == rhs.y,
            lhs == rhs.z,
        };
    }
};

__host__ __device__ __forceinline__ vec3<bool> operator&&(vec3<bool> const &lhs, vec3<bool> const &rhs)
{
    return {
        lhs.x && rhs.x,
        lhs.y && rhs.y,
        lhs.z && rhs.z,
    };
}

__host__ __device__ __forceinline__ vec3<bool> operator||(vec3<bool> const &lhs, vec3<bool> const &rhs)
{
    return {
        lhs.x || rhs.x,
        lhs.y || rhs.y,
        lhs.z || rhs.z,
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec3<T> sign(vec3<T> const &v)
{
    return vec3<T>(vec3<T>(0) < v) - vec3<T>(v < vec3<T>(0));
}

template <typename T>
static __host__ __device__ __forceinline__ vec3<T> ceil(vec3<T> const &v)
{
    return {
        ::ceil(v.x),
        ::ceil(v.y),
        ::ceil(v.z),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec3<T> floor(vec3<T> const &v)
{
    return {
        ::floor(v.x),
        ::floor(v.y),
        ::floor(v.z),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec3<T> min(vec3<T> const &u, vec3<T> const &v)
{
    return {
        ::fmin(u.x, v.x),
        ::fmin(u.y, v.y),
        ::fmin(u.z, v.z),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec3<T> min(vec3<T> const &v, T const s)
{
    return {
        ::fmin(v.x, s),
        ::fmin(v.y, s),
        ::fmin(v.z, s),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec3<T> min(T const s, vec3<T> const &v)
{
    return {
        ::fmin(s, v.x),
        ::fmin(s, v.y),
        ::fmin(s, v.z),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec3<T> max(vec3<T> const &u, vec3<T> const &v)
{
    return {
        ::fmax(u.x, v.x),
        ::fmax(u.y, v.y),
        ::fmax(u.z, v.z),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec3<T> max(vec3<T> const &v, T const s)
{
    return {
        ::fmax(v.x, s),
        ::fmax(v.y, s),
        ::fmax(v.z, s),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec3<T> max(T const s, vec3<T> const &v)
{
    return {
        ::fmax(s, v.x),
        ::fmax(s, v.y),
        ::fmax(s, v.z),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec3<T> abs(vec3<T> const &v)
{
    return {
        ::fabs(v.x),
        ::fabs(v.y),
        ::fabs(v.z),
    };
}

static __host__ __device__ __forceinline__ bool all(vec3<bool> const &v)
{
    return v.x && v.y && v.z;
}

static __host__ __device__ __forceinline__ bool any(vec3<bool> const &v)
{
    return v.x || v.y || v.z;
}

template <typename T>
static __host__ __device__ __forceinline__ T sum(vec3<T> const &v)
{
    return v.x + v.y + v.z;
}

template <typename T>
static __host__ __device__ __forceinline__ T prod(vec3<T> const &v)
{
    return v.x * v.y * v.z;
}

} /* namespace math */
} /* namespace vsplat3d */
