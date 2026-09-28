// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include "utils/decls.hpp"

#include "vec3.hpp"

namespace vxops {
namespace math {

template <typename T>
class vec4
{
private:
    T _v[4];

public:
    // scalar constructor, acting as default zero-init constructor
    __host__ __device__ __forceinline__ vec4(T const &s = T(0))
        : _v{s, s, s, s}
    {
    }

    // value constructor
    __host__ __device__ __forceinline__ vec4(T const x, T const y, T const z, T const w)
        : _v{x, y, z, w}
    {
    }

    // vec4-to-vec4 constructor
    __host__ __device__ __forceinline__ vec4(vec3<T> const xyz, T const w)
        : _v{xyz.x(), xyz.y(), xyz.z(), w}
    {
    }

    // casting constructor
    template <typename V>
    __host__ __device__ __forceinline__ explicit vec4(vec4<V> const &v)
        : _v{static_cast<T>(v.x()), static_cast<T>(v.y()), static_cast<T>(v.z())}
    {
    }

    __host__ __device__ __forceinline__ T &x()
    {
        return this->_v[0];
    }

    __host__ __device__ __forceinline__ T const &x() const
    {
        return this->_v[0];
    }

    __host__ __device__ __forceinline__ T &y()
    {
        return this->_v[1];
    }

    __host__ __device__ __forceinline__ T const &y() const
    {
        return this->_v[1];
    }

    __host__ __device__ __forceinline__ T &z()
    {
        return this->_v[2];
    }

    __host__ __device__ __forceinline__ T const &z() const
    {
        return this->_v[2];
    }

    __host__ __device__ __forceinline__ T &w()
    {
        return this->_v[3];
    }

    __host__ __device__ __forceinline__ T const &w() const
    {
        return this->_v[3];
    }

    __host__ __device__ __forceinline__ T &operator[](int index)
    {
        return this->_v[index];
    }

    __host__ __device__ __forceinline__ T const &operator[](int index) const
    {
        return this->_v[index];
    }

    __host__ __device__ __forceinline__ T dot(vec4<T> const &other) const
    {
        return (this->x() * other.x())
             + (this->y() * other.y())
             + (this->z() * other.z())
             + (this->w() * other.w());
    }

    // unary negate
    __host__ __device__ __forceinline__ friend vec4<T> operator-(vec4<T> const &rhs)
    {
        return {
            -rhs.x(),
            -rhs.y(),
            -rhs.z(),
            -rhs.w(),
        };
    }

    // vector add
    __host__ __device__ __forceinline__ friend vec4<T> operator+(vec4<T> const &lhs, vec4<T> const &rhs)
    {
        return {
            lhs.x() + rhs.x(),
            lhs.y() + rhs.y(),
            lhs.z() + rhs.z(),
            lhs.w() + rhs.w(),
        };
    }

    // vector subtract
    __host__ __device__ __forceinline__ friend vec4<T> operator-(vec4<T> const &lhs, vec4<T> const &rhs)
    {
        return {
            lhs.x() - rhs.x(),
            lhs.y() - rhs.y(),
            lhs.z() - rhs.z(),
            lhs.w() - rhs.w(),
        };
    }

    // scalar multiply
    __host__ __device__ __forceinline__ friend vec4<T> operator*(vec4<T> const &lhs, T const rhs)
    {
        return {
            lhs.x() * rhs,
            lhs.y() * rhs,
            lhs.z() * rhs,
            lhs.w() * rhs,
        };
    }

    // scalar multiply
    __host__ __device__ __forceinline__ friend vec4<T> operator*(T const lhs, vec4<T> const &rhs)
    {
        return vec4<T>{
            lhs * rhs.x(),
            lhs * rhs.y(),
            lhs * rhs.z(),
            lhs * rhs.w(),
        };
    }

    // vector component-wise multiply
    __host__ __device__ __forceinline__ friend vec4<T> operator*(vec4<T> const &lhs, vec4<T> const &rhs)
    {
        return vec4<T>{
            lhs.x() * rhs.x(),
            lhs.y() * rhs.y(),
            lhs.z() * rhs.z(),
            lhs.w() * rhs.w(),
        };
    }

    // scalar divide
    __host__ __device__ __forceinline__ friend vec4<T> operator/(vec4<T> const &lhs, T const rhs)
    {
        return vec4<T>{
            lhs.x() / rhs,
            lhs.y() / rhs,
            lhs.z() / rhs,
            lhs.w() / rhs,
        };
    }

    // scalar divide
    __host__ __device__ __forceinline__ friend vec4<T> operator/(T const lhs, vec4<T> const &rhs)
    {
        return vec4<T>{
            lhs / rhs.x(),
            lhs / rhs.y(),
            lhs / rhs.z(),
            lhs / rhs.w(),
        };
    }

    // component-wise divide
    __host__ __device__ __forceinline__ friend vec4<T> operator/(vec4<T> const &lhs, vec4<T> const &rhs)
    {
        return vec4<T>{
            lhs.x() / rhs.x(),
            lhs.y() / rhs.y(),
            lhs.z() / rhs.z(),
            lhs.w() / rhs.w(),
        };
    }

    // vector add assign
    __host__ __device__ __forceinline__ friend vec4<T> &operator+=(vec4<T> &lhs, vec4<T> const &rhs)
    {
        lhs.x() += rhs.x();
        lhs.y() += rhs.y();
        lhs.z() += rhs.z();
        lhs.w() += rhs.w();

        return lhs;
    }

    // vector sub assign
    __host__ __device__ __forceinline__ friend vec4<T> &operator-=(vec4<T> &lhs, vec4<T> const &rhs)
    {
        lhs.x() -= rhs.x();
        lhs.y() -= rhs.y();
        lhs.z() -= rhs.z();
        lhs.w() -= rhs.w();

        return lhs;
    }

    // scalar mutliply assign
    __host__ __device__ __forceinline__ friend vec4<T> &operator*=(vec4<T> &lhs, T const rhs)
    {
        lhs.x() *= rhs;
        lhs.y() *= rhs;
        lhs.z() *= rhs;
        lhs.w() *= rhs;

        return lhs;
    }

    // component-wise multiply assign
    __host__ __device__ __forceinline__ friend vec4<T> &operator*=(vec4<T> &lhs, vec4<T> const &rhs)
    {
        lhs.x() *= rhs.x();
        lhs.y() *= rhs.y();
        lhs.z() *= rhs.z();
        lhs.w() *= rhs.w();

        return lhs;
    }

    // scalar divide assign
    __host__ __device__ __forceinline__ friend vec4<T> &operator/=(vec4<T> &lhs, T const rhs)
    {
        lhs.x() /= rhs;
        lhs.y() /= rhs;
        lhs.z() /= rhs;
        lhs.w() /= rhs;

        return lhs;
    }

    // component-wise divide assign
    __host__ __device__ __forceinline__ friend vec4<T> &operator/=(vec4<T> &lhs, vec4<T> const &rhs)
    {
        lhs.x() /= rhs.x();
        lhs.y() /= rhs.y();
        lhs.z() /= rhs.z();
        lhs.w() /= rhs.w();

        return lhs;
    }

    // scalar less
    __host__ __device__ __forceinline__ friend vec4<bool> operator<(vec4<T> const &lhs, T const rhs)
    {
        return {
            lhs.x() < rhs,
            lhs.y() < rhs,
            lhs.z() < rhs,
            lhs.w() < rhs,
        };
    }

    // scalar less
    __host__ __device__ __forceinline__ friend vec4<bool> operator<(T const lhs, vec4<T> const &rhs)
    {
        return {
            lhs < rhs.x(),
            lhs < rhs.y(),
            lhs < rhs.z(),
            lhs < rhs.w(),
        };
    }

    // scalar less-equal
    __host__ __device__ __forceinline__ friend vec4<bool> operator<=(vec4<T> const &lhs, T const rhs)
    {
        return {
            lhs.x() <= rhs,
            lhs.y() <= rhs,
            lhs.z() <= rhs,
            lhs.w() <= rhs,
        };
    }

    // scalar less-equal
    __host__ __device__ __forceinline__ friend vec4<bool> operator<=(T const lhs, vec4<T> const &rhs)
    {
        return {
            lhs <= rhs.x(),
            lhs <= rhs.y(),
            lhs <= rhs.z(),
            lhs <= rhs.w(),
        };
    }

    // scalar greater
    __host__ __device__ __forceinline__ friend vec4<bool> operator>(vec4<T> const &lhs, T const rhs)
    {
        return {
            lhs.x() > rhs,
            lhs.y() > rhs,
            lhs.z() > rhs,
            lhs.w() > rhs,
        };
    }

    // scalar greater
    __host__ __device__ __forceinline__ friend vec4<bool> operator>(T const lhs, vec4<T> const &rhs)
    {
        return {
            lhs > rhs.x(),
            lhs > rhs.y(),
            lhs > rhs.z(),
            lhs > rhs.w(),
        };
    }

    // scalar greater-equal
    __host__ __device__ __forceinline__ friend vec4<bool> operator>=(vec4<T> const &lhs, T const rhs)
    {
        return {
            lhs.x() >= rhs,
            lhs.y() >= rhs,
            lhs.z() >= rhs,
            lhs.w() >= rhs,
        };
    }

    // scalar greater-equal
    __host__ __device__ __forceinline__ friend vec4<bool> operator>=(T const lhs, vec4<T> const &rhs)
    {
        return {
            lhs >= rhs.x(),
            lhs >= rhs.y(),
            lhs >= rhs.z(),
            lhs >= rhs.w(),
        };
    }

    // component-wise less
    __host__ __device__ __forceinline__ friend vec4<bool> operator<(vec4<T> const &lhs, vec4<T> const &rhs)
    {
        return {
            lhs.x() < rhs.x(),
            lhs.y() < rhs.y(),
            lhs.z() < rhs.z(),
            lhs.w() < rhs.w(),
        };
    }

    // component-wise less-equal
    __host__ __device__ __forceinline__ friend vec4<bool> operator<=(vec4<T> const &lhs, vec4<T> const &rhs)
    {
        return {
            lhs.x() <= rhs.x(),
            lhs.y() <= rhs.y(),
            lhs.z() <= rhs.z(),
            lhs.w() <= rhs.w(),
        };
    }

    // component-wise greater
    __host__ __device__ __forceinline__ friend vec4<bool> operator>(vec4<T> const &lhs, vec4<T> const &rhs)
    {
        return {
            lhs.x() > rhs.x(),
            lhs.y() > rhs.y(),
            lhs.z() > rhs.z(),
            lhs.w() > rhs.w(),
        };
    }

    // component-wise greater-equal
    __host__ __device__ __forceinline__ friend vec4<bool> operator>=(vec4<T> const &lhs, vec4<T> const &rhs)
    {
        return {
            lhs.x() >= rhs.x(),
            lhs.y() >= rhs.y(),
            lhs.z() >= rhs.z(),
            lhs.w() >= rhs.w(),
        };
    }

    // component-wise equal
    __host__ __device__ __forceinline__ friend vec4<bool> operator==(vec4<T> const &lhs, vec4<T> const &rhs)
    {
        return {
            lhs.x() == rhs.x(),
            lhs.y() == rhs.y(),
            lhs.z() == rhs.z(),
            lhs.w() == rhs.w(),
        };
    }

    // scalar equal
    __host__ __device__ __forceinline__ friend vec4<bool> operator==(vec4<T> const &lhs, T const rhs)
    {
        return {
            lhs.x() == rhs,
            lhs.y() == rhs,
            lhs.z() == rhs,
            lhs.w() == rhs,
        };
    }

    // scalar equal
    __host__ __device__ __forceinline__ friend vec4<bool> operator==(T const lhs, vec4<T> const &rhs)
    {
        return {
            lhs == rhs.x(),
            lhs == rhs.y(),
            lhs == rhs.z(),
            lhs == rhs.w(),
        };
    }
};

__host__ __device__ __forceinline__ vec4<bool> operator&&(vec4<bool> const &lhs, vec4<bool> const &rhs)
{
    return {
        lhs.x() && rhs.x(),
        lhs.y() && rhs.y(),
        lhs.z() && rhs.z(),
        lhs.w() && rhs.w(),
    };
}

__host__ __device__ __forceinline__ vec4<bool> operator||(vec4<bool> const &lhs, vec4<bool> const &rhs)
{
    return {
        lhs.x() || rhs.x(),
        lhs.y() || rhs.y(),
        lhs.z() || rhs.z(),
        lhs.w() || rhs.w(),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec4<T> sign(vec4<T> const &v)
{
    return vec4<T>(vec4<T>(0) < v) - vec4<T>(v < vec4<T>(0));
}

template <typename T>
static __host__ __device__ __forceinline__ vec4<T> ceil(vec4<T> const &v)
{
    return {
        ::ceil(v.x()),
        ::ceil(v.y()),
        ::ceil(v.z()),
        ::ceil(v.w()),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec4<T> floor(vec4<T> const &v)
{
    return {
        ::floor(v.x()),
        ::floor(v.y()),
        ::floor(v.z()),
        ::floor(v.w()),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec4<T> min(vec4<T> const &u, vec4<T> const &v)
{
    return {
        ::fmin(u.x(), v.x()),
        ::fmin(u.y(), v.y()),
        ::fmin(u.z(), v.z()),
        ::fmin(u.w(), v.w()),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec4<T> min(vec4<T> const &v, T const s)
{
    return {
        ::fmin(v.x(), s),
        ::fmin(v.y(), s),
        ::fmin(v.z(), s),
        ::fmin(v.w(), s),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec4<T> min(T const s, vec4<T> const &v)
{
    return {
        ::fmin(s, v.x()),
        ::fmin(s, v.y()),
        ::fmin(s, v.z()),
        ::fmin(s, v.w()),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec4<T> max(vec4<T> const &u, vec4<T> const &v)
{
    return {
        ::fmax(u.x(), v.x()),
        ::fmax(u.y(), v.y()),
        ::fmax(u.z(), v.z()),
        ::fmax(u.w(), v.w()),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec4<T> max(vec4<T> const &v, T const s)
{
    return {
        ::fmax(v.x(), s),
        ::fmax(v.y(), s),
        ::fmax(v.z(), s),
        ::fmax(v.w(), s),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec4<T> max(T const s, vec4<T> const &v)
{
    return {
        ::fmax(s, v.x()),
        ::fmax(s, v.y()),
        ::fmax(s, v.z()),
        ::fmax(s, v.w()),
    };
}

template <typename T>
static __host__ __device__ __forceinline__ vec4<T> abs(vec4<T> const &v)
{
    return {
        ::fabs(v.x()),
        ::fabs(v.y()),
        ::fabs(v.z()),
        ::fabs(v.w()),
    };
}

static __host__ __device__ __forceinline__ bool all(vec4<bool> const &v)
{
    return v.x() && v.y() && v.z() && v.w();
}

static __host__ __device__ __forceinline__ bool any(vec4<bool> const &v)
{
    return v.x() || v.y() || v.z() || v.w();
}

template <typename T>
static __host__ __device__ __forceinline__ T sum(vec4<T> const &v)
{
    return v.x() + v.y() + v.z() + v.w();
}

template <typename T>
static __host__ __device__ __forceinline__ T prod(vec4<T> const &v)
{
    return v.x() * v.y() * v.z() + v.w();
}

} /* namespace math */
} /* namespace vxops */
