// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <ATen/Dispatch.h>
#include <c10/core/ScalarType.h>

#define VXOPS_DISPATCH_CASE_FLOAT_TYPES(...)  \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VXOPS_DISPATCH_CASE_ALL_TYPES(...)  \
    AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Double, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VXOPS_DISPATCH_CASE_INDEX_TYPES(...)  \
    AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__) \
    AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define VXOPS_DISPATCH_FLOAT_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, VXOPS_DISPATCH_CASE_FLOAT_TYPES(__VA_ARGS__))

#define VXOPS_DISPATCH_ALL_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, VXOPS_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__))

#define VXOPS_DISPATCH_POINT_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, VXOPS_DISPATCH_CASE_FLOAT_TYPES(__VA_ARGS__))

#define VXOPS_DISPATCH_GRID_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, VXOPS_DISPATCH_CASE_ALL_TYPES(__VA_ARGS__))

#define VXOPS_DISPATCH_INDEX_TYPES(TYPE, NAME, ...) \
    AT_DISPATCH_SWITCH(TYPE, NAME, VXOPS_DISPATCH_CASE_INDEX_TYPES(__VA_ARGS__))
