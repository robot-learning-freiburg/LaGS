// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#include <ATen/Tensor.h>
#include <c10/util/Exception.h>

#define CHECK_CONTIGUOUS(tensor) \
    TORCH_CHECK(tensor.is_contiguous(), #tensor " must be contiguous")

#define CHECK_CUDA(tensor) \
    TORCH_CHECK(tensor.device().is_cuda(), #tensor " must be a CUDA tensor")

#define CHECK_CPU(tensor) \
    TORCH_CHECK(tensor.device().is_cpu(), #tensor " must be a CPU tensor")
