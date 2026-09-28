// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#pragma once

#ifndef __CUDACC__

#define __device__
#define __host__
#define __forceinline__ inline __attribute__((always_inline))

#endif /* __CUDACC__ */
