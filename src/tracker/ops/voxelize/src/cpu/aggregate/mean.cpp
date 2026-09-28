// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <ATen/ATen.h>

#include "utils/checks.hpp"
#include "utils/dispatch.hpp"

namespace vxops::cpu {

at::Tensor aggregate_mean(at::Tensor const features,
                          at::Tensor const offsets)
{
    CHECK_CPU(features);
    CHECK_CPU(offsets);

    auto const num_voxels = offsets.size(0) - 1;
    auto const num_dims = features.size(1);

    auto mean = at::empty({num_voxels, num_dims}, at::dtype(features.scalar_type()));

    VXOPS_DISPATCH_INDEX_TYPES(offsets.scalar_type(), "aggregate_mean", ([&] {
        using index_t = scalar_t;

        VXOPS_DISPATCH_FLOAT_TYPES(features.scalar_type(), "aggregate_mean_internal", ([&] {
            using feat_t = scalar_t;

            auto const features_ = features.accessor<feat_t, 2>();
            auto const offsets_ = offsets.accessor<index_t, 1>();
            auto mean_ = mean.accessor<feat_t, 2>();

            for (index_t voxel = 0; voxel < num_voxels; voxel++) {
                auto const i_begin = offsets_[voxel];
                auto const i_end = offsets_[voxel+1];
                auto const n = static_cast<feat_t>(i_end - i_begin);

                for (index_t dim = 0; dim < num_dims; dim++) {
                    feat_t sum = 0;

                    for (auto i = i_begin; i < i_end; i++) {
                        sum += features_[i][dim];
                    }

                    mean_[voxel][dim] = sum / n;
                }
            }
        }));
    }));

    return mean;
}

} /* namespace vxops::cpu */
