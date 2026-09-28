// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <functional>
#include <tuple>

#include <ATen/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/core/ScalarType.h>
#include <torch/extension.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "../math/rect3.hpp"
#include "../math/scalar.hpp"
#include "../math/smat3.hpp"
#include "../math/vec3.hpp"
#include "../utils/checks.hpp"
#include "../utils/dispatch.hpp"

#include "utils/cub.cuh"
#include "utils/loop.cuh"
#include "utils/mem.hpp"


namespace vsplat3d::cuda {

namespace detail {

namespace kernel {

constexpr float gauss_denom_pi = 0.06349363593424097;

template<typename T>
constexpr auto epsilon = T{1e-8};

template<>
constexpr auto epsilon<float> = 1e-8f;

template<>
constexpr auto epsilon<double> = 1e-15;

template<>
constexpr auto epsilon<at::Half> = 1e-5f;

template<>
constexpr auto epsilon<at::BFloat16> = 1e-7f;


template <typename T>
__forceinline__ __device__ math::rect3<T> get_rect(
    math::vec3<T> const& p, T const& radius, math::vec3<T> const& max)
{
    return math::rect3<T>{
        .min = math::vec3<T>{
            math::clamp(p.x - radius, T{0}, max.x),
            math::clamp(p.y - radius, T{0}, max.y),
            math::clamp(p.z - radius, T{0}, max.z),
        },
        .max = math::vec3<T>{
            math::clamp(p.x + radius + 1, T{0}, max.x),
            math::clamp(p.y + radius + 1, T{0}, max.y),
            math::clamp(p.z + radius + 1, T{0}, max.z),
        },
    };
}

template <typename index_t>
__global__ void compute_gaussian_voxel_counts(
    index_t const num_gaussians,
    index_t const* centers_xyz,
    index_t const* radii,
    math::vec3<index_t> const grid,
    index_t *counts)
{
    CUDA_1D_KERNEL_LOOP(idx, num_gaussians) {
        auto const point = math::vec3<index_t>::load(centers_xyz + 3 * idx);
        auto const rect = get_rect(point, radii[idx], grid);

        counts[idx] = rect.size();
    }
}

template <typename index_t>
__global__ void compute_gaussian_voxel_pairs(
    index_t const num_gaussians,
    index_t const* centers_xyz,
    index_t const* radii,
    index_t const* offsets,
    index_t* gaussian_indices,
    index_t* voxel_indices,
    math::vec3<index_t> const grid)
{
    index_t const idx = ((index_t) blockIdx.x) * blockDim.x + threadIdx.x;
    index_t const stride = (blockDim.x * gridDim.x) / C10_WARP_SIZE;

    index_t const warp_id = idx / C10_WARP_SIZE;
    index_t const tid_in_warp = idx % C10_WARP_SIZE;

    for (index_t i = warp_id; i < num_gaussians; i += stride) {
        index_t const start = (i == 0) ? 0 : offsets[i - 1];
        index_t const end = offsets[i];

        auto const point = math::vec3<index_t>::load(centers_xyz + 3 * i);
        auto const rect = get_rect(point, radii[i], grid);

        for (index_t j = start + tid_in_warp; j < end; j += C10_WARP_SIZE) {
            auto const local_idx = j - start;

            auto const rect_size = rect.dim();

            auto const x_offset = local_idx / (rect_size.y * rect_size.z);
            auto const y_offset = (local_idx / rect_size.z) % rect_size.y;
            auto const z_offset = local_idx % rect_size.z;

            auto const x = rect.min.x + x_offset;
            auto const y = rect.min.y + y_offset;
            auto const z = rect.min.z + z_offset;

            auto const voxel_index = x * grid.y * grid.z + y * grid.z + z;

            gaussian_indices[j] = i;
            voxel_indices[j] = voxel_index;
        }
    }
}

template <typename index_t>
__global__ void compute_voxel_ranges(
    index_t const num_pairs,
    index_t const* voxel_indices,
    index_t* ranges)
{
    CUDA_1D_KERNEL_LOOP(idx, num_pairs) {
        index_t const idx_prev = idx > 0 ? idx - 1 : num_pairs - 1;

        index_t const this_voxel = voxel_indices[idx];
        index_t const prev_voxel = voxel_indices[idx_prev];

        if (this_voxel != prev_voxel || idx == 0) {
            ranges[this_voxel * 2 + 0] = idx;
            ranges[prev_voxel * 2 + 1] = idx_prev + 1;
        }
    }
}

template <typename index_t>
__global__ void compute_voxel_to_point_map(
    index_t const num_points,
    index_t const* points_xyz,
    math::vec3<index_t> const grid,
    index_t* voxel_to_point_index)
{
    CUDA_1D_KERNEL_LOOP(idx, num_points) {
        auto const point = math::vec3<index_t>::load(points_xyz + 3 * idx);

        index_t const voxel_idx = point.x * grid.y * grid.z + point.y * grid.z + point.z;
        voxel_to_point_index[voxel_idx] = idx;
    }
}

template <typename index_t, typename float_t, typename data_t, int NUM_CHANNELS>
__global__ void render_forward_registers(
    index_t const num_samples,
    index_t const num_channels, // actual channel count (must be <= NUM_CHANNELS)
    float_t const* __restrict__ sample_points,
    index_t const* __restrict__ sample_points_int,
    float_t const* __restrict__ gaussian_means,
    float_t const* __restrict__ gaussian_icovs,
    float_t const* __restrict__ gaussian_idets,
    float_t const* __restrict__ gaussian_opacities,
    data_t const* __restrict__ gaussian_semantics,
    index_t const* __restrict__ gaussian_ranges,
    index_t const* __restrict__ gaussian_indices,
    math::vec3<index_t> const grid_size,
    data_t default_value,
    data_t* __restrict__ out_semantics,
    float_t* __restrict__ out_occupancy,
    float_t* __restrict__ out_density,
    float_t* __restrict__ out_probability)
{
    CUDA_1D_KERNEL_LOOP(idx, num_samples) {
        auto const point_int = math::vec3<index_t>::load(sample_points_int + idx * 3);
        auto const point = math::vec3<float_t>::load(sample_points + 3 * idx);

        auto const voxel_idx = point_int.x * grid_size.y * grid_size.z
            + point_int.y * grid_size.z
            + point_int.z;

        auto const range_start = gaussian_ranges[voxel_idx * 2 + 0];
        auto const range_end = gaussian_ranges[voxel_idx * 2 + 1];

        data_t semantics[NUM_CHANNELS];

        for (int j = 0; j < num_channels; j++) {
            semantics[j] = data_t{0};
        }

        float_t occupancy = 1.0;
        float_t density = 0.0;
        float_t prob_sum = 0.0;

        for (auto i = range_start; i < range_end; i++)
        {
            auto const gs_idx = gaussian_indices[i];

            auto const mean = math::vec3<float_t>::load(gaussian_means + gs_idx * 3);
            auto const cov = math::smat3<float_t>::load(gaussian_icovs + gs_idx * 6);
            auto const deter = gaussian_idets[gs_idx];
            auto const opacity = gaussian_opacities[gs_idx];

            auto const power = __expf(cov.neg_half_quadratic(mean - point));
            auto const denom = static_cast<float_t>(gauss_denom_pi) * __fsqrt_rn(deter);
            auto const prob = opacity * denom * power;

            auto const sem_base = gs_idx * num_channels;

            // unroll if small enough
            if (num_channels <= 32) {
                #pragma unroll
                for (int j = 0; j < NUM_CHANNELS; j++) {
                    if (j < num_channels) {
                        semantics[j] += gaussian_semantics[sem_base + j] * static_cast<data_t>(prob);
                    }
                }
            } else {
                for (int j = 0; j < num_channels; j++) {
                    semantics[j] += gaussian_semantics[sem_base + j] * static_cast<data_t>(prob);
                }
            }

            occupancy = (1 - power) * occupancy;
            density = power + density;
            prob_sum = prob + prob_sum;
        }

        auto const valid = prob_sum > epsilon<float_t>;
        auto const out_sem = out_semantics + idx * num_channels;

        if (num_channels <= 32) {
            #pragma unroll
            for (int j = 0; j < NUM_CHANNELS; j++) {
                if (j < num_channels) {
                    out_sem[j] = valid ? (semantics[j] / static_cast<data_t>(prob_sum)) : default_value;
                }
            }
        } else {
            for (int j = 0; j < num_channels; j++) {
                out_sem[j] = valid ? (semantics[j] / static_cast<data_t>(prob_sum)) : default_value;
            }
        }

        out_occupancy[idx] = 1 - occupancy;
        out_density[idx] = density;
        out_probability[idx] = prob_sum;
    }
}

// fallback kernel for very large channel counts
template <typename index_t, typename float_t, typename data_t>
__global__ void render_forward_global_memory(
    index_t const num_samples,
    index_t const num_channels,
    float_t const* __restrict__ sample_points,
    index_t const* __restrict__ sample_points_int,
    float_t const* __restrict__ gaussian_means,
    float_t const* __restrict__ gaussian_icovs,
    float_t const* __restrict__ gaussian_idets,
    float_t const* __restrict__ gaussian_opacities,
    data_t const* __restrict__ gaussian_semantics,
    index_t const* __restrict__ gaussian_ranges,
    index_t const* __restrict__ gaussian_indices,
    math::vec3<index_t> const grid_size,
    data_t default_value,
    data_t* __restrict__ out_semantics,
    float_t* __restrict__ out_occupancy,
    float_t* __restrict__ out_density,
    float_t* __restrict__ out_probability)
{
    CUDA_1D_KERNEL_LOOP(idx, num_samples) {
        auto const point_int = math::vec3<index_t>::load(sample_points_int + idx * 3);
        auto const point = math::vec3<float_t>::load(sample_points + 3 * idx);

        auto const voxel_idx = point_int.x * grid_size.y * grid_size.z
            + point_int.y * grid_size.z
            + point_int.z;

        auto const range_start = gaussian_ranges[voxel_idx * 2 + 0];
        auto const range_end = gaussian_ranges[voxel_idx * 2 + 1];

        auto const out_sem = out_semantics + idx * num_channels;

        for (int j = 0; j < num_channels; j++) {
            out_sem[j] = data_t{0};
        }

        float_t occupancy = 1.0;
        float_t density = 0.0;
        float_t prob_sum = 0.0;

        for (auto i = range_start; i < range_end; i++)
        {
            auto const gs_idx = gaussian_indices[i];

            auto const mean = math::vec3<float_t>::load(gaussian_means + gs_idx * 3);
            auto const cov = math::smat3<float_t>::load(gaussian_icovs + gs_idx * 6);
            auto const deter = gaussian_idets[gs_idx];
            auto const opacity = gaussian_opacities[gs_idx];

            auto const power = __expf(cov.neg_half_quadratic(mean - point));
            auto const denom = static_cast<float_t>(gauss_denom_pi) * __fsqrt_rn(deter);
            auto const prob = opacity * denom * power;

            auto const sem_base = gs_idx * num_channels;

            for (int j = 0; j < num_channels; j++) {
                out_sem[j] += gaussian_semantics[sem_base + j] * static_cast<data_t>(prob);
            }

            occupancy = (1 - power) * occupancy;
            density = power + density;
            prob_sum = prob + prob_sum;
        }

        auto const valid = prob_sum > epsilon<float_t>;

        for (int j = 0; j < num_channels; j++) {
            out_sem[j] = valid ? (out_sem[j] / static_cast<data_t>(prob_sum)) : default_value;
        }

        out_occupancy[idx] = 1 - occupancy;
        out_density[idx] = density;
        out_probability[idx] = prob_sum;
    }
}

template <typename index_t, typename float_t, typename data_t, int NUM_CHANNELS>
__global__ void render_backward_registers(
    index_t const num_gaussians,
    index_t const num_channels, // actual channel count (must be <= NUM_CHANNELS)
    index_t const* __restrict__ voxel_offsets,
    index_t const* __restrict__ voxel_indices,
    index_t const* __restrict__ voxel_to_point,
    float_t const* __restrict__ sample_points,
    float_t const* __restrict__ gaussian_means,
    float_t const* __restrict__ gaussian_icovs,
    float_t const* __restrict__ gaussian_idets,
    float_t const* __restrict__ gaussian_opacities,
    data_t const* __restrict__ gaussian_semantics,
    data_t const* __restrict__ out_semantics,
    float_t const* __restrict__ out_occupancy,
    float_t const* __restrict__ out_probability,
    data_t const* __restrict__ out_semantics_grad,
    float_t const* __restrict__ out_occupancy_grad,
    float_t const* __restrict__ out_density_grad,
    float_t* __restrict__ g_means_grad,
    float_t* __restrict__ g_icov_grad,
    float_t* __restrict__ g_idets_grad,
    float_t* __restrict__ g_opacities_grad,
    data_t* __restrict__ g_semantics_grad)
{
    CUDA_1D_KERNEL_LOOP(idx, num_gaussians) {
        index_t const start = (idx == 0) ? 0 : voxel_offsets[idx - 1];
        index_t const end = voxel_offsets[idx];

        auto const mean = math::vec3<float_t>::load(gaussian_means + 3 * idx);
        auto const cov = math::smat3<float_t>::load(gaussian_icovs + 6 * idx);
        auto const deter = gaussian_idets[idx];
        auto const opa = gaussian_opacities[idx];

        auto const sem_base = idx * num_channels;

        data_t sem[NUM_CHANNELS];
        data_t semantic_grad[NUM_CHANNELS];

        for (int j = 0; j < num_channels; j++)
        {
            sem[j] = gaussian_semantics[sem_base + j];
            semantic_grad[j] = data_t{0};
        }

        float_t means_grad[3] = {0};
        float_t opa_grad = 0;
        float_t cov_grad[6] = {0};
        float_t idets_grad = 0;

        for (auto i = start; i < end; i++)
        {
            auto const voxel_idx = voxel_indices[i];
            auto const pts_idx = voxel_to_point[voxel_idx];
            if (pts_idx >= 0)
            {
                auto const d = mean - math::vec3<float_t>::load(sample_points + pts_idx * 3);

                auto const power = __expf(cov.neg_half_quadratic(d));
                auto const denom = gauss_denom_pi * __fsqrt_rn(deter);
                auto const prob = denom * power;

                float_t power_grad = 0.;
                float_t deter_grad = 0.;
                float_t prob_grad = 0.;
                float_t prob_sum = out_probability[pts_idx];

                if (prob_sum > epsilon<float_t>)
                {
                    float_t const inv_prob_sum = 1.0f / prob_sum;
                    float_t const prob_opa_inv = prob * opa * inv_prob_sum;
                    float_t const prob_inv = prob * inv_prob_sum;
                    float_t const opa_inv = opa * inv_prob_sum;

                    auto const out_sem_base = pts_idx * num_channels;

                    // unroll if small enough
                    if (num_channels <= 32) {
                        #pragma unroll
                        for (int ch = 0; ch < NUM_CHANNELS; ch++)
                        {
                            if (ch < num_channels) {
                                data_t const logit_grad_ch = out_semantics_grad[out_sem_base + ch];
                                data_t const sem_diff = sem[ch] - out_semantics[out_sem_base + ch];

                                semantic_grad[ch] += logit_grad_ch * static_cast<data_t>(prob_opa_inv);
                                prob_grad += static_cast<float_t>(logit_grad_ch * sem_diff) * opa_inv;
                                opa_grad += static_cast<float_t>(logit_grad_ch * sem_diff) * prob_inv;
                            }
                        }
                    } else {
                        for (int ch = 0; ch < num_channels; ch++)
                        {
                            data_t const logit_grad_ch = out_semantics_grad[out_sem_base + ch];
                            data_t const sem_diff = sem[ch] - out_semantics[out_sem_base + ch];

                            semantic_grad[ch] += logit_grad_ch * static_cast<data_t>(prob_opa_inv);
                            prob_grad += static_cast<float_t>(logit_grad_ch * sem_diff) * opa_inv;
                            opa_grad += static_cast<float_t>(logit_grad_ch * sem_diff) * prob_inv;
                        }
                    }
                }
                power_grad += prob_grad * denom;
                power_grad += (1 - out_occupancy[pts_idx]) / (1 - power + epsilon<float_t>) * out_occupancy_grad[pts_idx];
                power_grad += out_density_grad[pts_idx];
                deter_grad += prob_grad * prob / 2 / deter;
                idets_grad += deter_grad;

                auto const power_grad_times_power = power_grad * power;
                means_grad[0] -= power_grad_times_power * (cov.a11 * d.x + cov.a12 * d.y + cov.a13 * d.z);
                means_grad[1] -= power_grad_times_power * (cov.a12 * d.x + cov.a22 * d.y + cov.a23 * d.z);
                means_grad[2] -= power_grad_times_power * (cov.a13 * d.x + cov.a23 * d.y + cov.a33 * d.z);

                cov_grad[0] += power_grad_times_power * (-0.5 * d.x * d.x) + deter_grad * (cov.a22 * cov.a33 - cov.a23 * cov.a23);
                cov_grad[1] += power_grad_times_power * (-0.5 * d.y * d.y) + deter_grad * (cov.a11 * cov.a33 - cov.a13 * cov.a13);
                cov_grad[2] += power_grad_times_power * (-0.5 * d.z * d.z) + deter_grad * (cov.a11 * cov.a22 - cov.a12 * cov.a12);
                cov_grad[3] += power_grad_times_power * (-d.x * d.y) + 2 * deter_grad * (cov.a23 * cov.a13 - cov.a33 * cov.a12);
                cov_grad[4] += power_grad_times_power * (-d.y * d.z) + 2 * deter_grad * (cov.a12 * cov.a13 - cov.a11 * cov.a23);
                cov_grad[5] += power_grad_times_power * (-d.x * d.z) + 2 * deter_grad * (cov.a12 * cov.a23 - cov.a22 * cov.a13);
            }
        }

        auto const p_mean_grad = g_means_grad + idx * 3;
        p_mean_grad[0] = means_grad[0];
        p_mean_grad[1] = means_grad[1];
        p_mean_grad[2] = means_grad[2];

        auto const p_cov_grad = g_icov_grad + idx * 6;
        p_cov_grad[0] = cov_grad[0];
        p_cov_grad[1] = cov_grad[1];
        p_cov_grad[2] = cov_grad[2];
        p_cov_grad[3] = cov_grad[3];
        p_cov_grad[4] = cov_grad[4];
        p_cov_grad[5] = cov_grad[5];

        g_idets_grad[idx] = idets_grad;
        g_opacities_grad[idx] = opa_grad;

        auto const p_sem_grad = g_semantics_grad + idx * num_channels;

        if (num_channels <= 32) {
            #pragma unroll
            for (int j = 0; j < NUM_CHANNELS; j++) {
                if (j < num_channels) {
                    p_sem_grad[j] = semantic_grad[j];
                }
            }
        } else {
            for (int j = 0; j < num_channels; j++) {
                p_sem_grad[j] = semantic_grad[j];
            }
        }
    }
}

// fallback kernel using global memory for very large channel counts
template <typename index_t, typename float_t, typename data_t>
__global__ void render_backward_global_memory(
    index_t const num_gaussians,
    index_t const num_channels,
    index_t const* __restrict__ voxel_offsets,
    index_t const* __restrict__ voxel_indices,
    index_t const* __restrict__ voxel_to_point,
    float_t const* __restrict__ sample_points,
    float_t const* __restrict__ gaussian_means,
    float_t const* __restrict__ gaussian_icovs,
    float_t const* __restrict__ gaussian_idets,
    float_t const* __restrict__ gaussian_opacities,
    data_t const* __restrict__ gaussian_semantics,
    data_t const* __restrict__ out_semantics,
    float_t const* __restrict__ out_occupancy,
    float_t const* __restrict__ out_probability,
    data_t const* __restrict__ out_semantics_grad,
    float_t const* __restrict__ out_occupancy_grad,
    float_t const* __restrict__ out_density_grad,
    float_t* __restrict__ g_means_grad,
    float_t* __restrict__ g_icov_grad,
    float_t* __restrict__ g_idets_grad,
    float_t* __restrict__ g_opacities_grad,
    data_t* __restrict__ g_semantics_grad)
{
    CUDA_1D_KERNEL_LOOP(idx, num_gaussians) {
        index_t const start = (idx == 0) ? 0 : voxel_offsets[idx - 1];
        index_t const end = voxel_offsets[idx];

        auto const mean = math::vec3<float_t>::load(gaussian_means + 3 * idx);
        auto const cov = math::smat3<float_t>::load(gaussian_icovs + 6 * idx);
        auto const deter = gaussian_idets[idx];
        auto const opa = gaussian_opacities[idx];

        auto const sem_base = idx * num_channels;
        auto const p_sem_grad = g_semantics_grad + idx * num_channels;

        for (int j = 0; j < num_channels; j++) {
            p_sem_grad[j] = data_t{0};
        }

        float_t means_grad[3] = {0};
        float_t opa_grad = 0;
        float_t cov_grad[6] = {0};
        float_t idets_grad = 0;

        for (auto i = start; i < end; i++)
        {
            auto const voxel_idx = voxel_indices[i];
            auto const pts_idx = voxel_to_point[voxel_idx];
            if (pts_idx >= 0)
            {
                auto const d = mean - math::vec3<float_t>::load(sample_points + pts_idx * 3);

                auto const power = __expf(cov.neg_half_quadratic(d));
                auto const denom = gauss_denom_pi * __fsqrt_rn(deter);
                auto const prob = denom * power;

                float_t power_grad = 0.;
                float_t deter_grad = 0.;
                float_t prob_grad = 0.;
                float_t prob_sum = out_probability[pts_idx];

                if (prob_sum > epsilon<float_t>)
                {
                    float_t const inv_prob_sum = 1.0f / prob_sum;
                    float_t const prob_opa_inv = prob * opa * inv_prob_sum;
                    float_t const prob_inv = prob * inv_prob_sum;
                    float_t const opa_inv = opa * inv_prob_sum;

                    auto const out_sem_base = pts_idx * num_channels;

                    for (int ch = 0; ch < num_channels; ch++)
                    {
                        data_t const sem_val = gaussian_semantics[sem_base + ch];
                        data_t const logit_grad_ch = out_semantics_grad[out_sem_base + ch];
                        data_t const sem_diff = sem_val - out_semantics[out_sem_base + ch];

                        p_sem_grad[ch] += logit_grad_ch * static_cast<data_t>(prob_opa_inv);
                        prob_grad += static_cast<float_t>(logit_grad_ch * sem_diff) * opa_inv;
                        opa_grad += static_cast<float_t>(logit_grad_ch * sem_diff) * prob_inv;
                    }
                }
                power_grad += prob_grad * denom;
                power_grad += (1 - out_occupancy[pts_idx]) / (1 - power + epsilon<float_t>) * out_occupancy_grad[pts_idx];
                power_grad += out_density_grad[pts_idx];
                deter_grad += prob_grad * prob / 2 / deter;
                idets_grad += deter_grad;

                auto const power_grad_times_power = power_grad * power;
                means_grad[0] -= power_grad_times_power * (cov.a11 * d.x + cov.a12 * d.y + cov.a13 * d.z);
                means_grad[1] -= power_grad_times_power * (cov.a12 * d.x + cov.a22 * d.y + cov.a23 * d.z);
                means_grad[2] -= power_grad_times_power * (cov.a13 * d.x + cov.a23 * d.y + cov.a33 * d.z);

                cov_grad[0] += power_grad_times_power * (-0.5 * d.x * d.x) + deter_grad * (cov.a22 * cov.a33 - cov.a23 * cov.a23);
                cov_grad[1] += power_grad_times_power * (-0.5 * d.y * d.y) + deter_grad * (cov.a11 * cov.a33 - cov.a13 * cov.a13);
                cov_grad[2] += power_grad_times_power * (-0.5 * d.z * d.z) + deter_grad * (cov.a11 * cov.a22 - cov.a12 * cov.a12);
                cov_grad[3] += power_grad_times_power * (-d.x * d.y) + 2 * deter_grad * (cov.a23 * cov.a13 - cov.a33 * cov.a12);
                cov_grad[4] += power_grad_times_power * (-d.y * d.z) + 2 * deter_grad * (cov.a12 * cov.a13 - cov.a11 * cov.a23);
                cov_grad[5] += power_grad_times_power * (-d.x * d.z) + 2 * deter_grad * (cov.a12 * cov.a23 - cov.a22 * cov.a13);
            }
        }

        auto const p_mean_grad = g_means_grad + idx * 3;
        p_mean_grad[0] = means_grad[0];
        p_mean_grad[1] = means_grad[1];
        p_mean_grad[2] = means_grad[2];

        auto const p_cov_grad = g_icov_grad + idx * 6;
        p_cov_grad[0] = cov_grad[0];
        p_cov_grad[1] = cov_grad[1];
        p_cov_grad[2] = cov_grad[2];
        p_cov_grad[3] = cov_grad[3];
        p_cov_grad[4] = cov_grad[4];
        p_cov_grad[5] = cov_grad[5];

        g_idets_grad[idx] = idets_grad;
        g_opacities_grad[idx] = opa_grad;
    }
}

} /* namespace kernel */

template<int... Channels>
struct ChannelList {};

// Define optimized channel counts here, must be ordered ascendingly
using CommonChannels = ChannelList<8, 16, 17, 18, 32, 64, 128, 256>;

template<typename ChannelListType>
struct Last {};

template<int... Channels>
struct Last<ChannelList<Channels...>> {
    static constexpr int value() {
        if constexpr (sizeof...(Channels) == 0) {
            return 0;
        } else {
            int arr[] = {Channels...};
            return arr[sizeof...(Channels) - 1];
        }
    }
};

template<typename ChannelListType>
constexpr int last() {
    return Last<ChannelListType>::value();
}

// Templated dispatcher wrapper
template<typename KernelWrapper, typename ChannelListType>
struct KernelDispatcher;

template<typename KernelWrapper, int... Channels>
struct KernelDispatcher<KernelWrapper, ChannelList<Channels...>> {
    template<typename... Args>
    static void dispatch(int num_channels, dim3 blocks, dim3 threads, c10::cuda::CUDAStream const& stream, Args&&... args) {
        bool dispatched = ((num_channels <= Channels ?
            (KernelWrapper::template launch_register<Channels>(blocks, threads, stream, std::forward<Args>(args)...), true) : false) || ...);

        if (!dispatched) {
            KernelWrapper::launch_global(blocks, threads, stream, std::forward<Args>(args)...);
        }

        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
};

// Convenience function
template<typename KernelWrapper, typename ChannelListType, typename... Args>
void dispatch_kernel(int num_channels, dim3 blocks, dim3 threads, c10::cuda::CUDAStream const& stream, Args&&... args) {
    KernelDispatcher<KernelWrapper, ChannelListType>::dispatch(
        num_channels, blocks, threads, stream, std::forward<Args>(args)...);
}

// Kernel wrapper for forward pass
template<typename index_t, typename float_t, typename data_t>
struct ForwardKernelWrapper {
    template<int N, typename... Args>
    static void launch_register(dim3 blocks, dim3 threads, c10::cuda::CUDAStream const& stream, Args&&... args) {
        kernel::render_forward_registers<index_t, float_t, data_t, N><<<blocks, threads, 0, stream>>>(
            std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void launch_global(dim3 blocks, dim3 threads, c10::cuda::CUDAStream const& stream, Args&&... args) {
        kernel::render_forward_global_memory<index_t, float_t, data_t><<<blocks, threads, 0, stream>>>(
            std::forward<Args>(args)...);
    }
};

// Kernel wrapper for backward pass
template<typename index_t, typename float_t, typename data_t>
struct BackwardKernelWrapper {
    template<int N, typename... Args>
    static void launch_register(dim3 blocks, dim3 threads, c10::cuda::CUDAStream const& stream, Args&&... args) {
        kernel::render_backward_registers<index_t, float_t, data_t, N><<<blocks, threads, 0, stream>>>(
            std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void launch_global(dim3 blocks, dim3 threads, c10::cuda::CUDAStream const& stream, Args&&... args) {
        kernel::render_backward_global_memory<index_t, float_t, data_t><<<blocks, threads, 0, stream>>>(
            std::forward<Args>(args)...);
    }
};


template <typename index_t>
inline at::Tensor compute_gaussian_voxel_counts(
    at::Tensor const& centers_xyz,
    at::Tensor const& radii,
    math::vec3<index_t> const& grid_size,
    c10::cuda::CUDAStream const& stream)
{
    auto const opts = at::TensorOptions()
            .device(centers_xyz.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto const num_gaussians = static_cast<index_t>(centers_xyz.size(0));
    auto counts = at::empty({num_gaussians}, opts);

    auto const threads = 512;
    auto const blocks = (num_gaussians + threads - 1) / threads;

    kernel::compute_gaussian_voxel_counts<<<blocks, threads, 0, stream>>>(
        num_gaussians,
        centers_xyz.template const_data_ptr<index_t>(),
        radii.template const_data_ptr<index_t>(),
        grid_size,
        counts.template mutable_data_ptr<index_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return counts;
}

template <typename index_t>
inline at::Tensor compute_gaussian_voxel_offsets(
    at::Tensor const& centers_xyz,
    at::Tensor const& radii,
    math::vec3<index_t> const& grid_size,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_gaussians = static_cast<index_t>(centers_xyz.size(0));

    auto const counts = compute_gaussian_voxel_counts<index_t>(
        centers_xyz, radii, grid_size, stream);

    auto offsets = at::empty_like(counts);

    cub::inclusive_sum(
        counts.template const_data_ptr<index_t>(),
        offsets.template mutable_data_ptr<index_t>(),
        num_gaussians, stream);

    return offsets;
}

template <typename index_t>
inline std::tuple<at::Tensor, at::Tensor>
compute_gaussian_voxel_pairs(
    at::Tensor const& centers_xyz,
    at::Tensor const& radii,
    at::Tensor const& offsets,
    math::vec3<index_t> const& grid_size,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_gaussians = static_cast<index_t>(centers_xyz.size(0));

    auto const num_pairs = utils::mem::load(
        (offsets.template const_data_ptr<index_t>()) + num_gaussians - 1, stream, false);

    auto const opts = at::TensorOptions()
            .device(centers_xyz.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto gaussian_indices = at::empty({num_pairs}, opts);
    auto voxel_indices = at::empty({num_pairs}, opts);

    auto const threads = 512;
    auto const warps = threads / at::cuda::warp_size();
    auto const blocks = std::min<int64_t>((num_gaussians + warps - 1) / warps, 2048L);

    kernel::compute_gaussian_voxel_pairs<<<blocks, threads, 0, stream>>>(
        num_gaussians,
        centers_xyz.template const_data_ptr<index_t>(),
        radii.template const_data_ptr<index_t>(),
        offsets.template const_data_ptr<index_t>(),
        gaussian_indices.template mutable_data_ptr<index_t>(),
        voxel_indices.template mutable_data_ptr<index_t>(),
        grid_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {gaussian_indices, voxel_indices};
}

template <typename index_t>
inline std::tuple<at::Tensor, at::Tensor>
sort_gaussian_voxel_pairs(
    at::Tensor const& voxel_indices,
    at::Tensor const& gaussian_indices,
    c10::cuda::CUDAStream const& stream)
{
    auto sorted_voxel_indices = at::empty_like(voxel_indices);
    auto sorted_gaussian_indices = at::empty_like(gaussian_indices);

    cub::radix_sort_pairs(
        voxel_indices.template const_data_ptr<index_t>(),
        sorted_voxel_indices.template mutable_data_ptr<index_t>(),
        gaussian_indices.template const_data_ptr<index_t>(),
        sorted_gaussian_indices.template mutable_data_ptr<index_t>(),
        voxel_indices.size(0),
        0, sizeof(index_t) * CHAR_BIT,
        stream);

    return {sorted_voxel_indices, sorted_gaussian_indices};
}

template <typename index_t>
inline at::Tensor compute_voxel_ranges(
    at::Tensor const& voxel_indices,
    math::vec3<index_t> const& grid_size,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_pairs = static_cast<index_t>(voxel_indices.size(0));

    auto const opts = at::TensorOptions()
            .device(voxel_indices.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto ranges = at::zeros({prod(grid_size), 2}, opts);

    if (num_pairs == 0)
        return ranges;

    auto const threads = 512;
    auto const blocks = (num_pairs + threads - 1) / threads;

    kernel::compute_voxel_ranges<<<blocks, threads, 0, stream>>>(
        num_pairs,
        voxel_indices.template const_data_ptr<index_t>(),
        ranges.template mutable_data_ptr<index_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return ranges;
}

template <typename index_t>
inline at::Tensor compute_voxel_to_point_map(
    at::Tensor const& points_xyz,
    math::vec3<index_t> const& grid_size,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_points = static_cast<index_t>(points_xyz.size(0));

    auto const opts = at::TensorOptions()
            .device(points_xyz.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto indices = at::full({prod(grid_size)}, -1, opts);

    auto const threads = 512;
    auto const blocks = (num_points + threads - 1) / threads;

    kernel::compute_voxel_to_point_map<<<blocks, threads, 0, stream>>>(
        num_points,
        points_xyz.template const_data_ptr<index_t>(),
        grid_size,
        indices.template mutable_data_ptr<index_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return indices;
}

template <typename index_t, typename float_t, typename data_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
render_forward(
    at::Tensor const& sample_points,
    at::Tensor const& sample_points_int,
    at::Tensor const& gaussian_means,
    at::Tensor const& gaussian_icovs,
    at::Tensor const& gaussian_idets,
    at::Tensor const& gaussian_opacities,
    at::Tensor const& gaussian_semantics,
    at::Tensor const& gaussian_ranges,
    at::Tensor const& gaussian_indices,
    math::vec3<index_t> const& grid_size,
    data_t default_value,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_samples = static_cast<index_t>(sample_points.size(0));
    auto const num_channels = static_cast<index_t>(gaussian_semantics.size(1));

    auto const data_opts = at::TensorOptions()
            .device(sample_points.device())
            .dtype(c10::CppTypeToScalarType<data_t>());

    auto const float_opts = at::TensorOptions()
            .device(sample_points.device())
            .dtype(c10::CppTypeToScalarType<float_t>());

    auto out_semantics = at::full({num_samples, num_channels}, 0.0, data_opts);
    auto out_occupancy = at::full({num_samples}, 0.0, float_opts);
    auto out_density = at::full({num_samples}, 0.0, float_opts);
    auto out_probability = at::full({num_samples}, 0.0, float_opts);

    auto const threads = [&]{
        if (num_channels <= 32)
            return 512;
        if (num_channels <= 128)
            return 256;
        if (num_channels <= last<CommonChannels>())
            return 128;
        return 512;
    }();
    auto const blocks = (num_samples + threads - 1) / threads;

    dispatch_kernel<ForwardKernelWrapper<index_t, float_t, data_t>, CommonChannels>(
        num_channels, blocks, threads, stream,
        num_samples,
        num_channels,
        sample_points.template const_data_ptr<float_t>(),
        sample_points_int.template const_data_ptr<index_t>(),
        gaussian_means.template const_data_ptr<float_t>(),
        gaussian_icovs.template const_data_ptr<float_t>(),
        gaussian_idets.template const_data_ptr<float_t>(),
        gaussian_opacities.template const_data_ptr<float_t>(),
        gaussian_semantics.template const_data_ptr<data_t>(),
        gaussian_ranges.template const_data_ptr<index_t>(),
        gaussian_indices.template const_data_ptr<index_t>(),
        grid_size,
        default_value,
        out_semantics.template mutable_data_ptr<data_t>(),
        out_occupancy.template mutable_data_ptr<float_t>(),
        out_density.template mutable_data_ptr<float_t>(),
        out_probability.template mutable_data_ptr<float_t>());

    return {out_semantics, out_occupancy, out_density, out_probability};
}

template <typename index_t, typename float_t, typename data_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
render_backward(
    at::Tensor const& voxel_offsets,
    at::Tensor const& voxel_indices,
    at::Tensor const& voxel_to_points,
    at::Tensor const& sample_points,
    at::Tensor const& gaussian_means,
    at::Tensor const& gaussian_icovs,
    at::Tensor const& gaussian_idets,
    at::Tensor const& gaussian_opacities,
    at::Tensor const& gaussian_semantics,
    at::Tensor const& out_semantics,
    at::Tensor const& out_occupancy,
    at::Tensor const& out_probability,
    at::Tensor const& out_semantics_grad,
    at::Tensor const& out_occupancy_grad,
    at::Tensor const& out_density_grad,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_samples = sample_points.size(0);
    auto const num_gaussians = gaussian_means.size(0);
    auto const num_channels = gaussian_semantics.size(1);

    auto const float_opts = at::TensorOptions()
            .device(gaussian_means.device())
            .dtype(c10::CppTypeToScalarType<float_t>());

    auto const data_opts = at::TensorOptions()
            .device(gaussian_means.device())
            .dtype(c10::CppTypeToScalarType<data_t>());

    auto means_grad = at::zeros({num_gaussians, 3}, float_opts);
    auto icov_grad = at::zeros({num_gaussians, 6}, float_opts);
    auto idets_grad = at::zeros({num_gaussians}, float_opts);
    auto opacities_grad = at::zeros({num_gaussians}, float_opts);
    auto semantics_grad = at::zeros({num_gaussians, num_channels}, data_opts);

    auto const threads = [&]{
        if (num_channels <= 32)
            return 512;
        if (num_channels <= 128)
            return 256;
        if (num_channels <= last<CommonChannels>())
            return 128;
        return 512;
    }();
    auto const blocks = (num_gaussians + threads - 1) / threads;

    dispatch_kernel<BackwardKernelWrapper<index_t, float_t, data_t>, CommonChannels>(
        static_cast<int>(num_channels), blocks, threads, stream,
        static_cast<index_t>(num_gaussians),
        static_cast<index_t>(num_channels),
        voxel_offsets.template const_data_ptr<index_t>(),
        voxel_indices.template const_data_ptr<index_t>(),
        voxel_to_points.template const_data_ptr<index_t>(),
        sample_points.template const_data_ptr<float_t>(),
        gaussian_means.template const_data_ptr<float_t>(),
        gaussian_icovs.template const_data_ptr<float_t>(),
        gaussian_idets.template const_data_ptr<float_t>(),
        gaussian_opacities.template const_data_ptr<float_t>(),
        gaussian_semantics.template const_data_ptr<data_t>(),
        out_semantics.template const_data_ptr<data_t>(),
        out_occupancy.template const_data_ptr<float_t>(),
        out_probability.template const_data_ptr<float_t>(),
        out_semantics_grad.template const_data_ptr<data_t>(),
        out_occupancy_grad.template const_data_ptr<float_t>(),
        out_density_grad.template const_data_ptr<float_t>(),
        means_grad.template mutable_data_ptr<float_t>(),
        icov_grad.template mutable_data_ptr<float_t>(),
        idets_grad.template mutable_data_ptr<float_t>(),
        opacities_grad.template mutable_data_ptr<float_t>(),
        semantics_grad.template mutable_data_ptr<data_t>());

    return {means_grad, icov_grad, idets_grad, opacities_grad, semantics_grad};
}

template <typename index_t, typename float_t, typename data_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
aggregate_forward(
    at::Tensor const& sample_points,
    at::Tensor const& gaussian_means,
    at::Tensor const& gaussian_icovs,
    at::Tensor const& gaussian_idets,
    at::Tensor const& gaussian_opacities,
    at::Tensor const& gaussian_semantics,
    at::Tensor const& sample_points_int,
    at::Tensor const& gaussian_means_int,
    at::Tensor const& gaussian_radii,
    math::vec3<index_t> const& grid_size,
    data_t default_value,
    c10::cuda::CUDAStream const& stream)
{
    auto voxel_offsets = compute_gaussian_voxel_offsets<index_t>(
        gaussian_means_int,
        gaussian_radii,
        grid_size,
        stream);

    auto const [
        gaussian_indices_unsorted,
        voxel_indices_unsorted
    ] = compute_gaussian_voxel_pairs<index_t>(
        gaussian_means_int,
        gaussian_radii,
        voxel_offsets,
        grid_size,
        stream);

    auto const [
        voxel_indices,
        gaussian_indices
    ] = sort_gaussian_voxel_pairs<index_t>(
        voxel_indices_unsorted,
        gaussian_indices_unsorted,
        stream);

    auto gaussian_ranges = compute_voxel_ranges<index_t>(
        voxel_indices, grid_size, stream);

    auto const [
        out_semantics,
        out_occupancy,
        out_density,
        out_probability
    ] = render_forward<index_t, float_t, data_t>(
        sample_points,
        sample_points_int,
        gaussian_means,
        gaussian_icovs,
        gaussian_idets,
        gaussian_opacities,
        gaussian_semantics,
        gaussian_ranges,
        gaussian_indices,
        grid_size,
        default_value,
        stream);

    return {
        out_semantics,
        out_occupancy,
        out_density,
        out_probability,
        voxel_offsets,
        voxel_indices_unsorted,
   };
}

template <typename index_t, typename float_t, typename data_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
aggregate_backward(
    at::Tensor const& voxel_offsets,
    at::Tensor const& voxel_indices,
    at::Tensor const& sample_points_int,
    at::Tensor const& sample_points,
    at::Tensor const& gaussian_means,
    at::Tensor const& gaussian_icovs,
    at::Tensor const& gaussian_idets,
    at::Tensor const& gaussian_opacities,
    at::Tensor const& gaussian_semantics,
    at::Tensor const& out_semantics,
    at::Tensor const& out_occupancy,
    at::Tensor const& out_probability,
    at::Tensor const& out_semantics_grad,
    at::Tensor const& out_occupancy_grad,
    at::Tensor const& out_density_grad,
    math::vec3<index_t> const& grid_size,
    c10::cuda::CUDAStream const& stream)
{
    auto const voxel_to_points = compute_voxel_to_point_map<index_t>(
        sample_points_int, grid_size, stream);

    return render_backward<index_t, float_t, data_t>(
        voxel_offsets,
        voxel_indices,
        voxel_to_points,
        sample_points,
        gaussian_means,
        gaussian_icovs,
        gaussian_idets,
        gaussian_opacities,
        gaussian_semantics,
        out_semantics,
        out_occupancy,
        out_probability,
        out_semantics_grad,
        out_occupancy_grad,
        out_density_grad,
        stream);
}

} /* namespace detail */


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
aggregate_forward(
    at::Tensor const& sample_points,
    at::Tensor const& gaussian_means,
    at::Tensor const& gaussian_icovs,
    at::Tensor const& gaussian_idets,
    at::Tensor const& gaussian_opacities,
    at::Tensor const& gaussian_semantics,
    at::Tensor const& sample_points_int,
    at::Tensor const& gaussian_means_int,
    at::Tensor const& gaussian_radii,
    int64_t const h,
    int64_t const w,
    int64_t const d,
    double const default_value)
{
    CHECK_CUDA(sample_points);
    CHECK_CUDA(gaussian_means);
    CHECK_CUDA(gaussian_icovs);
    CHECK_CUDA(gaussian_idets);
    CHECK_CUDA(gaussian_opacities);
    CHECK_CUDA(gaussian_semantics);
    CHECK_CUDA(sample_points_int);
    CHECK_CUDA(gaussian_means_int);
    CHECK_CUDA(gaussian_radii);

    CHECK_CONTIGUOUS(sample_points);
    CHECK_CONTIGUOUS(gaussian_means);
    CHECK_CONTIGUOUS(gaussian_icovs);
    CHECK_CONTIGUOUS(gaussian_idets);
    CHECK_CONTIGUOUS(gaussian_opacities);
    CHECK_CONTIGUOUS(gaussian_semantics);
    CHECK_CONTIGUOUS(sample_points_int);
    CHECK_CONTIGUOUS(gaussian_means_int);
    CHECK_CONTIGUOUS(gaussian_radii);

    return VSPLAT3D_DISPATCH_INDEX_TYPES(sample_points_int.scalar_type(), "aggregate_forward", ([&] {
        using index_t = scalar_t;

        return VSPLAT3D_DISPATCH_FLOAT_TYPES(sample_points.scalar_type(), "aggregate_forward_float", ([&] {
            using float_t = scalar_t;

            return VSPLAT3D_DISPATCH_FLOAT_TYPES(gaussian_semantics.scalar_type(), "aggregate_forward_data", ([&] {
                using data_t = scalar_t;

                auto const grid_size = math::vec3<index_t>{
                    static_cast<index_t>(h),
                    static_cast<index_t>(w),
                    static_cast<index_t>(d),
                };

                return detail::aggregate_forward<index_t, float_t, data_t>(
                    sample_points,
                    gaussian_means,
                    gaussian_icovs,
                    gaussian_idets,
                    gaussian_opacities,
                    gaussian_semantics,
                    sample_points_int,
                    gaussian_means_int,
                    gaussian_radii,
                    grid_size,
                    static_cast<data_t>(default_value),
                    at::cuda::getCurrentCUDAStream());
            }));
        }));
    }));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
agregate_backward(
    at::Tensor const& voxel_offsets,
    at::Tensor const& voxel_indices,
    at::Tensor const& sample_points_int,
    at::Tensor const& sample_points,
    at::Tensor const& gaussian_means,
    at::Tensor const& gaussian_icovs,
    at::Tensor const& gaussian_idets,
    at::Tensor const& gaussian_opacities,
    at::Tensor const& gaussian_semantics,
    at::Tensor const& out_semantics,
    at::Tensor const& out_occupancy,
    at::Tensor const& out_probability,
    at::Tensor const& out_semantics_grad,
    at::Tensor const& out_occupancy_grad,
    at::Tensor const& out_density_grad,
    int64_t const h,
    int64_t const w,
    int64_t const d)
{
    CHECK_CUDA(voxel_offsets);
    CHECK_CUDA(voxel_indices);
    CHECK_CUDA(sample_points_int);
    CHECK_CUDA(sample_points);
    CHECK_CUDA(gaussian_means);
    CHECK_CUDA(gaussian_icovs);
    CHECK_CUDA(gaussian_idets);
    CHECK_CUDA(gaussian_opacities);
    CHECK_CUDA(gaussian_semantics);
    CHECK_CUDA(out_semantics);
    CHECK_CUDA(out_occupancy);
    CHECK_CUDA(out_probability);
    CHECK_CUDA(out_semantics_grad);
    CHECK_CUDA(out_occupancy_grad);
    CHECK_CUDA(out_density_grad);

    CHECK_CONTIGUOUS(voxel_offsets);
    CHECK_CONTIGUOUS(voxel_indices);
    CHECK_CONTIGUOUS(sample_points_int);
    CHECK_CONTIGUOUS(sample_points);
    CHECK_CONTIGUOUS(gaussian_means);
    CHECK_CONTIGUOUS(gaussian_icovs);
    CHECK_CONTIGUOUS(gaussian_idets);
    CHECK_CONTIGUOUS(gaussian_opacities);
    CHECK_CONTIGUOUS(gaussian_semantics);
    CHECK_CONTIGUOUS(out_semantics);
    CHECK_CONTIGUOUS(out_occupancy);
    CHECK_CONTIGUOUS(out_probability);
    CHECK_CONTIGUOUS(out_semantics_grad);
    CHECK_CONTIGUOUS(out_occupancy_grad);
    CHECK_CONTIGUOUS(out_density_grad);

    return VSPLAT3D_DISPATCH_INDEX_TYPES(voxel_offsets.scalar_type(), "aggregate_backward", ([&] {
        using index_t = scalar_t;

        return VSPLAT3D_DISPATCH_FLOAT_TYPES(gaussian_means.scalar_type(), "aggregate_backward_float", ([&] {
            using float_t = scalar_t;

            return VSPLAT3D_DISPATCH_FLOAT_TYPES(gaussian_semantics.scalar_type(), "aggregate_backward_data", ([&] {
                using data_t = scalar_t;

                auto const grid_size = math::vec3<index_t>{
                    static_cast<index_t>(h),
                    static_cast<index_t>(w),
                    static_cast<index_t>(d),
                };

                return detail::aggregate_backward<index_t, float_t, data_t>(
                    voxel_offsets,
                    voxel_indices,
                    sample_points_int,
                    sample_points,
                    gaussian_means,
                    gaussian_icovs,
                    gaussian_idets,
                    gaussian_opacities,
                    gaussian_semantics,
                    out_semantics,
                    out_occupancy,
                    out_probability,
                    out_semantics_grad,
                    out_occupancy_grad,
                    out_density_grad,
                    grid_size,
                    at::cuda::getCurrentCUDAStream());
            }));
        }));
    }));
}

} /* namespace vsplat3d::cuda */
