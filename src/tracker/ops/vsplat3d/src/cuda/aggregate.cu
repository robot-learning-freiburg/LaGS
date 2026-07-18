// Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
// SPDX-License-Identifier: AGPL-3.0-or-later

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <tuple>
#include <vector>

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

// Phase-level timing for the aggregation pipeline, enabled by setting the
// VSPLAT3D_PROFILE environment variable. Prints per-phase times to stdout.
struct PhaseProfiler {
    bool enabled;
    cudaStream_t stream;
    std::vector<std::pair<char const*, cudaEvent_t>> marks;

    explicit PhaseProfiler(cudaStream_t stream)
        : enabled(std::getenv("VSPLAT3D_PROFILE") != nullptr)
        , stream(stream)
    {
        mark("start");
    }

    void mark(char const* name) {
        if (!enabled)
            return;

        cudaEvent_t event;
        cudaEventCreate(&event);
        cudaEventRecord(event, stream);
        marks.emplace_back(name, event);
    }

    ~PhaseProfiler() {
        if (!enabled)
            return;

        cudaEventSynchronize(marks.back().second);

        for (size_t i = 1; i < marks.size(); i++) {
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, marks[i - 1].second, marks[i].second);
            printf("[vsplat3d] %-24s %8.3f ms\n", marks[i].first, ms);
        }

        for (auto& [name, event] : marks)
            cudaEventDestroy(event);
    }
};

namespace kernel {

constexpr float gauss_denom_pi = 0.06349363593424097;

// The aggregation is bricked: voxels are grouped into bricks of 4x4x4, and the
// gaussian-to-space assignment (pair list, sort, ranges) works on bricks rather than
// individual voxels. A gaussian covering ~9^3 voxels covers only ~3^3 bricks, which
// shrinks the pair list and the sort by more than an order of magnitude, and all voxels
// of a brick share one gaussian list, which the render kernel exploits by staging that
// list through shared memory. Per-voxel coverage is re-established with an exact
// splat-rect test, so results match a per-voxel pair list bit for bit.
constexpr int brick_edge = 4;
constexpr int brick_volume = brick_edge * brick_edge * brick_edge;

// Per-pair gradient components outside the semantic channels:
// 3 (mean) + 6 (icov) + 1 (idet) + 1 (opacity).
constexpr int num_geometry_grads = 11;

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
    math::vec3<T> const& p, math::vec3<T> const& radius, math::vec3<T> const& max)
{
    return math::rect3<T>{
        .min = math::vec3<T>{
            math::clamp(p.x - radius.x, T{0}, max.x),
            math::clamp(p.y - radius.y, T{0}, max.y),
            math::clamp(p.z - radius.z, T{0}, max.z),
        },
        .max = math::vec3<T>{
            math::clamp(p.x + radius.x + 1, T{0}, max.x),
            math::clamp(p.y + radius.y + 1, T{0}, max.y),
            math::clamp(p.z + radius.z + 1, T{0}, max.z),
        },
    };
}

// Brick range covered by a voxel rect. rect.max is exclusive, so the last covered
// voxel is rect.max - 1 and the + brick_edge - 1 rounds up to the exclusive brick
// bound.
template <typename T>
__forceinline__ __device__ math::rect3<T> get_brick_rect(math::rect3<T> const& rect)
{
    return math::rect3<T>{
        .min = math::vec3<T>{
            rect.min.x / brick_edge,
            rect.min.y / brick_edge,
            rect.min.z / brick_edge,
        },
        .max = math::vec3<T>{
            (rect.max.x + brick_edge - 1) / brick_edge,
            (rect.max.y + brick_edge - 1) / brick_edge,
            (rect.max.z + brick_edge - 1) / brick_edge,
        },
    };
}

// Whether a gaussian's splat rect covers the given in-grid voxel. For an in-grid voxel
// this is exactly containment in get_rect(cell, radius, grid): the clamps there only
// ever cut the rect down to the grid, which the voxel satisfies by assumption.
template <typename index_t>
__forceinline__ __device__ bool splat_covers(
    math::vec3<index_t> const& cell,
    math::vec3<index_t> const& radius,
    math::vec3<index_t> const& voxel)
{
    return voxel.x - cell.x <= radius.x && cell.x - voxel.x <= radius.x
        && voxel.y - cell.y <= radius.y && cell.y - voxel.y <= radius.y
        && voxel.z - cell.z <= radius.z && cell.z - voxel.z <= radius.z;
}

template <typename index_t>
__global__ void compute_gaussian_brick_counts(
    index_t const num_gaussians,
    index_t const* gaussian_cells,
    index_t const* radii,
    math::vec3<index_t> const grid,
    index_t *counts)
{
    CUDA_1D_KERNEL_LOOP(idx, num_gaussians) {
        auto const cell = math::vec3<index_t>::load(gaussian_cells + 3 * idx);
        auto const rect = get_brick_rect(
            get_rect(cell, math::vec3<index_t>::load(radii + 3 * idx), grid));

        counts[idx] = rect.size();
    }
}

template <typename index_t>
__global__ void compute_gaussian_brick_pairs(
    index_t const num_gaussians,
    index_t const* gaussian_cells,
    index_t const* radii,
    index_t const* gaussian_elements,
    index_t const* offsets,
    index_t* pair_gaussians,
    index_t* pair_bricks,
    math::vec3<index_t> const grid,
    math::vec3<index_t> const brick_grid,
    index_t const bricks_per_element)
{
    index_t const idx = ((index_t) blockIdx.x) * blockDim.x + threadIdx.x;
    index_t const stride = (blockDim.x * gridDim.x) / C10_WARP_SIZE;

    index_t const warp_id = idx / C10_WARP_SIZE;
    index_t const tid_in_warp = idx % C10_WARP_SIZE;

    for (index_t i = warp_id; i < num_gaussians; i += stride) {
        index_t const start = (i == 0) ? 0 : offsets[i - 1];
        index_t const end = offsets[i];

        auto const cell = math::vec3<index_t>::load(gaussian_cells + 3 * i);
        auto const rect = get_brick_rect(
            get_rect(cell, math::vec3<index_t>::load(radii + 3 * i), grid));

        // Batching folds the element index into the key's high bits, so pairs of
        // different elements never mix and each element keeps its emission order.
        auto const element_base = gaussian_elements[i] * bricks_per_element;

        for (index_t j = start + tid_in_warp; j < end; j += C10_WARP_SIZE) {
            auto const local_idx = j - start;

            auto const rect_size = rect.dim();

            auto const x_offset = local_idx / (rect_size.y * rect_size.z);
            auto const y_offset = (local_idx / rect_size.z) % rect_size.y;
            auto const z_offset = local_idx % rect_size.z;

            auto const x = rect.min.x + x_offset;
            auto const y = rect.min.y + y_offset;
            auto const z = rect.min.z + z_offset;

            auto const brick_index = x * brick_grid.y * brick_grid.z + y * brick_grid.z + z;

            pair_gaussians[j] = i;
            pair_bricks[j] = element_base + brick_index;
        }
    }
}

// Start/end ranges per key slot from a sorted key list (brick indices here).
template <typename index_t>
__global__ void compute_key_ranges(
    index_t const num_pairs,
    index_t const* keys,
    index_t* ranges)
{
    CUDA_1D_KERNEL_LOOP(idx, num_pairs) {
        index_t const idx_prev = idx > 0 ? idx - 1 : num_pairs - 1;

        index_t const this_key = keys[idx];
        index_t const prev_key = keys[idx_prev];

        if (this_key != prev_key || idx == 0) {
            ranges[this_key * 2 + 0] = idx;
            ranges[prev_key * 2 + 1] = idx_prev + 1;
        }
    }
}

template <typename index_t>
__global__ void compute_voxel_to_point_map(
    index_t const num_points,
    index_t const* points_xyz,
    index_t const* point_elements,
    math::vec3<index_t> const grid,
    index_t const voxels_per_element,
    index_t* voxel_to_point_index)
{
    CUDA_1D_KERNEL_LOOP(idx, num_points) {
        auto const point = math::vec3<index_t>::load(points_xyz + 3 * idx);

        index_t const voxel_idx = point.x * grid.y * grid.z + point.y * grid.z + point.z;
        voxel_to_point_index[point_elements[idx] * voxels_per_element + voxel_idx] = idx;
    }
}

template <typename index_t, typename float_t, typename data_t, int NUM_CHANNELS>
__global__ void render_forward_bricks_registers(
    index_t const num_keys,
    index_t const num_channels, // actual channel count (must be <= NUM_CHANNELS)
    float_t const* __restrict__ sample_points,
    float_t const* __restrict__ gaussian_means,
    float_t const* __restrict__ gaussian_icovs,
    float_t const* __restrict__ gaussian_idets,
    float_t const* __restrict__ gaussian_opacities,
    data_t const* __restrict__ gaussian_semantics,
    index_t const* __restrict__ gaussian_cells,
    index_t const* __restrict__ gaussian_radii,
    index_t const* __restrict__ brick_ranges,
    index_t const* __restrict__ pair_gaussians,
    index_t const* __restrict__ pair_order,
    index_t const* __restrict__ voxel_to_point,
    math::vec3<index_t> const grid_size,
    math::vec3<index_t> const brick_grid,
    index_t const bricks_per_element,
    index_t const voxels_per_element,
    data_t default_value,
    bool const opacity_weighted,
    data_t* __restrict__ out_semantics,
    float_t* __restrict__ out_occupancy,
    float_t* __restrict__ out_density,
    float_t* __restrict__ out_probability)
{
    // One thread block per brick, one thread per voxel. All voxels of a brick walk the
    // same gaussian list, staged through shared memory one block-sized chunk at a time.
    __shared__ float_t staged_mean[3][brick_volume];
    __shared__ float_t staged_icov[6][brick_volume];
    __shared__ float_t staged_idet[brick_volume];
    __shared__ float_t staged_opacity[brick_volume];
    __shared__ index_t staged_cell[3][brick_volume];
    __shared__ index_t staged_radius[3][brick_volume];
    __shared__ index_t staged_index[brick_volume];

    auto const tid = static_cast<index_t>(threadIdx.x);

    for (index_t key = blockIdx.x; key < num_keys; key += gridDim.x) {
        auto const element = key / bricks_per_element;
        auto const brick = key % bricks_per_element;

        auto const brick_x = brick / (brick_grid.y * brick_grid.z);
        auto const brick_y = (brick / brick_grid.z) % brick_grid.y;
        auto const brick_z = brick % brick_grid.z;

        auto const voxel = math::vec3<index_t>{
            brick_x * brick_edge + tid / (brick_edge * brick_edge),
            brick_y * brick_edge + (tid / brick_edge) % brick_edge,
            brick_z * brick_edge + tid % brick_edge,
        };

        // Edge bricks may overhang a grid whose extent is not a multiple of brick_edge.
        auto const in_grid = voxel.x < grid_size.x
            && voxel.y < grid_size.y
            && voxel.z < grid_size.z;

        index_t pts_idx = -1;
        if (in_grid) {
            auto const voxel_idx = voxel.x * grid_size.y * grid_size.z
                + voxel.y * grid_size.z
                + voxel.z;

            pts_idx = voxel_to_point[element * voxels_per_element + voxel_idx];
        }

        // Note: block-uniform result, so the continue cannot split the barriers below.
        if (__syncthreads_count(pts_idx >= 0) == 0)
            continue;

        auto const range_start = brick_ranges[key * 2 + 0];
        auto const range_end = brick_ranges[key * 2 + 1];

        auto const point = pts_idx >= 0
            ? math::vec3<float_t>::load(sample_points + 3 * pts_idx)
            : math::vec3<float_t>::zero();

        data_t semantics[NUM_CHANNELS];

        for (int j = 0; j < num_channels; j++) {
            semantics[j] = data_t{0};
        }

        float_t occupancy = 1.0;
        float_t density = 0.0;
        float_t prob_sum = 0.0;

        for (auto chunk = range_start; chunk < range_end; chunk += brick_volume) {
            if (chunk + tid < range_end) {
                auto const gs_idx = pair_gaussians[pair_order[chunk + tid]];

                staged_index[tid] = gs_idx;
                staged_idet[tid] = gaussian_idets[gs_idx];
                staged_opacity[tid] = gaussian_opacities[gs_idx];

                for (int k = 0; k < 3; k++) {
                    staged_mean[k][tid] = gaussian_means[3 * gs_idx + k];
                    staged_cell[k][tid] = gaussian_cells[3 * gs_idx + k];
                    staged_radius[k][tid] = gaussian_radii[3 * gs_idx + k];
                }

                for (int k = 0; k < 6; k++) {
                    staged_icov[k][tid] = gaussian_icovs[6 * gs_idx + k];
                }
            }
            __syncthreads();

            auto const num_left = range_end - chunk;
            auto const num_staged = num_left < static_cast<index_t>(brick_volume)
                ? num_left : static_cast<index_t>(brick_volume);

            if (pts_idx >= 0) {
                for (index_t j = 0; j < num_staged; j++) {
                    // The pair list is brick-level, so re-check per-voxel coverage.
                    auto const cell = math::vec3<index_t>{
                        staged_cell[0][j], staged_cell[1][j], staged_cell[2][j]};
                    auto const radius = math::vec3<index_t>{
                        staged_radius[0][j], staged_radius[1][j], staged_radius[2][j]};

                    if (!splat_covers(cell, radius, voxel))
                        continue;

                    auto const mean = math::vec3<float_t>{
                        staged_mean[0][j], staged_mean[1][j], staged_mean[2][j]};

                    // Field order matches smat3::load: a11, a22, a33, a12, a23, a13.
                    auto const cov = math::smat3<float_t>{
                        staged_icov[0][j], staged_icov[1][j], staged_icov[2][j],
                        staged_icov[3][j], staged_icov[4][j], staged_icov[5][j]};

                    auto const deter = staged_idet[j];
                    auto const opacity = staged_opacity[j];

                    auto const power = __expf(cov.neg_half_quadratic(mean - point));
                    auto const denom = static_cast<float_t>(gauss_denom_pi) * __fsqrt_rn(deter);
                    auto const prob = opacity * denom * power;

                    // Opacity-weighted geometry: alpha = a * exp(-1/2 (x-m)^T S^-1 (x-m)).
                    // Without the weighting, occupancy cannot see opacity at all, so the only
                    // way to suppress a gaussian is to move/shrink it.
                    auto const alpha = opacity_weighted ? opacity * power : power;

                    auto const sem_base = staged_index[j] * num_channels;

                    // unroll if small enough
                    if (num_channels <= 32) {
                        #pragma unroll
                        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                            if (ch < num_channels) {
                                semantics[ch] += gaussian_semantics[sem_base + ch] * static_cast<data_t>(prob);
                            }
                        }
                    } else {
                        for (int ch = 0; ch < num_channels; ch++) {
                            semantics[ch] += gaussian_semantics[sem_base + ch] * static_cast<data_t>(prob);
                        }
                    }

                    occupancy = (1 - alpha) * occupancy;
                    density = alpha + density;
                    prob_sum = prob + prob_sum;
                }
            }
            __syncthreads();
        }

        if (pts_idx >= 0) {
            auto const valid = prob_sum > epsilon<float_t>;
            auto const out_sem = out_semantics + pts_idx * num_channels;

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

            out_occupancy[pts_idx] = 1 - occupancy;
            out_density[pts_idx] = density;
            out_probability[pts_idx] = prob_sum;
        }
    }
}

// fallback kernel for very large channel counts
template <typename index_t, typename float_t, typename data_t>
__global__ void render_forward_bricks_global_memory(
    index_t const num_keys,
    index_t const num_channels,
    float_t const* __restrict__ sample_points,
    float_t const* __restrict__ gaussian_means,
    float_t const* __restrict__ gaussian_icovs,
    float_t const* __restrict__ gaussian_idets,
    float_t const* __restrict__ gaussian_opacities,
    data_t const* __restrict__ gaussian_semantics,
    index_t const* __restrict__ gaussian_cells,
    index_t const* __restrict__ gaussian_radii,
    index_t const* __restrict__ brick_ranges,
    index_t const* __restrict__ pair_gaussians,
    index_t const* __restrict__ pair_order,
    index_t const* __restrict__ voxel_to_point,
    math::vec3<index_t> const grid_size,
    math::vec3<index_t> const brick_grid,
    index_t const bricks_per_element,
    index_t const voxels_per_element,
    data_t default_value,
    bool const opacity_weighted,
    data_t* __restrict__ out_semantics,
    float_t* __restrict__ out_occupancy,
    float_t* __restrict__ out_density,
    float_t* __restrict__ out_probability)
{
    __shared__ float_t staged_mean[3][brick_volume];
    __shared__ float_t staged_icov[6][brick_volume];
    __shared__ float_t staged_idet[brick_volume];
    __shared__ float_t staged_opacity[brick_volume];
    __shared__ index_t staged_cell[3][brick_volume];
    __shared__ index_t staged_radius[3][brick_volume];
    __shared__ index_t staged_index[brick_volume];

    auto const tid = static_cast<index_t>(threadIdx.x);

    for (index_t key = blockIdx.x; key < num_keys; key += gridDim.x) {
        auto const element = key / bricks_per_element;
        auto const brick = key % bricks_per_element;

        auto const brick_x = brick / (brick_grid.y * brick_grid.z);
        auto const brick_y = (brick / brick_grid.z) % brick_grid.y;
        auto const brick_z = brick % brick_grid.z;

        auto const voxel = math::vec3<index_t>{
            brick_x * brick_edge + tid / (brick_edge * brick_edge),
            brick_y * brick_edge + (tid / brick_edge) % brick_edge,
            brick_z * brick_edge + tid % brick_edge,
        };

        auto const in_grid = voxel.x < grid_size.x
            && voxel.y < grid_size.y
            && voxel.z < grid_size.z;

        index_t pts_idx = -1;
        if (in_grid) {
            auto const voxel_idx = voxel.x * grid_size.y * grid_size.z
                + voxel.y * grid_size.z
                + voxel.z;

            pts_idx = voxel_to_point[element * voxels_per_element + voxel_idx];
        }

        // Note: block-uniform result, so the continue cannot split the barriers below.
        if (__syncthreads_count(pts_idx >= 0) == 0)
            continue;

        auto const range_start = brick_ranges[key * 2 + 0];
        auto const range_end = brick_ranges[key * 2 + 1];

        auto const point = pts_idx >= 0
            ? math::vec3<float_t>::load(sample_points + 3 * pts_idx)
            : math::vec3<float_t>::zero();

        auto const out_sem = pts_idx >= 0
            ? out_semantics + pts_idx * num_channels
            : nullptr;

        if (pts_idx >= 0) {
            for (int j = 0; j < num_channels; j++) {
                out_sem[j] = data_t{0};
            }
        }

        float_t occupancy = 1.0;
        float_t density = 0.0;
        float_t prob_sum = 0.0;

        for (auto chunk = range_start; chunk < range_end; chunk += brick_volume) {
            if (chunk + tid < range_end) {
                auto const gs_idx = pair_gaussians[pair_order[chunk + tid]];

                staged_index[tid] = gs_idx;
                staged_idet[tid] = gaussian_idets[gs_idx];
                staged_opacity[tid] = gaussian_opacities[gs_idx];

                for (int k = 0; k < 3; k++) {
                    staged_mean[k][tid] = gaussian_means[3 * gs_idx + k];
                    staged_cell[k][tid] = gaussian_cells[3 * gs_idx + k];
                    staged_radius[k][tid] = gaussian_radii[3 * gs_idx + k];
                }

                for (int k = 0; k < 6; k++) {
                    staged_icov[k][tid] = gaussian_icovs[6 * gs_idx + k];
                }
            }
            __syncthreads();

            auto const num_left = range_end - chunk;
            auto const num_staged = num_left < static_cast<index_t>(brick_volume)
                ? num_left : static_cast<index_t>(brick_volume);

            if (pts_idx >= 0) {
                for (index_t j = 0; j < num_staged; j++) {
                    auto const cell = math::vec3<index_t>{
                        staged_cell[0][j], staged_cell[1][j], staged_cell[2][j]};
                    auto const radius = math::vec3<index_t>{
                        staged_radius[0][j], staged_radius[1][j], staged_radius[2][j]};

                    if (!splat_covers(cell, radius, voxel))
                        continue;

                    auto const mean = math::vec3<float_t>{
                        staged_mean[0][j], staged_mean[1][j], staged_mean[2][j]};

                    auto const cov = math::smat3<float_t>{
                        staged_icov[0][j], staged_icov[1][j], staged_icov[2][j],
                        staged_icov[3][j], staged_icov[4][j], staged_icov[5][j]};

                    auto const deter = staged_idet[j];
                    auto const opacity = staged_opacity[j];

                    auto const power = __expf(cov.neg_half_quadratic(mean - point));
                    auto const denom = static_cast<float_t>(gauss_denom_pi) * __fsqrt_rn(deter);
                    auto const prob = opacity * denom * power;

                    // See render_forward_bricks_registers for the rationale.
                    auto const alpha = opacity_weighted ? opacity * power : power;

                    auto const sem_base = staged_index[j] * num_channels;

                    for (int ch = 0; ch < num_channels; ch++) {
                        out_sem[ch] += gaussian_semantics[sem_base + ch] * static_cast<data_t>(prob);
                    }

                    occupancy = (1 - alpha) * occupancy;
                    density = alpha + density;
                    prob_sum = prob + prob_sum;
                }
            }
            __syncthreads();
        }

        if (pts_idx >= 0) {
            auto const valid = prob_sum > epsilon<float_t>;

            for (int j = 0; j < num_channels; j++) {
                out_sem[j] = valid ? (out_sem[j] / static_cast<data_t>(prob_sum)) : default_value;
            }

            out_occupancy[pts_idx] = 1 - occupancy;
            out_density[pts_idx] = density;
            out_probability[pts_idx] = prob_sum;
        }
    }
}

// The brick kernels render at most one sample point per voxel (whichever one the
// voxel-to-point map retained). Points sharing a voxel with another point are rendered
// here instead, one thread per point; with the usual one-point-per-voxel sampling this
// kernel is a no-op scan. Uses global-memory semantics accumulation for any channel
// count, which is fine for a path this cold.
template <typename index_t, typename float_t, typename data_t>
__global__ void render_forward_remainder(
    index_t const num_samples,
    index_t const num_channels,
    float_t const* __restrict__ sample_points,
    index_t const* __restrict__ sample_points_int,
    index_t const* __restrict__ point_elements,
    float_t const* __restrict__ gaussian_means,
    float_t const* __restrict__ gaussian_icovs,
    float_t const* __restrict__ gaussian_idets,
    float_t const* __restrict__ gaussian_opacities,
    data_t const* __restrict__ gaussian_semantics,
    index_t const* __restrict__ gaussian_cells,
    index_t const* __restrict__ gaussian_radii,
    index_t const* __restrict__ brick_ranges,
    index_t const* __restrict__ pair_gaussians,
    index_t const* __restrict__ pair_order,
    index_t const* __restrict__ voxel_to_point,
    math::vec3<index_t> const grid_size,
    math::vec3<index_t> const brick_grid,
    index_t const bricks_per_element,
    index_t const voxels_per_element,
    data_t default_value,
    bool const opacity_weighted,
    data_t* __restrict__ out_semantics,
    float_t* __restrict__ out_occupancy,
    float_t* __restrict__ out_density,
    float_t* __restrict__ out_probability)
{
    CUDA_1D_KERNEL_LOOP(idx, num_samples) {
        auto const point_int = math::vec3<index_t>::load(sample_points_int + idx * 3);

        auto const element = point_elements[idx];

        auto const voxel_idx = point_int.x * grid_size.y * grid_size.z
            + point_int.y * grid_size.z
            + point_int.z;

        // Rendered by the brick kernel already.
        if (voxel_to_point[element * voxels_per_element + voxel_idx] == static_cast<index_t>(idx))
            continue;

        auto const key = element * bricks_per_element
            + (point_int.x / brick_edge) * brick_grid.y * brick_grid.z
            + (point_int.y / brick_edge) * brick_grid.z
            + (point_int.z / brick_edge);

        auto const range_start = brick_ranges[key * 2 + 0];
        auto const range_end = brick_ranges[key * 2 + 1];

        auto const point = math::vec3<float_t>::load(sample_points + 3 * idx);
        auto const out_sem = out_semantics + idx * num_channels;

        for (int j = 0; j < num_channels; j++) {
            out_sem[j] = data_t{0};
        }

        float_t occupancy = 1.0;
        float_t density = 0.0;
        float_t prob_sum = 0.0;

        for (auto i = range_start; i < range_end; i++)
        {
            auto const gs_idx = pair_gaussians[pair_order[i]];

            auto const cell = math::vec3<index_t>::load(gaussian_cells + 3 * gs_idx);

            if (!splat_covers(cell, math::vec3<index_t>::load(gaussian_radii + 3 * gs_idx), point_int))
                continue;

            auto const mean = math::vec3<float_t>::load(gaussian_means + gs_idx * 3);
            auto const cov = math::smat3<float_t>::load(gaussian_icovs + gs_idx * 6);
            auto const deter = gaussian_idets[gs_idx];
            auto const opacity = gaussian_opacities[gs_idx];

            auto const power = __expf(cov.neg_half_quadratic(mean - point));
            auto const denom = static_cast<float_t>(gauss_denom_pi) * __fsqrt_rn(deter);
            auto const prob = opacity * denom * power;

            // See render_forward_bricks_registers for the rationale.
            auto const alpha = opacity_weighted ? opacity * power : power;

            auto const sem_base = gs_idx * num_channels;

            for (int j = 0; j < num_channels; j++) {
                out_sem[j] += gaussian_semantics[sem_base + j] * static_cast<data_t>(prob);
            }

            occupancy = (1 - alpha) * occupancy;
            density = alpha + density;
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

// Pair-parallel backward: one thread per gaussian-brick pair, each walking the up to
// 4x4x4 voxels of its brick. One thread per gaussian is only ~9k threads at typical
// scales and starves the GPU; the pair list is 30-60x larger and fills it.
//
// Threads follow the brick-sorted pair order, so a warp works on one or two bricks and
// nearly all per-voxel loads (point map, sample point, outputs, output gradients) hit
// the same addresses across the warp. Gradients stay atomic-free: each pair writes its
// partial sums to its emission-order slot, and reduce_backward_partials sums the
// contiguous per-gaussian slices.
template <typename index_t, typename float_t, typename data_t, int NUM_CHANNELS>
__global__ void render_backward_bricks(
    index_t const num_pairs,
    index_t const num_channels, // actual channel count (must be <= NUM_CHANNELS)
    index_t const* __restrict__ pair_gaussians,
    index_t const* __restrict__ pair_bricks,  // sorted by brick
    index_t const* __restrict__ pair_order,   // sorted position -> emission-order slot
    index_t const* __restrict__ voxel_to_point,
    float_t const* __restrict__ sample_points,
    float_t const* __restrict__ gaussian_means,
    float_t const* __restrict__ gaussian_icovs,
    float_t const* __restrict__ gaussian_idets,
    float_t const* __restrict__ gaussian_opacities,
    data_t const* __restrict__ gaussian_semantics,
    index_t const* __restrict__ gaussian_cells,
    index_t const* __restrict__ gaussian_radii,
    data_t const* __restrict__ out_semantics,
    float_t const* __restrict__ out_occupancy,
    float_t const* __restrict__ out_probability,
    data_t const* __restrict__ out_semantics_grad,
    float_t const* __restrict__ out_occupancy_grad,
    float_t const* __restrict__ out_density_grad,
    math::vec3<index_t> const grid_size,
    math::vec3<index_t> const brick_grid,
    index_t const bricks_per_element,
    index_t const voxels_per_element,
    bool const opacity_weighted,
    float_t* __restrict__ partial_geometry,  // [num_pairs, num_geometry_grads]
    data_t* __restrict__ partial_semantics)  // [num_pairs, num_channels]
{
    CUDA_1D_KERNEL_LOOP(p, num_pairs) {
        auto const slot = pair_order[p];
        auto const idx = pair_gaussians[slot];

        auto const key = pair_bricks[p];
        auto const element = key / bricks_per_element;
        auto const brick = key % bricks_per_element;

        auto const mean = math::vec3<float_t>::load(gaussian_means + 3 * idx);
        auto const cov = math::smat3<float_t>::load(gaussian_icovs + 6 * idx);
        auto const deter = gaussian_idets[idx];
        auto const opa = gaussian_opacities[idx];

        auto const cell = math::vec3<index_t>::load(gaussian_cells + 3 * idx);
        auto const radius = math::vec3<index_t>::load(gaussian_radii + 3 * idx);

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

        auto const brick_x = brick / (brick_grid.y * brick_grid.z);
        auto const brick_y = (brick / brick_grid.z) % brick_grid.y;
        auto const brick_z = brick % brick_grid.z;

        for (int v = 0; v < brick_volume; v++)
        {
            auto const voxel = math::vec3<index_t>{
                brick_x * brick_edge + static_cast<index_t>(v / (brick_edge * brick_edge)),
                brick_y * brick_edge + static_cast<index_t>((v / brick_edge) % brick_edge),
                brick_z * brick_edge + static_cast<index_t>(v % brick_edge),
            };

            if (voxel.x >= grid_size.x || voxel.y >= grid_size.y || voxel.z >= grid_size.z)
                continue;

            // The pair list is brick-level, so re-check per-voxel coverage.
            if (!splat_covers(cell, radius, voxel))
                continue;

            auto const voxel_idx = voxel.x * grid_size.y * grid_size.z
                + voxel.y * grid_size.z
                + voxel.z;

            auto const pts_idx = voxel_to_point[element * voxels_per_element + voxel_idx];
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
                // Note: prob here is denom * power, without opacity; the semantic path folds
                // opacity in separately via prob_opa_inv / opa_inv / prob_inv above.
                power_grad += prob_grad * denom;

                // Opacity-weighted geometry: alpha = a * power. Occupancy and density both
                // act on alpha, and alpha depends on both power and opacity, so the chain
                // rule forks here.
                //
                // This sits outside the prob_sum guard above on purpose: occupancy does not
                // depend on prob_sum, so a gaussian over a point with no probability mass
                // must still receive occupancy gradient.
                auto const alpha = opacity_weighted ? opa * power : power;

                float_t alpha_grad = 0.;
                alpha_grad += (1 - out_occupancy[pts_idx]) / (1 - alpha + epsilon<float_t>)
                    * out_occupancy_grad[pts_idx];
                alpha_grad += out_density_grad[pts_idx];

                if (opacity_weighted) {
                    power_grad += alpha_grad * opa;     // d(alpha)/d(power) = a
                    opa_grad += alpha_grad * power;     // d(alpha)/d(a) = power
                } else {
                    power_grad += alpha_grad;
                }

                deter_grad += prob_grad * prob / 2 / deter;
                idets_grad += deter_grad;

                auto const power_grad_times_power = power_grad * power;
                means_grad[0] -= power_grad_times_power * (cov.a11 * d.x + cov.a12 * d.y + cov.a13 * d.z);
                means_grad[1] -= power_grad_times_power * (cov.a12 * d.x + cov.a22 * d.y + cov.a23 * d.z);
                means_grad[2] -= power_grad_times_power * (cov.a13 * d.x + cov.a23 * d.y + cov.a33 * d.z);

                // Note: the forward takes the determinant as the separate gaussian_idets
                // input and never computes det(cov), so d(deter)/d(cov) = 0 and there is no
                // determinant term here. It reaches the caller solely via idets_grad.
                //
                // The factor of 2 on the off-diagonals comes from the quadratic form:
                // q = dx^2 a11 + dy^2 a22 + dz^2 a33 + 2 (dx dy a12 + dy dz a23 + dz dx a13).
                cov_grad[0] += power_grad_times_power * (-0.5 * d.x * d.x);
                cov_grad[1] += power_grad_times_power * (-0.5 * d.y * d.y);
                cov_grad[2] += power_grad_times_power * (-0.5 * d.z * d.z);
                cov_grad[3] += power_grad_times_power * (-d.x * d.y);
                cov_grad[4] += power_grad_times_power * (-d.y * d.z);
                cov_grad[5] += power_grad_times_power * (-d.x * d.z);
            }
        }

        auto const p_geom = partial_geometry + slot * num_geometry_grads;
        p_geom[0] = means_grad[0];
        p_geom[1] = means_grad[1];
        p_geom[2] = means_grad[2];
        p_geom[3] = cov_grad[0];
        p_geom[4] = cov_grad[1];
        p_geom[5] = cov_grad[2];
        p_geom[6] = cov_grad[3];
        p_geom[7] = cov_grad[4];
        p_geom[8] = cov_grad[5];
        p_geom[9] = idets_grad;
        p_geom[10] = opa_grad;

        auto const p_sem = partial_semantics + slot * num_channels;

        if (num_channels <= 32) {
            #pragma unroll
            for (int j = 0; j < NUM_CHANNELS; j++) {
                if (j < num_channels) {
                    p_sem[j] = semantic_grad[j];
                }
            }
        } else {
            for (int j = 0; j < num_channels; j++) {
                p_sem[j] = semantic_grad[j];
            }
        }
    }
}

// Sums the per-pair partial gradients into per-gaussian gradients. One thread per
// (gaussian, component); the pair list is grouped by gaussian in emission order, so
// each gaussian's partials form the contiguous slice given by pair_offsets.
template <typename index_t, typename float_t, typename data_t>
__global__ void reduce_backward_partials(
    index_t const num_gaussians,
    index_t const num_channels,
    index_t const* __restrict__ pair_offsets,
    float_t const* __restrict__ partial_geometry,
    data_t const* __restrict__ partial_semantics,
    float_t* __restrict__ g_means_grad,
    float_t* __restrict__ g_icov_grad,
    float_t* __restrict__ g_idets_grad,
    float_t* __restrict__ g_opacities_grad,
    data_t* __restrict__ g_semantics_grad)
{
    auto const num_components = num_geometry_grads + num_channels;

    CUDA_1D_KERNEL_LOOP(i, static_cast<size_t>(num_gaussians) * num_components) {
        index_t const idx = i / num_components;
        index_t const comp = i % num_components;

        index_t const start = (idx == 0) ? 0 : pair_offsets[idx - 1];
        index_t const end = pair_offsets[idx];

        if (comp < num_geometry_grads) {
            float_t acc = 0;

            for (auto p = start; p < end; p++)
                acc += partial_geometry[p * num_geometry_grads + comp];

            if (comp < 3)
                g_means_grad[idx * 3 + comp] = acc;
            else if (comp < 9)
                g_icov_grad[idx * 6 + (comp - 3)] = acc;
            else if (comp == 9)
                g_idets_grad[idx] = acc;
            else
                g_opacities_grad[idx] = acc;
        } else {
            auto const ch = comp - num_geometry_grads;
            data_t acc{0};

            for (auto p = start; p < end; p++)
                acc += partial_semantics[p * num_channels + ch];

            g_semantics_grad[idx * num_channels + ch] = acc;
        }
    }
}

// Warp-per-gaussian backward for channel counts too large for the pair-parallel
// kernel's partial buffers. The 32 lanes of each warp split the channel dimension
// (lane l owns channels l, l + 32, ...), so the per-channel accumulators stay in
// registers for any templated channel count, where a one-thread-per-gaussian version
// would spill them to local memory. The cross-channel sums feeding prob_grad and
// opa_grad become warp reductions; the rest of the per-voxel math is computed
// redundantly -- and identically -- in every lane, so all lane-uniform branches stay
// warp-uniform and the reductions are safe. One warp per gaussian also runs 32x the
// threads of one thread per gaussian, which at typical scales (~9k gaussians) is the
// difference between starving the GPU and filling it.
template <typename index_t, typename float_t, typename data_t, int NUM_CHANNELS>
__global__ void render_backward_warps(
    index_t const num_gaussians,
    index_t const num_channels, // actual channel count (must be <= NUM_CHANNELS)
    index_t const* __restrict__ gaussian_cells,
    index_t const* __restrict__ gaussian_radii,
    index_t const* __restrict__ gaussian_elements,
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
    math::vec3<index_t> const grid_size,
    index_t const voxels_per_element,
    bool const opacity_weighted,
    float_t* __restrict__ g_means_grad,
    float_t* __restrict__ g_icov_grad,
    float_t* __restrict__ g_idets_grad,
    float_t* __restrict__ g_opacities_grad,
    data_t* __restrict__ g_semantics_grad)
{
    constexpr int channels_per_lane = (NUM_CHANNELS + C10_WARP_SIZE - 1) / C10_WARP_SIZE;

    auto const warp = cooperative_groups::tiled_partition<C10_WARP_SIZE>(
        cooperative_groups::this_thread_block());

    index_t const thread_id = ((index_t) blockIdx.x) * blockDim.x + threadIdx.x;
    index_t const num_warps = (blockDim.x * gridDim.x) / C10_WARP_SIZE;
    index_t const lane = thread_id % C10_WARP_SIZE;

    for (index_t idx = thread_id / C10_WARP_SIZE; idx < num_gaussians; idx += num_warps) {
        auto const mean = math::vec3<float_t>::load(gaussian_means + 3 * idx);
        auto const cov = math::smat3<float_t>::load(gaussian_icovs + 6 * idx);
        auto const deter = gaussian_idets[idx];
        auto const opa = gaussian_opacities[idx];

        auto const cell = math::vec3<index_t>::load(gaussian_cells + 3 * idx);
        auto const rect = get_rect(
            cell, math::vec3<index_t>::load(gaussian_radii + 3 * idx), grid_size);

        auto const element = gaussian_elements[idx];

        auto const sem_base = idx * num_channels;

        data_t sem[channels_per_lane];
        data_t semantic_grad[channels_per_lane];

        #pragma unroll
        for (int k = 0; k < channels_per_lane; k++)
        {
            index_t const ch = lane + k * C10_WARP_SIZE;
            sem[k] = ch < num_channels ? gaussian_semantics[sem_base + ch] : data_t{0};
            semantic_grad[k] = data_t{0};
        }

        float_t means_grad[3] = {0};
        float_t opa_grad = 0;
        float_t cov_grad[6] = {0};
        float_t idets_grad = 0;

        for (auto x = rect.min.x; x < rect.max.x; x++)
        for (auto y = rect.min.y; y < rect.max.y; y++)
        for (auto z = rect.min.z; z < rect.max.z; z++)
        {
            auto const voxel_idx = x * grid_size.y * grid_size.z + y * grid_size.z + z;
            auto const pts_idx = voxel_to_point[element * voxels_per_element + voxel_idx];
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

                    float_t lane_sum = 0.;

                    #pragma unroll
                    for (int k = 0; k < channels_per_lane; k++)
                    {
                        index_t const ch = lane + k * C10_WARP_SIZE;
                        if (ch < num_channels) {
                            data_t const logit_grad_ch = out_semantics_grad[out_sem_base + ch];
                            data_t const sem_diff = sem[k] - out_semantics[out_sem_base + ch];

                            semantic_grad[k] += logit_grad_ch * static_cast<data_t>(prob_opa_inv);
                            lane_sum += static_cast<float_t>(logit_grad_ch * sem_diff);
                        }
                    }

                    // Sum of logit_grad * sem_diff over all channels; identical on
                    // every lane afterwards, so prob_grad and opa_grad stay uniform.
                    float_t const sem_diff_sum = cooperative_groups::reduce(
                        warp, lane_sum, cooperative_groups::plus<float_t>());

                    prob_grad += sem_diff_sum * opa_inv;
                    opa_grad += sem_diff_sum * prob_inv;
                }
                // Note: prob here is denom * power, without opacity; the semantic path folds
                // opacity in separately via prob_opa_inv / opa_inv / prob_inv above.
                power_grad += prob_grad * denom;

                // Opacity-weighted geometry: alpha = a * power. Occupancy and density both
                // act on alpha, and alpha depends on both power and opacity, so the chain
                // rule forks here.
                //
                // This sits outside the prob_sum guard above on purpose: occupancy does not
                // depend on prob_sum, so a gaussian over a point with no probability mass
                // must still receive occupancy gradient.
                auto const alpha = opacity_weighted ? opa * power : power;

                float_t alpha_grad = 0.;
                alpha_grad += (1 - out_occupancy[pts_idx]) / (1 - alpha + epsilon<float_t>)
                    * out_occupancy_grad[pts_idx];
                alpha_grad += out_density_grad[pts_idx];

                if (opacity_weighted) {
                    power_grad += alpha_grad * opa;     // d(alpha)/d(power) = a
                    opa_grad += alpha_grad * power;     // d(alpha)/d(a) = power
                } else {
                    power_grad += alpha_grad;
                }

                deter_grad += prob_grad * prob / 2 / deter;
                idets_grad += deter_grad;

                auto const power_grad_times_power = power_grad * power;
                means_grad[0] -= power_grad_times_power * (cov.a11 * d.x + cov.a12 * d.y + cov.a13 * d.z);
                means_grad[1] -= power_grad_times_power * (cov.a12 * d.x + cov.a22 * d.y + cov.a23 * d.z);
                means_grad[2] -= power_grad_times_power * (cov.a13 * d.x + cov.a23 * d.y + cov.a33 * d.z);

                // Note: the forward takes the determinant as the separate gaussian_idets
                // input and never computes det(cov), so d(deter)/d(cov) = 0 and there is no
                // determinant term here. It reaches the caller solely via idets_grad.
                //
                // The factor of 2 on the off-diagonals comes from the quadratic form:
                // q = dx^2 a11 + dy^2 a22 + dz^2 a33 + 2 (dx dy a12 + dy dz a23 + dz dx a13).
                cov_grad[0] += power_grad_times_power * (-0.5 * d.x * d.x);
                cov_grad[1] += power_grad_times_power * (-0.5 * d.y * d.y);
                cov_grad[2] += power_grad_times_power * (-0.5 * d.z * d.z);
                cov_grad[3] += power_grad_times_power * (-d.x * d.y);
                cov_grad[4] += power_grad_times_power * (-d.y * d.z);
                cov_grad[5] += power_grad_times_power * (-d.x * d.z);
            }
        }

        // The geometry gradients were accumulated identically in every lane; the
        // semantic gradients are partitioned, so each lane writes its own channels.
        if (lane == 0) {
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

        auto const p_sem_grad = g_semantics_grad + idx * num_channels;

        #pragma unroll
        for (int k = 0; k < channels_per_lane; k++)
        {
            index_t const ch = lane + k * C10_WARP_SIZE;
            if (ch < num_channels) {
                p_sem_grad[ch] = semantic_grad[k];
            }
        }
    }
}

// fallback kernel using global memory for very large channel counts
template <typename index_t, typename float_t, typename data_t>
__global__ void render_backward_global_memory(
    index_t const num_gaussians,
    index_t const num_channels,
    index_t const* __restrict__ gaussian_cells,
    index_t const* __restrict__ gaussian_radii,
    index_t const* __restrict__ gaussian_elements,
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
    math::vec3<index_t> const grid_size,
    index_t const voxels_per_element,
    bool const opacity_weighted,
    float_t* __restrict__ g_means_grad,
    float_t* __restrict__ g_icov_grad,
    float_t* __restrict__ g_idets_grad,
    float_t* __restrict__ g_opacities_grad,
    data_t* __restrict__ g_semantics_grad)
{
    CUDA_1D_KERNEL_LOOP(idx, num_gaussians) {
        auto const mean = math::vec3<float_t>::load(gaussian_means + 3 * idx);
        auto const cov = math::smat3<float_t>::load(gaussian_icovs + 6 * idx);
        auto const deter = gaussian_idets[idx];
        auto const opa = gaussian_opacities[idx];

        auto const cell = math::vec3<index_t>::load(gaussian_cells + 3 * idx);
        auto const rect = get_rect(
            cell, math::vec3<index_t>::load(gaussian_radii + 3 * idx), grid_size);

        auto const element = gaussian_elements[idx];

        auto const sem_base = idx * num_channels;
        auto const p_sem_grad = g_semantics_grad + idx * num_channels;

        for (int j = 0; j < num_channels; j++) {
            p_sem_grad[j] = data_t{0};
        }

        float_t means_grad[3] = {0};
        float_t opa_grad = 0;
        float_t cov_grad[6] = {0};
        float_t idets_grad = 0;

        for (auto x = rect.min.x; x < rect.max.x; x++)
        for (auto y = rect.min.y; y < rect.max.y; y++)
        for (auto z = rect.min.z; z < rect.max.z; z++)
        {
            auto const voxel_idx = x * grid_size.y * grid_size.z + y * grid_size.z + z;
            auto const pts_idx = voxel_to_point[element * voxels_per_element + voxel_idx];
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
                // Note: prob here is denom * power, without opacity; the semantic path folds
                // opacity in separately via prob_opa_inv / opa_inv / prob_inv above.
                power_grad += prob_grad * denom;

                // Opacity-weighted geometry: alpha = a * power. Occupancy and density both
                // act on alpha, and alpha depends on both power and opacity, so the chain
                // rule forks here.
                //
                // This sits outside the prob_sum guard above on purpose: occupancy does not
                // depend on prob_sum, so a gaussian over a point with no probability mass
                // must still receive occupancy gradient.
                auto const alpha = opacity_weighted ? opa * power : power;

                float_t alpha_grad = 0.;
                alpha_grad += (1 - out_occupancy[pts_idx]) / (1 - alpha + epsilon<float_t>)
                    * out_occupancy_grad[pts_idx];
                alpha_grad += out_density_grad[pts_idx];

                if (opacity_weighted) {
                    power_grad += alpha_grad * opa;     // d(alpha)/d(power) = a
                    opa_grad += alpha_grad * power;     // d(alpha)/d(a) = power
                } else {
                    power_grad += alpha_grad;
                }

                deter_grad += prob_grad * prob / 2 / deter;
                idets_grad += deter_grad;

                auto const power_grad_times_power = power_grad * power;
                means_grad[0] -= power_grad_times_power * (cov.a11 * d.x + cov.a12 * d.y + cov.a13 * d.z);
                means_grad[1] -= power_grad_times_power * (cov.a12 * d.x + cov.a22 * d.y + cov.a23 * d.z);
                means_grad[2] -= power_grad_times_power * (cov.a13 * d.x + cov.a23 * d.y + cov.a33 * d.z);

                // Note: the forward takes the determinant as the separate gaussian_idets
                // input and never computes det(cov), so d(deter)/d(cov) = 0 and there is no
                // determinant term here. It reaches the caller solely via idets_grad.
                //
                // The factor of 2 on the off-diagonals comes from the quadratic form:
                // q = dx^2 a11 + dy^2 a22 + dz^2 a33 + 2 (dx dy a12 + dy dz a23 + dz dx a13).
                cov_grad[0] += power_grad_times_power * (-0.5 * d.x * d.x);
                cov_grad[1] += power_grad_times_power * (-0.5 * d.y * d.y);
                cov_grad[2] += power_grad_times_power * (-0.5 * d.z * d.z);
                cov_grad[3] += power_grad_times_power * (-d.x * d.y);
                cov_grad[4] += power_grad_times_power * (-d.y * d.z);
                cov_grad[5] += power_grad_times_power * (-d.x * d.z);
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

// The pair-parallel backward keeps per-pair partial gradients, whose footprint scales
// with both the pair count and the channel count (~72 MB at 9k gaussians and 64
// channels); beyond this limit the warp-per-gaussian backward runs instead, which
// needs no partial buffers, and past the largest templated count the
// one-thread-per-gaussian global-memory fallback. Measured on an RTX 4060 Ti, the
// pair path was still 1.4x the warp path at 128 channels, but doubling this limit
// again was not worth doubling the buffers.
constexpr int pair_backward_channel_limit = 64;

using PairBackwardChannels = ChannelList<8, 16, 17, 18, 32, 64>;
using LargeBackwardChannels = ChannelList<128, 256>;

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
        kernel::render_forward_bricks_registers<index_t, float_t, data_t, N><<<blocks, threads, 0, stream>>>(
            std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void launch_global(dim3 blocks, dim3 threads, c10::cuda::CUDAStream const& stream, Args&&... args) {
        kernel::render_forward_bricks_global_memory<index_t, float_t, data_t><<<blocks, threads, 0, stream>>>(
            std::forward<Args>(args)...);
    }
};

// Kernel wrapper for the pair-parallel backward pass (small channel counts only; the
// dispatch below guards with pair_backward_channel_limit, so the global fallback is
// unreachable).
template<typename index_t, typename float_t, typename data_t>
struct PairBackwardKernelWrapper {
    template<int N, typename... Args>
    static void launch_register(dim3 blocks, dim3 threads, c10::cuda::CUDAStream const& stream, Args&&... args) {
        kernel::render_backward_bricks<index_t, float_t, data_t, N><<<blocks, threads, 0, stream>>>(
            std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void launch_global(dim3, dim3, c10::cuda::CUDAStream const&, Args&&...) {
        TORCH_INTERNAL_ASSERT(false, "pair-parallel backward dispatched beyond its channel limit");
    }
};

// Kernel wrapper for the warp-per-gaussian backward pass (the dispatch below guards
// with last<LargeBackwardChannels>(), so the global fallback is unreachable).
template<typename index_t, typename float_t, typename data_t>
struct WarpBackwardKernelWrapper {
    template<int N, typename... Args>
    static void launch_register(dim3 blocks, dim3 threads, c10::cuda::CUDAStream const& stream, Args&&... args) {
        kernel::render_backward_warps<index_t, float_t, data_t, N><<<blocks, threads, 0, stream>>>(
            std::forward<Args>(args)...);
    }

    template<typename... Args>
    static void launch_global(dim3, dim3, c10::cuda::CUDAStream const&, Args&&...) {
        TORCH_INTERNAL_ASSERT(false, "warp backward dispatched beyond its channel limit");
    }
};


template <typename index_t>
inline math::vec3<index_t> get_brick_grid(math::vec3<index_t> const& grid_size)
{
    return math::vec3<index_t>{
        (grid_size.x + kernel::brick_edge - 1) / kernel::brick_edge,
        (grid_size.y + kernel::brick_edge - 1) / kernel::brick_edge,
        (grid_size.z + kernel::brick_edge - 1) / kernel::brick_edge,
    };
}

template <typename index_t>
inline at::Tensor compute_gaussian_brick_counts(
    at::Tensor const& gaussian_cells,
    at::Tensor const& radii,
    math::vec3<index_t> const& grid_size,
    c10::cuda::CUDAStream const& stream)
{
    auto const opts = at::TensorOptions()
            .device(gaussian_cells.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto const num_gaussians = static_cast<index_t>(gaussian_cells.size(0));
    auto counts = at::empty({num_gaussians}, opts);

    auto const threads = 512;
    auto const blocks = (num_gaussians + threads - 1) / threads;

    kernel::compute_gaussian_brick_counts<<<blocks, threads, 0, stream>>>(
        num_gaussians,
        gaussian_cells.template const_data_ptr<index_t>(),
        radii.template const_data_ptr<index_t>(),
        grid_size,
        counts.template mutable_data_ptr<index_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return counts;
}

template <typename index_t>
inline at::Tensor compute_gaussian_brick_offsets(
    at::Tensor const& gaussian_cells,
    at::Tensor const& radii,
    math::vec3<index_t> const& grid_size,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_gaussians = static_cast<index_t>(gaussian_cells.size(0));

    auto const counts = compute_gaussian_brick_counts<index_t>(
        gaussian_cells, radii, grid_size, stream);

    auto offsets = at::empty_like(counts);

    cub::inclusive_sum(
        counts.template const_data_ptr<index_t>(),
        offsets.template mutable_data_ptr<index_t>(),
        num_gaussians, stream);

    return offsets;
}

template <typename index_t>
inline std::tuple<at::Tensor, at::Tensor>
compute_gaussian_brick_pairs(
    at::Tensor const& gaussian_cells,
    at::Tensor const& radii,
    at::Tensor const& gaussian_elements,
    at::Tensor const& offsets,
    math::vec3<index_t> const& grid_size,
    math::vec3<index_t> const& brick_grid,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_gaussians = static_cast<index_t>(gaussian_cells.size(0));

    auto const num_pairs = utils::mem::load(
        (offsets.template const_data_ptr<index_t>()) + num_gaussians - 1, stream, false);

    auto const opts = at::TensorOptions()
            .device(gaussian_cells.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto pair_gaussians = at::empty({num_pairs}, opts);
    auto pair_bricks = at::empty({num_pairs}, opts);

    auto const threads = 512;
    auto const warps = threads / at::cuda::warp_size();
    auto const blocks = std::min<int64_t>((num_gaussians + warps - 1) / warps, 2048L);

    kernel::compute_gaussian_brick_pairs<<<blocks, threads, 0, stream>>>(
        num_gaussians,
        gaussian_cells.template const_data_ptr<index_t>(),
        radii.template const_data_ptr<index_t>(),
        gaussian_elements.template const_data_ptr<index_t>(),
        offsets.template const_data_ptr<index_t>(),
        pair_gaussians.template mutable_data_ptr<index_t>(),
        pair_bricks.template mutable_data_ptr<index_t>(),
        grid_size,
        brick_grid,
        prod(brick_grid));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return {pair_gaussians, pair_bricks};
}

// Sorts the pair list by brick. The sorted values are the pairs' positions in the
// emission-ordered list ("slots") rather than the gaussian indices themselves: the
// backward walks pairs in brick order for memory locality but writes each pair's
// partial gradients to its slot, which keeps the per-gaussian slices contiguous for
// the reduction. The sort is stable and pairs are emitted in gaussian order, so slots
// ascend within each brick and per-voxel accumulation order is unchanged.
template <typename index_t>
inline std::tuple<at::Tensor, at::Tensor>
sort_gaussian_brick_pairs(
    at::Tensor const& pair_bricks,
    index_t const num_keys,
    c10::cuda::CUDAStream const& stream)
{
    auto const opts = at::TensorOptions()
            .device(pair_bricks.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto const slots = at::arange(pair_bricks.size(0), opts);

    auto sorted_bricks = at::empty_like(pair_bricks);
    auto sorted_slots = at::empty_like(slots);

    // Keys (element index x bricks + brick index) are dense in [0, num_keys), so the
    // sort only needs to look at the bits that can actually be set.
    int end_bit = 1;
    while ((int64_t{1} << end_bit) < static_cast<int64_t>(num_keys))
        end_bit++;

    cub::radix_sort_pairs(
        pair_bricks.template const_data_ptr<index_t>(),
        sorted_bricks.template mutable_data_ptr<index_t>(),
        slots.template const_data_ptr<index_t>(),
        sorted_slots.template mutable_data_ptr<index_t>(),
        pair_bricks.size(0),
        0, end_bit,
        stream);

    return {sorted_bricks, sorted_slots};
}

template <typename index_t>
inline at::Tensor compute_key_ranges(
    at::Tensor const& keys,
    index_t const num_slots,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_pairs = static_cast<index_t>(keys.size(0));

    auto const opts = at::TensorOptions()
            .device(keys.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto ranges = at::zeros({num_slots, 2}, opts);

    if (num_pairs == 0)
        return ranges;

    auto const threads = 512;
    auto const blocks = (num_pairs + threads - 1) / threads;

    kernel::compute_key_ranges<<<blocks, threads, 0, stream>>>(
        num_pairs,
        keys.template const_data_ptr<index_t>(),
        ranges.template mutable_data_ptr<index_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return ranges;
}

template <typename index_t>
inline at::Tensor compute_voxel_to_point_map(
    at::Tensor const& points_xyz,
    at::Tensor const& point_elements,
    index_t const num_elements,
    math::vec3<index_t> const& grid_size,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_points = static_cast<index_t>(points_xyz.size(0));

    auto const opts = at::TensorOptions()
            .device(points_xyz.device())
            .dtype(c10::CppTypeToScalarType<index_t>());

    auto indices = at::full({num_elements * prod(grid_size)}, -1, opts);

    auto const threads = 512;
    auto const blocks = (num_points + threads - 1) / threads;

    kernel::compute_voxel_to_point_map<<<blocks, threads, 0, stream>>>(
        num_points,
        points_xyz.template const_data_ptr<index_t>(),
        point_elements.template const_data_ptr<index_t>(),
        grid_size,
        prod(grid_size),
        indices.template mutable_data_ptr<index_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return indices;
}

template <typename index_t, typename float_t, typename data_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
render_forward(
    at::Tensor const& sample_points,
    at::Tensor const& sample_points_int,
    at::Tensor const& point_elements,
    at::Tensor const& gaussian_means,
    at::Tensor const& gaussian_icovs,
    at::Tensor const& gaussian_idets,
    at::Tensor const& gaussian_opacities,
    at::Tensor const& gaussian_semantics,
    at::Tensor const& gaussian_cells,
    at::Tensor const& gaussian_radii,
    at::Tensor const& brick_ranges,
    at::Tensor const& pair_gaussians,
    at::Tensor const& pair_order,
    at::Tensor const& voxel_to_point,
    math::vec3<index_t> const& grid_size,
    math::vec3<index_t> const& brick_grid,
    index_t const num_keys,
    data_t default_value,
    bool const opacity_weighted,
    PhaseProfiler& profiler,
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

    dispatch_kernel<ForwardKernelWrapper<index_t, float_t, data_t>, CommonChannels>(
        num_channels, num_keys, kernel::brick_volume, stream,
        num_keys,
        num_channels,
        sample_points.template const_data_ptr<float_t>(),
        gaussian_means.template const_data_ptr<float_t>(),
        gaussian_icovs.template const_data_ptr<float_t>(),
        gaussian_idets.template const_data_ptr<float_t>(),
        gaussian_opacities.template const_data_ptr<float_t>(),
        gaussian_semantics.template const_data_ptr<data_t>(),
        gaussian_cells.template const_data_ptr<index_t>(),
        gaussian_radii.template const_data_ptr<index_t>(),
        brick_ranges.template const_data_ptr<index_t>(),
        pair_gaussians.template const_data_ptr<index_t>(),
        pair_order.template const_data_ptr<index_t>(),
        voxel_to_point.template const_data_ptr<index_t>(),
        grid_size,
        brick_grid,
        prod(brick_grid),
        prod(grid_size),
        default_value,
        opacity_weighted,
        out_semantics.template mutable_data_ptr<data_t>(),
        out_occupancy.template mutable_data_ptr<float_t>(),
        out_density.template mutable_data_ptr<float_t>(),
        out_probability.template mutable_data_ptr<float_t>());

    profiler.mark("render");

    {
        auto const threads = 512;
        auto const blocks = (num_samples + threads - 1) / threads;

        kernel::render_forward_remainder<index_t, float_t, data_t><<<blocks, threads, 0, stream>>>(
            num_samples,
            num_channels,
            sample_points.template const_data_ptr<float_t>(),
            sample_points_int.template const_data_ptr<index_t>(),
            point_elements.template const_data_ptr<index_t>(),
            gaussian_means.template const_data_ptr<float_t>(),
            gaussian_icovs.template const_data_ptr<float_t>(),
            gaussian_idets.template const_data_ptr<float_t>(),
            gaussian_opacities.template const_data_ptr<float_t>(),
            gaussian_semantics.template const_data_ptr<data_t>(),
            gaussian_cells.template const_data_ptr<index_t>(),
            gaussian_radii.template const_data_ptr<index_t>(),
            brick_ranges.template const_data_ptr<index_t>(),
            pair_gaussians.template const_data_ptr<index_t>(),
            pair_order.template const_data_ptr<index_t>(),
            voxel_to_point.template const_data_ptr<index_t>(),
            grid_size,
            brick_grid,
            prod(brick_grid),
            prod(grid_size),
            default_value,
            opacity_weighted,
            out_semantics.template mutable_data_ptr<data_t>(),
            out_occupancy.template mutable_data_ptr<float_t>(),
            out_density.template mutable_data_ptr<float_t>(),
            out_probability.template mutable_data_ptr<float_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    profiler.mark("remainder");

    return {out_semantics, out_occupancy, out_density, out_probability};
}

template <typename index_t, typename float_t, typename data_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
render_backward(
    at::Tensor const& pair_offsets,
    at::Tensor const& pair_gaussians,
    at::Tensor const& pair_bricks,
    at::Tensor const& pair_order,
    at::Tensor const& gaussian_cells,
    at::Tensor const& gaussian_radii,
    at::Tensor const& gaussian_elements,
    at::Tensor const& voxel_to_point,
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
    math::vec3<index_t> const& brick_grid,
    bool const opacity_weighted,
    PhaseProfiler& profiler,
    c10::cuda::CUDAStream const& stream)
{
    auto const num_gaussians = gaussian_means.size(0);
    auto const num_channels = gaussian_semantics.size(1);
    auto const num_pairs = pair_gaussians.size(0);

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

    if (num_channels <= pair_backward_channel_limit) {
        auto partial_geometry = at::empty({num_pairs, kernel::num_geometry_grads}, float_opts);
        auto partial_semantics = at::empty({num_pairs, num_channels}, data_opts);

        {
            auto const threads = 256;
            auto const blocks = (num_pairs + threads - 1) / threads;

            dispatch_kernel<PairBackwardKernelWrapper<index_t, float_t, data_t>, PairBackwardChannels>(
                static_cast<int>(num_channels), blocks, threads, stream,
                static_cast<index_t>(num_pairs),
                static_cast<index_t>(num_channels),
                pair_gaussians.template const_data_ptr<index_t>(),
                pair_bricks.template const_data_ptr<index_t>(),
                pair_order.template const_data_ptr<index_t>(),
                voxel_to_point.template const_data_ptr<index_t>(),
                sample_points.template const_data_ptr<float_t>(),
                gaussian_means.template const_data_ptr<float_t>(),
                gaussian_icovs.template const_data_ptr<float_t>(),
                gaussian_idets.template const_data_ptr<float_t>(),
                gaussian_opacities.template const_data_ptr<float_t>(),
                gaussian_semantics.template const_data_ptr<data_t>(),
                gaussian_cells.template const_data_ptr<index_t>(),
                gaussian_radii.template const_data_ptr<index_t>(),
                out_semantics.template const_data_ptr<data_t>(),
                out_occupancy.template const_data_ptr<float_t>(),
                out_probability.template const_data_ptr<float_t>(),
                out_semantics_grad.template const_data_ptr<data_t>(),
                out_occupancy_grad.template const_data_ptr<float_t>(),
                out_density_grad.template const_data_ptr<float_t>(),
                grid_size,
                brick_grid,
                prod(brick_grid),
                prod(grid_size),
                opacity_weighted,
                partial_geometry.template mutable_data_ptr<float_t>(),
                partial_semantics.template mutable_data_ptr<data_t>());
        }

        profiler.mark("bwd_render");

        {
            auto const num_components = kernel::num_geometry_grads + num_channels;
            auto const total = num_gaussians * num_components;

            auto const threads = 512;
            auto const blocks = (total + threads - 1) / threads;

            kernel::reduce_backward_partials<index_t, float_t, data_t><<<blocks, threads, 0, stream>>>(
                static_cast<index_t>(num_gaussians),
                static_cast<index_t>(num_channels),
                pair_offsets.template const_data_ptr<index_t>(),
                partial_geometry.template const_data_ptr<float_t>(),
                partial_semantics.template const_data_ptr<data_t>(),
                means_grad.template mutable_data_ptr<float_t>(),
                icov_grad.template mutable_data_ptr<float_t>(),
                idets_grad.template mutable_data_ptr<float_t>(),
                opacities_grad.template mutable_data_ptr<float_t>(),
                semantics_grad.template mutable_data_ptr<data_t>());
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }

        profiler.mark("bwd_reduce");
    } else if (num_channels <= last<LargeBackwardChannels>()) {
        auto const threads = 256;
        auto const blocks = (num_gaussians * at::cuda::warp_size() + threads - 1) / threads;

        dispatch_kernel<WarpBackwardKernelWrapper<index_t, float_t, data_t>, LargeBackwardChannels>(
            static_cast<int>(num_channels), blocks, threads, stream,
            static_cast<index_t>(num_gaussians),
            static_cast<index_t>(num_channels),
            gaussian_cells.template const_data_ptr<index_t>(),
            gaussian_radii.template const_data_ptr<index_t>(),
            gaussian_elements.template const_data_ptr<index_t>(),
            voxel_to_point.template const_data_ptr<index_t>(),
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
            grid_size,
            prod(grid_size),
            opacity_weighted,
            means_grad.template mutable_data_ptr<float_t>(),
            icov_grad.template mutable_data_ptr<float_t>(),
            idets_grad.template mutable_data_ptr<float_t>(),
            opacities_grad.template mutable_data_ptr<float_t>(),
            semantics_grad.template mutable_data_ptr<data_t>());

        profiler.mark("bwd_render");
    } else {
        auto const threads = 512;
        auto const blocks = (num_gaussians + threads - 1) / threads;

        kernel::render_backward_global_memory<index_t, float_t, data_t><<<blocks, threads, 0, stream>>>(
            static_cast<index_t>(num_gaussians),
            static_cast<index_t>(num_channels),
            gaussian_cells.template const_data_ptr<index_t>(),
            gaussian_radii.template const_data_ptr<index_t>(),
            gaussian_elements.template const_data_ptr<index_t>(),
            voxel_to_point.template const_data_ptr<index_t>(),
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
            grid_size,
            prod(grid_size),
            opacity_weighted,
            means_grad.template mutable_data_ptr<float_t>(),
            icov_grad.template mutable_data_ptr<float_t>(),
            idets_grad.template mutable_data_ptr<float_t>(),
            opacities_grad.template mutable_data_ptr<float_t>(),
            semantics_grad.template mutable_data_ptr<data_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        profiler.mark("bwd_render");
    }

    return {means_grad, icov_grad, idets_grad, opacities_grad, semantics_grad};
}

template <typename index_t, typename float_t, typename data_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
aggregate_forward(
    at::Tensor const& sample_points,
    at::Tensor const& gaussian_means,
    at::Tensor const& gaussian_icovs,
    at::Tensor const& gaussian_idets,
    at::Tensor const& gaussian_opacities,
    at::Tensor const& gaussian_semantics,
    at::Tensor const& sample_points_int,
    at::Tensor const& point_elements,
    at::Tensor const& gaussian_means_int,
    at::Tensor const& gaussian_radii,
    at::Tensor const& gaussian_elements,
    index_t const num_elements,
    math::vec3<index_t> const& grid_size,
    data_t default_value,
    bool const opacity_weighted,
    c10::cuda::CUDAStream const& stream)
{
    PhaseProfiler profiler{stream};

    auto const brick_grid = get_brick_grid(grid_size);
    auto const num_keys = num_elements * prod(brick_grid);

    auto pair_offsets = compute_gaussian_brick_offsets<index_t>(
        gaussian_means_int,
        gaussian_radii,
        grid_size,
        stream);

    profiler.mark("counts+scan");

    auto const [
        pair_gaussians,
        pair_bricks
    ] = compute_gaussian_brick_pairs<index_t>(
        gaussian_means_int,
        gaussian_radii,
        gaussian_elements,
        pair_offsets,
        grid_size,
        brick_grid,
        stream);

    profiler.mark("pair_gen");

    auto const [
        sorted_bricks,
        sorted_slots
    ] = sort_gaussian_brick_pairs<index_t>(
        pair_bricks,
        num_keys,
        stream);

    profiler.mark("sort");

    auto brick_ranges = compute_key_ranges<index_t>(
        sorted_bricks, num_keys, stream);

    profiler.mark("ranges");

    auto const voxel_to_point = compute_voxel_to_point_map<index_t>(
        sample_points_int, point_elements, num_elements, grid_size, stream);

    profiler.mark("point_map");

    auto const [
        out_semantics,
        out_occupancy,
        out_density,
        out_probability
    ] = render_forward<index_t, float_t, data_t>(
        sample_points,
        sample_points_int,
        point_elements,
        gaussian_means,
        gaussian_icovs,
        gaussian_idets,
        gaussian_opacities,
        gaussian_semantics,
        gaussian_means_int,
        gaussian_radii,
        brick_ranges,
        pair_gaussians,
        sorted_slots,
        voxel_to_point,
        grid_size,
        brick_grid,
        num_keys,
        default_value,
        opacity_weighted,
        profiler,
        stream);

    return {
        out_semantics,
        out_occupancy,
        out_density,
        out_probability,
        pair_offsets,
        pair_gaussians,
        sorted_bricks,
        sorted_slots,
   };
}

template <typename index_t, typename float_t, typename data_t>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
aggregate_backward(
    at::Tensor const& pair_offsets,
    at::Tensor const& pair_gaussians,
    at::Tensor const& pair_bricks,
    at::Tensor const& pair_order,
    at::Tensor const& gaussian_means_int,
    at::Tensor const& gaussian_radii,
    at::Tensor const& gaussian_elements,
    at::Tensor const& sample_points_int,
    at::Tensor const& point_elements,
    index_t const num_elements,
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
    bool const opacity_weighted,
    c10::cuda::CUDAStream const& stream)
{
    PhaseProfiler profiler{stream};

    auto const brick_grid = get_brick_grid(grid_size);

    auto const voxel_to_point = compute_voxel_to_point_map<index_t>(
        sample_points_int, point_elements, num_elements, grid_size, stream);

    profiler.mark("bwd_point_map");

    return render_backward<index_t, float_t, data_t>(
        pair_offsets,
        pair_gaussians,
        pair_bricks,
        pair_order,
        gaussian_means_int,
        gaussian_radii,
        gaussian_elements,
        voxel_to_point,
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
        brick_grid,
        opacity_weighted,
        profiler,
        stream);
}

} /* namespace detail */


std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
aggregate_forward(
    at::Tensor const& sample_points,
    at::Tensor const& gaussian_means,
    at::Tensor const& gaussian_icovs,
    at::Tensor const& gaussian_idets,
    at::Tensor const& gaussian_opacities,
    at::Tensor const& gaussian_semantics,
    at::Tensor const& sample_points_int,
    at::Tensor const& point_elements,
    at::Tensor const& gaussian_means_int,
    at::Tensor const& gaussian_radii,
    at::Tensor const& gaussian_elements,
    int64_t const num_elements,
    int64_t const h,
    int64_t const w,
    int64_t const d,
    double const default_value,
    bool const opacity_weighted)
{
    CHECK_CUDA(sample_points);
    CHECK_CUDA(gaussian_means);
    CHECK_CUDA(gaussian_icovs);
    CHECK_CUDA(gaussian_idets);
    CHECK_CUDA(gaussian_opacities);
    CHECK_CUDA(gaussian_semantics);
    CHECK_CUDA(sample_points_int);
    CHECK_CUDA(point_elements);
    CHECK_CUDA(gaussian_means_int);
    CHECK_CUDA(gaussian_radii);
    CHECK_CUDA(gaussian_elements);

    CHECK_CONTIGUOUS(sample_points);
    CHECK_CONTIGUOUS(gaussian_means);
    CHECK_CONTIGUOUS(gaussian_icovs);
    CHECK_CONTIGUOUS(gaussian_idets);
    CHECK_CONTIGUOUS(gaussian_opacities);
    CHECK_CONTIGUOUS(gaussian_semantics);
    CHECK_CONTIGUOUS(sample_points_int);
    CHECK_CONTIGUOUS(point_elements);
    CHECK_CONTIGUOUS(gaussian_means_int);
    CHECK_CONTIGUOUS(gaussian_radii);
    CHECK_CONTIGUOUS(gaussian_elements);

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
                    point_elements,
                    gaussian_means_int,
                    gaussian_radii,
                    gaussian_elements,
                    static_cast<index_t>(num_elements),
                    grid_size,
                    static_cast<data_t>(default_value),
                    opacity_weighted,
                    at::cuda::getCurrentCUDAStream());
            }));
        }));
    }));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
agregate_backward(
    at::Tensor const& pair_offsets,
    at::Tensor const& pair_gaussians,
    at::Tensor const& pair_bricks,
    at::Tensor const& pair_order,
    at::Tensor const& gaussian_means_int,
    at::Tensor const& gaussian_radii,
    at::Tensor const& gaussian_elements,
    at::Tensor const& sample_points_int,
    at::Tensor const& point_elements,
    int64_t const num_elements,
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
    int64_t const d,
    bool const opacity_weighted)
{
    CHECK_CUDA(pair_offsets);
    CHECK_CUDA(pair_gaussians);
    CHECK_CUDA(pair_bricks);
    CHECK_CUDA(pair_order);
    CHECK_CUDA(gaussian_means_int);
    CHECK_CUDA(gaussian_radii);
    CHECK_CUDA(gaussian_elements);
    CHECK_CUDA(sample_points_int);
    CHECK_CUDA(point_elements);
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

    CHECK_CONTIGUOUS(pair_offsets);
    CHECK_CONTIGUOUS(pair_gaussians);
    CHECK_CONTIGUOUS(pair_bricks);
    CHECK_CONTIGUOUS(pair_order);
    CHECK_CONTIGUOUS(gaussian_means_int);
    CHECK_CONTIGUOUS(gaussian_radii);
    CHECK_CONTIGUOUS(gaussian_elements);
    CHECK_CONTIGUOUS(sample_points_int);
    CHECK_CONTIGUOUS(point_elements);
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

    return VSPLAT3D_DISPATCH_INDEX_TYPES(pair_offsets.scalar_type(), "aggregate_backward", ([&] {
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
                    pair_offsets,
                    pair_gaussians,
                    pair_bricks,
                    pair_order,
                    gaussian_means_int,
                    gaussian_radii,
                    gaussian_elements,
                    sample_points_int,
                    point_elements,
                    static_cast<index_t>(num_elements),
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
                    opacity_weighted,
                    at::cuda::getCurrentCUDAStream());
            }));
        }));
    }));
}

} /* namespace vsplat3d::cuda */
