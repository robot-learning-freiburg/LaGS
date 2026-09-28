# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

# WARNING: code is not tested...
# WARNING: code doesn't compile with torch.compile() currently...

# pylint: disable=protected-access
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._dynamo.config.capture_scalar_outputs = True


@torch.no_grad()
@torch.compile(mode="reduce-overhead", fullgraph=True)
def compute_gaussian_voxel_index_pairs(
    center_index: torch.Tensor,  # [N, 3], int
    radius: torch.Tensor,  # [N], int
    grid_size: torch.Tensor,  # [3], int
):
    # pylint: disable=too-many-locals

    device = center_index.device
    num_gaussians = center_index.shape[0]

    # Compute the bounding box for each gaussian
    zeros = torch.zeros(3, dtype=torch.int64, device=device)

    rect_min = center_index[:, :] - radius[:, None]
    rect_max = center_index[:, :] + radius[:, None] + 1

    rect_min = torch.clamp(rect_min, min=zeros[None, :], max=grid_size[None, :])
    rect_max = torch.clamp(rect_max, min=zeros[None, :], max=grid_size[None, :])

    # Compute the number of voxels touched by each gaussian
    size = rect_max - rect_min

    voxels_per_gaussian = size[:, 0] * size[:, 1] * size[:, 2]

    # Compute the total number of voxels and pairs
    total_voxels = torch.cumsum(voxels_per_gaussian, dim=0)

    # Compute the gaussian index for each gaussian-voxel pair
    gaussian_index = torch.arange(num_gaussians, dtype=torch.int64, device=device)
    gaussian_index = torch.repeat_interleave(gaussian_index, voxels_per_gaussian)

    # Compute the bounding-box local index for each gaussian-voxel pair
    num_pairs = gaussian_index.shape[0]

    local_voxel_index = torch.arange(num_pairs, dtype=torch.int64, device=device)
    local_voxel_index = total_voxels[gaussian_index] - local_voxel_index - 1

    # Translate the local bounding-box voxel index to the global 3D voxel index
    size = size[gaussian_index]
    rect_min = rect_min[gaussian_index]

    voxel_index_x = local_voxel_index // (size[:, 1] * size[:, 2])
    voxel_index_y = (local_voxel_index // size[:, 2]) % size[:, 1]
    voxel_index_z = local_voxel_index % size[:, 2]

    voxel_index_x += rect_min[:, 0]
    voxel_index_y += rect_min[:, 1]
    voxel_index_z += rect_min[:, 2]

    # Combine the indices into a single voxel index
    voxel_index = voxel_index_x * grid_size[1] * grid_size[2]
    voxel_index += voxel_index_y * grid_size[2]
    voxel_index += voxel_index_z

    return gaussian_index, voxel_index


@torch.no_grad()
@torch.compile(mode="reduce-overhead", fullgraph=True)
def filter_index_pairs(
    sample_index: torch.Tensor,  # [K, 3], int
    gaussian_index: torch.Tensor,  # [N]
    voxel_index: torch.Tensor,  # [N]
    grid_size: torch.Tensor,  # [3], int
):
    # pylint: disable=too-many-locals

    device = gaussian_index.device
    num_pairs = gaussian_index.shape[0]

    # Sort the index pairs by the voxel index
    voxel_index, order = torch.sort(voxel_index, dim=0)
    gaussian_index = gaussian_index[order]

    # Get the start indices/offsets and counts for each voxel
    mask = torch.diff(voxel_index, prepend=torch.tensor([-1], device=device)) != 0

    offsets = torch.nonzero(mask).squeeze(1)
    counts = torch.diff(
        offsets, append=torch.tensor([num_pairs], device=device, dtype=offsets.dtype)
    )

    # Create a grid-shaped buffer to store the sorted gaussian indices
    n_grid = grid_size.prod().item()

    grid_offsets = torch.zeros(n_grid, dtype=torch.int64, device=device)
    grid_counts = torch.zeros(n_grid, dtype=torch.int64, device=device)

    unique_voxels = voxel_index[offsets]
    grid_offsets[unique_voxels] = offsets
    grid_counts[unique_voxels] = counts

    # Get the offsets and counts for the sample indices
    sample_index = (
        sample_index[:, 0] * grid_size[1] * grid_size[2]
        + sample_index[:, 1] * grid_size[2]
        + sample_index[:, 2]
    )

    sample_offsets = grid_offsets[sample_index]
    sample_counts = grid_counts[sample_index]

    # Note: for each sample point, we now have an offset into the gaussian_index list
    # and a count of how many gaussians are associated with that sample point.

    return sample_offsets, sample_counts, gaussian_index


@torch.no_grad()
# @torch.compile(mode="max-autotune", fullgraph=True)
# Note: This doesn't want to compile :(
def compute_sample_gaussian_pairs(
    sample_point_index: torch.Tensor,  # [K, 3], int
    gaussian_center_index: torch.Tensor,  # [N, 3], int
    gaussian_radius: torch.Tensor,  # [N], int
    grid_size: torch.Tensor,  # [3], int
):
    num_samples = sample_point_index.shape[0]
    device = sample_point_index.device

    # Compute the gaussian-voxel pairs
    gaussian_index, voxel_index = compute_gaussian_voxel_index_pairs(
        gaussian_center_index,
        gaussian_radius,
        grid_size,
    )

    # Filter the pairs to only include those that are associated with the sample points
    sample_offsets, sample_counts, gaussian_index = filter_index_pairs(
        sample_point_index,
        gaussian_index,
        voxel_index,
        grid_size,
    )

    # Compute the sample-gaussian index pairs
    sample_index = torch.arange(num_samples, dtype=torch.int64, device=device)
    sample_index = torch.repeat_interleave(sample_index, sample_counts)

    num_pairs = sample_index.shape[0]

    sample_totals = torch.cumsum(sample_counts, dim=0)
    sample_arange = torch.arange(num_pairs, dtype=torch.int64, device=device)
    sample_arange = sample_totals[sample_index] - sample_arange - 1

    sample_offsets = sample_offsets[sample_index] + sample_arange

    gaussian_index = gaussian_index[sample_offsets]

    return sample_index, gaussian_index


@torch.compile(mode="reduce-overhead", fullgraph=True)
def evaluate_gaussians(
    sample_index: torch.Tensor,  # [K], int
    gaussian_index: torch.Tensor,  # [K], int
    sample_points: torch.Tensor,  # [N, 3], float
    gaussian_means: torch.Tensor,  # [M, 3], float
    gaussian_invcov: torch.Tensor,  # [M, 3, 3], float
    gaussian_opacities: torch.Tensor,  # [M], float
):
    # pylint: disable=too-many-locals

    # Collect the relevant sample points and gaussians
    sample_points = sample_points[sample_index]
    gaussian_means = gaussian_means[gaussian_index]
    gaussian_invcov = gaussian_invcov[gaussian_index]
    gaussian_opacities = gaussian_opacities[gaussian_index]

    # Evaluate gaussians at the sample points
    distance = gaussian_means - sample_points

    # Compute the exponent term of the gaussian
    # torch.einsum doesn't work well with torch.compile...
    power = torch.bmm(distance.unsqueeze(1), gaussian_invcov).squeeze(1)
    power = torch.sum(power * distance, dim=1)
    power = torch.exp(-0.5 * power)

    # Compute determinant manually to avoid linalg operations that cause
    # compilation issues
    a11, a12, a13 = (
        gaussian_invcov[:, 0, 0],
        gaussian_invcov[:, 0, 1],
        gaussian_invcov[:, 0, 2],
    )
    a21, a22, a23 = (
        gaussian_invcov[:, 1, 0],
        gaussian_invcov[:, 1, 1],
        gaussian_invcov[:, 1, 2],
    )
    a31, a32, a33 = (
        gaussian_invcov[:, 2, 0],
        gaussian_invcov[:, 2, 1],
        gaussian_invcov[:, 2, 2],
    )

    det = (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )
    sqrt_det = torch.sqrt(det)

    denom = 0.0634936392307281494  # 1 / (2 * pi)^(3/2)
    denom = denom * sqrt_det

    prob = denom * power * gaussian_opacities

    return prob, power  # [K], [K]


@torch.compile(mode="reduce-overhead", fullgraph=True)
def reduce(
    num_points: int,
    sample_index: torch.Tensor,  # [K], int
    gaussian_semantics: torch.Tensor,  # [K, C] float
    gaussian_prob: torch.Tensor,  # [K], float
    gaussian_power: torch.Tensor,  # [K], float
    epsilon: float = 1e-9,
):
    device = sample_index.device

    # Probability mass
    mass = torch.zeros(num_points, device=device, dtype=gaussian_prob.dtype)
    mass.index_add_(0, sample_index, gaussian_prob)

    # Density
    density = torch.zeros(num_points, device=device, dtype=gaussian_power.dtype)
    density.index_add_(0, sample_index, gaussian_power)

    # Binary occupancy
    occupancy = torch.ones(num_points, device=device, dtype=gaussian_power.dtype)
    occupancy.index_reduce_(
        0, sample_index, 1.0 - gaussian_power, reduce="prod", include_self=False
    )
    occupancy = 1.0 - occupancy

    # Reduce the gaussian semantics by weighted sum
    gaussian_semantics = gaussian_semantics * gaussian_prob[:, None]

    sem_shape = (num_points, gaussian_semantics.shape[1])
    sem = torch.zeros(sem_shape, device=device, dtype=gaussian_semantics.dtype)
    sem.index_add_(0, sample_index, gaussian_semantics)
    sem = sem / mass[:, None]

    # Handle cases where probability mass is too low
    valid = mass > epsilon

    sem = torch.where(valid[:, None], sem, 1.0 / (sem.shape[1] - 1))
    sem[~valid, :-1] = 0.0

    return sem, occupancy, density, mass


# @torch.compile(mode="reduce-overhead", fullgraph=True)
# Note: This doesn't want to compile because compute_sample_gaussian_pairs()
# doesn't compile
def render(
    sample_points: torch.Tensor,  # [N, 3], float
    sample_points_int: torch.Tensor,  # [N, 3], int
    gaussian_means: torch.Tensor,  # [M, 3], float
    gaussian_means_int: torch.Tensor,  # [M, 3], int
    gaussian_invcov: torch.Tensor,  # [M, 3, 3], float
    gaussian_opacities: torch.Tensor,  # [M], float
    gaussian_semantics: torch.Tensor,  # [M, C], float
    gaussian_radii: torch.Tensor,  # [M], int
    grid_size: torch.Tensor,  # [3], int
):
    # pylint: disable=too-many-locals

    # Get the sample-gaussian index pairs for which to evaluate
    sample_index, gaussian_index = compute_sample_gaussian_pairs(
        sample_points_int,
        gaussian_means_int,
        gaussian_radii,
        grid_size,
    )

    # Evaluate the gaussians at the sample points
    prob, power = evaluate_gaussians(
        sample_index,
        gaussian_index,
        sample_points,
        gaussian_means,
        gaussian_invcov,
        gaussian_opacities,
    )

    gaussian_semantics = gaussian_semantics[gaussian_index]

    # Reduce the evaluated gaussians to get the final rendered values
    sem, occ, density, mass = reduce(
        sample_points.shape[0],
        sample_index,
        gaussian_semantics,
        prob,
        power,
    )

    return sem, occ, density, mass
