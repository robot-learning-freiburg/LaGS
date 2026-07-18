# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
from torch import nn

from ... import utils
from ...utils.log import get_logger
from . import vsplat3d_ext as _ext

_aggregate_forward = torch.ops.vsplat3d.aggregate_forward
_aggregate_backward = torch.ops.vsplat3d.aggregate_backward

log = get_logger(__name__)


class _AggregateFn(torch.autograd.Function):
    # pylint: disable=abstract-method

    @staticmethod
    def forward(
        ctx,
        sample_points: torch.Tensor,
        gaussian_means: torch.Tensor,
        gaussian_icovs: torch.Tensor,
        gaussian_idets: torch.Tensor,
        gaussian_scales: torch.Tensor,
        gaussian_opacities: torch.Tensor,
        gaussian_semantics: torch.Tensor,
        voxel_range: torch.Tensor,
        voxel_size: torch.Tensor,
        grid_size: tuple[int, int, int],
        distance_threshold: float,
        min_radius: int,
        default_value: float,
        opacity_weighted: bool,
        anisotropic_radii: bool,
    ):
        # pylint: disable=arguments-differ
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        assert not sample_points.requires_grad

        # Leading dim is the batch of independent elements (batch entries, layers,
        # streams, ...), each with its own gaussians, all sharing one voxel grid.
        num_elements, num_points, _ = sample_points.shape
        num_gaussians = gaussian_means.shape[1]

        h, w, d = grid_size
        device = sample_points.device

        sample_points = sample_points.contiguous()
        gaussian_means = gaussian_means.contiguous()
        gaussian_icovs = gaussian_icovs.contiguous()
        gaussian_opacities = gaussian_opacities.contiguous()
        gaussian_semantics = gaussian_semantics.contiguous()

        # Note: ensure voxel_range and voxel_size are float32 to avoid precision issues
        voxel_range = torch.as_tensor(voxel_range, device=device, dtype=torch.float32)
        voxel_size = torch.as_tensor(voxel_size, device=device, dtype=torch.float32)

        sample_points_int = utils.torch.amp.upcast(sample_points, torch.float32)
        sample_points_int = (sample_points - voxel_range[:3]) / voxel_size
        sample_points_int = sample_points_int.to(torch.int)

        grid_max = torch.tensor([h, w, d], device=device)
        in_bounds = ((sample_points_int >= 0) & (sample_points_int < grid_max)).all()

        gaussian_means_int = utils.torch.amp.upcast(gaussian_means, torch.float32)
        gaussian_means_int = (gaussian_means_int - voxel_range[:3]) / voxel_size
        gaussian_means_int = gaussian_means_int.to(torch.int)
        gaussian_means_int[..., 0].clamp_(min=0, max=h - 1)
        gaussian_means_int[..., 1].clamp_(min=0, max=w - 1)
        gaussian_means_int[..., 2].clamp_(min=0, max=d - 1)

        # The kernel takes one splat radius per axis, in voxels. The default uses the
        # largest scale for all three axes; anisotropic_radii bounds each axis by its
        # own 3-sigma extent instead: sigma_i = sqrt(diag(Sigma)_i), and for the
        # symmetric inverse diag(Sigma)_i = adj(icov)_ii / det(icov), with det(icov)
        # supplied by the caller as gaussian_idets. That shrinks the splat boxes of
        # anisotropic gaussians, at the cost of no longer touching the tail voxels
        # beyond 3 sigma of the narrower axes, so results differ slightly there.
        if anisotropic_radii:
            a11, a22, a33, a12, a23, a13 = gaussian_icovs.unbind(-1)
            adj_diag = torch.stack(
                [
                    a22 * a33 - a23 * a23,
                    a11 * a33 - a13 * a13,
                    a11 * a22 - a12 * a12,
                ],
                dim=-1,
            )
            sigma = torch.sqrt(torch.clamp(adj_diag / gaussian_idets[..., None], min=0))
            gaussian_radii = sigma * distance_threshold / voxel_size
        else:
            gaussian_radii = gaussian_scales * distance_threshold / voxel_size
            gaussian_radii = torch.max(gaussian_radii, dim=-1, keepdim=True).values
            gaussian_radii = gaussian_radii.expand_as(gaussian_means)

        gaussian_radii = torch.ceil(gaussian_radii).to(torch.int)
        gaussian_radii = torch.clamp(gaussian_radii, min=min_radius).contiguous()

        # Flatten the element dim; the kernel is jagged-native and tracks each point's
        # and gaussian's element through explicit per-item element indices.
        sample_points = sample_points.reshape(-1, 3)
        gaussian_means = gaussian_means.reshape(-1, 3)
        gaussian_icovs = gaussian_icovs.reshape(-1, 6)
        gaussian_idets = gaussian_idets.reshape(-1)
        gaussian_opacities = gaussian_opacities.reshape(-1)
        gaussian_semantics = gaussian_semantics.reshape(
            num_elements * num_gaussians, -1
        )
        sample_points_int = sample_points_int.reshape(-1, 3)
        gaussian_means_int = gaussian_means_int.reshape(-1, 3)
        gaussian_radii = gaussian_radii.reshape(-1, 3)

        elements = torch.arange(num_elements, device=device, dtype=torch.int)
        point_elements = elements.repeat_interleave(num_points)
        gaussian_elements = elements.repeat_interleave(num_gaussians)

        # Filter out zero-opacity gaussians to avoid unnecessary kernel work.
        #
        # Note: this is only exact when opacity_weighted=True, where alpha = a * power
        # makes a zero-opacity gaussian contribute nothing to occupancy/density either.
        # With opacity_weighted=False, occupancy uses power alone, so a zero-opacity
        # gaussian still contributes and dropping it here alters the result.
        n_orig = gaussian_means.shape[0]
        active_mask = gaussian_opacities > 0

        # One combined host readback: read separately, the bounds check and the filter
        # decision would each synchronize with the device (the per-axis asserts this
        # covers used to cost six syncs per call).
        in_bounds, all_active = torch.stack([in_bounds, active_mask.all()]).tolist()
        assert in_bounds, "sample points must lie inside the voxel grid"

        active_idx = None

        if not all_active:
            active_idx = active_mask.nonzero(as_tuple=True)[0]

            # All gaussians filtered out — return defaults without calling kernel
            if active_idx.numel() == 0:
                n_pts = sample_points.shape[0]
                n_sem = gaussian_semantics.shape[-1]

                out_semantics = torch.full(
                    (n_pts, n_sem),
                    default_value,
                    dtype=gaussian_semantics.dtype,
                    device=device,
                )
                out_occupancy = torch.zeros(
                    n_pts, dtype=sample_points.dtype, device=device
                )
                out_density = torch.zeros(
                    n_pts, dtype=sample_points.dtype, device=device
                )
                out_prob_sum = torch.zeros(
                    n_pts, dtype=sample_points.dtype, device=device
                )

                ctx.h = h
                ctx.w = w
                ctx.d = d
                ctx.active_idx = active_idx
                ctx.n_orig = n_orig
                ctx.num_elements = num_elements
                ctx.num_points = num_points
                ctx.num_gaussians = num_gaussians
                ctx.opacity_weighted = opacity_weighted
                ctx.empty = True

                return (
                    out_semantics.reshape(num_elements, num_points, -1),
                    out_occupancy.reshape(num_elements, num_points),
                    out_density.reshape(num_elements, num_points),
                    # prob_sum has no backward — detach to prevent silent zero-gradient bugs
                    out_prob_sum.reshape(num_elements, num_points).detach(),
                )

            gaussian_means = gaussian_means[active_idx]
            gaussian_icovs = gaussian_icovs[active_idx]
            gaussian_idets = gaussian_idets[active_idx]
            gaussian_opacities = gaussian_opacities[active_idx]
            gaussian_semantics = gaussian_semantics[active_idx]
            gaussian_means_int = gaussian_means_int[active_idx]
            gaussian_radii = gaussian_radii[active_idx]
            gaussian_elements = gaussian_elements[active_idx]

        (
            out_semantics,
            out_occupancy,
            out_density,
            out_prob_sum,
            pair_offsets,
            pair_gaussians,
            pair_bricks,
            pair_order,
        ) = _aggregate_forward(
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
            num_elements,
            h,
            w,
            d,
            default_value,
            opacity_weighted,
        )

        ctx.h = h
        ctx.w = w
        ctx.d = d
        ctx.active_idx = active_idx
        ctx.n_orig = n_orig
        ctx.num_elements = num_elements
        ctx.num_points = num_points
        ctx.num_gaussians = num_gaussians
        ctx.opacity_weighted = opacity_weighted
        ctx.empty = False

        ctx.save_for_backward(
            pair_offsets,
            pair_gaussians,
            pair_bricks,
            pair_order,
            gaussian_means_int,
            gaussian_radii,
            gaussian_elements,
            sample_points_int,
            point_elements,
            sample_points,
            gaussian_means,
            gaussian_icovs,
            gaussian_idets,
            gaussian_opacities,
            gaussian_semantics,
            out_semantics,
            out_occupancy,
            out_prob_sum,
        )

        return (
            out_semantics.reshape(num_elements, num_points, -1),
            out_occupancy.reshape(num_elements, num_points),
            out_density.reshape(num_elements, num_points),
            # prob_sum has no backward — detach to prevent silent zero-gradient bugs
            out_prob_sum.reshape(num_elements, num_points).detach(),
        )

    @staticmethod
    def backward(
        ctx, out_semantics_grad, out_occupancy_grad, out_density_grad, out_prob_sum_grad
    ):  # pylint: disable=unused-argument
        # pylint: disable=arguments-differ
        # pylint: disable=too-many-locals

        n_orig = ctx.n_orig
        num_elements = ctx.num_elements
        num_points = ctx.num_points
        num_gaussians = ctx.num_gaussians

        # All gaussians were filtered out — return zero gradients
        if ctx.empty:
            device = out_semantics_grad.device
            dtype = out_semantics_grad.dtype
            n_sem = out_semantics_grad.shape[-1]
            shape = (num_elements, num_gaussians)
            return (
                None,  # sample_points
                torch.zeros((*shape, 3), dtype=dtype, device=device),
                torch.zeros((*shape, 6), dtype=dtype, device=device),
                torch.zeros(shape, dtype=dtype, device=device),
                None,  # gaussian_scales
                torch.zeros(shape, dtype=dtype, device=device),
                torch.zeros((*shape, n_sem), dtype=dtype, device=device),
                None,  # voxel_range
                None,  # voxel_size
                None,  # grid_size
                None,  # distance_threshold
                None,  # min_radius
                None,  # default_value
                None,  # opacity_weighted
                None,  # anisotropic_radii
            )

        out_semantics_grad = out_semantics_grad.reshape(
            num_elements * num_points, -1
        ).contiguous()
        out_occupancy_grad = out_occupancy_grad.reshape(-1).contiguous()
        out_density_grad = out_density_grad.reshape(-1).contiguous()

        h, w, d = ctx.h, ctx.w, ctx.d
        active_idx = ctx.active_idx

        (
            pair_offsets,
            pair_gaussians,
            pair_bricks,
            pair_order,
            gaussian_means_int,
            gaussian_radii,
            gaussian_elements,
            sample_points_int,
            point_elements,
            sample_points,
            gaussian_means,
            gaussian_icovs,
            gaussian_idets,
            gaussian_opacities,
            gaussian_semantics,
            out_semantics,
            out_occupancy,
            out_probability,
        ) = ctx.saved_tensors

        (
            means_grad,
            icovs_grad,
            idets_grad,
            opacities_grad,
            semantics_grad,
        ) = _aggregate_backward(
            pair_offsets,
            pair_gaussians,
            pair_bricks,
            pair_order,
            gaussian_means_int,
            gaussian_radii,
            gaussian_elements,
            sample_points_int,
            point_elements,
            num_elements,
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
            h,
            w,
            d,
            ctx.opacity_weighted,
        )

        # Scatter gradients back to original positions if we filtered
        if active_idx is not None:

            def _scatter_back(grad):
                full = torch.zeros(
                    (n_orig, *grad.shape[1:]),
                    dtype=grad.dtype,
                    device=grad.device,
                )
                full[active_idx] = grad
                return full

            means_grad = _scatter_back(means_grad)
            icovs_grad = _scatter_back(icovs_grad)
            idets_grad = _scatter_back(idets_grad)
            opacities_grad = _scatter_back(opacities_grad)
            semantics_grad = _scatter_back(semantics_grad)

        means_grad = means_grad.reshape(num_elements, num_gaussians, 3)
        icovs_grad = icovs_grad.reshape(num_elements, num_gaussians, 6)
        idets_grad = idets_grad.reshape(num_elements, num_gaussians)
        opacities_grad = opacities_grad.reshape(num_elements, num_gaussians)
        semantics_grad = semantics_grad.reshape(num_elements, num_gaussians, -1)

        return (
            None,  # sample_points
            means_grad,  # gaussian_means
            icovs_grad,  # gaussian_icovs
            idets_grad,  # gaussian_idets
            None,  # gaussian_scales
            opacities_grad,  # gaussian_opacities
            semantics_grad,  # gaussian_semantics
            None,  # voxel_range
            None,  # voxel_size
            None,  # grid_size
            None,  # distance_threshold
            None,  # min_radius
            None,  # default_value
            None,  # opacity_weighted
            None,  # anisotropic_radii
        )


def aggregate(
    sample_points: torch.Tensor,
    gaussian_means: torch.Tensor,
    gaussian_icovs: torch.Tensor,
    gaussian_idets: torch.Tensor,
    gaussian_scales: torch.Tensor,
    gaussian_opacities: torch.Tensor,
    gaussian_semantics: torch.Tensor,
    voxel_range: torch.Tensor,
    voxel_size: torch.Tensor,
    grid_size: tuple[int, int, int],
    distance_threshold: float,
    min_radius: int,
    default_value: float,
    opacity_weighted: bool = False,
    anisotropic_radii: bool = False,
):
    # pylint: disable=too-many-locals
    gaussian_icovs = gaussian_icovs.flatten(2)[..., [0, 4, 8, 1, 5, 2]]

    dtype = gaussian_means.dtype

    sample_points = sample_points.to(dtype)
    gaussian_icovs = gaussian_icovs.to(dtype)
    gaussian_idets = gaussian_idets.to(dtype)
    gaussian_scales = gaussian_scales.to(dtype)
    gaussian_opacities = gaussian_opacities.to(dtype)

    voxel_range = voxel_range.to(dtype)
    voxel_size = voxel_size.to(dtype)

    return _AggregateFn.apply(
        sample_points,
        gaussian_means,
        gaussian_icovs,
        gaussian_idets,
        gaussian_scales,
        gaussian_opacities,
        gaussian_semantics,
        voxel_range,
        voxel_size,
        grid_size,
        distance_threshold,
        min_radius,
        default_value,
        opacity_weighted,
        anisotropic_radii,
    )


class Aggregator(nn.Module):
    voxel_size: torch.Tensor
    voxel_range: torch.Tensor

    def __init__(
        self,
        distance_threshold,
        grid_size,
        voxel_range,
        voxel_size,
        min_radius=1,
        opacity_weighted=False,
        anisotropic_radii=False,
    ):
        super().__init__()

        self.distance_threshold = distance_threshold
        self.min_radius = min_radius
        self.grid_size = grid_size
        self.opacity_weighted = opacity_weighted
        self.anisotropic_radii = anisotropic_radii

        voxel_range = torch.as_tensor(voxel_range, dtype=torch.float)
        voxel_size = torch.as_tensor(voxel_size, dtype=torch.float)

        self.register_buffer("voxel_range", voxel_range, persistent=False)
        self.register_buffer("voxel_size", voxel_size, persistent=False)

    def forward(
        self,
        sample_points: torch.Tensor,
        gaussian_means: torch.Tensor,
        gaussian_icovs: torch.Tensor,
        gaussian_idets: torch.Tensor,
        gaussian_scales: torch.Tensor,
        gaussian_opacities: torch.Tensor,
        gaussian_semantics: torch.Tensor,
        default_value: float = 0.0,
    ):
        return aggregate(
            sample_points,
            gaussian_means,
            gaussian_icovs,
            gaussian_idets,
            gaussian_scales,
            gaussian_opacities,
            gaussian_semantics,
            self.voxel_range,
            self.voxel_size,
            self.grid_size,
            self.distance_threshold,
            self.min_radius,
            default_value,
            self.opacity_weighted,
            self.anisotropic_radii,
        )
