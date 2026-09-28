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
    ):
        # pylint: disable=arguments-differ
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements

        assert sample_points.shape[0] == 1, "Batch size > 1 not supported yet"
        assert not sample_points.requires_grad

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

        assert sample_points_int[..., 0].min() >= 0
        assert sample_points_int[..., 1].min() >= 0
        assert sample_points_int[..., 2].min() >= 0
        assert sample_points_int[..., 0].max() < h
        assert sample_points_int[..., 1].max() < w
        assert sample_points_int[..., 2].max() < d

        gaussian_means_int = utils.torch.amp.upcast(gaussian_means, torch.float32)
        gaussian_means_int = (gaussian_means_int - voxel_range[:3]) / voxel_size
        gaussian_means_int = gaussian_means_int.to(torch.int)
        gaussian_means_int[..., 0].clamp_(min=0, max=h - 1)
        gaussian_means_int[..., 1].clamp_(min=0, max=w - 1)
        gaussian_means_int[..., 2].clamp_(min=0, max=d - 1)

        gaussian_radii = gaussian_scales * distance_threshold / voxel_size
        gaussian_radii = torch.max(gaussian_radii, dim=-1).values
        gaussian_radii = torch.ceil(gaussian_radii).to(torch.int)
        gaussian_radii = torch.clamp(gaussian_radii, min=min_radius)

        sample_points = sample_points.squeeze(0)
        gaussian_means = gaussian_means.squeeze(0)
        gaussian_icovs = gaussian_icovs.squeeze(0)
        gaussian_idets = gaussian_idets.squeeze(0)
        gaussian_opacities = gaussian_opacities.squeeze(0)
        gaussian_semantics = gaussian_semantics.squeeze(0)
        sample_points_int = sample_points_int.squeeze(0)
        gaussian_means_int = gaussian_means_int.squeeze(0)
        gaussian_radii = gaussian_radii.squeeze(0)

        # Filter out zero-opacity gaussians to avoid unnecessary kernel work
        n_orig = gaussian_means.shape[0]
        active_mask = gaussian_opacities > 0
        active_idx = None

        if not active_mask.all():
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
                ctx.empty = True

                return (
                    out_semantics.unsqueeze(0),
                    out_occupancy.unsqueeze(0),
                    out_density.unsqueeze(0),
                    # prob_sum has no backward — detach to prevent silent zero-gradient bugs
                    out_prob_sum.unsqueeze(0).detach(),
                )

            gaussian_means = gaussian_means[active_idx]
            gaussian_icovs = gaussian_icovs[active_idx]
            gaussian_idets = gaussian_idets[active_idx]
            gaussian_opacities = gaussian_opacities[active_idx]
            gaussian_semantics = gaussian_semantics[active_idx]
            gaussian_means_int = gaussian_means_int[active_idx]
            gaussian_radii = gaussian_radii[active_idx]

        (
            out_semantics,
            out_occupancy,
            out_density,
            out_prob_sum,
            voxel_offsets,
            voxel_indices,
        ) = _aggregate_forward(
            sample_points,
            gaussian_means,
            gaussian_icovs,
            gaussian_idets,
            gaussian_opacities,
            gaussian_semantics,
            sample_points_int,
            gaussian_means_int,
            gaussian_radii,
            h,
            w,
            d,
            default_value,
        )

        ctx.h = h
        ctx.w = w
        ctx.d = d
        ctx.active_idx = active_idx
        ctx.n_orig = n_orig
        ctx.empty = False

        ctx.save_for_backward(
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
            out_prob_sum,
        )

        return (
            out_semantics.unsqueeze(0),
            out_occupancy.unsqueeze(0),
            out_density.unsqueeze(0),
            # prob_sum has no backward — detach to prevent silent zero-gradient bugs
            out_prob_sum.unsqueeze(0).detach(),
        )

    @staticmethod
    def backward(
        ctx, out_semantics_grad, out_occupancy_grad, out_density_grad, out_prob_sum_grad
    ):  # pylint: disable=unused-argument
        # pylint: disable=arguments-differ
        # pylint: disable=too-many-locals

        n_orig = ctx.n_orig

        # All gaussians were filtered out — return zero gradients
        if ctx.empty:
            device = out_semantics_grad.device
            dtype = out_semantics_grad.dtype
            n_sem = out_semantics_grad.shape[-1]
            return (
                None,  # sample_points
                torch.zeros((1, n_orig, 3), dtype=dtype, device=device),
                torch.zeros((1, n_orig, 6), dtype=dtype, device=device),
                torch.zeros((1, n_orig), dtype=dtype, device=device),
                None,  # gaussian_scales
                torch.zeros((1, n_orig), dtype=dtype, device=device),
                torch.zeros((1, n_orig, n_sem), dtype=dtype, device=device),
                None,  # voxel_range
                None,  # voxel_size
                None,  # grid_size
                None,  # distance_threshold
                None,  # min_radius
                None,  # default_value
            )

        out_semantics_grad = out_semantics_grad.squeeze(0).contiguous()
        out_occupancy_grad = out_occupancy_grad.squeeze(0).contiguous()
        out_density_grad = out_density_grad.squeeze(0).contiguous()

        h, w, d = ctx.h, ctx.w, ctx.d
        active_idx = ctx.active_idx

        (
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
        ) = ctx.saved_tensors

        (
            means_grad,
            icovs_grad,
            idets_grad,
            opacities_grad,
            semantics_grad,
        ) = _aggregate_backward(
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
            h,
            w,
            d,
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

        means_grad = means_grad.unsqueeze(0)
        icovs_grad = icovs_grad.unsqueeze(0)
        idets_grad = idets_grad.unsqueeze(0)
        opacities_grad = opacities_grad.unsqueeze(0)
        semantics_grad = semantics_grad.unsqueeze(0)

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
):
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
    )


class Aggregator(nn.Module):
    voxel_size: torch.Tensor
    voxel_range: torch.Tensor

    def __init__(
        self, distance_threshold, grid_size, voxel_range, voxel_size, min_radius=1
    ):
        super().__init__()

        self.distance_threshold = distance_threshold
        self.min_radius = min_radius
        self.grid_size = grid_size

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
        )
