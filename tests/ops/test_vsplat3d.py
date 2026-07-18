"""Tests for the vsplat3d gaussian-to-voxel aggregation op.

The kernel's backward pass is hand-derived analytically, so the tests here compare it
against a dense [N, M] reference whose gradients come from autograd. The reference
deliberately mirrors the kernel's own conventions (the six-component symmetric icov
layout, the splat-rect masking, the `valid` epsilon) so that any disagreement is a real
one rather than a difference in formulation.
"""

import math

import pytest
import torch

from tracker.ops import vsplat3d
from tracker.ops.vsplat3d import impl as vsplat3d_impl

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="vsplat3d is a CUDA-only op"
)

VOXEL_SIZE = (0.4, 0.4, 0.4)
DISTANCE_THRESHOLD = 3.0
MIN_RADIUS = 1
DEFAULT_VALUE = 0.25

# Matches epsilon<float> in src/cuda/aggregate.cu.
EPSILON = 1e-8


def _voxel_range(grid_size):
    ext = [n * s for n, s in zip(grid_size, VOXEL_SIZE)]
    return (-ext[0] / 2, -ext[1] / 2, -ext[2] / 2, ext[0] / 2, ext[1] / 2, ext[2] / 2)


def _voxel_centers(grid_size, voxel_range, device):
    """All voxel centers, ordered to match the kernel's x * ny * nz + y * nz + z."""
    axes = [
        (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * s + voxel_range[i]
        for i, (n, s) in enumerate(zip(grid_size, VOXEL_SIZE))
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3)


def _quaternion_to_matrix(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def _make_scene(num_gaussians, grid_size, num_channels, seed=0, device="cuda"):
    """A small scene with gaussians on a sub-voxel offset from voxel centers.

    The 0.13 * voxel_size offset matters for gradcheck: it keeps means ~0.148m from a
    voxel boundary (~15x the 1e-2 step in GRADCHECK_EPS) so perturbations cannot flip
    the integer bin and jump the splat rect, while keeping d != 0 so `power` stays away
    from 1 and off the (1 - occ) / (1 - alpha + eps) singularity in the backward.
    """
    # pylint: disable=too-many-locals
    gen = torch.Generator(device=device).manual_seed(seed)

    def rand(*shape, lo=0.0, hi=1.0):
        return torch.rand(*shape, generator=gen, device=device) * (hi - lo) + lo

    voxel_range = _voxel_range(grid_size)
    centers = _voxel_centers(grid_size, voxel_range, device)

    voxel_size_t = torch.tensor(VOXEL_SIZE, device=device)

    # Place gaussians on distinct voxel centers, offset into the voxel interior.
    pick = torch.randperm(centers.shape[0], generator=gen, device=device)
    means = centers[pick[:num_gaussians]] + 0.13 * voxel_size_t

    # Scales chosen so ceil(scale * 3.0 / 0.4) == 3 for the whole range, keeping the
    # splat radius off the ceil() boundary (0.40 would land exactly on 3.0).
    scales = rand(num_gaussians, 3, lo=0.35, hi=0.39)

    rotations = torch.randn(num_gaussians, 4, generator=gen, device=device)
    rot = _quaternion_to_matrix(rotations)

    icovs = rot @ torch.diag_embed(1.0 / (scales * scales)) @ rot.transpose(-1, -2)
    # Symmetrize exactly: the kernel reads only the upper components, so a numerically
    # asymmetric matrix would make the reference and the kernel disagree spuriously.
    icovs = 0.5 * (icovs + icovs.transpose(-1, -2))
    idets = 1.0 / torch.prod(scales * scales, dim=-1)

    # Kept clear of 0 so the `opacity > 0` filter in _AggregateFn cannot engage, and
    # clear of 1 so alpha stays away from the backward singularity.
    opacities = rand(num_gaussians, lo=0.2, hi=0.8)
    semantics = torch.randn(num_gaussians, num_channels, generator=gen, device=device)

    return {
        "sample_points": centers,
        "means": means,
        "scales": scales,
        "rotations": rot,
        "icovs": icovs,
        "idets": idets,
        "opacities": opacities,
        "semantics": semantics,
        "grid_size": grid_size,
        "voxel_range": voxel_range,
    }


def _build_icovs_idets(scales, rot):
    """Derive both from scales exactly as aggregator.py:72-83 does.

    This is the caller's invariant: idets == det(icovs), with both reachable from the
    same `scales` leaf. det(R S^-2 R^T) == det(S^-2) == 1 / prod(s^2).
    """
    icovs = rot @ torch.diag_embed(1.0 / (scales * scales)) @ rot.transpose(-1, -2)
    icovs = 0.5 * (icovs + icovs.transpose(-1, -2))
    idets = 1.0 / torch.prod(scales * scales, dim=-1)
    return icovs, idets


def _call_kernel(scene, opacity_weighted, anisotropic_radii=False):
    """Invoke the op under test, adding the batch dim it expects."""
    voxel_range = torch.tensor(scene["voxel_range"], device=scene["means"].device)
    voxel_size = torch.tensor(VOXEL_SIZE, device=scene["means"].device)

    return vsplat3d.aggregate(
        scene["sample_points"][None],
        scene["means"][None],
        scene["icovs"][None],
        scene["idets"][None],
        scene["scales"][None],
        scene["opacities"][None],
        scene["semantics"][None],
        voxel_range,
        voxel_size,
        scene["grid_size"],
        DISTANCE_THRESHOLD,
        MIN_RADIUS,
        DEFAULT_VALUE,
        opacity_weighted,
        anisotropic_radii,
    )


def _splat_mask(scene, anisotropic_radii=False):
    """[N, M] bool: which gaussians reach which sample points.

    Mirrors get_rect() in the kernel plus the quantization in _AggregateFn.forward,
    including its per-axis 3-sigma radii when anisotropic_radii is set.
    """
    # pylint: disable=too-many-locals
    device = scene["means"].device
    grid_size = scene["grid_size"]

    vr = torch.tensor(scene["voxel_range"], device=device)
    vs = torch.tensor(VOXEL_SIZE, device=device)
    gs = torch.tensor(grid_size, device=device, dtype=torch.int64)

    pts_int = ((scene["sample_points"] - vr[:3]) / vs).to(torch.int64)

    means_int = ((scene["means"].detach() - vr[:3]) / vs).to(torch.int64)
    means_int = torch.minimum(torch.clamp(means_int, min=0), gs[None, :] - 1)

    if anisotropic_radii:
        icovs = scene["icovs"]
        a11, a22, a33 = icovs[:, 0, 0], icovs[:, 1, 1], icovs[:, 2, 2]
        a12, a23, a13 = icovs[:, 0, 1], icovs[:, 1, 2], icovs[:, 0, 2]
        adj_diag = torch.stack(
            [a22 * a33 - a23 * a23, a11 * a33 - a13 * a13, a11 * a22 - a12 * a12],
            dim=-1,
        )
        sigma = torch.sqrt(torch.clamp(adj_diag / scene["idets"][:, None], min=0))
        radii = sigma * DISTANCE_THRESHOLD / vs
    else:
        radii = (
            (scene["scales"] * DISTANCE_THRESHOLD / vs).max(dim=-1, keepdim=True).values
        )
        radii = radii.expand(-1, 3)

    radii = torch.ceil(radii).to(torch.int64).clamp(min=MIN_RADIUS)

    zeros = torch.zeros(3, dtype=torch.int64, device=device)
    rect_min = torch.clamp(means_int - radii, min=zeros, max=gs)
    rect_max = torch.clamp(means_int + radii + 1, min=zeros, max=gs)

    inside = (pts_int[:, None, :] >= rect_min[None]) & (
        pts_int[:, None, :] < rect_max[None]
    )
    return inside.all(dim=-1)


def _dense_reference(scene, opacity_weighted, anisotropic_radii=False):
    """Dense [N, M] restatement of the kernel forward; gradients come from autograd."""
    # pylint: disable=too-many-locals
    mask = _splat_mask(scene, anisotropic_radii)

    d = scene["means"][None, :, :] - scene["sample_points"][:, None, :]
    dx, dy, dz = d.unbind(-1)

    icovs = scene["icovs"]
    a11, a22, a33 = icovs[:, 0, 0], icovs[:, 1, 1], icovs[:, 2, 2]
    a12, a23, a13 = icovs[:, 0, 1], icovs[:, 1, 2], icovs[:, 0, 2]

    # Matches smat3::neg_half_quadratic, which reads only the six upper components.
    quad = (
        dx * dx * a11[None]
        + dy * dy * a22[None]
        + dz * dz * a33[None]
        + 2 * (dx * dy * a12[None] + dy * dz * a23[None] + dz * dx * a13[None])
    )
    power = torch.exp(-0.5 * quad)

    denom = (2 * math.pi) ** -1.5 * torch.sqrt(scene["idets"])
    prob = scene["opacities"][None] * denom[None] * power
    alpha = scene["opacities"][None] * power if opacity_weighted else power

    prob = prob * mask
    alpha = alpha * mask

    prob_sum = prob.sum(dim=1)
    density = alpha.sum(dim=1)
    occupancy = 1.0 - (1.0 - alpha).prod(dim=1)

    sem = (prob[:, :, None] * scene["semantics"][None]).sum(dim=1)
    sem = sem / prob_sum.clamp(min=EPSILON)[:, None]
    sem = torch.where(
        prob_sum[:, None] > EPSILON, sem, torch.full_like(sem, DEFAULT_VALUE)
    )

    return sem, occupancy, density, prob_sum


@requires_cuda
@pytest.mark.parametrize("opacity_weighted", [True, False])
def test_forward_matches_dense_reference(opacity_weighted):
    scene = _make_scene(num_gaussians=50, grid_size=(16, 16, 8), num_channels=18)

    sem, occ, density, prob_sum = _call_kernel(scene, opacity_weighted)
    ref_sem, ref_occ, ref_density, ref_prob_sum = _dense_reference(
        scene, opacity_weighted
    )

    # Loose enough for __expf, which is the ex2.approx.f32 intrinsic rather than expf.
    torch.testing.assert_close(sem[0], ref_sem, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(occ[0], ref_occ, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(density[0], ref_density, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(prob_sum[0], ref_prob_sum, rtol=1e-4, atol=1e-5)


@requires_cuda
def test_forward_opacity_weighting_changes_occupancy():
    """Guards against the flag being silently inert."""
    scene = _make_scene(num_gaussians=50, grid_size=(16, 16, 8), num_channels=18)

    _, occ_on, density_on, prob_on = _call_kernel(scene, True)
    _, occ_off, density_off, prob_off = _call_kernel(scene, False)

    # alpha = a * power <= power, so weighting can only lower occupancy/density.
    assert (occ_on <= occ_off + 1e-6).all()
    assert occ_on.max() > 0, "degenerate scene: no occupancy anywhere"
    assert not torch.allclose(occ_on, occ_off)
    assert not torch.allclose(density_on, density_off)

    # The semantic mixture already weighted by opacity, so it must not move.
    torch.testing.assert_close(prob_on, prob_off, rtol=1e-6, atol=1e-7)


def _grads(outputs, inputs, grad_outputs):
    return torch.autograd.grad(
        outputs=[o for o, g in zip(outputs, grad_outputs) if g is not None],
        inputs=inputs,
        grad_outputs=[g for g in grad_outputs if g is not None],
        allow_unused=True,
    )


@requires_cuda
@pytest.mark.parametrize("opacity_weighted", [True, False])
def test_backward_matches_dense_reference(opacity_weighted):
    """The kernel's analytic backward vs autograd through the dense reference.

    This is what validates the hand-derived opa_grad += alpha_grad * power term.
    """
    scene = _make_scene(num_gaussians=50, grid_size=(16, 16, 8), num_channels=18)

    gen = torch.Generator(device="cuda").manual_seed(7)
    n_pts = scene["sample_points"].shape[0]
    n_ch = scene["semantics"].shape[1]

    grad_sem = torch.randn(n_pts, n_ch, generator=gen, device="cuda")
    grad_occ = torch.randn(n_pts, generator=gen, device="cuda")
    grad_density = torch.randn(n_pts, generator=gen, device="cuda")

    names = ["means", "icovs", "idets", "opacities", "semantics"]

    def run(fn, batched):
        scn = dict(scene)
        for name in names:
            scn[name] = scene[name].clone().requires_grad_(True)
        outs = fn(scn)
        outs = [o[0] for o in outs] if batched else outs
        return (
            _grads(
                outs[:3],
                [scn[n] for n in names],
                [grad_sem, grad_occ, grad_density],
            ),
            outs,
        )

    kernel_grads, _ = run(lambda s: _call_kernel(s, opacity_weighted), batched=True)
    ref_grads, _ = run(lambda s: _dense_reference(s, opacity_weighted), batched=False)

    for name, got, want in zip(names, kernel_grads, ref_grads):
        assert got is not None, f"{name} received no gradient from the kernel"
        torch.testing.assert_close(
            got, want, rtol=2e-3, atol=1e-5, msg=lambda m, n=name: f"{n} grad:\n{m}"
        )


@requires_cuda
def test_backward_occupancy_alone_drives_opacity():
    """Isolate the occupancy -> opacity path.

    With only out_occupancy_grad driven, the pre-existing semantic opacity gradient is
    zero, so any opacity gradient here comes solely from the new alpha fork. Under
    opacity_weighted=False that path does not exist and the gradient must vanish.
    """
    scene = _make_scene(num_gaussians=50, grid_size=(16, 16, 8), num_channels=18)
    n_pts = scene["sample_points"].shape[0]

    gen = torch.Generator(device="cuda").manual_seed(11)
    grad_occ = torch.randn(n_pts, generator=gen, device="cuda")

    def opacity_grad(opacity_weighted, fn, batched):
        scn = dict(scene)
        scn["opacities"] = scene["opacities"].clone().requires_grad_(True)
        outs = fn(scn, opacity_weighted)
        occ = outs[1][0] if batched else outs[1]
        (grad,) = torch.autograd.grad(occ, scn["opacities"], grad_outputs=grad_occ)
        return grad

    on = opacity_grad(True, _call_kernel, True)
    ref_on = opacity_grad(True, _dense_reference, False)
    torch.testing.assert_close(on, ref_on, rtol=2e-3, atol=1e-5)
    assert on.abs().max() > 1e-4, "occupancy loss is not reaching opacity"

    off = opacity_grad(False, _call_kernel, True)
    assert off.abs().max() == 0.0, "opacity must not see occupancy when flag is off"


@requires_cuda
@pytest.mark.parametrize("opacity_weighted", [True, False])
def test_backward_scales_under_caller_invariant(opacity_weighted):
    """Total d(loss)/d(scales) when icovs and idets both come from one `scales` leaf.

    The other backward tests treat icovs and idets as independent, which is the op's
    actual contract but not how it is called: aggregator.py derives both from scales, so
    idets == det(icovs) always holds and autograd sums the two routes.

    That makes this the configuration where a determinant counted twice shows up -- once
    via idets_grad and once via any det path folded into icovs_grad -- since both reach
    scales as the same function. It is how the double-count was originally caught.
    """
    scene = _make_scene(num_gaussians=50, grid_size=(16, 16, 8), num_channels=18)
    rot = scene["rotations"]

    gen = torch.Generator(device="cuda").manual_seed(13)
    probe = torch.randn(
        scene["sample_points"].shape[0],
        scene["semantics"].shape[1],
        generator=gen,
        device="cuda",
    )

    # Guard the premise: the parametrization really does satisfy idets == det(icovs).
    icovs, idets = _build_icovs_idets(scene["scales"], rot)
    # pylint: disable-next=not-callable
    det = torch.linalg.det(icovs)
    torch.testing.assert_close(det, idets, rtol=1e-4, atol=1e-4)

    def scales_grad(fn, batched):
        scales = scene["scales"].clone().requires_grad_(True)
        scn = dict(scene)
        scn["scales"] = scales
        scn["icovs"], scn["idets"] = _build_icovs_idets(scales, rot)
        outs = fn(scn)
        sem = outs[0][0] if batched else outs[0]
        return torch.autograd.grad(sem, scales, grad_outputs=probe)[0]

    got = scales_grad(lambda s: _call_kernel(s, opacity_weighted), batched=True)
    want = scales_grad(lambda s: _dense_reference(s, opacity_weighted), batched=False)

    torch.testing.assert_close(got, want, rtol=2e-3, atol=1e-4)


@requires_cuda
@pytest.mark.parametrize("opacity_weighted", [True, False])
def test_forward_matches_impl_reference(opacity_weighted):
    """Keep impl.py, the maintained pure-torch twin, in lockstep with the kernel.

    Note impl.py derives the determinant from the inverse covariance rather than taking
    it as a separate input, so this only holds while idets == det(icovs).
    """
    # pylint: disable=too-many-locals
    scene = _make_scene(num_gaussians=20, grid_size=(8, 8, 4), num_channels=8)
    device = scene["means"].device

    vr = torch.tensor(scene["voxel_range"], device=device)
    vs = torch.tensor(VOXEL_SIZE, device=device)
    gs = torch.tensor(scene["grid_size"], device=device, dtype=torch.int64)

    pts_int = ((scene["sample_points"] - vr[:3]) / vs).to(torch.int64)
    means_int = ((scene["means"] - vr[:3]) / vs).to(torch.int64)
    means_int = torch.minimum(torch.clamp(means_int, min=0), gs[None, :] - 1)

    radii = (scene["scales"] * DISTANCE_THRESHOLD / vs).max(dim=-1).values
    radii = torch.ceil(radii).to(torch.int64).clamp(min=MIN_RADIUS)

    sem, occ, density, mass = vsplat3d_impl.render(
        scene["sample_points"],
        pts_int,
        scene["means"],
        means_int,
        scene["icovs"],
        scene["opacities"],
        scene["semantics"],
        radii,
        gs,
        DEFAULT_VALUE,
        opacity_weighted,
    )

    k_sem, k_occ, k_density, k_mass = _call_kernel(scene, opacity_weighted)

    torch.testing.assert_close(k_sem[0], sem, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(k_occ[0], occ, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(k_density[0], density, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(k_mass[0], mass, rtol=1e-4, atol=1e-5)


@requires_cuda
@pytest.mark.parametrize("opacity_weighted", [True, False])
def test_grid_not_divisible_by_brick_edge(opacity_weighted):
    """Forward and backward on a grid that is no multiple of the 4-voxel brick edge.

    The kernel groups voxels into 4x4x4 bricks, so grids like this one leave bricks
    overhanging the grid on every axis; those out-of-grid voxels must be skipped both
    when rendering and when accumulating gradients.
    """
    # pylint: disable=too-many-locals
    scene = _make_scene(num_gaussians=30, grid_size=(13, 11, 6), num_channels=18)

    sem, occ, density, prob_sum = _call_kernel(scene, opacity_weighted)
    ref_sem, ref_occ, ref_density, ref_prob_sum = _dense_reference(
        scene, opacity_weighted
    )

    torch.testing.assert_close(sem[0], ref_sem, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(occ[0], ref_occ, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(density[0], ref_density, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(prob_sum[0], ref_prob_sum, rtol=1e-4, atol=1e-5)

    gen = torch.Generator(device="cuda").manual_seed(17)
    n_pts = scene["sample_points"].shape[0]
    n_ch = scene["semantics"].shape[1]

    grad_sem = torch.randn(n_pts, n_ch, generator=gen, device="cuda")
    grad_occ = torch.randn(n_pts, generator=gen, device="cuda")
    grad_density = torch.randn(n_pts, generator=gen, device="cuda")

    names = ["means", "icovs", "idets", "opacities", "semantics"]

    def run(fn, batched):
        scn = dict(scene)
        for name in names:
            scn[name] = scene[name].clone().requires_grad_(True)
        outs = fn(scn)
        outs = [o[0] for o in outs] if batched else outs
        return _grads(
            outs[:3], [scn[n] for n in names], [grad_sem, grad_occ, grad_density]
        )

    kernel_grads = run(lambda s: _call_kernel(s, opacity_weighted), batched=True)
    ref_grads = run(lambda s: _dense_reference(s, opacity_weighted), batched=False)

    for name, got, want in zip(names, kernel_grads, ref_grads):
        torch.testing.assert_close(
            got, want, rtol=2e-3, atol=1e-5, msg=lambda m, n=name: f"{n} grad:\n{m}"
        )


@requires_cuda
@pytest.mark.parametrize("num_channels", [33, 64, 300])
@pytest.mark.parametrize("opacity_weighted", [True, False])
def test_large_channel_counts(num_channels, opacity_weighted):
    """Forward and backward beyond the 32-channel pair-parallel backward limit.

    The other tests all use <= 32 channels, so without this the warp-per-gaussian
    backward (33..256 channels; 33 exercises its partially-filled last lane pass, 64 a
    full one) and the global-memory fallbacks past the largest templated channel count
    (300) would be untested.
    """
    # pylint: disable=too-many-locals
    scene = _make_scene(
        num_gaussians=30, grid_size=(16, 16, 8), num_channels=num_channels
    )

    sem, occ, density, prob_sum = _call_kernel(scene, opacity_weighted)
    ref_sem, ref_occ, ref_density, ref_prob_sum = _dense_reference(
        scene, opacity_weighted
    )

    torch.testing.assert_close(sem[0], ref_sem, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(occ[0], ref_occ, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(density[0], ref_density, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(prob_sum[0], ref_prob_sum, rtol=1e-4, atol=1e-5)

    gen = torch.Generator(device="cuda").manual_seed(19)
    n_pts = scene["sample_points"].shape[0]

    grad_sem = torch.randn(n_pts, num_channels, generator=gen, device="cuda")
    grad_occ = torch.randn(n_pts, generator=gen, device="cuda")
    grad_density = torch.randn(n_pts, generator=gen, device="cuda")

    names = ["means", "icovs", "idets", "opacities", "semantics"]

    def run(fn, batched):
        scn = dict(scene)
        for name in names:
            scn[name] = scene[name].clone().requires_grad_(True)
        outs = fn(scn)
        outs = [o[0] for o in outs] if batched else outs
        return _grads(
            outs[:3], [scn[n] for n in names], [grad_sem, grad_occ, grad_density]
        )

    kernel_grads = run(lambda s: _call_kernel(s, opacity_weighted), batched=True)
    ref_grads = run(lambda s: _dense_reference(s, opacity_weighted), batched=False)

    for name, got, want in zip(names, kernel_grads, ref_grads):
        torch.testing.assert_close(
            got, want, rtol=2e-3, atol=1e-5, msg=lambda m, n=name: f"{n} grad:\n{m}"
        )


@requires_cuda
@pytest.mark.parametrize("opacity_weighted", [True, False])
def test_anisotropic_radii_match_dense_reference(opacity_weighted):
    """anisotropic_radii=True bounds each splat axis by its own 3-sigma extent.

    The scene is rebuilt with strongly anisotropic scales so the per-axis boxes
    genuinely differ from the default max-scale cubes, and the dense reference mirrors
    the same radii quantisation, so this checks that the kernel touches exactly the
    tightened voxel set, forward and backward.
    """
    # pylint: disable=too-many-locals
    scene = _make_scene(num_gaussians=40, grid_size=(16, 16, 8), num_channels=18)

    gen = torch.Generator(device="cuda").manual_seed(23)
    scales = torch.rand(40, 3, generator=gen, device="cuda") * 0.45 + 0.15
    scene["scales"] = scales
    scene["icovs"], scene["idets"] = _build_icovs_idets(scales, scene["rotations"])

    # Guard against the test silently degenerating into the isotropic one.
    tight = _splat_mask(scene, anisotropic_radii=True)
    assert (tight != _splat_mask(scene)).any(), "scene too isotropic to test anything"
    assert (tight & ~_splat_mask(scene)).sum() == 0, "per-axis boxes must only shrink"

    sem, occ, density, prob_sum = _call_kernel(
        scene, opacity_weighted, anisotropic_radii=True
    )
    ref_sem, ref_occ, ref_density, ref_prob_sum = _dense_reference(
        scene, opacity_weighted, anisotropic_radii=True
    )

    torch.testing.assert_close(sem[0], ref_sem, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(occ[0], ref_occ, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(density[0], ref_density, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(prob_sum[0], ref_prob_sum, rtol=1e-4, atol=1e-5)

    gen = torch.Generator(device="cuda").manual_seed(29)
    n_pts = scene["sample_points"].shape[0]
    n_ch = scene["semantics"].shape[1]

    grad_sem = torch.randn(n_pts, n_ch, generator=gen, device="cuda")
    grad_occ = torch.randn(n_pts, generator=gen, device="cuda")
    grad_density = torch.randn(n_pts, generator=gen, device="cuda")

    names = ["means", "icovs", "idets", "opacities", "semantics"]

    def run(fn, batched):
        scn = dict(scene)
        for name in names:
            scn[name] = scene[name].clone().requires_grad_(True)
        outs = fn(scn)
        outs = [o[0] for o in outs] if batched else outs
        return _grads(
            outs[:3], [scn[n] for n in names], [grad_sem, grad_occ, grad_density]
        )

    kernel_grads = run(
        lambda s: _call_kernel(s, opacity_weighted, anisotropic_radii=True),
        batched=True,
    )
    ref_grads = run(
        lambda s: _dense_reference(s, opacity_weighted, anisotropic_radii=True),
        batched=False,
    )

    for name, got, want in zip(names, kernel_grads, ref_grads):
        torch.testing.assert_close(
            got, want, rtol=2e-3, atol=1e-5, msg=lambda m, n=name: f"{n} grad:\n{m}"
        )


@requires_cuda
def test_batched_elements_match_individual_calls():
    """A batched call must reproduce every element's individual result exactly.

    Elements are independent scenes over a shared grid. The element index rides in the
    high bits of the stable pair sort, so each element keeps the pair order -- and
    thus the float accumulation order -- of a lone call, making the results bitwise
    equal, forward and backward. One scene carries a zero-opacity gaussian so the
    active-gaussian filter runs with per-element counts that differ after filtering.
    """
    # pylint: disable=too-many-locals
    scenes = [
        _make_scene(num_gaussians=40, grid_size=(16, 16, 8), num_channels=18, seed=s)
        for s in (0, 1, 2)
    ]
    scenes[1]["opacities"][7] = 0.0

    voxel_range = torch.tensor(scenes[0]["voxel_range"], device="cuda")
    voxel_size = torch.tensor(VOXEL_SIZE, device="cuda")

    names = ["means", "icovs", "idets", "opacities", "semantics"]
    stacked = {
        k: torch.stack([s[k] for s in scenes])
        for k in names + ["sample_points", "scales"]
    }
    for name in names:
        stacked[name].requires_grad_(True)

    batched = vsplat3d.aggregate(
        stacked["sample_points"],
        stacked["means"],
        stacked["icovs"],
        stacked["idets"],
        stacked["scales"],
        stacked["opacities"],
        stacked["semantics"],
        voxel_range,
        voxel_size,
        scenes[0]["grid_size"],
        DISTANCE_THRESHOLD,
        MIN_RADIUS,
        DEFAULT_VALUE,
        False,
    )

    gen = torch.Generator(device="cuda").manual_seed(31)
    n_pts = scenes[0]["sample_points"].shape[0]
    n_ch = scenes[0]["semantics"].shape[1]

    grad_sem = torch.randn(len(scenes), n_pts, n_ch, generator=gen, device="cuda")
    grad_occ = torch.randn(len(scenes), n_pts, generator=gen, device="cuda")
    grad_density = torch.randn(len(scenes), n_pts, generator=gen, device="cuda")

    batched_grads = _grads(
        batched[:3],
        [stacked[n] for n in names],
        [grad_sem, grad_occ, grad_density],
    )

    for i, scene in enumerate(scenes):
        scn = dict(scene)
        for name in names:
            scn[name] = scene[name].clone().requires_grad_(True)

        single = _call_kernel(scn, opacity_weighted=False)

        for out_name, got, want in zip(
            ["sem", "occ", "density", "prob_sum"], batched, single
        ):
            assert torch.equal(got[i], want[0]), f"{out_name} differs for element {i}"

        single_grads = _grads(
            [o[0] for o in single[:3]],
            [scn[n] for n in names],
            [grad_sem[i], grad_occ[i], grad_density[i]],
        )

        for name, got, want in zip(names, batched_grads, single_grads):
            assert torch.equal(got[i], want), f"{name} grad differs for element {i}"


@requires_cuda
def test_forward_multiple_sample_points_per_voxel():
    """Several sample points sharing a voxel must each be rendered at their own
    coordinates.

    The bricked render maps each voxel to at most one sample point; points that lose
    that mapping go through a separate remainder pass, which this exercises. (The
    backward has always kept only one point per voxel -- see compute_voxel_to_point_map
    -- so only the forward is compared here.)
    """
    scene = _make_scene(num_gaussians=50, grid_size=(16, 16, 8), num_channels=18)

    # Duplicate some voxels' sample points at a different position inside the voxel, so
    # the duplicates must produce genuinely different outputs.
    voxel_size_t = torch.tensor(VOXEL_SIZE, device=scene["sample_points"].device)
    extra = scene["sample_points"][:100] + 0.21 * voxel_size_t
    scene["sample_points"] = torch.cat([scene["sample_points"], extra], dim=0)

    sem, occ, density, prob_sum = _call_kernel(scene, opacity_weighted=False)
    ref_sem, ref_occ, ref_density, ref_prob_sum = _dense_reference(scene, False)

    torch.testing.assert_close(sem[0], ref_sem, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(occ[0], ref_occ, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(density[0], ref_density, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(prob_sum[0], ref_prob_sum, rtol=1e-4, atol=1e-5)


# Finite-difference step per input, chosen relative to that input's magnitude.
#
# The kernel calls __expf/__fsqrt_rn, so its internals run at float precision (~1e-6
# relative) whatever dtype is dispatched. That puts the numerical jacobian in a
# noise-dominated regime where its error scales as noise/eps rather than eps^2, so eps
# must be far larger than gradcheck's 1e-6 default (measured: 1.5e-3 error at eps=1e-3,
# 3.7e-5 at eps=1e-2). gradcheck applies one absolute eps to every input, but these
# inputs span opacities ~0.5 to idets ~544, so each is checked on its own instead.
#
# Means are the one input with an upper bound too: they are truncated to voxel bins,
# which carries no gradient, so a step that flips a bin would jump the splat rect and
# show up as a cliff. _make_scene keeps means 0.148m from a boundary, ~15x this step.
GRADCHECK_EPS = {
    "means": 1e-2,  # metres; scene keeps a 0.148m margin to the nearest voxel bound
    "icovs": 1e-2,  # ~8 for the scene's scales
    "idets": 1e0,  # ~544; the forward is very insensitive to it in absolute terms
    "opacities": 1e-2,  # in [0.2, 0.8]
    "semantics": 1e-2,  # ~N(0, 1); sem is linear in this, so differences are exact
}


@requires_cuda
@pytest.mark.parametrize("name", list(GRADCHECK_EPS))
@pytest.mark.parametrize("opacity_weighted", [True, False])
def test_gradcheck(name, opacity_weighted):
    """Finite-difference check of the kernel against itself, independent of any oracle.

    Unlike the dense-reference tests this does not presume a correct oracle, so it
    catches a backward that disagrees with our own forward. It cannot catch a wrong
    alpha in both -- that is what the dense-reference tests are for.

    gaussian_scales is left non-differentiable on purpose: _AggregateFn returns None for
    its gradient by design, and it feeds the ceil() in gaussian_radii.
    """
    scene = _make_scene(num_gaussians=8, grid_size=(8, 8, 4), num_channels=4)

    voxel_range = torch.tensor(scene["voxel_range"], device="cuda")
    voxel_size = torch.tensor(VOXEL_SIZE, device="cuda")

    fixed = {k: scene[k].double() for k in GRADCHECK_EPS}
    target = fixed.pop(name).clone().requires_grad_(True)

    def fn(x):
        args = dict(fixed, **{name: x})
        sem, occ, density, _ = vsplat3d.aggregate(
            scene["sample_points"].double()[None],
            args["means"][None],
            args["icovs"][None],
            args["idets"][None],
            scene["scales"].double()[None],
            args["opacities"][None],
            args["semantics"][None],
            voxel_range,
            voxel_size,
            scene["grid_size"],
            DISTANCE_THRESHOLD,
            MIN_RADIUS,
            DEFAULT_VALUE,
            opacity_weighted,
        )
        # prob_sum is detached by design, so it is not part of the check.
        return sem, occ, density

    assert torch.autograd.gradcheck(
        fn, (target,), eps=GRADCHECK_EPS[name], rtol=1e-2, atol=1e-3, nondet_tol=1e-6
    )
