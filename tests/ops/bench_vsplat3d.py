"""Benchmark for the vsplat3d aggregation op at the scale of S2GO (arXiv:2506.05473)
table 4: 9k gaussians splatted onto a 200x200x16 grid (640k voxels) with 18 channels.

Not collected by pytest; run directly:

    uv run python tests/ops/bench_vsplat3d.py

For a per-phase breakdown of the pipeline (pair generation, sort, render, ...), set
VSPLAT3D_PROFILE=1 -- the op then prints CUDA-event timings for every phase.
"""

import time

import torch

from tracker.mod.lags.occupancy.gaussian.aggregator import GaussianSemanticAggregator

VOXEL_RANGE = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
VOXEL_SIZE = [0.4, 0.4, 0.4]
NUM_GAUSSIANS = 9000
NUM_CHANNELS = 18


def make_inputs(generator):
    def rand(*shape, lo=0.0, hi=1.0):
        r = torch.rand(*shape, generator=generator, device="cuda")
        return r * (hi - lo) + lo

    n = NUM_GAUSSIANS
    return {
        "logits": torch.randn(
            1, 1, n, NUM_CHANNELS, generator=generator, device="cuda"
        ).requires_grad_(True),
        "centers": torch.stack(
            [
                rand(n, lo=-40.0, hi=40.0),
                rand(n, lo=-40.0, hi=40.0),
                rand(n, lo=-1.0, hi=5.4),
            ],
            dim=-1,
        )[None, None].requires_grad_(True),
        "scales": rand(1, 1, n, 3, lo=0.2, hi=0.6).requires_grad_(True),
        "rotations": torch.randn(
            1, 1, n, 4, generator=generator, device="cuda"
        ).requires_grad_(True),
        "opacities": rand(1, 1, n, lo=0.2, hi=0.8).requires_grad_(True),
    }


def bench(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / n * 1000


def run_case(inputs, opacity_weighted):
    aggregator = GaussianSemanticAggregator(
        VOXEL_SIZE,
        VOXEL_RANGE,
        scale_multiplier=3.0,
        opacity_weighted=opacity_weighted,
    ).cuda()

    def step():
        for value in inputs.values():
            value.grad = None
        outputs = aggregator(**inputs)
        (outputs[0].sum() + outputs[1].sum()).backward()

    forward = bench(lambda: aggregator(**inputs))
    combined = bench(step)

    print(
        f"opacity_weighted={opacity_weighted!s:5}  "
        f"forward {forward:6.3f} ms   "
        f"forward+backward {combined:6.3f} ms   "
        f"backward {combined - forward:6.3f} ms"
    )


def main():
    generator = torch.Generator(device="cuda").manual_seed(0)
    inputs = make_inputs(generator)

    for opacity_weighted in (False, True):
        run_case(inputs, opacity_weighted)


if __name__ == "__main__":
    main()
