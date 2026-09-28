# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import shutil
from pathlib import Path

from setuptools import setup
from torch.utils import cpp_extension

root = Path(__file__).parent


def _enable_ccache() -> None:
    """Route the CUDA (nvcc) compiles through ccache when it is on PATH.

    Runs automatically as part of the build (uv/pip alike), so rebuilds reuse
    cached objects with no environment setup. We only wrap nvcc — via torch's
    ``PYTORCH_NVCC`` hook, which torch uses solely as the ninja compile command
    (never exec'd as a single argv), so it's safe. The host compiler (``CXX``)
    is deliberately left alone: torch runs it as ``[compiler, "-dumpversion"]``
    for its ABI check, which a ``"ccache c++"`` value would break. ``.cu``
    compiles dominate build time anyway. No-op when ccache is absent, and any
    pre-set values win.
    """
    if not shutil.which("ccache"):
        return
    os.environ.setdefault("PYTORCH_NVCC", "ccache nvcc")
    # nvcc's long-form dependency flags (--generate-dependencies-with-compile,
    # --dependency-output) break ccache (ccache#1663), so disable them to keep
    # caching effective. torch >= 2.11 fixes this natively by emitting short-form
    # -MD/-MF (pytorch/pytorch@c7b8eaa) — drop this line once the torch pin moves.
    os.environ.setdefault("TORCH_EXTENSION_SKIP_NVCC_GEN_DEPENDENCIES", "1")
    # uv builds each extension in a fresh random /tmp/…build-temp/ directory. By
    # default ccache hashes the working directory, so that random CWD makes every
    # rebuild a cache miss. Our sources compile with absolute paths and no debug
    # info, so the objects don't depend on the CWD — tell ccache to ignore it.
    os.environ.setdefault("CCACHE_NOHASHDIR", "1")


_enable_ccache()

if __name__ == "__main__":
    setup(
        name="tracker",
        packages=["tracker"],
        package_dir={"": "src"},
        ext_modules=[
            cpp_extension.CUDAExtension(
                "tracker.ops.voxelize.voxelize_ext",
                [
                    "src/tracker/ops/voxelize/src/cpu/aggregate/mean.cpp",
                    "src/tracker/ops/voxelize/src/cpu/voxelize/assign.cpp",
                    "src/tracker/ops/voxelize/src/cpu/voxelize/trace.cpp",
                    "src/tracker/ops/voxelize/src/cuda/aggregate/mean.cu",
                    "src/tracker/ops/voxelize/src/cuda/voxelize/assign.cu",
                    "src/tracker/ops/voxelize/src/cuda/voxelize/trace.cu",
                    "src/tracker/ops/voxelize/src/module.cpp",
                ],
                include_dirs=[
                    root / "src/tracker/ops/voxelize/src/",
                ],
            ),
            cpp_extension.CUDAExtension(
                "tracker.ops.vsplat3d.vsplat3d_ext",
                [
                    "src/tracker/ops/vsplat3d/src/cuda/aggregate.cu",
                    "src/tracker/ops/vsplat3d/src/module.cpp",
                ],
                include_dirs=[
                    root / "src/tracker/ops/vsplat3d/src/",
                ],
            ),
            cpp_extension.CUDAExtension(
                "tracker.ops.bev_pool_v2.bev_pool_v2_ext",
                [
                    "src/tracker/ops/bev_pool_v2/src/cpu/bev_pool.cpp",
                    "src/tracker/ops/bev_pool_v2/src/cuda/bev_pool.cu",
                    "src/tracker/ops/bev_pool_v2/src/module.cpp",
                ],
                include_dirs=[
                    root / "src/tracker/ops/bev_pool_v2/src/",
                ],
            ),
        ],
        cmdclass={"build_ext": cpp_extension.BuildExtension},
        zip_safe=False,
    )
