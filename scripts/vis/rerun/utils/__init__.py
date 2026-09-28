# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reusable Rerun rendering for panoptic occupancy.

Rerun rendering (cameras, intensity point clouds, per-class panoptic occupancy,
boxes/trajectories, predicted gaussians), factored so the visualization
entrypoint stays thin. The full-sensor dataset builder is shared via
:mod:`scripts.vis.common.sensors` and the class-colour palette via
:mod:`scripts.common.datasets`.
"""

from scripts.vis.rerun.utils import blueprint, gaussians, predictions, render

__all__ = [
    "blueprint",
    "gaussians",
    "predictions",
    "render",
]
