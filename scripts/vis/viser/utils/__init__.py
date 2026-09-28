# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared building blocks for the viser occupancy figure scripts.

Occupancy voxel + annotation (box/trajectory) renderers, viser server plumbing
(lighting, frame navigation, screenshots) and scene selection, factored so the
entrypoints stay thin. The occupancy pipeline, palette and prediction I/O are
reused from :mod:`scripts.common.datasets`; the visual filter from
:mod:`scripts.vis.common`.
"""

from scripts.vis.viser.utils import annotations, mesh, occupancy, scenes, viewer

__all__ = ["annotations", "mesh", "occupancy", "scenes", "viewer"]
