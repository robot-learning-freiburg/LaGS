# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cube mesh geometry shared by the viser voxel renderers."""

import numpy as np


def create_cube_mesh_geometry(voxel_size, spacing=0.001):
    """Return ``(vertices, faces)`` for one voxel cube of ``voxel_size`` ([x, y, z]).

    ``spacing`` shrinks the cube slightly so neighbouring voxels don't z-fight.
    """
    vertices = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],  # bottom
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],  # top
        ]
    ) * (voxel_size - spacing)

    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # bottom (counter-clockwise when viewed from outside)
            [4, 5, 6],
            [4, 6, 7],  # top
            [0, 1, 5],
            [0, 5, 4],  # front
            [2, 7, 6],
            [2, 3, 7],  # back
            [0, 7, 3],
            [0, 4, 7],  # left
            [1, 2, 6],
            [1, 6, 5],  # right
        ]
    )

    return vertices, faces
