# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


def normalize_angle(
    angle: torch.Tensor, offset: float = 0.5, period: float = 2 * torch.pi
) -> torch.Tensor:
    """
    Normalizes an angle to a specified range.

    The function ensures that the given angle is mapped to the range
    [-(period / 2), (period / 2)) by default. With the default values
    (offset=0.5, period=2π), this results in normalization to the range
    [-π, π).

    Args:
        angle (torch.Tensor): Input angle(s) in radians.
        offset (float, optional): Determines the center of the target range.
            Defaults to 0.5, which centers the range around zero.
        period (float, optional): The full period of the angle,
            typically 2π for radians. Defaults to 2π.

    Returns:
        torch.Tensor: The normalized angle(s) within the target range.
    """
    return angle - torch.floor(angle / period + offset) * period


def rotate_2d(points: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rotates a set of 2D points counterclockwise by a given angle.

    Args:
        points (torch.Tensor): A tensor of shape (N, 2) representing N points in 2D space.
        angle (float): The rotation angle in radians.

    Returns:
        torch.Tensor: A tensor of shape (N, 2) containing the rotated points.

    Note:
        - The function applies a standard 2D rotation matrix:
          [[ cos(θ), sin(θ)]
           [-sin(θ), cos(θ)]]
        - The rotation is counterclockwise about the origin (0,0).
    """
    angle = torch.as_tensor(angle)

    sin = torch.sin(angle)
    cos = torch.cos(angle)

    # Construct the transposed 2D rotation matrix
    m_t = torch.tensor(
        [
            [cos, sin],
            [-sin, cos],
        ],
        device=points.device,
    )

    # Apply the rotation matrix
    return torch.mm(points, m_t)


def rotate_3d_z(points: torch.Tensor, angle: float) -> torch.Tensor:
    """
    Rotates a set of 3D points counterclockwise around the Z-axis.

    Args:
        points (torch.Tensor): A tensor of shape (N, 3) representing N points in 3D space.
        angle (float): The rotation angle in radians.

    Returns:
        torch.Tensor: A tensor of shape (N, 3) containing the rotated points.

    Note:
        - The function applies a standard 3D rotation matrix around the Z-axis:
          [[ cos(θ), sin(θ), 0]
           [-sin(θ), cos(θ), 0]
           [     0,      0,  1]]
        - The rotation is counterclockwise when looking down the positive Z-axis.
        - The operation preserves the Z-coordinates of the input points.
    """
    angle = torch.as_tensor(angle)

    sin = torch.sin(angle)
    cos = torch.cos(angle)

    # Construct the transposed 3D rotation matrix
    m_t = torch.tensor(
        [
            [cos, sin, 0.0],
            [-sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=points.device,
    )

    # Apply the rotation matrix
    return torch.mm(points, m_t)


def box_coords(center: torch.Tensor, dim: torch.Tensor, rot: torch.Tensor):
    """
    Computes the corner coordinates (vertices) of 2D (BEV) or 3D bounding boxes.

    This function generates the corner coordinates of axis-aligned bounding
    boxes, applies scaling based on the given dimensions, rotates them by the
    specified angles, and translates them to their final positions.

    Args:
        center (torch.Tensor): A tensor of shape (N, D) representing the center
            coordinates of N bounding boxes in a D-dimensional space (D=2 for 2D, D=3 for 3D).
        dim (torch.Tensor): A tensor of shape (N, D) representing the dimensions
            (width, length, height in 3D or width, length in 2D) of each box.
        rot (torch.Tensor): A tensor of shape (N,) representing the rotation
            angles of the bounding boxes in radians. Rotation is counterclockwise
            around the origin.

    Returns:
        torch.Tensor: A tensor of shape (N, 2**D, D) containing the corner
        coordinates of each bounding box. Here:
            - N is the number of bounding boxes.
            - 2**D is the number of corners per box (4 for 2D, 8 for 3D).
            - D is the number of spatial dimensions.

    Note:
        - The function first constructs the untransformed box corners in local
          coordinates, then scales them according to `dim`, rotates them, and
          finally translates them to the correct position using `center`.
        - The rotation is applied around the box center.
    """
    ndim = dim.shape[1]

    # Compute generic box corners (binary encoding for 2D: (0,0), (0,1), (1,0), (1,1))
    coords = torch.unravel_index(torch.arange(2**ndim), [2] * ndim)
    coords = torch.stack(coords, dim=1)
    coords = torch.flip(coords, (-1,))

    # Apply dimensions (scale and center around origin)
    coords = (coords - 0.5)[None, :, :] * dim[:, None, :]

    # Apply rotation
    rot = rot - torch.pi / 2.0
    sin, cos = torch.sin(rot), torch.cos(rot)

    m_t = torch.eye(ndim).repeat(dim.shape[0], 1, 1)
    m_t[:, 0, 0] = cos
    m_t[:, 0, 1] = sin
    m_t[:, 1, 0] = -sin
    m_t[:, 1, 1] = cos

    coords = torch.bmm(coords, m_t)

    # Apply center offset
    coords = coords[:, :, :] + center[:, None, :]

    return coords


def box_planes(center: torch.Tensor, dim: torch.Tensor, rot: torch.Tensor):
    """
    Computes the plane equations (normals and offsets) for a set of oriented
    bounding boxes (2D or 3D).

    Given the center, dimensions, and rotation of multiple boxes, this function computes
    the normal vectors of their bounding planes and the corresponding offsets for half-space
    tests. This is useful for collision detection, point-in-box tests, or geometric queries.

    Args:
        center (torch.Tensor): A tensor of shape (N, ndim) representing the center positions
            of N boxes in an ndim-dimensional space.
        dim (torch.Tensor): A tensor of shape (N, ndim) representing the box dimensions
            (width, length, height in 3D or width, height in 2D).
        rot (torch.Tensor): A tensor of shape (N,) representing the rotation angles of each
            box. The rotation is assumed to be counterclockwise and applied around the origin.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - normals (torch.Tensor): A tensor of shape (N, ndim, ndim) representing
              the normal vectors of the box planes.
            - d_min (torch.Tensor): A tensor of shape (N, ndim) containing the minimum
              offset values for the planes.
            - d_max (torch.Tensor): A tensor of shape (N, ndim) containing the maximum
              offset values for the planes.

    Note:
        - The function computes `ndim` normals per box, while the remaining normals
          are simply their negations.
        - The computed plane equations allow for efficient point-in-box checks by
          projecting a point onto the normals and checking if it falls within `[d_min, d_max]`.
    """
    ndim = dim.shape[1]

    # Construct generic axis coordinates (e.g., as (0, 0), (0, 1), (1, 0) in 2D).
    coords = torch.cat(
        (
            torch.zeros(1, ndim, device=center.device),
            torch.eye(ndim, device=center.device),
        ),
        dim=0,
    )

    # Scale by box dimensions and center around 0.
    coords = (coords - 0.5)[None, :, :] * dim[:, None, :]

    # Apply rotation.
    rot = rot - torch.pi / 2.0
    sin, cos = torch.sin(rot), torch.cos(rot)

    m_t = torch.eye(ndim, device=center.device).repeat(dim.shape[0], 1, 1)
    m_t[:, 0, 0] = cos
    m_t[:, 0, 1] = sin
    m_t[:, 1, 0] = -sin
    m_t[:, 1, 1] = cos

    coords = torch.bmm(coords, m_t)

    # Compute normals from corner coordinates.
    #
    # Note: We only need to compute ndim normals. The remaining normals are the
    # negatives of these (shape [n_boxes, ndim (axes), ndim (x/y/z)]).
    normals = coords[:, 1:, :] - coords[:, 0:1, :]

    # Compute the plane offsets. For the first half of the normals, these will
    # be the negative offsets (-d), for the second half, these will be positive
    # (+d). For point-box checks we can therefore just project the point onto
    # the normal and check whether it is in range [d_min, d_max].
    d_min = torch.einsum("bij,bij->bi", normals, coords[:, 0:1, :] + center[:, None, :])
    d_max = torch.einsum("bij,bij->bi", normals, coords[:, 1:, :] + center[:, None, :])

    return normals, d_min, d_max


def _intersect_box_poly_half(box_verts: torch.Tensor, poly_verts: torch.Tensor):
    """
    Partial box-polygon intersection check.

    This checks all box normals for overlap with the specified polygon. Note
    that for a full intersection check, all polygon normals need to be checked
    too. Meaning for a box-box check, this function would have to be called
    twice.

    Args:
        box_verts (tensor [N, 2**D, D])
            box vertices, where N is the number of boxes and D the dimension
        poly_verts (tensor [M, K, D])
            polygon vertices, with M polygons having K vertices each

    Returns:
        A bool tensor [N, M] where element [i, j] is True when the polygon j is
        entirely outside of the box i, meaning the box normals/edges do not
        overlap with the polygon.
    """
    n_box, _, n_dim = box_verts.shape
    n_poly, n_vert, n_dim = poly_verts.shape

    # Compute the box normals. Note that since we are dealing with proper
    # boxes, we only need to compute half the normals. The other half are the
    # negatives of these. (final shape: [n_box, n_dim (axes), n_dim (x/y/z)]).
    corners = torch.stack([box_verts[:, 2**i, :] for i in range(n_dim)], dim=1)
    normals = corners - box_verts[:, 0:1, :]

    # Compute the plane offsets. For the first half of the normals, these will
    # be the negative offsets (-d), for the second half, these will be positive
    # (+d).
    d_min = torch.einsum("bij,bij->bi", normals, box_verts[:, 0:1, :])
    d_max = torch.einsum("bij,bij->bi", normals, corners)

    # Project the polygon vertices onto the box normals.
    poly_verts = poly_verts.view(n_poly * n_vert, n_dim)
    poly_verts = poly_verts.expand(n_box, n_poly * n_vert, n_dim)

    proj = torch.bmm(poly_verts, normals.permute(0, 2, 1))
    proj = proj.view(n_box, n_poly, n_vert, n_dim)

    # Check for overlap on the projected axes. Note that this now simplifies to
    # checking for overlaps of the projected polygon vertices and the range
    # [d_min, d_max] on each axis.
    outside_box = (proj[:, :, :, :] < d_min[:, None, None, :]).all(2)
    outside_box |= (proj[:, :, :, :] > d_max[:, None, None, :]).all(2)

    # If there is any axis on which there is no overlap, the polygon must be
    # outside.
    outside_box = outside_box.any(2)

    return outside_box


def interesect_box_box(
    box_a_center: torch.Tensor,
    box_a_dim: torch.Tensor,
    box_a_rot: torch.Tensor,
    box_b_center: torch.Tensor,
    box_b_dim: torch.Tensor,
    box_b_rot: torch.Tensor,
):
    """
    Check whether the given object boxes (center, dimension, rotation)
    intersect.

    This check tests whether/how two given sets of bounding boxes interesect.
    To compute self-intersections, it is possible to pass the same set twice.
    Note, however, that in that case, all bounding boxes will be marked as
    intersecting with itself.

    Note: Implemented as check by separating axes. Works for both 3D and 2D
    (BEV) boxes (i.e., D=3 or D=2).

    Args:
        box_a_center (tensor [N, D]):
            center point of the first set of bounding boxes
        box_a_dim (tensor [N, D]):
            dimensions (width, length, height) of the first set of bounding boxes
        box_a_rot (tensor [N]):
            rotation angle of the first set of bounding boxes
        box_b_center (tensor [M, D]):
            center point of the second set of bounding boxes
        box_b_dim (tensor [M, D]):
            dimensions (width, length, height) of the second set of bounding boxes
        box_b_rot (tensor [M]):
            rotation angle of the second set of bounding boxes

    Returns:
        A bool tensor of shape [N, M], where N is size of the first and M the
        size of the secodn set of boxes. Entry [i, j] is set to True if box i
        of the first set intersects with box j of the second set.
    """

    coords_a = box_coords(box_a_center, box_a_dim, box_a_rot)
    coords_b = box_coords(box_b_center, box_b_dim, box_b_rot)

    outside_a = _intersect_box_poly_half(coords_a, coords_b)
    outside_b = _intersect_box_poly_half(coords_b, coords_a)

    return ~(outside_a | outside_b.T)


def interesect_box_aabb(
    box_center: torch.Tensor,
    box_dim: torch.Tensor,
    box_rot: torch.Tensor,
    aabb_min: torch.Tensor,
    aabb_max: torch.Tensor,
):
    """
    Check whether the given object boxes (center, dimension, rotation)
    intersect with the given axis-aligned bounding box volume.

    Note: Implemented as check by separating axes. Works for both 3D and 2D
    (BEV) boxes (i.e., D=3 or D=2).

    Args:
        box_center (tensor [N, D]):
            center point of the bounding boxes
        box_dim (tensor [N, D]):
            dimensions (width, length, height) of the bounding boxes
        box_rot (tensor [N]):
            rotation angle of the bounding boxes
        aabb_min (tensor [D]):
            minimum bounds of the axis aligned bounding box volume
        aabb_max (tensor [D]):
            maximum bounds of the axis aligned bounding box volume

    Returns:
        A bool tensor of shape [N], where N is the number of boxes. The i-th
        entry is set to True if the box intersects (meaning is at least
        partially inside) the specified axis-aligned bounding box.
    """
    # pylint: disable=too-many-locals

    _, ndim = box_dim.shape

    # Step 1: Check the AABB axes for overlap.
    #
    # Note: We don't have to do any projections here and can just use the
    # coordinates since we're dealing with an axis-aligned bounding box.

    # 1.1: Get coordinates for all (non-aligned) boxes.
    coords = box_coords(box_center, box_dim, box_rot)

    # 1.2: Check if the boxes overlap with the AABB. If all points on an axis
    # are outside (smaller than min or larger than max), there is no overlap on
    # that axis.
    outside_aabb = (coords[:, :, :] < aabb_min).all(1)
    outside_aabb |= (coords[:, :, :] > aabb_max).all(1)

    # 1.3: If there is any axis on which there is no overlap, the box must be
    # outside.
    outside_aabb = outside_aabb.any(1)

    # NB: If the object boxes were axis aligned as well, we would be done here.

    # Step 2: Check the box axes for overlap.
    #
    # Note: In contrast to the AABB, the object boxes are not axis aligned.
    # Therefore, we need to project the AABB corner points onto the box axes.

    # 2.1: Construct the AABB corners.
    aabb_coords = torch.unravel_index(torch.arange(2**ndim), [2] * ndim)
    aabb_coords = torch.stack(aabb_coords, dim=1).to(dtype=coords.dtype)
    aabb_coords = (aabb_max - aabb_min) * aabb_coords + aabb_min

    # 2.2: Check for overlap of the AABB box the normals.
    outside_box = _intersect_box_poly_half(coords, aabb_coords[None, :, :])
    outside_box = outside_box.squeeze(1)

    # Step 3: If there is any non-overlapping segment at all, the boxes do not
    # intersect with the AABB.
    return ~(outside_aabb | outside_box)


def intersect_box_voxel(
    box_center: torch.Tensor,
    box_dim: torch.Tensor,
    box_rot: torch.Tensor,
    voxel_indices: torch.Tensor,
    voxel_size: torch.Tensor,
    voxel_offset: torch.Tensor,
):
    """
    Check whether the given object boxes (center, dimension, rotation)
    intersect with the given voxel grid.

    Note: Voxel indices denote the lower left corner of the voxel, not the
    center.

    Args:
        box_center (tensor [N, D]):
            center point of the bounding boxes
        box_dim (tensor [N, D]):
            dimensions (width, length, height) of the bounding boxes
        box_rot (tensor [N]):
            rotation angle of the bounding boxes
        voxel_indices (tensor [M, D]):
            indices of the occupied voxels
        voxel_size (tensor [D]):
            size of the voxels
        voxel_offset (tensor [D]):
            offset of the voxel grid

    Returns:
        A bool tensor of shape [N, M], where N is the number of boxes and M the
        number of voxels. Entry [i, j] is set to True if box i intersects with
        voxel j.
    """

    # Step 1: Check the voxel axes for overlap.
    #
    # Note: We don't have to do any projections here and can just use the
    # coordinates since voxels are axis-aligned.

    # 1.1: Get coordinates for all (non-aligned) boxes.
    coords = box_coords(box_center, box_dim, box_rot)  # [N, 2**D, D]

    # 1.2: Get the voxel bounds.
    voxel_min, voxel_max = voxel_bounds(voxel_indices, voxel_size, voxel_offset)

    # 1.3: Check if the boxes overlap with the voxels. If all points on an axis
    # are outside (smaller than min or larger than max), there is no overlap on
    # that axis.
    outside_voxel = (coords[:, None, :, :] < voxel_min[None, :, None, :]).all(2)
    outside_voxel |= (coords[:, None, :, :] > voxel_max[None, :, None, :]).all(2)

    # 1.4: If there is any axis on which there is no overlap, the box must be
    # outside.
    outside_voxel = outside_voxel.any(2)  # [N, M]

    # Step 2: Check the box axes for overlap.

    # 2.1: Construct the voxel corners.
    voxel_corners = voxel_coords(voxel_indices, voxel_size, voxel_offset)
    voxel_corners = voxel_corners.contiguous()

    # 2.2: Check for overlap of the voxels with the box normals.
    outside_box = _intersect_box_poly_half(coords, voxel_corners)

    # Step 3: If there is any non-overlapping segment at all, the boxes do not
    # intersect with the voxels.
    return ~(outside_voxel | outside_box)


def intersect_points_box(
    points: torch.Tensor,
    box_center: torch.Tensor,
    box_dim: torch.Tensor,
    box_rot: torch.Tensor,
):
    """
    Test whether the given points are inside the given boxes.

    Args:
        points (tensor [N, D]):
            list of N points of dimension D to test
        box_center (tensor [M, D]):
            center point of the M bounding boxes
        box_dim (tensor [M, D]):
            dimensions (width, length, height) of the bounding boxes
        box_rot (tensor [M]):
            rotation angle of the bounding boxes

    Returns:
        Returns a bool tensor of shape [M, N] where M is the number of boxes
        and N the number of points. The entry [i, j] is set to True if point j
        is inside of box i.
    """

    # Compute the normals of the bounding boxes and the corresponding minimum
    # and maximum offsets (normals: [nbox, ndim (normals), ndim (x/y/z)],
    # d_{min,max}: [nbox, ndim]).
    normals, d_min, d_max = box_planes(box_center, box_dim, box_rot)

    # Project the points onto the normals (shape [nbox, ndim (normals), npoints]).
    proj = torch.einsum("bna,pa->bnp", normals, points)

    # Check whether points are in range [d_min, d_max] for all axes.
    inside = (proj >= d_min[:, :, None]) & (proj <= d_max[:, :, None])
    inside = inside.all(dim=1)

    return inside


def voxel_centers(
    voxel_indices: torch.Tensor,
    voxel_size: torch.Tensor,
    voxel_offset: torch.Tensor,
):
    """
    Compute the center coordinates of the given voxel indices.

    Args:
        voxel_indices (tensor [N, D]):
            indices of the voxels
        voxel_size (tensor [D]):
            size of the voxels
        voxel_offset (tensor [D]):
            offset of the voxel grid

    Returns:
        A tensor of shape [N, D] containing the center coordinates of the voxels.
    """
    return (voxel_indices + 0.5) * voxel_size + voxel_offset


def voxel_coords(
    voxel_indices: torch.Tensor,
    voxel_size: torch.Tensor,
    voxel_offset: torch.Tensor,
):
    """
    Compute the corner coordinates of the given voxel indices.

    Args:
        voxel_indices (tensor [N, D]):
            indices of the voxels
        voxel_size (tensor [D]):
            size of the voxels
        voxel_offset (tensor [D]):
            offset of the voxel grid

    Returns:
        A tensor of shape [N, 2**D, D] containing the corner coordinates of the voxels.
    """
    ndim = voxel_indices.shape[1]

    # Compute generic box corners (binary encoding for 2D: (0,0), (0,1), (1,0), (1,1))
    coords = torch.unravel_index(torch.arange(2**ndim), [2] * ndim)
    coords = torch.stack(coords, dim=1)
    coords = torch.flip(coords, (-1,))

    # Expand the indices to all corners
    coords = coords[None, :, :] + voxel_indices[:, None, :]

    # Transform indices to coordinates
    coords = coords * voxel_size[None, None, :] + voxel_offset[None, None, :]

    return coords


def voxel_bounds(
    voxel_indices: torch.Tensor,
    voxel_size: torch.Tensor,
    voxel_offset: torch.Tensor,
):
    """
    Compute the bounds of the given voxel indices.

    Args:
        voxel_indices (tensor [N, D]):
            indices of the voxels
        voxel_size (tensor [D]):
            size of the voxels
        voxel_offset (tensor [D]):
            offset of the voxel grid

    Returns:
        A tuple of two tensors [N, D], where the first tensor contains the
        minimum bounds and the second tensor contains the maximum bounds of the
        voxels.
    """
    bounds_min = voxel_indices * voxel_size + voxel_offset
    bounds_max = bounds_min + voxel_size

    return bounds_min, bounds_max


def distance_points_box(
    points: torch.Tensor,
    box_center: torch.Tensor,
    box_dim: torch.Tensor,
    box_rot: torch.Tensor,
):
    """
    Computes the shortest distance from each point to each box.

    The distance represents the distance of the point to the closest point on
    the box surface. If the point is inside the box, the distance is zero.

    Args:
        points (torch.Tensor):
            A tensor of shape (N, ndim) representing N points.
        center (torch.Tensor):
            A tensor of shape (M, ndim) representing the center positions of M boxes.
        dim (torch.Tensor):
            A tensor of shape (M, ndim) representing the box dimensions.
        rot (torch.Tensor):
            A tensor of shape (M,) representing the rotation angles of each box.

    Returns:
        torch.Tensor:
            A float tensor of shape (M, N) representing the minimum distance
            from each point to the closest plane of each box (or zero if the
            point lies inside the box).
    """
    normals, d_min, d_max = box_planes(box_center, box_dim, box_rot)

    # Project the points onto the normals (shape [nbox, ndim (normals), npoints]).
    proj = torch.einsum("bna,pa->bnp", normals, points)

    # Compute distances to the bounding planes
    norm = torch.linalg.vector_norm(normals, dim=2)  # pylint: disable=not-callable
    dist_min = torch.abs(proj - d_min[:, :, None]) / norm[:, :, None]
    dist_max = torch.abs(proj - d_max[:, :, None]) / norm[:, :, None]

    # Select the minimum distance for each box-relative axis (i.e., normal)
    dist = torch.minimum(dist_min, dist_max)

    # Set the distance to zero for points inside the relative axis/normal bounds
    inside = (proj >= d_min[:, :, None]) & (proj <= d_max[:, :, None])
    dist = dist * ~inside

    # Compute the euclidean distance
    return torch.linalg.vector_norm(dist, dim=1)  # pylint: disable=not-callable


def distance_points_box_axis_relative(
    points: torch.Tensor,
    box_center: torch.Tensor,
    box_dim: torch.Tensor,
    box_rot: torch.Tensor,
):
    """
    Computes the shortest non-normalized per-axis distance from each point to
    each box.

    The distance represents the distance of the point to the closest point on
    the box surface. If the point is inside the box, the distance is zero.

    The distances are relative to the box dimensions/axes, meaning that the
    distance is not normalized and distances for different boxes have different
    scales.

    Args:
        points (torch.Tensor):
            A tensor of shape (N, ndim) representing N points.
        center (torch.Tensor):
            A tensor of shape (M, ndim) representing the center positions of M boxes.
        dim (torch.Tensor):
            A tensor of shape (M, ndim) representing the box dimensions.
        rot (torch.Tensor):
            A tensor of shape (M,) representing the rotation angles of each box.

    Returns:
        torch.Tensor:
            A float tensor of shape (M, N, ndim) representing the minimum
            distance from each point to the closest plane of each box (or zero
            if the point lies inside the box), separated per axis.
    """
    normals, d_min, d_max = box_planes(box_center, box_dim, box_rot)

    # Project the points onto the normals (shape [nbox, ndim (normals), npoints]).
    proj = torch.einsum("bna,pa->bpn", normals, points)

    # Compute distances to the bounding planes
    dist_min = torch.abs(proj - d_min[:, None, :])
    dist_max = torch.abs(proj - d_max[:, None, :])

    # Select the minimum distance for each box-relative axis (i.e., normal)
    dist = torch.minimum(dist_min, dist_max)

    # Set the distance to zero for points inside the relative axis/normal bounds
    inside = (proj >= d_min[:, None, :]) & (proj <= d_max[:, None, :])

    return dist * ~inside


def distance_points_voxel(
    points: torch.Tensor,
    voxel_indices: torch.Tensor,
    voxel_size: torch.Tensor,
    voxel_offset: torch.Tensor,
):
    """
    Computes the shortest distance from each point to each voxel.

    The distance represents the distance of the closest point on the voxel
    surface to the point. If the point is inside the voxel, the distance is zero.

    Args:
        points (torch.Tensor):
            A tensor of shape (N, ndim) representing N points.
        voxel_indices (torch.Tensor):
            A tensor of shape (M, ndim) representing the indices of the voxels.
        voxel_size (torch.Tensor):
            A tensor of shape (ndim,) representing the size of the voxels.
        voxel_offset (torch.Tensor):
            A tensor of shape (ndim,) representing the offset of the voxel grid.

    Returns:
        torch.Tensor:
            A float tensor of shape (M, N) representing the minimum distance
            from each point to the closest plane of each voxel (or zero if the
            point lies inside the voxel).
    """

    # Compute the voxel bounds
    d_min, d_max = voxel_bounds(voxel_indices, voxel_size, voxel_offset)

    # Compute the per-axis distances to the voxel bounds
    dist_min = torch.abs(points[None, :, :] - d_min[:, None, :])
    dist_max = torch.abs(points[None, :, :] - d_max[:, None, :])

    # Select the minimum distance for each axis
    dist = torch.minimum(dist_min, dist_max)  # [nvox, npoints, ndim]

    # Set the distance to zero for points inside the voxel bounds
    inside = points[None, :, :] >= d_min[:, None, :]
    inside &= points[None, :, :] <= d_max[:, None, :]
    dist = dist * ~inside

    # Compute the euclidean distance
    return torch.linalg.vector_norm(dist, dim=2)  # pylint: disable=not-callable
