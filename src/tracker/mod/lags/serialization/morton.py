# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch


def _dilate_bits_2d(n: torch.Tensor) -> torch.Tensor:
    """Dilate bits by inserting 1 zero between each bit for 2D Morton encoding."""
    n = (n | (n << 16)) & 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def _undilate_bits_2d(n: torch.Tensor) -> torch.Tensor:
    """Undilate bits by removing 1 zero between each bit for 2D Morton decoding."""
    n = n & 0x55555555
    n = (n | (n >> 1)) & 0x33333333
    n = (n | (n >> 2)) & 0x0F0F0F0F
    n = (n | (n >> 4)) & 0x00FF00FF
    n = (n | (n >> 8)) & 0x0000FFFF
    n = (n | (n >> 16)) & 0x0000FFFF
    return n


def _dilate_bits_3d(n: torch.Tensor) -> torch.Tensor:
    """Dilate bits by inserting 2 zeros between each bit for 3D Morton encoding."""
    n = (n | (n << 16)) & 0x030000FF
    n = (n | (n << 8)) & 0x0300F00F
    n = (n | (n << 4)) & 0x030C30C3
    n = (n | (n << 2)) & 0x09249249
    return n


def _undilate_bits_3d(n: torch.Tensor) -> torch.Tensor:
    """Undilate bits by removing 2 zeros between each bit for 3D Morton decoding."""
    n = n & 0x09249249
    n = (n | (n >> 2)) & 0x030C30C3
    n = (n | (n >> 4)) & 0x0300F00F
    n = (n | (n >> 8)) & 0x030000FF
    n = (n | (n >> 16)) & 0x000000FF
    return n


@torch.compile(mode="reduce-overhead", fullgraph=True)
def encode(points: torch.Tensor, dims: int, depth: int):
    """Encode coordinates to Morton keys using magic number bit manipulation.

    Supports both 2D and 3D encoding based on dims parameter.

    Args:
      points (torch.Tensor): Coordinates with shape (..., dims) where dims is 2 or 3.
      dims (int): Number of dimensions (2 or 3).
      depth (int): The depth of the shuffled key. Max: 32 for 2D, 21 for 3D.

    Returns:
      torch.Tensor: Morton keys with same shape as points[..., :-1].
    """

    if points.shape[-1] != dims:
        raise ValueError(
            f"Expected points.shape[-1] == dims, got {points.shape[-1]} != {dims}"
        )

    if dims not in [2, 3]:
        raise ValueError(f"dims must be 2 or 3, got {dims}")

    if dims * depth > 64:
        raise ValueError(f"dims * depth = {dims * depth} exceeds 64-bit limit")

    # Extract coordinates
    coords = points.long()
    mask = (1 << depth) - 1

    if dims == 2:
        x, y = coords[..., 0] & mask, coords[..., 1] & mask
        key = _dilate_bits_2d(y) | (_dilate_bits_2d(x) << 1)
    else:  # dims == 3
        x, y, z = coords[..., 0] & mask, coords[..., 1] & mask, coords[..., 2] & mask
        key = _dilate_bits_3d(z) | (_dilate_bits_3d(y) << 1) | (_dilate_bits_3d(x) << 2)

    return key


@torch.compile(mode="reduce-overhead", fullgraph=True)
def decode(keys: torch.Tensor, dims: int, depth: int):
    """Decode Morton keys to coordinates using magic number bit manipulation.

    Supports both 2D and 3D decoding based on dims parameter.

    Args:
      keys (torch.Tensor): Morton keys to decode.
      dims (int): Number of dimensions (2 or 3).
      depth (int): The depth of the shuffled key. Max: 32 for 2D, 21 for 3D.

    Returns:
      torch.Tensor: Coordinates with shape (*keys.shape, dims).
    """

    if dims not in [2, 3]:
        raise ValueError(f"dims must be 2 or 3, got {dims}")

    if dims * depth > 64:
        raise ValueError(f"dims * depth = {dims * depth} exceeds 64-bit limit")

    # Handle the case where we got handed a naked integer
    keys = torch.atleast_1d(keys)
    orig_shape = keys.shape

    # Mask to depth
    mask = (1 << depth) - 1

    if dims == 2:
        # 2D Morton decoding: extract x and y bits
        x = _undilate_bits_2d(keys >> 1) & mask
        y = _undilate_bits_2d(keys) & mask
        coords = torch.stack([x, y], dim=-1)
    else:  # dims == 3
        # 3D Morton decoding: extract x, y, z bits
        x = _undilate_bits_3d(keys >> 2) & mask
        y = _undilate_bits_3d(keys >> 1) & mask
        z = _undilate_bits_3d(keys) & mask
        coords = torch.stack([x, y, z], dim=-1)

    return coords.reshape((*orig_shape, dims))
