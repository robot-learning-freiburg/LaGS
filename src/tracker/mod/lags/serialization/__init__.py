# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from .hilbert import decode as _hilbert_decode
from .hilbert import encode as _hilbert_encode
from .morton import decode as _morton_decode
from .morton import encode as _morton_encode


@torch.compile(mode="reduce-overhead", fullgraph=True)
def _encode_with_batch(encode_func, points, batch, dims, depth):
    code = encode_func(points, dims, depth)

    if batch is not None:
        assert dims * depth < 64
        code = code | (batch << (dims * depth))

    return code


@torch.compile(mode="reduce-overhead", fullgraph=True)
def _decode_with_batch(decode_func, keys, dims, depth):
    batch = keys >> (dims * depth)
    keys = keys & ((1 << (dims * depth)) - 1)

    return decode_func(keys, dims, depth), batch


@torch.compile(mode="reduce-overhead", fullgraph=True)
def hilbert_encode(points, batch=None, dims=3, depth=16):
    return _encode_with_batch(_hilbert_encode, points, batch, dims, depth)


@torch.compile(mode="reduce-overhead", fullgraph=True)
def hilbert_decode(keys, dims=3, depth=16):
    return _decode_with_batch(_hilbert_decode, keys, dims, depth)


@torch.compile(mode="reduce-overhead", fullgraph=True)
def morton_encode(points, batch=None, dims=3, depth=16):
    return _encode_with_batch(_morton_encode, points, batch, dims, depth)


@torch.compile(mode="reduce-overhead", fullgraph=True)
def morton_decode(keys, dims=3, depth=16):
    return _decode_with_batch(_morton_decode, keys, dims, depth)


@torch.no_grad()
@torch.compile(mode="reduce-overhead", fullgraph=True)
def encode(
    points: torch.Tensor,
    batch: torch.Tensor | None = None,
    dims: int | None = None,
    depth: int = 16,
    order: str = "hilbert",
    shuffle: tuple[int] | None = None,
    flip: tuple[bool] | None = None,
):
    if dims is None:
        dims = points.shape[-1]

    # Apply coordinate flipping first
    if flip is not None:
        points = _apply_flip(points, flip, dims, depth)

    # Apply coordinate shuffling
    if shuffle is not None:
        points = points[..., shuffle]

    if order == "hilbert":
        return hilbert_encode(points, batch, dims, depth)

    if order == "morton":
        return morton_encode(points, batch, dims, depth)

    raise ValueError(f"Unknown order: {order}")


@torch.no_grad()
@torch.compile(mode="reduce-overhead", fullgraph=True)
def decode(
    keys: torch.Tensor,
    dims: int = 3,
    depth: int = 16,
    order: str = "hilbert",
    unshuffle: tuple[int] | None = None,
    flip: tuple[bool] | None = None,
):
    if order == "hilbert":
        points, batch = hilbert_decode(keys, dims, depth)
    elif order == "morton":
        points, batch = morton_decode(keys, dims, depth)
    else:
        raise ValueError(f"Unknown order: {order}")

    # Reverse coordinate shuffling
    if unshuffle is not None:
        points = points[..., unshuffle]

    # Reverse coordinate flipping (same operation as encoding)
    if flip is not None:
        points = _apply_flip(points, flip, dims, depth)

    return points, batch


def _parse_shuffle(shuffle_spec: str):
    """Parse coordinate shuffle specification."""
    if not shuffle_spec:
        return None

    table = {
        "xy": None,
        "yx": (1, 0),
        "xyz": None,
        "xzy": (0, 2, 1),
        "yxz": (1, 0, 2),
        "yzx": (1, 2, 0),
        "zxy": (2, 0, 1),
        "zyx": (2, 1, 0),
    }
    return table[shuffle_spec]


def _parse_shuffle_reverse(shuffle_spec: str):
    """Parse coordinate shuffle specification for decoding (reverse mapping)."""
    if not shuffle_spec:
        return None

    table = {
        "xy": None,
        "yx": (1, 0),
        "xyz": None,
        "xzy": (0, 2, 1),
        "yxz": (1, 0, 2),
        "yzx": (2, 0, 1),
        "zxy": (1, 2, 0),
        "zyx": (2, 1, 0),
    }
    return table[shuffle_spec]


def _parse_flip(flip_spec: str):
    """Parse coordinate flip specification."""
    if not flip_spec:
        return None

    if not all(c in "+-" for c in flip_spec):
        raise ValueError(
            f"Invalid flip characters in '{flip_spec}', use only '+' and '-'"
        )

    return tuple(c == "-" for c in flip_spec)


def _parse_order_spec(spec: str, for_decode: bool = False):
    """Parse order specification into components."""
    parts = spec.split(":")
    order = parts[0]

    shuffle_spec = parts[1] if len(parts) > 1 else ""
    flip_spec = parts[2] if len(parts) > 2 else ""

    if for_decode:
        shuffle = _parse_shuffle_reverse(shuffle_spec)
    else:
        shuffle = _parse_shuffle(shuffle_spec)

    flip = _parse_flip(flip_spec)

    return order, shuffle, flip


def _apply_flip(points, flip, dims, depth):
    """Apply coordinate flipping."""
    if len(flip) != dims:
        raise ValueError(f"Flip string length ({len(flip)}) must match dims ({dims})")

    points = points.clone()  # Avoid modifying input
    max_coord = (1 << depth) - 1
    for i, should_flip in enumerate(flip):
        if should_flip:
            points[..., i] = max_coord - points[..., i]

    return points


@torch.no_grad()
def encode_named(points, batch=None, dims=None, depth=16, order="hilbert"):
    order, shuffle, flip = _parse_order_spec(order, for_decode=False)

    return encode(
        points,
        batch=batch,
        dims=dims,
        depth=depth,
        order=order,
        shuffle=shuffle,
        flip=flip,
    )


@torch.no_grad()
def decode_named(keys, dims=3, depth=16, order="hilbert"):
    return decode(
        keys,
        dims=dims,
        depth=depth,
        order=order.split(":")[0],
    )


class Curve:
    """Space-filling curve encoder/decoder with named order specifications.

    Args:
      dims (int): Number of dimensions (2 or 3).
      depth (int): The depth of the shuffled key. Max: 32 for 2D, 21 for 3D.
      order (str): Curve type and optional shuffle/flip, e.g. "hilbert:xyz:-++".
    """

    def __init__(
        self,
        dims: int = 3,
        depth: int = 16,
        order: str = "hilbert",
    ):
        order, shuffle, flip = _parse_order_spec(order, for_decode=False)
        order, unshuffle, flip = _parse_order_spec(order, for_decode=True)

        self.dims = dims
        self.depth = depth
        self.order = order
        self.shuffle = shuffle
        self.unshuffle = unshuffle
        self.flip = flip

    def encode(self, points, batch=None):
        return encode(
            points,
            batch=batch,
            dims=self.dims,
            depth=self.depth,
            order=self.order,
            shuffle=self.shuffle,
            flip=self.flip,
        )

    def decode(self, keys):
        return decode(
            keys,
            dims=self.dims,
            depth=self.depth,
            order=self.order,
            unshuffle=self.unshuffle,
            flip=self.flip,
        )
