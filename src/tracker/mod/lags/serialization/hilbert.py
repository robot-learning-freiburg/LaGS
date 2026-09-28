# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * numpy-hilbert-curve (https://github.com/PrincetonLIPS/numpy-hilbert-curve),
#   Copyright (c) 2020 Princeton Laboratory for Intelligent Probabilistic
#   Systems, licensed under MIT.
# The original NumPy implementation was adapted to use PyTorch.
# See the LICENSES/ directory for full license texts.

"""
Hilbert curve encoding and decoding.

Based on the work by John Skilling as described in:

  Skilling, J. (2004, April). Programming the Hilbert curve. In AIP Conference
    Proceedings (Vol. 707, No. 1, pp. 381-387). American Institute of Physics.
"""

import math

import torch


def right_shift(binary, k=1, axis=-1):
    # If we're shifting the whole thing, just return zeros.
    if binary.shape[axis] <= k:
        return torch.zeros_like(binary)

    # Determine the padding pattern.
    # padding = [(0,0)] * len(binary.shape)
    # padding[axis] = (k,0)

    # Determine the slicing pattern to eliminate just the last one.
    slicing = [slice(None)] * len(binary.shape)
    slicing[axis] = slice(None, -k)
    shifted = torch.nn.functional.pad(
        binary[tuple(slicing)], (k, 0), mode="constant", value=0
    )

    return shifted


def binary2gray(binary, axis=-1):
    shifted = right_shift(binary, axis=axis)

    # Do the X ^ (X >> 1) trick.
    gray = torch.logical_xor(binary, shifted)

    return gray


def gray2binary(gray, axis=-1):
    # Loop the log2(bits) number of times necessary, with shift and xor.
    shift = 2 ** (math.ceil(math.log2(gray.shape[axis])) - 1)
    while shift > 0:
        gray = torch.logical_xor(gray, right_shift(gray, shift))
        shift = shift // 2
    return gray


@torch.compile(mode="reduce-overhead", fullgraph=True)
def encode(points, dims, depth):
    # pylint: disable=too-many-locals

    # Keep around the original shape for later.
    orig_shape = points.shape
    bitpack_mask = 1 << torch.arange(0, 8, device=points.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    if orig_shape[-1] != dims:
        raise ValueError(
            f"Expected points.shape[-1] == dims, got {orig_shape[-1]} != {dims}"
        )

    if dims * depth > 63:
        raise ValueError(
            f"dims={dims} and depth={depth} for {dims * depth} "
            f"bits total exceeds int64 capacity"
        )

    # Treat the location integers as 64-bit unsigned and then split them up into
    # a sequence of uint8s.  Preserve the association by dimension.
    points_uint8 = points.long().view(torch.uint8).reshape((-1, dims, 8)).flip(-1)

    # Now turn these into bits and truncate to num_bits.
    gray = (
        points_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[..., -depth:]
    )

    # Run the decoding process the other way.
    # Iterate forwards through the bits.
    for bit in range(0, depth):
        # Iterate forwards through the dimensions.
        for dim in range(0, dims):
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]).repeat(1, gray.shape[2] - bit - 1),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Now flatten out.
    gray = gray.swapaxes(1, 2).reshape((-1, depth * dims))

    # Convert Gray back to binary.
    keys = gray2binary(gray)

    # Pad back out to 64 bits.
    extra_dims = 64 - depth * dims
    keys = torch.nn.functional.pad(keys, (extra_dims, 0), "constant", 0)

    # Convert binary values into uint8s.
    keys = (
        (keys.flip(-1).reshape((-1, 8, 8)) * bitpack_mask)
        .sum(2)
        .squeeze()
        .to(dtype=torch.uint8)
    )

    # Convert uint8s into uint64s.
    keys = keys.view(torch.int64).squeeze()

    # Return them in the expected shape.
    return keys.view(*orig_shape[:-1])


@torch.compile(mode="reduce-overhead", fullgraph=True)
def decode(keys, dims, depth):
    # pylint: disable=too-many-locals

    if dims * depth > 64:
        raise ValueError(
            f"num_dims={dims} and num_bits={depth} for {dims * depth} "
            f"bits total exceeds uint64 capacity"
        )

    # Keep around the shape for later.
    orig_shape = keys.shape
    bitpack_mask = 2 ** torch.arange(0, 8, device=keys.device)
    bitpack_mask_rev = bitpack_mask.flip(-1)

    # Treat each of the hilberts as a sequence of eight uint8.
    # This treats all of the inputs as uint64 and makes things uniform.
    key_uint8 = (
        keys.ravel().to(dtype=torch.int64).view(torch.uint8).reshape((-1, 8)).flip(-1)
    )

    # Turn these lists of uints into lists of bits and then truncate to the size
    # we actually need for using Skilling's procedure.
    key_bits = (
        key_uint8.unsqueeze(-1)
        .bitwise_and(bitpack_mask_rev)
        .ne(0)
        .byte()
        .flatten(-2, -1)[:, -dims * depth :]
    )

    # Take the sequence of bits and Gray-code it.
    gray = binary2gray(key_bits)
    gray = gray.reshape((-1, depth, dims)).swapaxes(1, 2)

    # Iterate backwards through the bits.
    for bit in range(depth - 1, -1, -1):
        # Iterate backwards through the dimensions.
        for dim in range(dims - 1, -1, -1):
            # Identify which ones have this bit active.
            mask = gray[:, dim, bit]

            # Where this bit is on, invert the 0 dimension for lower bits.
            gray[:, 0, bit + 1 :] = torch.logical_xor(
                gray[:, 0, bit + 1 :], mask[:, None]
            )

            # Where the bit is off, exchange the lower bits with the 0 dimension.
            to_flip = torch.logical_and(
                torch.logical_not(mask[:, None]),
                torch.logical_xor(gray[:, 0, bit + 1 :], gray[:, dim, bit + 1 :]),
            )
            gray[:, dim, bit + 1 :] = torch.logical_xor(
                gray[:, dim, bit + 1 :], to_flip
            )
            gray[:, 0, bit + 1 :] = torch.logical_xor(gray[:, 0, bit + 1 :], to_flip)

    # Pad back out to 64 bits.
    extra_dims = 64 - depth
    points = torch.nn.functional.pad(gray, (extra_dims, 0), "constant", 0)

    # Now chop these up into blocks of 8.
    points = points.flip(-1).reshape((-1, dims, 8, 8))

    # Take those blocks and turn them unto uint8s.
    points = (points * bitpack_mask).sum(3).squeeze().to(dtype=torch.uint8)

    # Finally, treat these as uint64s.
    points = points.view(torch.int64)

    # Return them in the expected shape.
    return points.view((*orig_shape, dims))
