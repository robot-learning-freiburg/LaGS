# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping, Sequence

import einops
import torch
import torch.nn.attention.flex_attention as flex
from torch import nn
from torch.nn import functional as F

from .. import serialization
from .base import Operation, StatefulOperation, attention_ops


class SlidingWindowSelfAttentionLayer(nn.Module):
    """Sliding window self-attention using FlexAttention.

    This layer applies sliding window self-attention to the input features.

    Args:
      embed_dim (int): Dimension of the input features.
      num_heads (int): Number of attention heads.
      window_size (int): Size of the sliding window.
      dropout (float): Dropout rate for attention weights. Default is 0.0.
      block_size (int): Block size for FlexAttention. Default is 128.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0,
        block_size: int = flex._DEFAULT_SPARSE_BLOCK_SIZE,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.block_size = block_size

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        self._cache_args = None
        self._cache_mask = None

    @torch.no_grad()
    @torch.compile(mode="default", dynamic=True)
    def _create_block_mask(
        self,
        window_size: int,
        seq_len: int,
        block_size: int,
        device: torch.device,
    ) -> flex.BlockMask:
        args = (window_size, seq_len, block_size, device)
        if args == self._cache_args:
            return self._cache_mask

        def sliding_window_mask(b, h, q_idx, kv_idx):
            # pylint: disable=unused-argument
            return torch.abs(q_idx - kv_idx) <= window_size // 2

        mask = flex.create_block_mask(
            sliding_window_mask,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            BLOCK_SIZE=block_size,
        )

        self._cache_args = args
        self._cache_mask = mask

        return mask

    @torch.compile(mode="default", dynamic=True)
    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = query.shape

        # Apply positional encoding if provided
        query_with_pos = query + query_pos if query_pos is not None else query

        # Apply projections
        q = self.q_proj(query_with_pos)
        k = self.k_proj(query_with_pos)
        v = self.v_proj(query)

        # Reshape for multi-head attention: [B, S, D] -> [B, H, S, D//H]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Create block mask
        block_mask = self._create_block_mask(
            window_size=self.window_size,
            seq_len=seq_len,
            block_size=self.block_size,
            device=query.device,
        )

        # Perform flex attention
        attn_out = flex.flex_attention(query=q, key=k, value=v, block_mask=block_mask)

        # Reshape back: [B, H, S, D//H] -> [B, S, D]
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        )

        # Apply output projection
        query_upd = self.out_proj(attn_out)
        query_upd = self.drop(query_upd)

        return self.norm(query + query_upd)


@attention_ops.register(key="SlidingWindowSelfAttention")
class SlidingWindowSelfAttentionOp(Operation):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int,
        dropout: float = 0.0,
        args_group: str | None = None,
    ):
        super().__init__(
            args_group=(
                args_group if args_group is not None else "SlidingWindowSelfAttention"
            )
        )

        self.op = SlidingWindowSelfAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.op(query, query_pos)


class BlockSparseSelfAttentionLayer(nn.Module):
    """Block-sparse self-attention using FlexAttention.

    This layer applies block-sparse self-attention where each block attends to
    itself and N consecutive neighboring blocks on each side.

    Args:
      embed_dim (int): Dimension of the input features.
      num_heads (int): Number of attention heads.
      num_neighbor_blocks (int): Number of neighboring blocks to attend to on each side.
      dropout (float): Dropout rate for attention weights. Default is 0.0.
      block_size (int): Block size for FlexAttention. Default is 128.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_neighbor_blocks: int,
        dropout: float = 0.0,
        block_size: int = flex._DEFAULT_SPARSE_BLOCK_SIZE,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_neighbor_blocks = num_neighbor_blocks
        self.dropout = dropout
        self.block_size = block_size

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        self._cache_args = None
        self._cache_mask = None

    @torch.no_grad()
    @torch.compile(mode="default", dynamic=True)
    def _create_block_sparse_mask(
        self,
        num_neighbor_blocks: int,
        seq_len: int,
        block_size: int,
        device: torch.device,
    ) -> flex.BlockMask:
        # pylint: disable=too-many-locals

        # Cache masks for given arguments
        args = (num_neighbor_blocks, seq_len, block_size, device)
        if args == self._cache_args:
            return self._cache_mask

        # Create mask_mod
        def block_sparse_mask(b, h, q_idx, kv_idx):
            # pylint: disable=unused-argument
            # Convert token indices to block indices
            q_block = q_idx // block_size
            kv_block = kv_idx // block_size

            # Check if blocks are within the neighbor range
            block_diff = torch.abs(q_block - kv_block)
            return block_diff <= num_neighbor_blocks

        # TODO: this can be optimized by creating the mask manually
        mask = flex.create_block_mask(
            block_sparse_mask,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            BLOCK_SIZE=block_size,
        )

        self._cache_args = args
        self._cache_mask = mask

        return mask

    @torch.compile(mode="default", dynamic=True)
    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = query.shape

        # Apply positional encoding if provided
        query_with_pos = query + query_pos if query_pos is not None else query

        # Apply projections
        q = self.q_proj(query_with_pos)
        k = self.k_proj(query_with_pos)
        v = self.v_proj(query)

        # Reshape for multi-head attention: [B, S, D] -> [B, H, S, D//H]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Create block sparse mask
        block_mask = self._create_block_sparse_mask(
            num_neighbor_blocks=self.num_neighbor_blocks,
            seq_len=seq_len,
            block_size=self.block_size,
            device=query.device,
        )

        # Perform flex attention
        attn_out = flex.flex_attention(query=q, key=k, value=v, block_mask=block_mask)

        # Reshape back: [B, H, S, D//H] -> [B, S, D]
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        )

        # Apply output projection
        query_upd = self.out_proj(attn_out)
        query_upd = self.drop(query_upd)

        return self.norm(query + query_upd)


@attention_ops.register(key="BlockSparseSelfAttention")
class BlockSparseSelfAttentionOp(Operation):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_neighbor_blocks: int,
        dropout: float = 0.0,
        block_size: int = flex._DEFAULT_SPARSE_BLOCK_SIZE,
        args_group: str | None = None,
    ):
        super().__init__(
            args_group=(
                args_group if args_group is not None else "BlockSparseSelfAttention"
            )
        )

        self.op = BlockSparseSelfAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_neighbor_blocks=num_neighbor_blocks,
            dropout=dropout,
            block_size=block_size,
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.op(query, query_pos)


@torch.compile(mode="default")
def _pad_images_to_blocks(
    images: torch.Tensor,  # [b, num_cams, c, h, w]
    block_size: tuple[int, int],  # (block_size_h, block_size_w)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad images to complete block boundaries and create padding masks."""
    b, num_cams, _, h, w = images.shape
    block_size_h, block_size_w = block_size

    # Calculate padding needed
    pad_h = (block_size_h - h % block_size_h) % block_size_h
    pad_w = (block_size_w - w % block_size_w) % block_size_w

    # Pad images
    if pad_h > 0 or pad_w > 0:
        padded_images = F.pad(images, (0, pad_w, 0, pad_h), value=0.0)
    else:
        padded_images = images

    # Create padding mask (True for valid pixels, False for padding)
    new_h, new_w = padded_images.shape[-2:]
    mask = torch.ones(b, num_cams, new_h, new_w, device=images.device, dtype=torch.bool)
    if pad_h > 0:
        mask[:, :, h:, :] = False
    if pad_w > 0:
        mask[:, :, :, w:] = False

    return padded_images, mask


@torch.compile(mode="default")
def _linearize_image_blocks(
    images: torch.Tensor,  # [b, num_cams, c, h, w]
    block_size: tuple[int, int],  # (block_size_h, block_size_w)
) -> torch.Tensor:
    """Linearize images so that each block is stored consecutively in memory."""
    _b, _num_cams, _c, h, w = images.shape
    block_size_h, block_size_w = block_size

    # Ensure dimensions are divisible by block sizes
    assert (
        h % block_size_h == 0 and w % block_size_w == 0
    ), f"Image dims ({h}, {w}) must be divisible by block_sizes {block_size}"

    # Make block elements contiguous in memory
    blocks = einops.rearrange(
        images,
        "b cam c (bh block_h) (bw block_w) -> b cam bh bw (block_h block_w) c",
        block_h=block_size_h,
        block_w=block_size_w,
    )

    # Flatten to token sequence
    blocks = einops.rearrange(
        blocks, "b cam bh bw block_tokens c -> b (cam bh bw block_tokens) c"
    )

    return blocks


@torch.compile(mode="default")
def _compute_block_hits(
    reference_points: torch.Tensor,  # [b, num_cams, num_query, 2]
    reference_points_mask: torch.Tensor,  # [b, num_cams, num_query]
    image_size: tuple[int, int],  # (h, w)
    block_size: tuple[int, int],  # (block_size_h, block_size_w)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute which blocks are hit by reference points.

    Returns:
        block_indices: [b, num_cams, num_query, 2] - (block_h, block_w) indices
        valid_hits: [b, num_cams, num_query] - whether the hit is valid
    """
    h, w = image_size
    block_size_h, block_size_w = block_size

    # Convert from normalized [-1, 1] to pixel coordinates
    pts_pixel = torch.stack(
        [
            (reference_points[..., 0] + 1) * w / 2,  # x coordinate
            (reference_points[..., 1] + 1) * h / 2,  # y coordinate
        ],
        dim=-1,
    )

    # Convert to block coordinates
    pts_block = torch.stack(
        [
            torch.floor(pts_pixel[..., 0] / block_size_w).long(),  # block_w index
            torch.floor(pts_pixel[..., 1] / block_size_h).long(),  # block_h index
        ],
        dim=-1,
    )

    # Check bounds
    num_blocks_h = (h + block_size_h - 1) // block_size_h
    num_blocks_w = (w + block_size_w - 1) // block_size_w
    valid = (
        reference_points_mask
        & (pts_block[..., 0] >= 0)
        & (pts_block[..., 0] < num_blocks_w)
        & (pts_block[..., 1] >= 0)
        & (pts_block[..., 1] < num_blocks_h)
    )

    return pts_block, valid


class BlockSparseMultiViewCrossAttentionLayer(nn.Module):
    """Block-sparse multi-view image cross-attention using FlexAttention.

    This layer divides images into blocks, linearizes them block-wise, and performs
    efficient cross-attention where each query attends to the block hit by its
    reference coordinate plus N neighboring blocks.

    Args:
        embed_dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads.
        image_block_size (int | tuple[int, int]): Image block size. If int, broadcasts to (h, w).
        num_neighbor_blocks (int): Number of neighboring blocks to attend to on each side.
        dropout (float): Dropout rate for attention weights. Default is 0.0.
        block_size (int): Block size for FlexAttention. Default is 128.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        image_block_size: int | tuple[int, int] = 8,
        num_neighbor_blocks: int = 1,
        dropout: float = 0.0,
        block_size: int = flex._DEFAULT_SPARSE_BLOCK_SIZE,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Handle broadcasting: int -> (int, int)
        if isinstance(image_block_size, int):
            image_block_size = (image_block_size, image_block_size)

        self.image_block_size = image_block_size

        self.num_neighbor_blocks = num_neighbor_blocks
        self.dropout = dropout
        self.block_size = block_size

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    @torch.no_grad()
    @torch.compile(mode="default")
    def _create_block_sparse_cross_mask(
        self,
        reference_points: torch.Tensor,  # [b, num_cams, num_query, 2]
        reference_points_mask: torch.Tensor,  # [b, num_cams, num_query]
        image_size: tuple[int, int],  # (h, w)
        image_size_padded: tuple[int, int],  # (padded_h, padded_w)
        image_block_size: tuple[int, int],  # (bh, bw)
        image_mask: torch.Tensor,  # [b, num_cams, h, w]
        num_neighbor_blocks: int,
        block_size: int | tuple[int, int],
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> flex.BlockMask:
        """Create block mask for cross-attention."""
        # pylint: disable=too-many-locals

        _, num_cams, _, _ = reference_points.shape

        # Compute block hits
        block_hits, valid_hits = _compute_block_hits(
            reference_points, reference_points_mask, image_size, image_block_size
        )

        # Precompute all block coordinates and attention patterns
        h, w = image_size_padded
        block_size_h, block_size_w = image_block_size
        h_blocks = (h + block_size_h - 1) // block_size_h
        w_blocks = (w + block_size_w - 1) // block_size_w
        tokens_per_block = block_size_h * block_size_w
        blocks_per_cam = h_blocks * w_blocks
        tokens_per_cam = blocks_per_cam * tokens_per_block

        def cross_attention_mask(b_idx, h_idx, q_idx, kv_idx):
            # pylint: disable=unused-argument

            # Determine which camera and block this kv_idx belongs to
            cam_idx = kv_idx // tokens_per_cam
            masked = cam_idx < num_cams

            # Check if this query has a valid hit in this camera
            masked = masked & valid_hits[b_idx, cam_idx, q_idx]

            # Get hit block for this query in this camera
            hit_w, hit_h = block_hits[b_idx, cam_idx, q_idx]

            # Convert kv_idx to block coordinates and pixel position
            local_token_idx = kv_idx % tokens_per_cam
            local_block_idx = local_token_idx // tokens_per_block
            block_h = local_block_idx // w_blocks
            block_w = local_block_idx % w_blocks

            # Check if within neighbor range
            h_diff = torch.abs(block_h - hit_h)
            w_diff = torch.abs(block_w - hit_w)

            masked = masked & (h_diff <= num_neighbor_blocks)
            masked = masked & (w_diff <= num_neighbor_blocks)

            # Get pixel position within the block
            pixel_in_block = local_token_idx % tokens_per_block
            pixel_h_in_block = pixel_in_block // block_size_w
            pixel_w_in_block = pixel_in_block % block_size_w

            # Convert to global pixel coordinates
            pixel_h = block_h * block_size_h + pixel_h_in_block
            pixel_w = block_w * block_size_w + pixel_w_in_block

            # Check if this pixel is in the valid (non-padded) region
            masked = masked & image_mask[b_idx, cam_idx, pixel_h, pixel_w]

            return masked

        return flex.create_block_mask(
            cross_attention_mask,
            B=block_hits.shape[0],
            H=None,
            Q_LEN=query_len,
            KV_LEN=key_len,
            device=device,
            BLOCK_SIZE=block_size,
        )

    # @torch.compile(mode="default")
    def forward(
        self,
        query: torch.Tensor,  # [b, num_query, embed_dim]
        feats: torch.Tensor,  # [b, num_cams, c, h, w]
        reference_points: torch.Tensor,  # [b, num_cams, num_query, 2]
        reference_points_mask: torch.Tensor,  # [b, num_cams, num_query]
        query_pos: torch.Tensor | None = None,
        feats_pos: torch.Tensor | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        # pylint: disable=too-many-locals
        _, _, _, h, w = feats.shape

        batch_size, num_query, embed_dim = query.shape

        # Pad images to block boundaries
        feats, padding_mask = _pad_images_to_blocks(feats, self.image_block_size)

        if feats_pos is not None:
            feats_pos, _ = _pad_images_to_blocks(feats_pos, self.image_block_size)

        # Linearize image blocks
        kv_seq = _linearize_image_blocks(feats, self.image_block_size)

        if feats_pos is not None:
            feats_pos = _linearize_image_blocks(feats_pos, self.image_block_size)

        # Apply projections and add positional encodings
        q = self.q_proj(query + query_pos if query_pos is not None else query)
        k = self.k_proj(kv_seq + feats_pos if feats_pos is not None else kv_seq)
        v = self.v_proj(kv_seq)

        # Reshape for multi-head attention
        q = q.view(batch_size, num_query, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Create block sparse mask
        padded_h, padded_w = feats.shape[-2:]
        block_mask = self._create_block_sparse_cross_mask(
            reference_points=reference_points,
            reference_points_mask=reference_points_mask,
            image_size=(h, w),
            image_size_padded=(padded_h, padded_w),
            image_block_size=self.image_block_size,
            image_mask=padding_mask,
            num_neighbor_blocks=self.num_neighbor_blocks,
            block_size=self.block_size,
            query_len=num_query,
            key_len=kv_seq.shape[1],
            device=query.device,
        )

        # Perform flex attention
        attn_out = flex.flex_attention(query=q, key=k, value=v, block_mask=block_mask)

        # Reshape back
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, num_query, embed_dim)
        )

        # Apply output projection
        query_upd = self.out_proj(attn_out)
        query_upd = self.drop(query_upd)

        return self.norm(query + query_upd)


@attention_ops.register(key="BlockSparseMultiViewCrossAttention")
class BlockSparseMultiViewImageCrossAttentionOp(Operation):
    def __init__(
        self,
        embed_dim: int,
        features: str,
        num_heads: int,
        image_block_size: int | tuple[int, int] = 8,
        num_neighbor_blocks: int = 1,
        dropout: float = 0.0,
        block_size: int = flex._DEFAULT_SPARSE_BLOCK_SIZE,
        args_group: str | None = None,
    ):
        super().__init__(
            args_group=(
                args_group
                if args_group is not None
                else "BlockSparseImageCrossAttention"
            )
        )

        self.features = features

        self.op = BlockSparseMultiViewCrossAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            image_block_size=image_block_size,
            num_neighbor_blocks=num_neighbor_blocks,
            dropout=dropout,
            block_size=block_size,
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # pylint: disable=unused-argument
        assert features is not None
        assert self.features in features

        return self.op(
            query=query,
            query_pos=query_pos,
            **features[self.features],
        )


class SlidingWindowMultiViewCrossAttentionLayer(nn.Module):
    """Window-based multi-view image cross-attention using FlexAttention.

    This layer performs cross-attention where each query attends to a fixed MxN window
    around its reference coordinate in each camera view.

    Args:
        embed_dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads.
        window_size (int | tuple[int, int]): Window size. If int, broadcasts to (h, w).
        dropout (float): Dropout rate for attention weights. Default is 0.0.
        block_size (int): Block size for FlexAttention. Default is 128.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        window_size: int | tuple[int, int] = 7,
        dropout: float = 0.0,
        block_size: int = flex._DEFAULT_SPARSE_BLOCK_SIZE,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Handle broadcasting: int -> (int, int)
        if isinstance(window_size, int):
            window_size = (window_size, window_size)

        self.window_size = window_size
        self.dropout = dropout
        self.block_size = block_size

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    @torch.no_grad()
    @torch.compile(mode="default")
    def _create_window_cross_mask(
        self,
        reference_points: torch.Tensor,  # [b, num_cams, num_query, 2] in [-1, 1]
        reference_points_mask: torch.Tensor,  # [b, num_cams, num_query]
        image_size: tuple[int, int],  # (h, w)
        window_size: tuple[int, int],  # (window_h, window_w)
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> flex.BlockMask:
        """Create window mask for cross-attention."""
        # pylint: disable=too-many-locals

        batch_size, num_cams, _, _ = reference_points.shape
        h, w = image_size
        window_h, window_w = window_size

        # Convert reference points from normalized [-1, 1] to pixel coordinates
        pts_pixel = torch.stack(
            [
                (reference_points[..., 0] + 1) * w / 2,  # x coordinate
                (reference_points[..., 1] + 1) * h / 2,  # y coordinate
            ],
            dim=-1,
        )

        # Clamp to valid pixel ranges
        pts_pixel = torch.stack(
            [
                torch.clamp(pts_pixel[..., 0], 0, w - 1),  # x coordinate
                torch.clamp(pts_pixel[..., 1], 0, h - 1),  # y coordinate
            ],
            dim=-1,
        )

        def window_attention_mask(b_idx, h_idx, q_idx, kv_idx):
            # pylint: disable=unused-argument

            # Determine which camera this kv_idx belongs to
            tokens_per_cam = h * w
            cam_idx = kv_idx // tokens_per_cam

            # Check if camera index is valid
            masked = cam_idx < num_cams

            # Check if this query has valid reference points in this camera
            masked = masked & reference_points_mask[b_idx, cam_idx, q_idx]

            # Get reference pixel coordinates for this query in this camera
            ref_x, ref_y = pts_pixel[b_idx, cam_idx, q_idx]

            # Convert kv_idx to pixel coordinates
            local_token_idx = kv_idx % tokens_per_cam
            pixel_y = local_token_idx // w
            pixel_x = local_token_idx % w

            # Check if pixel is within the window around reference point
            window_half_h = window_h // 2
            window_half_w = window_w // 2

            y_diff = torch.abs(pixel_y - ref_y)
            x_diff = torch.abs(pixel_x - ref_x)

            masked = masked & (y_diff <= window_half_h) & (x_diff <= window_half_w)

            return masked

        return flex.create_block_mask(
            window_attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=query_len,
            KV_LEN=key_len,
            device=device,
            BLOCK_SIZE=self.block_size,
        )

    # @torch.compile(mode="default")
    def forward(
        self,
        query: torch.Tensor,  # [b, num_query, embed_dim]
        feats: torch.Tensor,  # [b, num_cams, c, h, w]
        reference_points: torch.Tensor,  # [b, num_cams, num_query, 2]
        reference_points_mask: torch.Tensor,  # [b, num_cams, num_query]
        query_pos: torch.Tensor | None = None,
        feats_pos: torch.Tensor | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        # pylint: disable=too-many-locals
        batch_size, num_query, embed_dim = query.shape
        _, _, _, h, w = feats.shape

        # Flatten image features: [b, num_cams, c, h, w] -> [b, num_cams*h*w, c]
        kv_seq = feats.flatten(start_dim=3).transpose(2, 3)  # [b, num_cams, h*w, c]
        kv_seq = kv_seq.flatten(start_dim=1, end_dim=2)  # [b, num_cams*h*w, c]

        if feats_pos is not None:
            feats_pos = feats_pos.flatten(start_dim=3).transpose(2, 3)
            feats_pos = feats_pos.flatten(start_dim=1, end_dim=2)

        # Apply projections and add positional encodings
        q = self.q_proj(query + query_pos if query_pos is not None else query)
        k = self.k_proj(kv_seq + feats_pos if feats_pos is not None else kv_seq)
        v = self.v_proj(kv_seq)

        # Reshape for multi-head attention
        q = q.view(batch_size, num_query, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Create window mask
        block_mask = self._create_window_cross_mask(
            reference_points=reference_points,
            reference_points_mask=reference_points_mask,
            image_size=(h, w),
            window_size=self.window_size,
            query_len=num_query,
            key_len=kv_seq.shape[1],
            device=query.device,
        )

        # Perform flex attention
        attn_out = flex.flex_attention(query=q, key=k, value=v, block_mask=block_mask)

        # Reshape back
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, num_query, embed_dim)
        )

        # Apply output projection
        query_upd = self.out_proj(attn_out)
        query_upd = self.drop(query_upd)

        return self.norm(query + query_upd)


@attention_ops.register(key="SlidingWindowMultiViewCrossAttention")
class SlidingWindowMultiViewCrossAttentionOp(Operation):
    def __init__(
        self,
        embed_dim: int,
        features: str,
        num_heads: int,
        window_size: int | tuple[int, int] = 7,
        dropout: float = 0.0,
        block_size: int = flex._DEFAULT_SPARSE_BLOCK_SIZE,
        args_group: str | None = None,
    ):
        super().__init__(
            args_group=(
                args_group
                if args_group is not None
                else "WindowBasedMultiViewCrossAttention"
            )
        )

        self.features = features

        self.op = SlidingWindowMultiViewCrossAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            dropout=dropout,
            block_size=block_size,
        )

    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # pylint: disable=unused-argument
        assert features is not None
        assert self.features in features

        return self.op(
            query=query,
            query_pos=query_pos,
            **features[self.features],
        )


# ==============================================================================
# Hierarchical Gaussian Processing: Cross-Stream Attention
# ==============================================================================


class SpatialWindowCrossStreamAttentionLayer(nn.Module):
    """3D spatial window cross-stream attention using FlexAttention.

    Allows queries in one stream (e.g., fine) to attend to context in another
    stream (e.g., coarse) within a 3D spatial window.

    This replaces the O(n²) torch.cdist() operations in CrossStreamAggregationOp
    and CrossStreamPropagationOp with efficient O(n*w) windowed attention.

    Args:
        embed_dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads.
        spatial_window (tuple[float, float, float]): Window size in meters (x, y, z).
            Example: (4.0, 4.0, 2.0) allows queries to attend within ±4m in x/y, ±2m in z.
        dropout (float): Dropout rate for attention weights. Default is 0.0.
        block_size (int): Block size for FlexAttention. Default is 128.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        spatial_window: tuple[float, float, float] | list[float] = (4.0, 4.0, 2.0),
        dropout: float = 0.0,
        block_size: int = flex._DEFAULT_SPARSE_BLOCK_SIZE,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # (window_x, window_y, window_z) in meters
        self.spatial_window = tuple(spatial_window)

        self.dropout = dropout
        self.block_size = block_size

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        # Projections (same as other attention layers)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    @torch.no_grad()
    @torch.compile(mode="default")
    def _create_spatial_window_mask(
        self,
        query_coords: torch.Tensor,  # [b, n_query, 3] in meters
        context_coords: torch.Tensor,  # [b, n_context, 3] in meters
        spatial_window: tuple[float, float, float],
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> flex.BlockMask:
        """Create spatial window mask for 3D cross-stream attention.

        Similar to SlidingWindowMultiViewCrossAttentionLayer._create_window_cross_mask
        but for 3D voxel coordinates instead of 2D image coordinates.
        """
        batch_size = query_coords.shape[0]
        window_x, window_y, window_z = spatial_window

        def spatial_window_attention_mask(b_idx, h_idx, q_idx, kv_idx):
            # pylint: disable=unused-argument

            # Get query and context coordinates
            q_coord = query_coords[b_idx, q_idx]  # [3]
            kv_coord = context_coords[b_idx, kv_idx]  # [3]

            # Compute spatial distance in each dimension
            dx = torch.abs(q_coord[0] - kv_coord[0])
            dy = torch.abs(q_coord[1] - kv_coord[1])
            dz = torch.abs(q_coord[2] - kv_coord[2])

            # Check if within spatial window
            masked = (dx <= window_x) & (dy <= window_y) & (dz <= window_z)

            return masked

        return flex.create_block_mask(
            spatial_window_attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=query_len,
            KV_LEN=key_len,
            device=device,
            BLOCK_SIZE=self.block_size,
        )

    def forward(
        self,
        query: torch.Tensor,  # [b, n_query, embed_dim]
        context: torch.Tensor,  # [b, n_context, embed_dim]
        query_coords: torch.Tensor,  # [b, n_query, 3]
        context_coords: torch.Tensor,  # [b, n_context, 3]
        query_pos: torch.Tensor | None = None,
        context_pos: torch.Tensor | None = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Forward pass for spatial window cross-stream attention.

        Args:
            query: Query features from one stream (e.g., fine)
            context: Context features from another stream (e.g., coarse)
            query_coords: 3D coordinates of queries in meters
            context_coords: 3D coordinates of context in meters
            query_pos: Position embeddings for queries
            context_pos: Position embeddings for context

        Returns:
            Updated query features
        """
        # pylint: disable=too-many-locals

        batch_size, num_query, embed_dim = query.shape
        num_context = context.shape[1]

        # Apply projections and add positional encodings
        q = self.q_proj(query + query_pos if query_pos is not None else query)
        k = self.k_proj(context + context_pos if context_pos is not None else context)
        v = self.v_proj(context)

        # Reshape for multi-head attention
        q = q.view(batch_size, num_query, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)

        k = k.view(batch_size, num_context, self.num_heads, self.head_dim)
        k = k.transpose(1, 2)

        v = v.view(batch_size, num_context, self.num_heads, self.head_dim)
        v = v.transpose(1, 2)

        # Create spatial window mask
        block_mask = self._create_spatial_window_mask(
            query_coords=query_coords,
            context_coords=context_coords,
            spatial_window=self.spatial_window,
            query_len=num_query,
            key_len=num_context,
            device=query.device,
        )

        # Perform flex attention
        attn_out = flex.flex_attention(query=q, key=k, value=v, block_mask=block_mask)

        # Reshape back
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, num_query, embed_dim)
        )

        # Apply output projection
        query_upd = self.out_proj(attn_out)
        query_upd = self.drop(query_upd)

        return self.norm(query + query_upd)


@attention_ops.register(key="SpatialWindowCrossStreamAttention")
class SpatialWindowCrossStreamAttentionOp(StatefulOperation):
    """Windowed cross-stream attention operation for hierarchical gaussians.

    This replaces the inefficient CrossStreamPropagationOp that uses torch.cdist.
    Instead of O(n²) distance computation, uses FlexAttention with spatial windows
    for O(n*w) complexity where w = window size.

    Config example:
        - stream: fine
          type: SpatialWindowCrossStreamAttention
          context_stream: medium
          num_heads: 8
          spatial_window: [4.0, 4.0, 2.0]  # window size in meters
    """

    def __init__(
        self,
        embed_dim: int,
        context_stream: str,  # name of context stream
        num_heads: int = 8,
        spatial_window: tuple[float, float, float] | list[float] = (4.0, 4.0, 2.0),
        dropout: float = 0.0,
        block_size: int = flex._DEFAULT_SPARSE_BLOCK_SIZE,
        args_group: str | Sequence[str] | None = None,
    ):
        super().__init__(
            args_group=(
                args_group if args_group is not None else "SpatialWindowCrossStream"
            )
        )

        self.context_stream = context_stream

        self.op = SpatialWindowCrossStreamAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            spatial_window=spatial_window,
            dropout=dropout,
            block_size=block_size,
        )

    def forward(
        self,
        query: torch.Tensor,  # [b, n_query, c]
        query_pos: torch.Tensor | None = None,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        query_coords: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[dict[str, Any], Mapping[str, Mapping[str, Any]]]:
        """Forward pass that accesses context stream from features dict.

        Expects features[context_stream] to contain:
            - 'feats': context features [b, n_context, c]
            - 'feats_coords': context coords [b, n_context, 3]
            - 'feats_pos': context position embeddings [b, n_context, c] (optional)
        """
        assert features is not None
        assert self.context_stream in features
        assert query_coords is not None

        context = features[self.context_stream]
        context_features = context["feats"]
        context_coords = context["feats_coords"]
        context_pos = context.get("feats_pos")

        # Perform windowed cross-stream attention
        query = self.op(
            query=query,
            context=context_features,
            query_coords=query_coords,
            context_coords=context_coords,
            query_pos=query_pos,
            context_pos=context_pos,
        )

        # Return updated inputs (query changed, other state unchanged)
        inputs = {
            "query": query,
            "query_pos": query_pos,
            "query_coords": query_coords,
            **kwargs,
        }

        return inputs, features


# ==============================================================================
# Hierarchical Gaussian Processing: Serialized Multi-Level Attention
# ==============================================================================


class SerializedMultiLevelSelfAttentionLayer(nn.Module):
    """Self-attention over concatenated multi-level queries with re-serialization.

    Instead of separate cross-stream operations, all hierarchy levels are:
    1. Concatenated into single sequence
    2. Re-serialized using space-filling curve (Hilbert/Morton)
    3. Attended uniformly with optional masking
    4. Deserialized back to original order
    5. Split back into separate streams

    This enables flexible cross-level communication without explicit cross-stream ops.

    Args:
        embed_dim (int): Dimension of the input features.
        num_heads (int): Number of attention heads.
        level_names (list[str]): Names of hierarchy levels (e.g., ["coarse", "medium", "fine"]).
        use_level_embeddings (bool): Add level ID embeddings to distinguish levels.
        use_per_level_projections (bool): Use separate Q/K/V projections per level instead of shared.
        serialize (bool): Enable re-serialization (critical for cross-level communication).
        sfc_order (str): Space-filling curve type ("hilbert" or "morton").
        sfc_depth (int): Curve depth (16 for 3D coords).
        coord_range (list[float]): Coordinate range [x_min, y_min, z_min, x_max, y_max, z_max]
            for normalization before encoding.
        attention_mask_type (str): Mask type after serialization:
            - "sliding-window": Sequential neighbors in serialized order (efficient!)
            - "block-sparse": Block-sparse pattern
            - "full": Full attention (expensive)
        window_size (int): Window size for sliding window attention. Ignored for other types.
        num_neighbor_blocks (int): Number of neighbor blocks for block-sparse. Ignored for others.
        dropout (float): Dropout rate for attention weights. Default is 0.0.
        block_size (int): Block size for FlexAttention. Default is 128.
    """  # pylint: disable=line-too-long

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        level_names: list[str],
        use_level_embeddings: bool = True,
        use_per_level_projections: bool = False,
        serialize: bool = True,
        sfc_order: str = "hilbert",
        sfc_depth: int = 16,
        coord_range: list[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        attention_mask_type: str = "sliding-window",
        window_size: int = 128,
        num_neighbor_blocks: int = 1,
        dropout: float = 0.0,
        block_size: int = flex._DEFAULT_SPARSE_BLOCK_SIZE,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.level_names = level_names
        self.num_levels = len(level_names)
        self.use_level_embeddings = use_level_embeddings
        self.use_per_level_projections = use_per_level_projections
        self.serialize = serialize
        self.sfc_order = sfc_order
        self.sfc_depth = sfc_depth
        self.attention_mask_type = attention_mask_type
        self.window_size = window_size
        self.num_neighbor_blocks = num_neighbor_blocks
        self.dropout = dropout
        self.block_size = block_size

        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads

        # Coordinate range for normalization
        coord_range = torch.as_tensor(coord_range, dtype=torch.float32)
        self.register_buffer("coord_range", coord_range, persistent=False)

        # Level embeddings to distinguish hierarchy levels
        if use_level_embeddings:
            self.level_embeddings = nn.Embedding(self.num_levels, embed_dim)
        else:
            self.level_embeddings = None

        # Q/K/V projections - either per-level or shared
        if use_per_level_projections:
            self.q_proj = nn.ModuleDict(
                {name: nn.Linear(embed_dim, embed_dim) for name in level_names}
            )
            self.k_proj = nn.ModuleDict(
                {name: nn.Linear(embed_dim, embed_dim) for name in level_names}
            )
            self.v_proj = nn.ModuleDict(
                {name: nn.Linear(embed_dim, embed_dim) for name in level_names}
            )
            # Per-level output projections as well
            self.out_proj = nn.ModuleDict(
                {name: nn.Linear(embed_dim, embed_dim) for name in level_names}
            )
        else:
            # Shared projections across all levels
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

        # Cache for attention masks
        self._cache_args = None
        self._cache_mask = None

        # Space-filling curve for serialization
        if serialize:
            self.curve = serialization.Curve(dims=3, depth=sfc_depth, order=sfc_order)

    @torch.no_grad()
    @torch.compile(mode="reduce-overhead", fullgraph=True)
    def _normalize_coords(
        self, coords: torch.Tensor  # [b, n, 3]
    ) -> torch.Tensor:  # [b, n, 3] in [0, 2^depth)
        """Normalize coordinates to [0, 2^depth) integer range for space-filling curve encoding."""
        # Normalize to [0, 1]
        coord_min = self.coord_range[:3]  # [x_min, y_min, z_min]
        coord_max = self.coord_range[3:]  # [x_max, y_max, z_max]
        coords_norm = (coords - coord_min) / (coord_max - coord_min)
        coords_norm = torch.clamp(coords_norm, 0.0, 1.0)

        # Scale to [0, 2^depth)
        max_coord = (1 << self.sfc_depth) - 1
        coords_int = (coords_norm * max_coord).long()

        return coords_int

    @torch.no_grad()
    @torch.compile(mode="default", dynamic=True)
    def _create_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> flex.BlockMask | None:
        """Create attention mask based on mask type."""
        # Check cache
        args = (seq_len, device)
        if args == self._cache_args:
            return self._cache_mask

        if self.attention_mask_type == "sliding-window":
            # Sliding window mask
            def sliding_window_mask(b, h, q_idx, kv_idx):
                # pylint: disable=unused-argument
                return torch.abs(q_idx - kv_idx) <= self.window_size // 2

            mask = flex.create_block_mask(
                sliding_window_mask,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                BLOCK_SIZE=self.block_size,
            )

            self._cache_args = args
            self._cache_mask = mask
            return mask

        if self.attention_mask_type == "block-sparse":
            # Block sparse mask
            def block_sparse_mask(b, h, q_idx, kv_idx):
                # pylint: disable=unused-argument
                q_block = q_idx // self.block_size
                kv_block = kv_idx // self.block_size
                block_diff = torch.abs(q_block - kv_block)
                return block_diff <= self.num_neighbor_blocks

            mask = flex.create_block_mask(
                block_sparse_mask,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                BLOCK_SIZE=self.block_size,
            )

            self._cache_args = args
            self._cache_mask = mask
            return mask

        if self.attention_mask_type == "full":
            # No mask - full attention
            self._cache_args = args
            self._cache_mask = None
            return None

        raise ValueError(f"Unknown attention_mask_type: {self.attention_mask_type}")

    @torch.compile(mode="default")
    def _gather_and_add_level_embeddings(
        self,
        inputs_per_level: dict[str, dict[str, torch.Tensor]],
    ) -> tuple[
        list[int], list[torch.Tensor], list[torch.Tensor | None], list[torch.Tensor]
    ]:
        """Gather features from all levels and add level embeddings.

        Returns:
            level_lengths: List of sequence lengths for each level
            all_feats_list: List of features (with level embeddings added)
            all_pos_list: List of positional encodings
            all_coords_list: List of coordinates
        """
        level_lengths = []
        all_feats_list = []
        all_pos_list = []
        all_coords_list = []

        for name in self.level_names:
            level_input = inputs_per_level[name]
            level_lengths.append(level_input["query"].shape[1])
            all_feats_list.append(level_input["query"])
            all_pos_list.append(level_input.get("query_pos"))
            all_coords_list.append(level_input["query_coords"])

        # Add level embeddings (before projections)
        if self.use_level_embeddings and self.level_embeddings is not None:
            for idx in range(len(self.level_names)):
                level_emb = self.level_embeddings.weight[idx]  # [c]
                all_feats_list[idx] = all_feats_list[idx] + level_emb

        return level_lengths, all_feats_list, all_pos_list, all_coords_list

    @torch.compile(mode="default")
    def _apply_qkv_projections(
        self,
        all_feats_list: list[torch.Tensor],
        all_pos_list: list[torch.Tensor | None],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Apply Q/K/V projections per level (before concatenation).

        Returns:
            q_list: List of query projections
            k_list: List of key projections
            v_list: List of value projections
        """
        q_list, k_list, v_list = [], [], []

        for idx, name in enumerate(self.level_names):
            feats = all_feats_list[idx]
            pos = all_pos_list[idx]
            feats_with_pos = feats + pos if pos is not None else feats

            # Use per-level or shared projections
            if self.use_per_level_projections:
                q_list.append(self.q_proj[name](feats_with_pos))
                k_list.append(self.k_proj[name](feats_with_pos))
                v_list.append(self.v_proj[name](feats))
            else:
                q_list.append(self.q_proj(feats_with_pos))
                k_list.append(self.k_proj(feats_with_pos))
                v_list.append(self.v_proj(feats))

        return q_list, k_list, v_list

    @torch.compile(mode="default")
    def _serialize_features(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        all_coords: torch.Tensor,
        batch_size: int,
        seq_len: int,
        embed_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Re-serialize Q/K/V using spatial ordering if enabled.

        Returns:
            q: Serialized query tensor
            k: Serialized key tensor
            v: Serialized value tensor
            inverse_indices: Indices to restore original order (None if not serializing)
        """
        # pylint: disable=too-many-locals

        if not self.serialize:
            return q, k, v, None

        with torch.no_grad():
            # Compute space-filling curve codes
            coords_int = self._normalize_coords(all_coords)
            sfc_codes = self.curve.encode(coords_int)

            # Sort by SFC codes
            sort_indices = torch.argsort(sfc_codes, dim=1)  # [b, n_total]

            # Compute inverse indices
            inverse_indices = torch.empty_like(sort_indices)
            position_indices = torch.arange(
                seq_len, device=sort_indices.device, dtype=sort_indices.dtype
            )
            position_indices = position_indices.unsqueeze(0).expand(batch_size, -1)
            inverse_indices.scatter_(1, sort_indices, position_indices)

        # Reorder Q/K/V
        sort_indices_expanded = sort_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        q = torch.gather(q, 1, sort_indices_expanded)
        k = torch.gather(k, 1, sort_indices_expanded)
        v = torch.gather(v, 1, sort_indices_expanded)

        return q, k, v, inverse_indices

    @torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=True)
    def _perform_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        embed_dim: int,
        block_mask: flex.BlockMask | None,
    ) -> torch.Tensor:
        """Perform multi-head attention with block mask.

        Returns:
            attn_out: Attention output [b, seq_len, embed_dim]
        """
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform flex attention
        attn_out = flex.flex_attention(query=q, key=k, value=v, block_mask=block_mask)

        # Reshape back
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        )

        return attn_out

    @torch.compile(mode="default")
    def _deserialize_features(
        self,
        attn_out: torch.Tensor,
        inverse_indices: torch.Tensor | None,
        embed_dim: int,
    ) -> torch.Tensor:
        """Deserialize attention output if needed.

        Returns:
            attn_out: Deserialized attention output
        """
        if not self.serialize or inverse_indices is None:
            return attn_out

        # Restore original order using inverse indices
        attn_out = torch.gather(
            attn_out,
            1,
            inverse_indices.unsqueeze(-1).expand(-1, -1, embed_dim),
        )

        return attn_out

    @torch.compile(mode="reduce-overhead", fullgraph=True)
    def _split_and_apply_residual(
        self,
        attn_out: torch.Tensor,
        all_feats_list: list[torch.Tensor],
        level_lengths: list[int],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Split back into levels and apply per-level output projection + residual.

        Returns:
            outputs: Dict mapping level names to updated features
        """
        outputs = {}
        offset = 0

        for idx, (name, length) in enumerate(zip(self.level_names, level_lengths)):
            # Extract attention output for this level
            level_attn_out = attn_out[:, offset : offset + length, :]

            # Apply output projection (per-level or shared)
            if self.use_per_level_projections:
                level_out = self.out_proj[name](level_attn_out)
            else:
                level_out = self.out_proj(level_attn_out)

            # Apply dropout
            level_out = self.drop(level_out)

            # Apply residual connection with original input and layer norm
            level_out = self.norm(all_feats_list[idx] + level_out)

            outputs[name] = {"query": level_out}
            offset += length

        return outputs

    @torch.compile(mode="default")
    def forward(
        self,
        inputs_per_level: dict[
            str, dict[str, torch.Tensor]
        ],  # level_name -> {'query', 'query_pos', 'query_coords'}
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Forward pass with multi-level inputs.

        Args:
            inputs_per_level: Dict mapping level names to their inputs:
                {
                    'coarse': {'query': [b, n_coarse, c], 'query_pos': [b, n_coarse, c], 'query_coords': [b, n_coarse, 3]},
                    'medium': {'query': [b, n_medium, c], 'query_pos': [b, n_medium, c], 'query_coords': [b, n_medium, 3]},
                    'fine': {'query': [b, n_fine, c], 'query_pos': [b, n_fine, c], 'query_coords': [b, n_fine, c]},
                }

        Returns:
            Dict mapping level names to updated features:
                {'coarse': {'query': [b, n_coarse, c]}, 'medium': {...}, 'fine': {...}}
        """  # pylint: disable=line-too-long
        # pylint: disable=too-many-locals

        # 1. Gather features and add level embeddings
        level_lengths, all_feats_list, all_pos_list, all_coords_list = (
            self._gather_and_add_level_embeddings(inputs_per_level)
        )

        # 2. Apply Q/K/V projections per level
        q_list, k_list, v_list = self._apply_qkv_projections(
            all_feats_list, all_pos_list
        )

        # 3. Concatenate projected Q/K/V and coordinates
        q = torch.cat(q_list, dim=1)  # [b, n_total, c]
        k = torch.cat(k_list, dim=1)  # [b, n_total, c]
        v = torch.cat(v_list, dim=1)  # [b, n_total, c]
        all_coords = torch.cat(all_coords_list, dim=1)  # [b, n_total, 3]

        batch_size, seq_len, embed_dim = q.shape

        # 4. Re-serialize Q/K/V using spatial ordering if enabled
        q, k, v, inverse_indices = self._serialize_features(
            q, k, v, all_coords, batch_size, seq_len, embed_dim
        )

        # 5. Perform attention
        block_mask = self._create_attention_mask(seq_len=seq_len, device=q.device)
        attn_out = self._perform_attention(
            q, k, v, batch_size, seq_len, embed_dim, block_mask
        )

        # 6. Deserialize if needed
        attn_out = self._deserialize_features(attn_out, inverse_indices, embed_dim)

        # 7. Split back into levels and apply per-level output projection + residual
        outputs = self._split_and_apply_residual(
            attn_out, all_feats_list, level_lengths
        )

        return outputs


@attention_ops.register(key="SerializedMultiLevelSelfAttention")
class SerializedMultiLevelSelfAttentionOp(StatefulOperation):
    """Serialized multi-level self-attention operation.

    Concatenates all hierarchy levels, re-serializes using space-filling curves,
    applies self-attention, deserializes, and splits back to streams.

    Config example:
        - type: SerializedMultiLevelSelfAttention
          embed_dim: 256
          num_heads: 8
          level_names: ["coarse", "medium", "fine"]
          use_level_embeddings: true
          use_per_level_projections: false  # true for separate Q/K/V per level
          serialize: true
          sfc_order: "hilbert"
          sfc_depth: 16
          coord_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
          attention_mask_type: "sliding-window"
          window_size: 128
    """

    def __init__(
        self,
        embed_dim: int,
        level_names: list[str],
        num_heads: int = 8,
        use_level_embeddings: bool = True,
        use_per_level_projections: bool = False,
        serialize: bool = True,
        sfc_order: str = "hilbert",
        sfc_depth: int = 16,
        coord_range: list[float] = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        attention_mask_type: str = "sliding-window",
        window_size: int = 128,
        num_neighbor_blocks: int = 1,
        dropout: float = 0.0,
        block_size: int = flex._DEFAULT_SPARSE_BLOCK_SIZE,
        args_group: str | Sequence[str] | None = None,
    ):
        # pylint: disable=too-many-locals

        super().__init__(
            args_group=(
                args_group
                if args_group is not None
                else "SerializedMultiLevelSelfAttention"
            )
        )

        self.level_names = level_names

        self.op = SerializedMultiLevelSelfAttentionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            level_names=level_names,
            use_level_embeddings=use_level_embeddings,
            use_per_level_projections=use_per_level_projections,
            serialize=serialize,
            sfc_order=sfc_order,
            sfc_depth=sfc_depth,
            coord_range=coord_range,
            attention_mask_type=attention_mask_type,
            window_size=window_size,
            num_neighbor_blocks=num_neighbor_blocks,
            dropout=dropout,
            block_size=block_size,
        )

    def forward(
        self,
        features: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs,  # pylint: disable=arguments-differ
    ) -> tuple[dict[str, Any], Mapping[str, Mapping[str, Any]]]:
        """Forward pass that gathers features from all streams.

        Expects features to contain keys matching level_names:
            features['coarse'] = {'feats': [b, n_c, c], 'feats_coords': [b, n_c, 3], 'feats_pos': [b, n_c, c]}
            features['medium'] = {'feats': [b, n_m, c], 'feats_coords': [b, n_m, 3], 'feats_pos': [b, n_m, c]}
            features['fine'] = {'feats': [b, n_f, c], 'feats_coords': [b, n_f, 3], 'feats_pos': [b, n_f, c]}

        Note: This operation should typically be used WITHOUT a stream tag in the config,
        as it operates on ALL streams simultaneously.
        """  # pylint: disable=line-too-long
        assert features is not None

        # Gather inputs from all levels
        inputs_per_level = {}
        for level_name in self.level_names:
            assert level_name in features, f"Level '{level_name}' not found in features"

            level_feats = features[level_name]
            inputs_per_level[level_name] = {
                "query": level_feats["feats"],
                "query_pos": level_feats.get("feats_pos"),
                "query_coords": level_feats["feats_coords"],
            }

        # Perform multi-level attention
        outputs_per_level = self.op(inputs_per_level)

        # Update features dict with new features
        feats = dict(features)
        for level_name in self.level_names:
            feats[level_name] = dict(feats[level_name])
            feats[level_name]["feats"] = outputs_per_level[level_name]["query"]

        # Return unchanged current stream inputs, but updated features
        return {**kwargs}, feats
