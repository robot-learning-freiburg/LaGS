# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import functools
from typing import Any, Mapping

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from ....utils.torch import amp
from ..head import LearnedSinePosEnc3d, MultiViewImagePosEnc, MultiViewPerspectivePosEnc
from .multistream import MultiStreamTransformerLayerSequence


def _get_op_features(ops):
    """Get the set of feature types used by the given operations."""

    return set(op.features for op in ops if hasattr(op, "features"))


@torch.autocast("cuda", enabled=False)
def _project_grid(
    points: torch.Tensor,  # [b, n, (x,y,z)]
    tx_project: torch.Tensor,  # [b, n, 4, 4]
    img_shape: tuple[int, int],
    min_depth: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    *_, h, w = img_shape

    points = amp.upcast(points, dtype=torch.float32)
    tx_project = amp.upcast(tx_project, dtype=torch.float32)

    points = torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)
    points = torch.einsum("b n i j, b k j -> b n k i", tx_project, points)

    img_depth = points[..., 2]

    points = points[..., :2] / img_depth.clamp(min=min_depth)[..., None]
    points = points / torch.tensor([w, h], device=points.device)
    points = points * 2 - 1  # [-1, 1]

    mask = img_depth > min_depth
    mask = mask & (points[..., 0] > -1) & (points[..., 0] < 1)
    mask = mask & (points[..., 1] > -1) & (points[..., 1] < 1)

    return points, mask


def _compute_image_padding_mask(tgt_shape, img_shape, img_padding, device):
    *_, img_h, img_w = img_shape
    *n, out_h, out_w = tgt_shape
    left, right, top, bottom = img_padding

    # build mask (1 for padded area, 0 for image area)
    mask = torch.ones((1, 1, img_h, img_w), dtype=torch.uint8, device=device)
    mask[..., top : img_h - bottom, left : img_w - right] = 0

    # interpolate it to the feature shape
    mask = F.interpolate(mask, size=(out_h, out_w), mode="nearest")
    mask = mask.bool()

    # expand to feature shape
    mask = mask.expand(*n, out_h, out_w)

    return mask


class OccupancyTransformer(nn.Module):
    # pylint: disable=too-many-instance-attributes

    # buffers:
    voxel_range: torch.Tensor  # [6]
    voxel_size: torch.Tensor  # [3]

    def __init__(
        self,
        num_layers: int,
        image_dim: int | list[int],
        embed_dim: int,
        voxel_range: tuple[float, float, float, float, float, float],
        voxel_size: tuple[float, float, float],
        operations: list[OmegaConf | Mapping[str, Any]],
        use_keypoint_posemb: bool = True,
    ) -> None:
        super().__init__()

        # Normalize image_dim to list
        if isinstance(image_dim, int):
            image_dim = [image_dim]

        self.image_dim = image_dim
        self.num_image_levels = len(image_dim)
        self.embed_dim = embed_dim

        voxel_range = torch.as_tensor(voxel_range)
        self.register_buffer("voxel_range", voxel_range, persistent=False)

        voxel_size = torch.as_tensor(voxel_size)
        self.register_buffer("voxel_size", voxel_size, persistent=False)

        self.features = _get_op_features(operations)

        # Determine which image levels are needed
        needed_levels = set()
        for feat in self.features:
            if feat.startswith("img-l"):
                # Extract level number from "img-l0", "img-l1", etc.
                level = int(feat.split("-l")[1])
                needed_levels.add(level)
            elif feat in ["img", "img-keypoint"]:
                # Default to level 0
                needed_levels.add(0)
            elif feat == "img-sca":
                # SpatialCrossAttention uses all levels
                needed_levels.update(range(self.num_image_levels))

        self.needed_image_levels = sorted(needed_levels)

        # Input projections (only for needed levels)
        self.image_proj = nn.ModuleDict(
            {
                f"l{i}": nn.Conv2d(image_dim[i], embed_dim, kernel_size=1)
                for i in self.needed_image_levels
            }
        )

        # positional encodings for query keypoints
        if use_keypoint_posemb:
            self.keypoint_pos_emb = LearnedSinePosEnc3d(
                num_feats=128,
                embed_dim=embed_dim,
            )
        else:
            self.keypoint_pos_emb = lambda coords: torch.zeros(
                *coords.shape[:-1], embed_dim, device=coords.device
            )

        # positional encodings for image features (only if needed)
        needs_img_posenc = any(
            feat in self.features
            for feat in ["img", "img-keypoint"]
            + [f"img-l{i}" for i in range(self.num_image_levels)]
            + [f"img-keypoint-l{i}" for i in range(self.num_image_levels)]
        )

        if needs_img_posenc:
            self.posenc_perspective = MultiViewPerspectivePosEnc(
                embed_dim=embed_dim,
                depth_bins=64,
                depth_range=(1, 45),
                position_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
            )

            self.posenc_image = MultiViewImagePosEnc(
                num_feats=128,
                embed_dim=embed_dim,
                normalize=True,
            )
        else:
            self.posenc_perspective = None
            self.posenc_image = None

        # main transformer
        self.transformer = MultiStreamTransformerLayerSequence(
            num_layers=num_layers,
            embed_dim=embed_dim,
            operations=operations,
            return_intermediate=True,
        )

    def _prepare_image_features(
        self,
        feats: torch.Tensor | list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Project image features to embedding dimension.

        Args:
            feats: Single tensor [b, n, c, h, w] or list of tensors for multi-level

        Returns:
            Dict mapping level names (e.g., 'l0', 'l1') to projected features
              [b, n, embed_dim, h, w]
        """
        # Normalize to list
        if isinstance(feats, torch.Tensor):
            feats = [feats]

        projected = {}
        for i in self.needed_image_levels:
            level_feats = feats[i]
            b, n, c, h, w = level_feats.shape

            # Project to embedding dimension
            level_feats = level_feats.view(b * n, c, h, w)
            level_feats = self.image_proj[f"l{i}"](level_feats)
            level_feats = level_feats.view(b, n, self.embed_dim, h, w)

            projected[f"l{i}"] = level_feats

        return projected

    def _compute_image_ref(
        self,
        points: torch.Tensor,  # [b, m, 3]  in [0, 1]
        tx_project: torch.Tensor,  # [b, n, 4, 4]
        img_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_min, v_max = self.voxel_range[None, :3], self.voxel_range[None, 3:]
        keypoint_wcoords = points * (v_max - v_min) + v_min

        # compute reference points for image features
        image_ref, image_ref_mask = _project_grid(
            points=keypoint_wcoords,
            tx_project=tx_project,
            img_shape=img_shape,
        )

        return image_ref, image_ref_mask

    def forward(
        self,
        image_feats: torch.Tensor | list[torch.Tensor],
        keypoint_feats: torch.Tensor,  # [b, m, c]
        keypoint_coords: torch.Tensor,  # [b, m, 3] in [0, 1]
        tx_project: torch.Tensor,  # [b, n, 4, 4]
        tx_unproject: torch.Tensor,  # [b, n, 4, 4]
        image_shape: tuple[int],
        image_padding: tuple[int],
    ) -> torch.Tensor:
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements

        # Project image features per level
        image_feats_projected = self._prepare_image_features(image_feats)

        # Get batch size and num cameras from first level
        first_level = f"l{self.needed_image_levels[0]}"
        b, n, _, _, _ = image_feats_projected[first_level].shape

        # Compute positional encodings
        keypoint_pos = self.keypoint_pos_emb(keypoint_coords)

        # Compute per-level image position embeddings and masks (if needed)
        image_pos_embs = {}
        image_masks = {}

        if self.posenc_perspective is not None:
            for level_name, level_feats in image_feats_projected.items():
                _, _, _, h, w = level_feats.shape

                # Generate image padding mask
                image_mask = _compute_image_padding_mask(
                    tgt_shape=(b, n, h, w),
                    img_shape=image_shape,
                    img_padding=image_padding,
                    device=level_feats.device,
                )
                image_mask = image_mask.contiguous()

                # Generate position embeddings
                image_sin_emb = self.posenc_image(image_mask)
                image_pos_emb = self.posenc_perspective(
                    tgt_shape=level_feats.shape,
                    img_shape=image_shape,
                    unproject_tx=tx_unproject,
                    device=level_feats.device,
                )
                image_pos_emb = image_pos_emb + image_sin_emb

                image_pos_embs[level_name] = image_pos_emb
                image_masks[level_name] = image_mask

        # Prepare reference points for cross-attention (computed once, shared by all levels)
        image_ref_fn = functools.partial(
            self._compute_image_ref,
            tx_project=tx_project,
            img_shape=image_shape,
        )
        image_ref, image_ref_mask = image_ref_fn(keypoint_coords)

        # Populate features dict (only for requested features)
        features = {}

        # Handle per-level "img" features
        for level_idx in self.needed_image_levels:
            level_name = f"l{level_idx}"
            feat_name = f"img-{level_name}"

            if feat_name in self.features:
                level_feats = image_feats_projected[level_name]
                level_pos = image_pos_embs[level_name]
                level_mask = image_masks[level_name]

                features[feat_name] = {
                    "feats": rearrange(level_feats, "b n c h w -> b (n h w) c"),
                    "feats_pos": rearrange(level_pos, "b n c h w -> b (n h w) c"),
                    "key_padding_mask": rearrange(level_mask, "b n h w -> b (n h w)"),
                }

        # Backward compatibility: "img" maps to "img-l0"
        if "img" in self.features:
            level_name = "l0"
            level_feats = image_feats_projected[level_name]
            level_pos = image_pos_embs[level_name]
            level_mask = image_masks[level_name]

            features["img"] = {
                "feats": rearrange(level_feats, "b n c h w -> b (n h w) c"),
                "feats_pos": rearrange(level_pos, "b n c h w -> b (n h w) c"),
                "key_padding_mask": rearrange(level_mask, "b n h w -> b (n h w)"),
            }

        # Handle per-level "img-keypoint" features
        for level_idx in self.needed_image_levels:
            level_name = f"l{level_idx}"
            feat_name = f"img-keypoint-{level_name}"

            if feat_name in self.features:
                level_feats = image_feats_projected[level_name]
                level_pos = image_pos_embs[level_name]

                features[feat_name] = {
                    "feats": level_feats,
                    "feats_pos": level_pos,
                    "reference_points": image_ref,
                    "reference_points_mask": image_ref_mask,
                    "reference_points_fn": image_ref_fn,
                }

        # Backward compatibility: "img-keypoint" maps to "img-keypoint-l0"
        if "img-keypoint" in self.features:
            level_name = "l0"
            level_feats = image_feats_projected[level_name]
            level_pos = image_pos_embs[level_name]

            features["img-keypoint"] = {
                "feats": level_feats,
                "feats_pos": level_pos,
                "reference_points": image_ref,
                "reference_points_mask": image_ref_mask,
                "reference_points_fn": image_ref_fn,
            }

        # Handle "img-sca" (uses all levels)
        if "img-sca" in self.features:
            # Collect all level features in order
            sca_feats = tuple(
                image_feats_projected[f"l{i}"] for i in self.needed_image_levels
            )

            features["img-sca"] = {
                "value": sca_feats,
                "value_mask": None,
                "reference_points": image_ref,
                "reference_points_mask": image_ref_mask,
                "reference_points_fn": image_ref_fn,
            }

        # Run transformer
        output = self.transformer(
            inputs={
                "default": {
                    "query": keypoint_feats,
                    "query_pos": keypoint_pos,
                    "query_coords": keypoint_coords,
                },
            },
            features=features,
            return_state=("query", "query_coords"),
        )

        query = output["default"]["query"]  # [b, l, n_q, emb_dim]
        query_coords = output["default"]["query_coords"]  # [b, l, n_q, 3] in [0, 1]

        return query, query_coords


class HierarchicalOccupancyTransformer(nn.Module):
    """
    Multi-stream occupancy transformer with hierarchical gaussians.

    Extends OccupancyTransformer to support multiple streams (e.g., coarse/medium/fine)
    with different numbers of queries and different attention patterns per stream.

    Args:
        num_layers: Number of transformer layers
        image_dim: Dimension of image features
        embed_dim: Embedding dimension for all streams
        voxel_range: Voxel coordinate range
        voxel_size: Voxel size
        stream_names: Names of streams (e.g., ['coarse', 'medium', 'fine'])
        operations: List of operations with 'stream' tags
        use_keypoint_posemb: Whether to use position embeddings
        share_keypoint_posemb: If True, all streams share the same position embedding.
            If False, each stream has its own independent position embedding. Default: False.
        input_channels: Dict mapping stream names to input channel dimensions.
            This is needed when sampling from pyramid levels with varying channels.
            Example: {'coarse': 96, 'medium': 64, 'fine': 32}
            If None, assumes all inputs already have embed_dim channels. Default: None.
    """

    # pylint: disable=too-many-instance-attributes

    # buffers:
    voxel_range: torch.Tensor  # [6]
    voxel_size: torch.Tensor  # [3]

    def __init__(
        self,
        num_layers: int,
        image_dim: int | list[int],
        embed_dim: int,
        voxel_range: tuple[float, float, float, float, float, float],
        voxel_size: tuple[float, float, float],
        stream_names: list[str],
        operations: list[OmegaConf | Mapping[str, Any]],
        use_keypoint_posemb: bool = True,
        share_keypoint_posemb: bool = False,
    ) -> None:
        # pylint: disable=too-many-locals
        super().__init__()

        # Normalize image_dim to list
        if isinstance(image_dim, int):
            image_dim = [image_dim]

        self.image_dim = image_dim
        self.num_image_levels = len(image_dim)
        self.embed_dim = embed_dim
        self.stream_names = stream_names
        self.share_keypoint_posemb = share_keypoint_posemb

        voxel_range = torch.as_tensor(voxel_range)
        self.register_buffer("voxel_range", voxel_range, persistent=False)

        voxel_size = torch.as_tensor(voxel_size)
        self.register_buffer("voxel_size", voxel_size, persistent=False)

        self.features = _get_op_features(operations)

        # Determine which image levels are needed
        needed_levels = set()
        for feat in self.features:
            if feat.startswith("img-l"):
                level = int(feat.split("-l")[1])
                needed_levels.add(level)
            elif feat in ["img", "img-keypoint"]:
                needed_levels.add(0)
            elif feat == "img-sca":
                needed_levels.update(range(self.num_image_levels))

        self.needed_image_levels = sorted(needed_levels)

        # Input projections (only for needed levels)
        self.image_proj = nn.ModuleDict(
            {
                f"l{i}": nn.Conv2d(image_dim[i], embed_dim, kernel_size=1)
                for i in self.needed_image_levels
            }
        )

        # Positional encodings (per-stream or shared)
        if use_keypoint_posemb:
            if share_keypoint_posemb:
                shared_pos_emb = LearnedSinePosEnc3d(
                    num_feats=128,
                    embed_dim=embed_dim,
                )
                self.keypoint_pos_embs = nn.ModuleDict(
                    {name: shared_pos_emb for name in stream_names}
                )
            else:
                self.keypoint_pos_embs = nn.ModuleDict(
                    {
                        name: LearnedSinePosEnc3d(
                            num_feats=128,
                            embed_dim=embed_dim,
                        )
                        for name in stream_names
                    }
                )
        else:

            def zero_emb(coords):
                return torch.zeros(*coords.shape[:-1], embed_dim, device=coords.device)

            self.keypoint_pos_embs = nn.ModuleDict(
                {name: zero_emb for name in stream_names}
            )

        # Image positional encodings (conditional)
        needs_img_posenc = any(
            feat in self.features
            for feat in ["img", "img-keypoint"]
            + [f"img-l{i}" for i in range(self.num_image_levels)]
            + [f"img-keypoint-l{i}" for i in range(self.num_image_levels)]
        )

        if needs_img_posenc:
            self.posenc_perspective = MultiViewPerspectivePosEnc(
                embed_dim=embed_dim,
                depth_bins=64,
                depth_range=(1, 45),
                position_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
            )

            self.posenc_image = MultiViewImagePosEnc(
                num_feats=128,
                embed_dim=embed_dim,
                normalize=True,
            )
        else:
            self.posenc_perspective = None
            self.posenc_image = None

        # Multi-stream transformer
        self.transformer = MultiStreamTransformerLayerSequence(
            num_layers=num_layers,
            embed_dim=embed_dim,
            operations=operations,
            return_intermediate=True,
        )

    def _prepare_image_features(
        self,
        feats: torch.Tensor | list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Project image features to embedding dimension.

        Args:
            feats: Single tensor [b, n, c, h, w] or list of tensors for multi-level

        Returns:
            Dict mapping level names (e.g., 'l0', 'l1') to projected features
              [b, n, embed_dim, h, w]
        """
        # Normalize to list
        if isinstance(feats, torch.Tensor):
            feats = [feats]

        projected = {}
        for i in self.needed_image_levels:
            level_feats = feats[i]
            b, n, c, h, w = level_feats.shape

            # Project to embedding dimension
            level_feats = level_feats.view(b * n, c, h, w)
            level_feats = self.image_proj[f"l{i}"](level_feats)
            level_feats = level_feats.view(b, n, self.embed_dim, h, w)

            projected[f"l{i}"] = level_feats

        return projected

    def _compute_image_ref(
        self,
        points: torch.Tensor,  # [b, m, 3]  in [0, 1]
        tx_project: torch.Tensor,  # [b, n, 4, 4]
        img_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v_min, v_max = self.voxel_range[None, :3], self.voxel_range[None, 3:]
        keypoint_wcoords = points * (v_max - v_min) + v_min

        # compute reference points for image features
        image_ref, image_ref_mask = _project_grid(
            points=keypoint_wcoords,
            tx_project=tx_project,
            img_shape=img_shape,
        )

        return image_ref, image_ref_mask

    def forward(
        self,
        image_feats: torch.Tensor | list[torch.Tensor],
        queries: dict[str, dict[str, torch.Tensor]],
        tx_project: torch.Tensor,  # [b, n, 4, 4]
        tx_unproject: torch.Tensor,  # [b, n, 4, 4]
        image_shape: tuple[int],
        image_padding: tuple[int],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Forward pass with multi-stream queries.

        Args:
            image_feats: Image features [b, n, c, h, w] or list of multi-level features
            queries: Dict mapping stream names to their queries and coords
              Format:
              {
                  'coarse': {'query': [b, n_q1, c], 'query_coords': [b, n_q1, 3]},
                  'medium': {'query': [b, n_q2, c], 'query_coords': [b, n_q2, 3]},
                  'fine': {'query': [b, n_q3, c], 'query_coords': [b, n_q3, 3]},
              }
            tx_project: Projection transforms [b, n, 4, 4]
            tx_unproject: Unprojection transforms [b, n, 4, 4]
            image_shape: Original image shape (H, W)
            image_padding: Image padding

        Returns:
            Dict mapping stream names to their outputs:
            {
                'coarse': {
                    'query': [b, num_layers, n_queries, c],
                    'query_coords': [b, num_layers, n_queries, 3],
                },
                'medium': {...},
                'fine': {...},
            }
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements

        # Project image features per level
        image_feats_projected = self._prepare_image_features(image_feats)

        # Get batch size and num cameras from first level
        first_level = f"l{self.needed_image_levels[0]}"
        b, n, _, _, _ = image_feats_projected[first_level].shape

        # Compute per-level image position embeddings and masks (if needed)
        image_pos_embs = {}
        image_masks = {}

        if self.posenc_perspective is not None:
            for level_name, level_feats in image_feats_projected.items():
                _, _, _, h, w = level_feats.shape

                # Generate image padding mask
                image_mask = _compute_image_padding_mask(
                    tgt_shape=(b, n, h, w),
                    img_shape=image_shape,
                    img_padding=image_padding,
                    device=level_feats.device,
                )
                image_mask = image_mask.contiguous()

                # Generate position embeddings
                image_sin_emb = self.posenc_image(image_mask)
                image_pos_emb = self.posenc_perspective(
                    tgt_shape=level_feats.shape,
                    img_shape=image_shape,
                    unproject_tx=tx_unproject,
                    device=level_feats.device,
                )
                image_pos_emb = image_pos_emb + image_sin_emb

                image_pos_embs[level_name] = image_pos_emb
                image_masks[level_name] = image_mask

        # Prepare per-stream inputs
        inputs = {}
        for stream_name in self.stream_names:
            stream_queries = queries[stream_name]

            keypoint_feats = stream_queries["query"]
            keypoint_coords = stream_queries["query_coords"]

            # Compute position embeddings
            keypoint_pos = self.keypoint_pos_embs[stream_name](keypoint_coords)

            inputs[stream_name] = {
                "query": keypoint_feats,
                "query_pos": keypoint_pos,
                "query_coords": keypoint_coords,
            }

        # Prepare image features dict (shared across streams)
        features = {}

        # Handle per-level "img" features
        for level_idx in self.needed_image_levels:
            level_name = f"l{level_idx}"
            feat_name = f"img-{level_name}"

            if feat_name in self.features:
                level_feats = image_feats_projected[level_name]
                level_pos = image_pos_embs[level_name]
                level_mask = image_masks[level_name]

                features[feat_name] = {
                    "feats": rearrange(level_feats, "b n c h w -> b (n h w) c"),
                    "feats_pos": rearrange(level_pos, "b n c h w -> b (n h w) c"),
                    "key_padding_mask": rearrange(level_mask, "b n h w -> b (n h w)"),
                }

        # Backward compatibility: "img" maps to "img-l0"
        if "img" in self.features:
            level_name = "l0"
            level_feats = image_feats_projected[level_name]
            level_pos = image_pos_embs[level_name]
            level_mask = image_masks[level_name]

            features["img"] = {
                "feats": rearrange(level_feats, "b n c h w -> b (n h w) c"),
                "feats_pos": rearrange(level_pos, "b n c h w -> b (n h w) c"),
                "key_padding_mask": rearrange(level_mask, "b n h w -> b (n h w)"),
            }

        # Handle per-level "img-keypoint" features
        for level_idx in self.needed_image_levels:
            level_name = f"l{level_idx}"
            feat_name = f"img-keypoint-{level_name}"

            if feat_name in self.features:
                level_feats = image_feats_projected[level_name]
                level_pos = image_pos_embs[level_name]

                # Pre-compute reference points for ALL streams
                refpoints_per_stream = {}
                refpoints_mask_per_stream = {}

                for stream_name in self.stream_names:
                    ref_points, ref_mask = self._compute_image_ref(
                        points=inputs[stream_name]["query_coords"],
                        tx_project=tx_project,
                        img_shape=image_shape,
                    )
                    refpoints_per_stream[stream_name] = ref_points
                    refpoints_mask_per_stream[stream_name] = ref_mask

                features[feat_name] = {
                    "feats": level_feats,
                    "feats_pos": level_pos,
                    # Store per-stream reference points - multistream layer will inject them
                    "reference_points_per_stream": refpoints_per_stream,
                    "reference_points_mask_per_stream": refpoints_mask_per_stream,
                }

        # Backward compatibility: "img-keypoint" maps to "img-keypoint-l0"
        if "img-keypoint" in self.features:
            level_name = "l0"
            level_feats = image_feats_projected[level_name]
            level_pos = image_pos_embs[level_name]

            # Pre-compute reference points for ALL streams
            refpoints_per_stream = {}
            refpoints_mask_per_stream = {}

            for stream_name in self.stream_names:
                ref_points, ref_mask = self._compute_image_ref(
                    points=inputs[stream_name]["query_coords"],
                    tx_project=tx_project,
                    img_shape=image_shape,
                )
                refpoints_per_stream[stream_name] = ref_points
                refpoints_mask_per_stream[stream_name] = ref_mask

            features["img-keypoint"] = {
                "feats": level_feats,
                "feats_pos": level_pos,
                # Store per-stream reference points - multistream layer will inject them
                "reference_points_per_stream": refpoints_per_stream,
                "reference_points_mask_per_stream": refpoints_mask_per_stream,
            }

        # Handle "img-sca" (uses all levels)
        if "img-sca" in self.features:
            # Collect all level features in order
            sca_feats = tuple(
                image_feats_projected[f"l{i}"] for i in self.needed_image_levels
            )

            # Pre-compute reference points for ALL streams
            refpoints_per_stream = {}
            refpoints_mask_per_stream = {}

            for stream_name in self.stream_names:
                ref_points, ref_mask = self._compute_image_ref(
                    points=inputs[stream_name]["query_coords"],
                    tx_project=tx_project,
                    img_shape=image_shape,
                )
                refpoints_per_stream[stream_name] = ref_points
                refpoints_mask_per_stream[stream_name] = ref_mask

            features["img-sca"] = {
                "value": sca_feats,
                "value_mask": None,
                # Store per-stream reference points - multistream layer will inject them
                "reference_points_per_stream": refpoints_per_stream,
                "reference_points_mask_per_stream": refpoints_mask_per_stream,
            }

        # Run multi-stream transformer
        outputs = self.transformer(
            inputs=inputs,
            features=features,
            return_state=("query", "query_coords"),
        )

        return outputs
