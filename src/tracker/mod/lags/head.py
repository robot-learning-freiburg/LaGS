# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

from ...utils.types import MetaDict
from . import transformer as transformers
from .bbox_coder import TrackNMSFreeCoder
from .posenc.fourier import MultiFrameSinePosEnc, embedding3d
from .utils import inverse_sigmoid, unproject_image_rays


# pylint: disable-next=too-many-instance-attributes
class OccupancyTrackingHead(nn.Module):
    def __init__(
        self,
        num_instance_classes: int,
        num_semantic_classes: int,
        in_channels: int,
        num_semantic_queries: int,
        num_decoder_layers: int = 2,
        bbox_coder=None,
        embed_dims=256,
        voxel_feature_dims=None,
        depth_num=64,
        depth_start=1,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        position_range=(-65, -65, -8.0, 65, 65, 8.0),
        voxel_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        clamp_voxel_reference_points=False,
        transformer=None,
        force_mask_type: torch.dtype | str | None = None,
        use_keypoint_features: bool = True,
    ):
        # pylint: disable=too-many-locals
        super().__init__()

        self.num_instance_classes = num_instance_classes
        self.num_semantic_classes = num_semantic_classes
        self.num_semantic_queries = num_semantic_queries
        self.in_channels = in_channels
        self.num_decoder_layers = num_decoder_layers
        self.embed_dims = embed_dims
        self.voxel_feature_dims = voxel_feature_dims or embed_dims
        self.position_range = position_range
        self.voxel_range = voxel_range
        self.clamp_voxel_reference_points = clamp_voxel_reference_points
        self.transformer = transformer
        self.use_keypoint_features = use_keypoint_features

        self.depth_range = (depth_start, self.position_range[3])
        self.depth_bins = depth_num

        self.bbox_coder = TrackNMSFreeCoder(**bbox_coder)
        self.pc_range = pc_range

        if force_mask_type is not None:
            if isinstance(force_mask_type, str):
                force_mask_type = getattr(torch, force_mask_type)
                assert isinstance(force_mask_type, torch.dtype)

        self.force_mask_type = force_mask_type

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        # semantic queries
        self.semantic_queries = nn.Embedding(self.num_semantic_queries, self.embed_dims)
        self.semantic_query_embeddings = nn.Embedding(
            self.num_semantic_queries, self.embed_dims
        )
        self.semantic_reference_points = nn.Embedding(self.num_semantic_queries, 3)

        nn.init.uniform_(self.semantic_reference_points.weight.data, 0, 1)

        # 3D perspective positional encodings
        self.posenc_perspective = MultiViewPerspectivePosEnc(
            embed_dim=self.embed_dims,
            depth_bins=self.depth_bins,
            depth_range=self.depth_range,
            position_range=self.position_range,
        )

        # image position encodings
        self.posenc_image = MultiViewImagePosEnc(
            num_feats=128,
            embed_dim=self.embed_dims,
            normalize=True,
        )

        if self.use_keypoint_features:
            self.posenc_keypoint = LearnedSinePosEnc3d(
                embed_dim=self.embed_dims,
                num_feats=128,
            )

        # input feature projection
        self.input_proj = nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=1)

        # main transformer decoder
        self.transformer = transformers.build(self.transformer)

        # mask embedding decoder
        self.mask_decoder_instance = QueryDecoder(
            embed_dim=self.embed_dims,
            output_dim=self.voxel_feature_dims,
            num_layers=self.num_decoder_layers,
            act=nn.ReLU,
            norm=None,
        )

        self.mask_decoder_semantic = QueryDecoder(
            embed_dim=self.embed_dims,
            output_dim=self.voxel_feature_dims,
            num_layers=self.num_decoder_layers,
            act=nn.ReLU,
            norm=None,
        )

        # class score decoder for instance queries
        self.cls_decoder_instance = QueryDecoder(
            embed_dim=self.embed_dims,
            output_dim=self.num_instance_classes,
            num_layers=self.num_decoder_layers,
            act=nn.ReLU,
            norm=nn.LayerNorm,
        )

        # class score decoder for semantic queries
        self.cls_decoder_semantic = QueryDecoder(
            embed_dim=self.embed_dims,
            output_dim=self.num_semantic_classes,
            num_layers=self.num_decoder_layers,
            act=nn.ReLU,
            norm=nn.LayerNorm,
        )

        # box property regression decoder for queries
        self.reg_decoder_instance = QueryDecoder(
            embed_dim=self.embed_dims,
            output_dim=self.bbox_coder.code_size,
            num_layers=self.num_decoder_layers,
            act=nn.ReLU,
            norm=None,
        )

    def _init_weights(self):
        # init bias of last class probability decoder layer via prior probability
        bias_prior = 0.01
        bias_init = -math.log((1.0 - bias_prior) / bias_prior)
        nn.init.constant_(self.cls_decoder_instance[-1].bias, bias_init)
        nn.init.constant_(self.cls_decoder_semantic[-1].bias, bias_init)

    # pylint: disable-next=too-many-locals
    def forward(
        self,
        features: torch.Tensor,
        voxel_features: tuple[torch.Tensor],
        img_meta: MetaDict,
        query_targets: torch.Tensor,
        query_embs: torch.Tensor,
        reference_points: torch.Tensor,
        keypoint_features: torch.Tensor,  # [b, n_keypoints, embed_dims]
        keypoint_coords: torch.Tensor,  # [b, n_keypoints, 3] in [0, 1] range
    ):
        b, n, c, h, w = features.shape

        # apply input feature projection
        features = features.view(b * n, c, h, w)
        features = self.input_proj(features)
        features = features.view(b, n, self.embed_dims, h, w)

        # generate padding masks
        # NOTE: this assumes that padding is the same for all images...
        mask = self.padding_mask(
            tgt_shape=(b, n, h, w),
            img_shape=img_meta.shape[0],
            img_padding=img_meta.padding[0],
            device=features.device,
        )
        mask = mask.contiguous()

        # generate position embeddings
        sin_emb = self.posenc_image(mask)
        pos_emb = self.posenc_perspective(
            tgt_shape=features.shape,
            img_shape=img_meta.shape[0],
            unproject_tx=img_meta.transforms.image_to_ego,
            device=features.device,
        )
        pos_emb = pos_emb + sin_emb

        # re-scale reference points to voxel range
        msda_ref = self.get_query_reference_points_for_voxels(reference_points)

        # set up semantic queries
        refpts_sem = self.semantic_reference_points.weight.clone()
        query_embs_sem = self.semantic_query_embeddings.weight.clone()
        query_targets_sem = self.semantic_queries.weight.clone()
        msda_ref_sem = self.get_query_reference_points_for_voxels(refpts_sem)

        # set up keypoints
        if self.use_keypoint_features:
            keypoint_embs = self.posenc_keypoint(keypoint_coords)
        else:
            keypoint_features = None
            keypoint_embs = None

        # run transformer for semantic queries
        q_semantic, q_instance = self.transformer(
            instance_query=query_targets,
            instance_query_embs=query_embs,
            instance_reference_points=msda_ref,
            semantic_query=query_targets_sem,
            semantic_query_embs=query_embs_sem,
            semantic_reference_points=msda_ref_sem,
            image_features=features,
            image_features_mask=mask,
            image_features_embs=pos_emb,
            voxel_features=voxel_features,
            keypoint_features=keypoint_features,
            keypoint_embs=keypoint_embs,
        )  # [b, num_layers, (n_qi | n_qs), embed_dim]

        # decode semantic and instance queries
        semantic_cls_scores, semantic_mask_preds = self.decode_semantic_queries(
            query=q_semantic,
            voxel_features=voxel_features[0],
        )

        instance_cls_scores, bbox_preds, instance_mask_preds, reference_points = (
            self.decode_instance_queries(
                query=q_instance,
                reference_points=reference_points,
                voxel_features=voxel_features[0],
            )
        )

        return {
            "all_cls_scores": instance_cls_scores,
            "all_bbox_preds": bbox_preds,
            "all_mask_preds": instance_mask_preds,
            "semantic_cls_scores": semantic_cls_scores,
            "semantic_mask_preds": semantic_mask_preds,
            "query_feats": q_instance[:, -1],
            "reference_points": reference_points,
        }

    def decode_semantic_queries(
        self,
        query: torch.Tensor,
        voxel_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # decode class scores [b, num_layers, n_queries, num_classes]
        class_scores = self.cls_decoder_semantic(query)

        # decode semantic masks
        masks = self.mask_decoder_semantic(query)  # [b, num_layers, n_queries, emb_dim]

        if self.force_mask_type:
            masks = masks.to(dtype=self.force_mask_type)
            voxel_features = voxel_features.to(dtype=self.force_mask_type)

        masks = torch.einsum(
            "blnc,bczyx -> blnzyx", masks, voxel_features
        )  # [b, num_layers, n_queries, z, y, x]

        return class_scores, masks

    def decode_instance_queries(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        voxel_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # decode class scores [b, num_layers, n_queries, num_classes]
        class_scores = self.cls_decoder_instance(query)

        # decode box properties, add reference positions to coordinates
        reference = inverse_sigmoid(reference_points.clone())
        reference = reference[None, None, :, :]  # [1, 1, n_queries, 3]

        bbox_preds = self.reg_decoder_instance(query)  # [b, num_layers, n_queries, k]
        bbox_preds[..., 0:2] = torch.sigmoid(bbox_preds[..., 0:2] + reference[..., 0:2])
        bbox_preds[..., 4:5] = torch.sigmoid(bbox_preds[..., 4:5] + reference[..., 2:3])

        # collect last coordinates as new reference points
        reference_points = bbox_preds[:, -1, :, (0, 1, 4)].clone()  # [b, n_queries, 3]

        # bring coordinates from [0, 1] back to full range
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        bbox_preds[..., 0] = bbox_preds[..., 0] * (x_max - x_min) + x_min
        bbox_preds[..., 1] = bbox_preds[..., 1] * (y_max - y_min) + y_min
        bbox_preds[..., 4] = bbox_preds[..., 4] * (z_max - z_min) + z_min

        # decode instance masks
        masks = self.mask_decoder_instance(query)

        if self.force_mask_type:
            masks = masks.to(dtype=self.force_mask_type)
            voxel_features = voxel_features.to(dtype=self.force_mask_type)

        masks = torch.einsum(
            "blnc,bczyx -> blnzyx", masks, voxel_features
        )  # [b, num_layers, n_queries, z, y, x]

        return class_scores, bbox_preds, masks, reference_points

    def get_query_reference_points_for_voxels(self, reference):
        """
        Convert the reference points from [0, 1] point cloud range to the
        normalized 3D occupancy range [-1, 1].
        """
        reference = reference.clone()

        # convert coordinates from [0, 1] detector frame back to full range
        x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
        reference[..., 0] = reference[..., 0] * (x_max - x_min) + x_min
        reference[..., 1] = reference[..., 1] * (y_max - y_min) + y_min
        reference[..., 2] = reference[..., 2] * (z_max - z_min) + z_min

        # convert to normalized [-1, 1] 3d occupancy range
        x_min, y_min, z_min, x_max, y_max, z_max = self.voxel_range
        reference[..., 0] = (reference[..., 0] - x_min) / (x_max - x_min) * 2 - 1
        reference[..., 1] = (reference[..., 1] - y_min) / (y_max - y_min) * 2 - 1
        reference[..., 2] = (reference[..., 2] - z_min) / (z_max - z_min) * 2 - 1

        if self.clamp_voxel_reference_points:
            reference = reference.clamp(min=-1.0, max=1.0)

        return reference

    def padding_mask(self, tgt_shape, img_shape, img_padding, device):
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

    def get_bboxes(self, preds):
        # ``decode`` returns a dict of PackedTensors (batched, ragged across the
        # variable number of surviving boxes per sample): "boxes", "scores",
        # "labels", and optionally "instances" (tracking) and "forecasting".
        decoded = self.bbox_coder.decode(preds)

        out = MetaDict()
        out.boxes = decoded["boxes"].to(dtype=torch.float32)
        out.scores = decoded["scores"].to(dtype=torch.float32)
        out.labels = decoded["labels"]

        if "instances" in decoded:
            out.instances = decoded["instances"]

        if "forecasting" in decoded:
            out.forecasting = decoded["forecasting"].to(dtype=torch.float32)

        return out


class QueryDecoder(nn.Sequential):
    def __init__(
        self,
        embed_dim: int,
        output_dim: int,
        num_layers: int = 2,
        act: Callable[[], nn.Module] = nn.ReLU,
        norm: Callable[[int], nn.Module] | None = None,
    ) -> None:
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # create decoder layers
        decoder_layers = []
        for _ in range(num_layers):
            decoder_layers += [
                nn.Linear(embed_dim, embed_dim),
                norm(embed_dim) if norm is not None else nn.Identity(),
                act(),
            ]

        super().__init__(
            *decoder_layers,
            nn.Linear(embed_dim, output_dim),
        )


class LearnedSinePosEnc3d(nn.Module):
    def __init__(self, embed_dim: int, num_feats: int = 128) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_feats = num_feats

        self.adapter = nn.Sequential(
            nn.Linear(num_feats * 3, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.adapter(embedding3d(coords))


class MultiViewImagePosEnc(nn.Module):
    def __init__(
        self,
        num_feats: int,
        embed_dim: int,
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.num_feats = num_feats
        self.embed_dim = embed_dim
        self.normalize = normalize

        self.encoder = MultiFrameSinePosEnc(
            num_feats=num_feats,
            normalize=normalize,
        )

        self.adapter = nn.Sequential(
            nn.Conv2d(self.num_feats * 3, self.embed_dim * 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim * 4, self.embed_dim, kernel_size=1),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        b, n, h, w = mask.shape

        emb = self.encoder(mask)
        emb = self.adapter(emb.flatten(0, 1)).view(b, n, self.embed_dim, h, w)

        return emb


class MultiViewPerspectivePosEnc(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth_bins: int,
        depth_range: tuple[float, float],
        position_range: tuple[float, float, float, float, float, float],
        scaling: str = "quadratic",
    ) -> None:
        super().__init__()

        self.embed_dims = embed_dim
        self.depth_bins = depth_bins
        self.depth_range = depth_range
        self.position_range = position_range
        self.scaling = scaling

        self.adapter = nn.Sequential(
            nn.Conv2d(depth_bins * 3, embed_dim * 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
        )

    # pylint: disable-next=too-many-locals
    def forward(self, tgt_shape, img_shape, unproject_tx, device):
        # get unprojected view ray coordinates
        coords = unproject_image_rays(
            img_shape=img_shape,
            tgt_shape=tgt_shape,
            depth_range=self.depth_range,
            num_depth_bins=self.depth_bins,
            unproject_tx=unproject_tx,
            device=device,
            scaling=self.scaling,
        )

        b, n, w, h, d, _3 = coords.shape

        # normalize coordinates
        x_min, y_min, z_min, x_max, y_max, z_max = self.position_range
        coords[..., 0] = (coords[..., 0] - x_min) / (x_max - x_min)
        coords[..., 1] = (coords[..., 1] - y_min) / (y_max - y_min)
        coords[..., 2] = (coords[..., 2] - z_min) / (z_max - z_min)

        # bring into image feature shape
        coords = coords.permute(0, 1, 4, 5, 3, 2)  # [b, n, d, 3, h, w]
        coords = coords.reshape(b * n, d * 3, h, w)

        # encode coordinates (apply inverse sigmoid first)
        embs = inverse_sigmoid(coords)
        embs = self.adapter(embs)
        embs = embs.view(b, n, self.embed_dims, h, w)

        return embs
