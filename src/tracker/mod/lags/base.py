# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

# pylint: disable=too-many-lines
import math
from contextlib import nullcontext
from typing import Mapping

import einops
import lightning.pytorch as L
import torch
from lightning.pytorch.trainer import call
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from ... import config
from ...utils.types import MetaDict, PackedTensor, Sample
from ..module import Base, registry
from . import depth as depthnet
from . import (
    img_backbones,
    img_necks,
    matcher,
    occupancy,
    serialization,
    transformer,
    voxel_backbones,
    voxel_necks,
)
from .depth.stereo import StereoDepthNet, StereoState
from .head import OccupancyTrackingHead, QueryDecoder
from .losses import (
    MemBankLoss,
    PredictionLoss,
    SemanticGaussianLoss,
    SemanticMaskLoss,
    SemanticOccupancyLoss,
    SingleFrameLoss,
    gaussian_fov,
)
from .losses.gaussian_depth import GaussianDepthLoss
from .losses.hierarchical_occupancy import (
    HierarchicalSemanticGaussianLossHard,
    HierarchicalSemanticGaussianLossSoft,
)
from .losses.object_motion import compute_dynamic_targets
from .losses.temporal_consistency import TemporalGaussianConsistencyLoss
from .occupancy.gaussian import (
    GaussianFeatureAggregator,
    HierarchicalGaussianAggregator,
    sampling_priority,
)
from .occupancy.gaussian.temporal import (
    GaussianInstances,
    GaussianStreamInstances,
    TemporalState,
)
from .occupancy.gaussian.temporal import confidence as confidence_module
from .occupancy.gaussian.temporal import gating as visibility_gating
from .occupancy.gaussian.temporal import visibility
from .occupancy.gaussian.temporal.density_transform import DensityEgoMotionCompensation
from .posenc.fourier import embedding3d
from .runtime_tracker import RunTimeTracker
from .st_reasoner import SpatialTemporalReasoner
from .utils import Instances, build_label_map, inverse_sigmoid


@registry.register
class LatentGaussianOccupancyTracker(Base):
    # pylint: disable=too-many-ancestors
    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-public-methods

    def __init__(self, model, optimizer, lr_scheduler):
        # pylint: disable=too-many-statements,too-many-branches,too-many-locals
        super().__init__(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler)

        self.tracking = self.conf.tracking
        self.train_backbone = self.conf.train_backbone
        self.num_queries = self.conf.num_queries
        self.use_ego_update = self.conf.use_ego_update
        self.use_motion_prediction = self.conf.use_motion_prediction
        self.use_motion_prediction_ref_update = (
            self.conf.use_motion_prediction_ref_update
        )

        # When disabled, the ST-Refiner (SpatialTemporalReasoner) is not
        # constructed or executed at all; the tracking rollout falls back to
        # plain temporal bookkeeping without any learnable spatio-temporal
        # refinement.
        self.use_st_reasoner = self.conf.get("use_st_reasoner", False)

        self.pc_range = self.conf.pc_range
        self.image_base_feat_level = self.conf.get("image_base_feat_level", 0)

        self.num_gaussian_decoder_layers = self.conf.num_gaussian_decoder_layers
        self.gaussian_query_nmspool = self.conf.gaussian_query_nmspool
        self.gaussian_query_sampling = self.conf.gaussian_query_sampling
        self.supervise_intermediate_gaussians = self.conf.get(
            "supervise_intermediate_gaussians", True
        )
        self.gaussian_center_decode = self.conf.get("gaussian_center_decode", True)

        self.voxel_range = self.conf.voxel_range
        self.voxel_size = self.conf.voxel_size

        # Parse num_gaussian_queries: can be int (single stream) or dict (hierarchical)
        num_gaussian_queries = self.conf.num_gaussian_queries
        if isinstance(num_gaussian_queries, Mapping):
            self.use_hierarchical_gaussians = True

            self.num_gaussian_queries = dict(num_gaussian_queries)
            self.gaussian_scale_range = dict(self.conf.gaussian_scale_range)
            self.gaussian_sample_source_level = dict(self.conf.gaussian_sample_source)

            self.gaussian_sample_mode = self.conf.gaussian_sample_mode
            assert self.gaussian_sample_mode in ["independent", "top-down"]

            if self.gaussian_sample_mode == "top-down":
                # For top-down sampling, we need explicit ordering (finest to coarsest)
                self.gaussian_sample_order = list(self.conf.gaussian_sample_order)
                # radius for top-down sampling: dict mapping target_stream -> radius
                self.gaussian_sample_radius = dict(self.conf.gaussian_sample_radius)

            # Option to share gaussian decoders across streams
            self.share_gaussian_decoders = self.conf.get(
                "share_gaussian_decoders", False
            )
        else:
            # Single stream: integer (uses "default" stream internally)
            self.use_hierarchical_gaussians = False
            self.num_gaussian_queries = {"default": int(num_gaussian_queries)}
            self.gaussian_scale_range = {"default": self.conf.gaussian_scale_range}
            self.share_gaussian_decoders = False  # N/A for single stream

        assert self.gaussian_query_sampling in ["topk", "multinomial"]

        # Gaussian aggregation mode: how to aggregate multi-stream gaussians
        # - "hierarchical": Per-stream aggregation with multi-scale fusion (default)
        # - "merged": Concatenate all streams, then aggregate once
        self.gaussian_aggregation_mode = self.conf.get(
            "gaussian_aggregation_mode", "hierarchical"
        )
        assert self.gaussian_aggregation_mode in ["hierarchical", "merged"]

        # Selective stream decoding: which streams to decode into gaussians
        # All streams still participate in transformer, but only specified streams are decoded
        # Default: decode all streams (backward compatible)
        if "decode_gaussian_streams" in self.conf:
            self.decode_gaussian_streams = list(self.conf.decode_gaussian_streams)
            # Validate that specified streams exist
            for stream in self.decode_gaussian_streams:
                if stream not in self.num_gaussian_queries:
                    raise ValueError(
                        f"decode_gaussian_streams contains unknown stream '{stream}'. "
                        f"Available streams: {list(self.num_gaussian_queries.keys())}"
                    )
        else:
            # Default: decode all streams
            self.decode_gaussian_streams = list(self.num_gaussian_queries.keys())

        # Selective stream concatenation: which streams to concatenate for tracking_head
        # All streams still participate in transformer and decoding, but only specified streams
        # are concatenated and passed to tracking_head as keypoint features
        # Default: use all decoded streams (backward compatible)
        if "tracking_head_streams" in self.conf:
            self.tracking_head_streams = list(self.conf.tracking_head_streams)
            # Validate that specified streams exist
            for stream in self.tracking_head_streams:
                if stream not in self.num_gaussian_queries:
                    raise ValueError(
                        f"tracking_head_streams contains unknown stream '{stream}'. "
                        f"Available streams: {list(self.num_gaussian_queries.keys())}"
                    )
        else:
            # Default: use all decoded streams
            self.tracking_head_streams = self.decode_gaussian_streams

        # Whether to pass emb_decoder_gauss-projected features to the tracking head
        # (True) or raw query features (False). Only affects decoded streams.
        self.tracking_head_projected_keypoints = self.conf.get(
            "tracking_head_projected_keypoints", True
        )

        # optional filtering of the track classes for inference
        filter_classes = None
        if "filter_track_classes" in self.conf:
            filter_classes = self.conf.filter_track_classes
            if filter_classes is False:
                filter_classes = None

            if filter_classes is not None:
                class_ids = {n: i for i, n in enumerate(self.conf.instance_labels)}
                filter_classes = [class_ids[n] for n in filter_classes]
                filter_classes = torch.tensor(filter_classes, dtype=torch.long)

        if filter_classes is not None:
            self.register_buffer(
                "filter_track_classes", filter_classes, persistent=False
            )
        else:
            self.filter_track_classes = None

        # map for occupancy classes to gaussian semantic classes
        gaussian_class_map = build_label_map(
            self.conf.occupancy_labels,
            self.conf.gaussian_labels,
        )
        self.num_gaussian_classes = torch.sum(gaussian_class_map >= 0).item()
        self.register_buffer("gaussian_class_map", gaussian_class_map, persistent=False)

        # map for occupancy classes to mask semantic classes
        semantic_class_map = build_label_map(
            self.conf.occupancy_labels,
            self.conf.semantic_labels,
        )
        self.register_buffer("semantic_class_map", semantic_class_map, persistent=False)

        # initialize the components
        self.img_backbone = img_backbones.build(self.conf.img_backbone)
        self.img_neck = img_necks.build(self.conf.img_neck)

        self.depth_net = depthnet.build(self.conf.depth_net)
        self.lift_and_pool = depthnet.build(self.conf.lift_and_pool)

        if "depth" in self.conf.losses and self.conf.losses.depth is not None:
            self.depth_loss = depthnet.build(self.conf.losses.depth)
        else:
            self.depth_loss = None

        self.voxel_backbone = voxel_backbones.build(self.conf.voxel_backbone)
        self.voxel_aggregate = voxel_necks.build(self.conf.voxel_aggregate)

        # Create keypoint projections for gaussian queries
        # These project from pyramid channel dims to embed_dim
        pyramid_channels = self.voxel_backbone.num_channels

        if self.use_hierarchical_gaussians:
            # Map each stream to its pyramid channel dim based on sample source level
            stream_input_dims = {
                stream: pyramid_channels[self.gaussian_sample_source_level[stream]]
                for stream in self.num_gaussian_queries.keys()
            }
        else:
            # Non-hierarchical: single "default" stream
            # Check if gaussian_sample_source is specified, otherwise use level 0
            if hasattr(self.conf, "gaussian_sample_source"):
                default_level = int(self.conf.gaussian_sample_source)
            else:
                default_level = 0  # Default to finest level

            self.gaussian_sample_source_level = {"default": default_level}
            stream_input_dims = {"default": pyramid_channels[default_level]}

        # Create projection layers: pyramid_channels -> embed_dim
        self.gaussian_pyramid_projs = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(
                        stream_input_dims[name],
                        self.conf.voxel_transformer.embed_dim,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        self.conf.voxel_transformer.embed_dim,
                        self.conf.voxel_transformer.embed_dim,
                    ),
                )
                for name in self.num_gaussian_queries.keys()
            }
        )

        if "voxel_transformer" in self.conf and self.conf.voxel_transformer is not None:
            self.voxel_transformer = transformer.build(self.conf.voxel_transformer)
        else:
            self.voxel_transformer = None

        if "occupancy" in self.conf.losses and self.conf.losses.occupancy is not None:
            self.voxel_loss = SemanticOccupancyLoss(**self.conf.losses.occupancy)
            self.voxel_predictor = nn.Sequential(
                nn.Conv3d(self.voxel_aggregate.out_channels, 64, kernel_size=1),
                nn.Softplus(),
                nn.Conv3d(64, self.conf.losses.occupancy.num_classes, kernel_size=1),
            )
        else:
            self.voxel_loss = None
            self.voxel_predictor = None

        # Optional: Dense voxel-based gaussian losses
        if (
            "gaussian_voxel" in self.conf.losses
            and self.conf.losses.gaussian_voxel is not None
        ):
            loss_cfg = dict(self.conf.losses.gaussian_voxel)
            self.gaussian_voxel_source = loss_cfg.pop("source", "fullres_occ")

            # Extract semantic_labels from loss config for class mapping
            gaussian_voxel_semantic_labels = loss_cfg.pop("semantic_labels", None)
            if gaussian_voxel_semantic_labels is None:
                raise ValueError(
                    "gaussian_voxel loss config must specify 'semantic_labels' parameter"
                )

            # Build class map for gaussian_voxel loss
            gaussian_voxel_class_map = build_label_map(
                self.conf.occupancy_labels,
                gaussian_voxel_semantic_labels,
            )
            self.register_buffer(
                "gaussian_voxel_class_map", gaussian_voxel_class_map, persistent=False
            )

            self.gaussian_voxel_loss = SemanticOccupancyLoss(**loss_cfg)
            # Predictor projects gaussian features to semantic scores
            # Input channels from voxel_transformer (gaussian features)
            self.gaussian_voxel_predictor = nn.Sequential(
                nn.Conv3d(self.voxel_transformer.embed_dim, 64, kernel_size=1),
                nn.Softplus(),
                nn.Conv3d(64, loss_cfg["num_classes"], kernel_size=1),
            )
            # Store which feature source to use (gaussian_occ or fullres_occ)
        else:
            self.gaussian_voxel_loss = None
            self.gaussian_voxel_predictor = None
            self.gaussian_voxel_source = None

        # Optional: Binary occupancy loss on bin_scores
        if (
            "gaussian_occupancy" in self.conf.losses
            and self.conf.losses.gaussian_occupancy is not None
        ):
            loss_cfg = dict(self.conf.losses.gaussian_occupancy)

            # Extract semantic_labels from loss config for class mapping
            gaussian_voxel_occ_labels = loss_cfg.pop("nonfree_labels", None)
            if gaussian_voxel_occ_labels is None:
                raise ValueError(
                    "gaussian_occupancy loss config must specify 'nonfree_labels' parameter"
                )

            # Build class map for gaussian_occupancy loss
            gaussian_voxel_occ_map = build_label_map(
                self.conf.occupancy_labels,
                gaussian_voxel_occ_labels,
                default=-1,
            )
            self.register_buffer(
                "gaussian_voxel_occ_map", gaussian_voxel_occ_map, persistent=False
            )

            # Binary occupancy loss uses existing SemanticOccupancyLoss with 2 classes
            loss_cfg["num_classes"] = 2  # Binary: free vs occupied
            self.gaussian_occupancy_loss = SemanticOccupancyLoss(**loss_cfg)
        else:
            self.gaussian_occupancy_loss = None

        # Optional: Gaussian depth rendering loss (supervises gaussian placement via gsplat)
        if (
            "gaussian_depth" in self.conf.losses
            and self.conf.losses.gaussian_depth is not None
        ):
            loss_cfg = dict(self.conf.losses.gaussian_depth)
            self.gaussian_depth_loss = GaussianDepthLoss(**loss_cfg)
        else:
            self.gaussian_depth_loss = None

        if (
            "gaussian_serialization" in self.conf
            and self.conf.gaussian_serialization is not None
        ):
            self.gaussian_serialization = serialization.Curve(
                dims=3, **self.conf.gaussian_serialization
            )
        else:
            self.gaussian_serialization = None

        # (Hierarchical) Gaussian aggregator
        scale_multiplier = self.conf.get("gaussian_aggregator_scale", 3.0)
        min_radius = self.conf.get("gaussian_aggregator_min_radius", 1)

        # Choose aggregator based on aggregation mode
        if (
            self.gaussian_aggregation_mode == "hierarchical"
            and self.use_hierarchical_gaussians
        ):
            # Per-stream aggregation with multi-scale fusion
            self.gaussian_aggregator = HierarchicalGaussianAggregator(
                voxel_size=self.voxel_size,
                voxel_range=self.voxel_range,
                streams=self.num_gaussian_queries.keys(),
                scale_multiplier=scale_multiplier,
                fusion_mode=self.conf.gaussian_aggregator_fusion,
                num_channels=self.voxel_transformer.embed_dim,
                min_radius=min_radius,
            )
        else:
            # Single-stream aggregation (used for non-hierarchical or merged mode)
            self.gaussian_aggregator = GaussianFeatureAggregator(
                voxel_size=self.voxel_size,
                voxel_range=self.voxel_range,
                scale_multiplier=scale_multiplier,
                min_radius=min_radius,
            )

        self.voxel_fuse = nn.Sequential(
            nn.Conv3d(
                self.voxel_aggregate.out_channels + self.voxel_transformer.embed_dim,
                self.voxel_aggregate.out_channels,
                kernel_size=1,
            ),
            nn.BatchNorm3d(self.voxel_aggregate.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                self.voxel_aggregate.out_channels,
                self.voxel_aggregate.out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        # Gaussian temporal aggregation
        self.conf = config.utils.copy(self.conf, readonly=False)
        temporal_cfg = self.conf.get("gaussian_temporal", {})
        self.use_gaussian_temporal = temporal_cfg.get("enabled", False)

        if self.use_gaussian_temporal:
            # Build ego-motion compensation module
            self.ego_motion_compensation = occupancy.gaussian.temporal.emc.build(
                temporal_cfg.emc,
                voxel_range=self.voxel_range,
            )

            self.gaussian_pruning = occupancy.gaussian.temporal.pruning.build_sequence(
                temporal_cfg.pruning,
            )

            # Temporal aggregation mode
            # - "context": Use temporal gaussians only as context in transformer,
            #              decode only fresh queries (fixed budget)
            # - "track": Track both temporal and fresh gaussians through decoding
            #            (variable budget)
            self.gaussian_temporal_mode = temporal_cfg.get("mode", "context")
            assert self.gaussian_temporal_mode in ["context", "track"], (
                f"Invalid temporal mode: {self.gaussian_temporal_mode}. "
                "Must be 'context' or 'track'"
            )

            # Fresh query sampling strategy
            # - "fixed": Always sample num_gaussian_queries[stream] fresh queries
            # - "adaptive": Sample (num_gaussian_queries[stream] - n_temporal) fresh queries
            self.gaussian_temporal_sampling = temporal_cfg.get("sampling", "adaptive")
            assert self.gaussian_temporal_sampling in ["fixed", "adaptive"], (
                f"Invalid temporal sampling: {self.gaussian_temporal_sampling}. "
                "Must be 'fixed' or 'adaptive'"
            )

            # Reconcile query_coords with decoded centers before storing temporal state
            # When enabled, query_coords are set to normalized centers, so that:
            # 1. EMC feature adaptation uses actual gaussian displacement
            # 2. Transformer positional encoding reflects actual gaussian position
            # Default: False for backwards compatibility
            self.gaussian_temporal_reconcile_coords = temporal_cfg.get(
                "reconcile_coords", False
            )

            # Age embedding for temporal queries
            # Provides learned signal about query age for the transformer
            self.gaussian_age_embedding = (
                occupancy.gaussian.temporal.age_embedding.build(
                    temporal_cfg.get("age_embedding", None),
                    embed_dim=self.conf.voxel_transformer.embed_dim,
                )
            )

            # Visibility-based gating for transformer updates
            gating_cfg = temporal_cfg.get("gating", None)
            gating_cfg = config.utils.copy(gating_cfg, readonly=False)

            self.temporal_gate_visibility = visibility.build(
                (gating_cfg or {}).pop("visibility", None)
            )
            self.temporal_gate = visibility_gating.build(gating_cfg)

            # Confidence tracking for temporal gaussians
            confidence_cfg = temporal_cfg.get("confidence", {})
            if confidence_cfg.pop("enabled", False):
                self.confidence_visibility = visibility.build(
                    confidence_cfg.pop("visibility", None)
                )

                # Extract loss config before building tracker
                confidence_loss_cfg = confidence_cfg.pop("loss", None)
                dynamic_labels = confidence_cfg.pop("dynamic_labels", None)
                self.confidence_tracker = confidence_module.build(
                    confidence_cfg,
                    gaussian_labels=(
                        list(self.conf.gaussian_labels) if dynamic_labels else None
                    ),
                    dynamic_labels=list(dynamic_labels) if dynamic_labels else None,
                )

                # Learned confidence loss (optional)
                if confidence_loss_cfg is not None:
                    self.learned_confidence_loss = (
                        confidence_module.LearnedConfidenceLoss(
                            weight=confidence_loss_cfg.get("weight", 1.0),
                        )
                    )
                else:
                    self.learned_confidence_loss = None
            else:
                self.confidence_visibility = None
                self.confidence_tracker = None
                self.learned_confidence_loss = None
        else:
            self.ego_motion_compensation = None
            self.gaussian_temporal_mode = None
            self.gaussian_temporal_sampling = None
            self.gaussian_temporal_reconcile_coords = False
            self.gaussian_age_embedding = None
            self.temporal_gate_visibility = None
            self.temporal_gate = None
            self.confidence_tracker = None
            self.confidence_visibility = None
            self.learned_confidence_loss = None

        # Gaussian FoV/visibility losses
        loss_gaussian_fov = self.conf.losses.get("gaussian_fov", None)
        if loss_gaussian_fov is not None:
            self.loss_gaussian_fov = gaussian_fov.build(loss_gaussian_fov.losses)
            self.loss_gaussian_fov_vis = visibility.build(loss_gaussian_fov.visibility)
        else:
            self.loss_gaussian_fov = None
            self.loss_gaussian_fov_vis = None

        # Sampling priority computer for density-aware sampling
        sampling_priority_cfg = self.conf.get(
            "gaussian_sample_priority", {"type": "feature-magnitude"}
        )
        self.sampling_priority_computer = sampling_priority.build(sampling_priority_cfg)

        # Density transform (only for density-aware mode)
        if sampling_priority_cfg.get("type") == "temporal-density-aware":
            assert self.use_gaussian_temporal

            self.density_transform = DensityEgoMotionCompensation(
                voxel_range=self.voxel_range,
                voxel_size=self.voxel_size,
            )
        else:
            self.density_transform = None

        # Temporal consistency loss (only when gaussian_temporal is enabled)
        if self.use_gaussian_temporal:
            consistency_cfg = temporal_cfg.get("consistency_loss", {})
            if consistency_cfg.get("enabled", False):
                gaussian_labels = list(self.conf.gaussian_labels)
                dynamic_labels = consistency_cfg.get("dynamic_labels", None)

                self.temporal_consistency_loss = TemporalGaussianConsistencyLoss(
                    center_weight=consistency_cfg.get("center_weight", 1.0),
                    scale_weight=consistency_cfg.get("scale_weight", 0.5),
                    rotation_weight=consistency_cfg.get("rotation_weight", 0.5),
                    opacity_weight=consistency_cfg.get("opacity_weight", 0.1),
                    logits_weight=consistency_cfg.get("logits_weight", 0.0),
                    gaussian_labels=gaussian_labels,
                    dynamic_labels=dynamic_labels,
                    static_weighting_mode=consistency_cfg.get(
                        "static_weighting_mode", "soft"
                    ),
                    static_class_weight=consistency_cfg.get("static_class_weight", 1.0),
                    dynamic_class_weight=consistency_cfg.get(
                        "dynamic_class_weight", 1.0
                    ),
                    visibility_boost=consistency_cfg.get("visibility_boost", 0.0),
                    confidence_boost=consistency_cfg.get("confidence_boost", 0.0),
                )

                # Dynamic supervision config
                dynamic_cfg = consistency_cfg.get("dynamic_supervision", {})
                self.dynamic_supervision_enabled = dynamic_cfg.get("enabled", False)
                self.dynamic_max_match_distance = dynamic_cfg.get(
                    "max_match_distance", 2.0
                )
            else:
                self.temporal_consistency_loss = None
                self.dynamic_supervision_enabled = False
                self.dynamic_max_match_distance = 2.0
        else:
            self.temporal_consistency_loss = None
            self.dynamic_supervision_enabled = False
            self.dynamic_max_match_distance = 2.0

        self.tracking_head = OccupancyTrackingHead(**self.conf.head)
        self.num_classes = self.tracking_head.num_instance_classes
        self.embed_dims = self.tracking_head.embed_dims

        if self.use_st_reasoner:
            self.st_reasoner = SpatialTemporalReasoner(
                **self.conf.spatial_temporal_reason
            )
            self.hist_len = self.st_reasoner.hist_len
            self.fut_len = self.st_reasoner.fut_len
            self.st_history_reasoning = self.st_reasoner.history_reasoning
            self.st_future_reasoning = self.st_reasoner.future_reasoning
        else:
            # No ST-Refiner: keep the temporal buffer dimensions (still used by
            # the fallback bookkeeping) but disable all learnable reasoning. The
            # ego-motion and motion-prediction updates live in the ST-Refiner,
            # so they must be disabled too.
            self.st_reasoner = None
            self.hist_len = self.conf.spatial_temporal_reason.get("hist_len", 3)
            self.fut_len = self.conf.spatial_temporal_reason.get("fut_len", 4)
            self.st_history_reasoning = False
            self.st_future_reasoning = False
            assert (
                not self.use_ego_update and not self.use_motion_prediction
            ), "use_ego_update / use_motion_prediction require use_st_reasoner=true"

        self.runtime_tracker = RunTimeTracker(**self.conf.runtime_tracker)

        self.matcher = matcher.build(self.conf.matcher)
        self.semantic_mask_matcher = matcher.build(self.conf.semantic_mask_matcher)

        # Voxel subsampling for instance occupancy matching + mask loss. Bounds
        # the per-query mask cost/loss memory on datasets with very large valid
        # masks. Disabled (no-op) when the fraction is None. ``fraction`` is of
        # the full grid; ``pos_share`` is the budget reserved for occupied
        # (positive) voxels (see ``_subsample_match_voxels``).
        self.match_voxel_fraction = self.conf.get(
            "match_voxel_subsample_fraction", None
        )
        self.match_voxel_pos_share = self.conf.get(
            "match_voxel_subsample_pos_share", 0.5
        )

        self.single_frame_loss = SingleFrameLoss(
            num_classes=self.num_classes,
            **self.conf.losses.single_frame,
        )

        # Optional: Per-gaussian semantic loss
        if (
            "semantic_gauss" in self.conf.losses
            and self.conf.losses.semantic_gauss is not None
        ):
            # Choose gaussian loss based on type field
            loss_config = dict(self.conf.losses.semantic_gauss)
            loss_type = loss_config.pop("type", "single")

            if loss_type == "single":
                # Single-scale loss (concatenates all streams)
                self.semantic_gauss_loss = SemanticGaussianLoss(**loss_config)
                self.semantic_gauss_loss_is_hierarchical = False
            elif loss_type == "hierarchical-soft":
                # Hierarchical loss with soft labels (per-stream supervision)
                self.semantic_gauss_loss = HierarchicalSemanticGaussianLossSoft(
                    **loss_config
                )
                self.semantic_gauss_loss_is_hierarchical = True
            elif loss_type == "hierarchical-hard":
                # Hierarchical loss with hard labels (per-stream supervision)
                self.semantic_gauss_loss = HierarchicalSemanticGaussianLossHard(
                    **loss_config
                )
                self.semantic_gauss_loss_is_hierarchical = True
            else:
                raise ValueError(f"Unknown semantic_gauss loss type: {loss_type}")
        else:
            self.semantic_gauss_loss = None
            self.semantic_gauss_loss_is_hierarchical = False

        self.semantic_mask_loss = SemanticMaskLoss(
            **self.conf.losses.semantic_mask,
        )

        self.mem_bank_loss = MemBankLoss(
            num_classes=self.num_classes,
            **self.conf.losses.mem_bank,
        )

        self.prediction_loss = PredictionLoss(
            **self.conf.losses.prediction,
        )

        self.occupancy_predictor = occupancy.mask.build(self.conf.occupancy_predictor)

        # optimization mode
        self.optimization_mode = OmegaConf.select(
            self.conf, "optimization.mode", default="default"
        )

        if self.optimization_mode == "detached":
            self.automatic_optimization = False

            optim = self.conf.optimization

            self.optimizer = Optimizer(
                gradient_clip_algorithm=optim.get("gradient_clip_algorithm", None),
                gradient_clip_val=optim.get("gradient_clip_val", None),
            )

        # Temporal warmup configuration
        # Allows running N-K warmup frames without gradients before K training frames
        warmup_cfg = self.conf.get("temporal_warmup", {})
        if warmup_cfg.get("enabled", False):
            self.num_warmup_frames = warmup_cfg.get("warmup_frames", 0)
        else:
            self.num_warmup_frames = 0

        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        """Generate the instances for tracking, especially the object queries"""
        # pylint: disable=too-many-branches, too-many-locals, too-many-statements
        # query initialization for detection
        # reference points, mapping fourier encoding to embed_dims
        self.reference_points = nn.Embedding(self.num_queries, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(128 * 3, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

        # embedding initialization for tracking
        if self.tracking:
            self.query_feat_embedding = nn.Embedding(self.num_queries, self.embed_dims)
            nn.init.zeros_(self.query_feat_embedding.weight)

        # build coordinate grids for each pyramid level if using hierarchical gaussians
        if self.use_hierarchical_gaussians:
            num_pyramid_levels = max(self.gaussian_sample_source_level.values()) + 1

            for level in range(num_pyramid_levels):
                scale_factor = 2**level

                level_coords = self._build_voxel_coords(
                    voxel_range=self.voxel_range,
                    voxel_size=[s * scale_factor for s in self.voxel_size],
                )
                self.register_buffer(
                    f"voxel_coords_level{level}", level_coords, persistent=False
                )
        else:
            # build voxel coordinates for finest level
            coords = self._build_voxel_coords(
                voxel_range=self.voxel_range,
                voxel_size=self.voxel_size,
            )
            self.register_buffer("voxel_coords", coords, persistent=False)

        # Only create decoders for streams that will be decoded
        decoded_stream_names = list(self.decode_gaussian_streams)

        # Check if we need class decoder (only needed for semantic_gauss loss)
        use_cls_decoder = (
            "semantic_gauss" in self.conf.losses
            and self.conf.losses.semantic_gauss is not None
        )

        if self.share_gaussian_decoders:
            # Shared decoders: create one decoder and share across all streams
            if use_cls_decoder:
                shared_cls = QueryDecoder(
                    embed_dim=self.embed_dims,
                    output_dim=self.num_gaussian_classes,
                    num_layers=self.num_gaussian_decoder_layers,
                    act=nn.ReLU,
                    norm=nn.LayerNorm,
                )

            k = 3 if self.gaussian_center_decode else 0
            shared_reg = QueryDecoder(
                embed_dim=self.embed_dims,
                output_dim=k + 3 + 4 + 1,  # (center +) size + quat. + opacity
                num_layers=self.num_gaussian_decoder_layers,
                act=nn.ReLU,
                norm=None,
            )

            shared_emb = QueryDecoder(
                embed_dim=self.embed_dims,
                output_dim=self.embed_dims,
                num_layers=self.num_gaussian_decoder_layers,
                act=nn.ReLU,
                norm=nn.LayerNorm,
            )

            # Point decoded stream names to the same shared decoder
            if use_cls_decoder:
                self.cls_decoder_gauss = nn.ModuleDict(
                    {name: shared_cls for name in decoded_stream_names}
                )
            else:
                self.cls_decoder_gauss = None

            self.reg_decoder_gauss = nn.ModuleDict(
                {name: shared_reg for name in decoded_stream_names}
            )
            self.emb_decoder_gauss = nn.ModuleDict(
                {name: shared_emb for name in decoded_stream_names}
            )

            # Init bias for shared decoder
            if use_cls_decoder:
                bias_prior = 0.01
                bias_init = -math.log((1.0 - bias_prior) / bias_prior)
                nn.init.constant_(shared_cls[-1].bias, bias_init)

        else:
            # Per-stream decoders (or single "default" stream for non-hierarchical)
            # Only create decoders for streams that will be decoded
            if use_cls_decoder:
                self.cls_decoder_gauss = nn.ModuleDict(
                    {
                        name: QueryDecoder(
                            embed_dim=self.embed_dims,
                            output_dim=self.num_gaussian_classes,
                            num_layers=self.num_gaussian_decoder_layers,
                            act=nn.ReLU,
                            norm=nn.LayerNorm,
                        )
                        for name in decoded_stream_names
                    }
                )
            else:
                self.cls_decoder_gauss = None

            k = 3 if self.gaussian_center_decode else 0
            self.reg_decoder_gauss = nn.ModuleDict(
                {
                    name: QueryDecoder(
                        embed_dim=self.embed_dims,
                        output_dim=k + 3 + 4 + 1,  # (center +) size + quat. + opacity
                        num_layers=self.num_gaussian_decoder_layers,
                        act=nn.ReLU,
                        norm=None,
                    )
                    for name in decoded_stream_names
                }
            )

            self.emb_decoder_gauss = nn.ModuleDict(
                {
                    name: QueryDecoder(
                        embed_dim=self.embed_dims,
                        output_dim=self.embed_dims,
                        num_layers=self.num_gaussian_decoder_layers,
                        act=nn.ReLU,
                        norm=nn.LayerNorm,
                    )
                    for name in decoded_stream_names
                }
            )

            # Init bias of last class probability decoder layer via prior probability
            if use_cls_decoder:
                bias_prior = 0.01
                bias_init = -math.log((1.0 - bias_prior) / bias_prior)
                for name in decoded_stream_names:
                    nn.init.constant_(self.cls_decoder_gauss[name][-1].bias, bias_init)

        # Aggregation opacity mode: how to determine opacity for 3D feature splatting
        # - "decoded": use the opacity decoded from gaussian queries (default)
        # - "fixed": use fixed opacity=1.0 (decouples 3D aggregation from depth loss)
        # - "confidence": use max(opacity, confidence) - well-observed gaussians stay active
        # - "local": use separate learned local opacity decoder
        self.aggregation_opacity_mode = self.conf.get(
            "aggregation_opacity_mode", "decoded"
        )
        assert self.aggregation_opacity_mode in [
            "decoded",
            "fixed",
            "confidence",
            "local",
        ], (
            f"Invalid aggregation_opacity_mode: {self.aggregation_opacity_mode}. "
            "Must be 'decoded', 'fixed', 'confidence', or 'local'"
        )

        # Opacity threshold for feature splatting: gaussians with opacity below
        # this value are zeroed out before aggregation (effectively excluded).
        # Default 0.0 means no thresholding (backward compatible).
        self.aggregation_opacity_threshold = self.conf.get(
            "aggregation_opacity_threshold", 0.0
        )

        if self.aggregation_opacity_mode == "local":
            local_opacity_layers = self.conf.get("local_opacity_layers", 1)
            self.local_opacity_decoder = nn.ModuleDict(
                {
                    name: QueryDecoder(
                        embed_dim=self.embed_dims,
                        output_dim=1,
                        num_layers=local_opacity_layers,
                    )
                    for name in decoded_stream_names
                }
            )
        else:
            self.local_opacity_decoder = None

    def _build_voxel_coords(self, voxel_range: list[float], voxel_size: list[float]):
        # pylint: disable=too-many-locals

        x_min, y_min, z_min, x_max, y_max, z_max = voxel_range
        sx, sy, sz = voxel_size

        nx = int(round((x_max - x_min) / sx))
        ny = int(round((y_max - y_min) / sy))
        nz = int(round((z_max - z_min) / sz))

        # create grid coordinates for voxel centers
        x = torch.arange(nx, dtype=torch.float32) + 0.5
        y = torch.arange(ny, dtype=torch.float32) + 0.5
        z = torch.arange(nz, dtype=torch.float32) + 0.5

        # map grid coordinates to [0, 1]
        x = x / nx
        y = y / ny
        z = z / nz

        x = x.view(1, 1, nx, 1).expand(nz, ny, nx, 1)
        y = y.view(1, ny, 1, 1).expand(nz, ny, nx, 1)
        z = z.view(nz, 1, 1, 1).expand(nz, ny, nx, 1)

        return torch.cat((x, y, z), dim=-1)

    def _normalize_centers_to_coords(self, centers: torch.Tensor) -> torch.Tensor:
        """
        Normalize world-space centers to [0, 1] query_coords space.

        Args:
            centers: Gaussian centers [b, n, 3] in world coordinates (meters)

        Returns:
            Normalized coordinates [b, n, 3] in [0, 1]
        """
        vx_range = torch.as_tensor(self.voxel_range, device=centers.device)
        vx_min, vx_max = vx_range[:3], vx_range[3:]

        # Normalize: (centers - min) / (max - min)
        normalized = (centers - vx_min) / (vx_max - vx_min)

        return normalized

    def _init_weights(self):
        # freeze backbone
        if not self.train_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False

    def generate_empty_instance(self):
        # pylint: disable=too-many-locals
        """Build the initial per-query track instances for a new sequence.

        Every field the tracking rollout consumes is created up front, zero- or
        identity-initialized. Query positions, embeddings and (when tracking)
        features are seeded from the learned detection queries; all prediction,
        cache and history/future buffers start empty, with the temporal padding
        masks marking every slot as unfilled.
        """
        device = self.reference_points.weight.device
        num_queries = self.reference_points.weight.shape[0]
        embed_dims = self.embed_dims
        num_classes = self.num_classes
        hist_len = self.hist_len
        fut_len = self.fut_len

        # Seed values derived from the learned detection queries. Clone before
        # storing so the embedding weights are never aliased into the instances.
        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(embedding3d(reference_points))
        if self.tracking:
            query_feats = self.query_feat_embedding.weight.clone()
        else:
            query_feats = torch.zeros_like(query_embeds)

        inst = Instances()

        # Assign a length-``num_queries`` field first to fix the instance length.
        inst.reference_points = reference_points.clone()
        inst.query_embeds = query_embeds.clone()
        inst.query_feats = query_feats.clone()

        # Cache seeds mirror the query seeds.
        inst.cache_reference_points = reference_points.clone()
        inst.cache_query_embeds = query_embeds.clone()
        inst.cache_query_feats = query_feats.clone()

        # Integer/boolean per-query bookkeeping. ``obj_idxes == -1`` marks a
        # query as not yet assigned to a tracked object.
        inst.obj_idxes = torch.full((num_queries,), -1, dtype=torch.long, device=device)
        inst.disappear_time = torch.zeros(num_queries, dtype=torch.long, device=device)
        inst.track_query_mask = torch.zeros(
            num_queries, dtype=torch.bool, device=device
        )

        # All remaining tensors are zero-initialized floats; only their trailing
        # shape differs, so build them from a table.
        zero_fields = [
            ("logits", (num_classes,)),
            ("bboxes", (10,)),
            ("scores", ()),
            ("motion_predictions", (fut_len, 3)),
            ("cache_logits", (num_classes,)),
            ("cache_bboxes", (10,)),
            ("cache_scores", ()),
            ("cache_motion_predictions", (fut_len, 3)),
            ("hist_embeds", (hist_len, embed_dims)),
            ("hist_xyz", (hist_len, 3)),
            ("hist_position_embeds", (hist_len, embed_dims)),
            ("hist_bboxes", (hist_len, 10)),
            ("hist_logits", (hist_len, num_classes)),
            ("hist_scores", (hist_len,)),
            ("fut_embeds", (fut_len, embed_dims)),
            ("fut_xyz", (fut_len, 3)),
            ("fut_position_embeds", (fut_len, embed_dims)),
            ("fut_bboxes", (fut_len, 10)),
            ("fut_logits", (fut_len, num_classes)),
            ("fut_scores", (fut_len,)),
        ]
        for name, trailing in zero_fields:
            inst.set(
                name,
                torch.zeros(
                    (num_queries, *trailing), dtype=torch.float32, device=device
                ),
            )

        # Temporal padding masks start all-True: every history/future slot is
        # empty (1 = padded).
        inst.hist_padding_masks = torch.ones(
            (num_queries, hist_len), dtype=torch.bool, device=device
        )
        inst.fut_padding_masks = torch.ones(
            (num_queries, fut_len), dtype=torch.bool, device=device
        )

        return inst

    def spatial_temporal_reason(self, track_instances: Instances) -> Instances:
        """Run the ST-Refiner, or its non-refiner fallback when disabled.

        With the ST-Refiner enabled this delegates to it. When disabled, we only
        advance the per-track temporal buffers by one step (no learnable
        reasoning), which is all the downstream tracking rollout requires.
        """
        if self.use_st_reasoner:
            return self.st_reasoner(track_instances)
        return self._shift_track_history(track_instances)

    def sync_pos_embedding(self, track_instances: Instances) -> Instances:
        """Refresh positional embeddings, via the ST-Refiner or the fallback."""
        if self.use_st_reasoner:
            return self.st_reasoner.sync_pos_embedding(
                track_instances, self.query_embedding
            )

        # Non-refiner fallback: recompute the query positional embedding from the
        # current reference points, and keep the history positional embeds in
        # sync for buffer consistency. No learnable refinement is applied.
        track_instances.query_embeds = self.query_embedding(
            embedding3d(track_instances.reference_points)
        )
        track_instances.hist_position_embeds = self.query_embedding(
            embedding3d(track_instances.hist_xyz)
        )
        return track_instances

    def _shift_track_history(self, track_instances: Instances) -> Instances:
        """Advance the per-track temporal buffers by one frame (no reasoning).

        Non-refiner fallback for the ST-Refiner's frame shift: it rolls every
        history/future buffer forward and appends the current cached state (for
        history) or a zero/pad slice (for future, since without the refiner
        there are no forecasts). This keeps the buffer bookkeeping the tracking
        rollout expects without executing any separately licensed code.
        """
        ti = track_instances
        device = ti.query_feats.device

        def roll_append(buf, new):
            # drop the oldest step, append ``new`` ([N, ...]) as the newest
            return torch.cat((buf[:, 1:], new[:, None]), dim=1)

        # History buffers: append the current frame's cached state.
        ti.hist_embeds = roll_append(ti.hist_embeds.clone(), ti.cache_query_feats)
        ti.hist_xyz = roll_append(ti.hist_xyz, ti.cache_reference_points)
        ti.hist_position_embeds = roll_append(
            ti.hist_position_embeds, ti.cache_query_embeds
        )
        ti.hist_bboxes = roll_append(ti.hist_bboxes, ti.cache_bboxes)
        ti.hist_logits = roll_append(ti.hist_logits, ti.cache_logits)
        ti.hist_scores = torch.cat(
            (ti.hist_scores[:, 1:], ti.cache_scores[:, None]), dim=1
        )
        ti.hist_padding_masks = torch.cat(
            (
                ti.hist_padding_masks[:, 1:],
                torch.zeros((len(ti), 1), dtype=torch.bool, device=device),
            ),
            dim=1,
        )

        # Future buffers: roll forward and pad (no forecasts without the refiner).
        for field in (
            "motion_predictions",
            "fut_embeds",
            "fut_xyz",
            "fut_position_embeds",
            "fut_bboxes",
            "fut_logits",
        ):
            buf = getattr(ti, field)
            setattr(ti, field, roll_append(buf, torch.zeros_like(buf[:, 0])))
        ti.fut_scores = torch.cat(
            (ti.fut_scores[:, 1:], torch.zeros_like(ti.fut_scores[:, 0:1])), dim=1
        )
        ti.fut_padding_masks = torch.cat(
            (
                ti.fut_padding_masks[:, 1:],
                torch.ones_like(ti.fut_padding_masks[:, 0:1]).bool(),
            ),
            dim=1,
        )
        return ti

    def load_detection_output_into_cache(self, track_instances: Instances, out):
        """Copy the detection head's output into the track instances' cache.

        Writes the last-decoder-layer, single-batch predictions from ``out`` into
        the ``cache_*`` fields of ``track_instances`` in place and returns it. The
        query features and reference points are popped from ``out`` (they are
        consumed here, not by later heads); the cached positional embedding is
        recomputed from the cached reference points.
        """
        # last decoder layer of the single batch element
        cls = out["all_cls_scores"][0, -1]  # [Q, num_classes]
        boxes = out["all_bbox_preds"][0, -1]  # [Q, 10]
        feats = out.pop("query_feats")[0]  # [Q, embed_dims]
        refs = out.pop("reference_points")[0]  # [Q, 3]

        with torch.no_grad():
            scores = cls.sigmoid().amax(dim=-1)  # per-query max class probability

        track_instances.cache_logits = cls.clone()
        track_instances.cache_scores = scores.clone()
        track_instances.cache_bboxes = boxes.clone()
        track_instances.cache_query_feats = feats.clone()
        track_instances.cache_reference_points = refs.clone()
        track_instances.cache_query_embeds = self.query_embedding(embedding3d(refs))

        return track_instances

    def frame_summarization(self, inst, tracking=False):
        """Commit the cached, post-reasoning predictions into the current-frame
        fields of the track instances.

        An entry is treated as active when its cached score passes the tracker's
        record threshold (inference) or unconditionally during training (where
        the current boxes/logits/scores are also seeded from the cache). For each
        active entry the cached query features, positional embeddings, logits,
        scores, motion predictions, boxes, and reference points overwrite the
        current-frame values. When future reasoning is enabled, the future
        positions and boxes are additionally rolled out from the current state by
        accumulating the predicted per-step motion.
        """

        # inference mode
        if tracking:
            active = inst.cache_scores >= self.runtime_tracker.record_threshold

        # training mode
        else:
            inst.bboxes = inst.cache_bboxes.clone()
            inst.logits = inst.cache_logits.clone()
            inst.scores = inst.cache_scores.clone()
            active = inst.cache_scores >= 0.0

        inst.query_feats[active] = inst.cache_query_feats[active].to(
            dtype=inst.query_feats.dtype
        )
        inst.query_embeds[active] = inst.cache_query_embeds[active].to(
            dtype=inst.query_embeds.dtype
        )
        inst.logits[active] = inst.cache_logits[active].to(dtype=inst.logits.dtype)
        inst.scores[active] = inst.cache_scores[active].to(dtype=inst.scores.dtype)
        inst.motion_predictions[active] = inst.cache_motion_predictions[active].to(
            dtype=inst.motion_predictions.dtype
        )
        inst.bboxes[active] = inst.cache_bboxes[active].to(dtype=inst.bboxes.dtype)
        inst.reference_points[active] = inst.cache_reference_points[active].to(
            dtype=inst.reference_points.dtype
        )

        if self.st_future_reasoning:
            motion_predictions = inst.motion_predictions[active]
            inst.fut_xyz[active] = (
                inst.reference_points[active]
                .clone()[:, None, :]
                .repeat(1, self.fut_len, 1)
            )
            inst.fut_bboxes[active] = (
                inst.bboxes[active]
                .clone()[:, None, :]
                .repeat(1, self.fut_len, 1)
                .to(dtype=inst.fut_bboxes.dtype)
            )

            motion_add = torch.cumsum(motion_predictions.clone().detach(), dim=1)
            motion_add_normalized = motion_add.clone()
            motion_add_normalized[..., 0] /= self.pc_range[3] - self.pc_range[0]
            motion_add_normalized[..., 1] /= self.pc_range[4] - self.pc_range[1]

            inst.fut_xyz[active, :, 0] += motion_add_normalized[..., 0]
            inst.fut_xyz[active, :, 1] += motion_add_normalized[..., 1]
            inst.fut_bboxes[active, :, 0] += motion_add[..., 0]
            inst.fut_bboxes[active, :, 1] += motion_add[..., 1]

        return inst

    def update_assignments(
        self,
        track_instances: Instances,
        assignments: torch.Tensor,
        labels: MetaDict,
    ) -> Instances:
        gt_instance_ids = labels.instance_ids.get(0)
        assignments = assignments[0]

        valid = assignments >= 0

        assigned_iids = torch.full_like(track_instances.obj_idxes, -1)
        assigned_iids[valid] = gt_instance_ids[assignments[valid]]

        # Note: We only update the object indices that have not been
        # matched yet, keeping any object indices for discontinued or
        # temporarily occluded tracks unchanged.
        unmatched = track_instances.obj_idxes < 0
        track_instances.obj_idxes[unmatched] = assigned_iids[unmatched]

        return track_instances

    def extract_image_features(
        self,
        images: torch.Tensor,
    ) -> list[torch.Tensor]:
        b, num_cams, c, h, w = images.shape

        context = nullcontext if self.train_backbone else torch.no_grad
        with context():
            feats = self.img_backbone(images.view(b * num_cams, c, h, w))

        feats = self.img_neck(feats)
        feats = [f.view(b, num_cams, *f.shape[1:]) for f in feats]

        return feats

    def _compute_depth(
        self,
        image_feats: list[torch.Tensor],
        img_meta: MetaDict,
        ego_to_global: torch.Tensor,
        ego_to_image: torch.Tensor,
        image_to_ego: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute depth and context features, handling both mono and stereo.

        Args:
            image_feats: FPN features from extract_image_features
            img_meta: Camera metadata for depth_net
            ego_to_global: Current frame's ego-to-global transform [4, 4]
            ego_to_image: Current frame's ego-to-image transform [b, ncams, 4, 4]
            image_to_ego: Current frame's image-to-ego transform [b, ncams, 4, 4]

        Returns:
            depth: [b, ncams, depth_bins, h, w]
            context: [b, ncams, context_channels, h, w]
        """
        image_feats = image_feats[self.image_base_feat_level]

        if isinstance(self.depth_net, StereoDepthNet):
            # Build stereo_meta from runtime_tracker state
            stereo_meta = None
            if self.runtime_tracker.stereo_state is not None:
                ss = self.runtime_tracker.stereo_state
                stereo_meta = {
                    "prev_features": ss.features,
                    "prev_ego_to_image": ss.ego_to_image,
                    "curr_image_to_ego": image_to_ego,
                    "curr_ego_to_global": ego_to_global,
                    "prev_ego_to_global": ss.ego_to_global,
                }

            depth, context = self.depth_net(image_feats, img_meta, stereo_meta)

            # Store features for next frame
            self.runtime_tracker.stereo_state = StereoState(
                features=self.depth_net.get_cv_features(),
                ego_to_global=ego_to_global.detach(),
                ego_to_image=ego_to_image.detach(),
            )
        else:
            depth, context = self.depth_net(image_feats, img_meta)

        return depth, context

    def sample_gaussian_queries(
        self,
        pyramid_features: list[torch.Tensor],
        num_queries: dict[str, int] | None = None,
        past_density: torch.Tensor | None = None,
    ) -> dict[str, MetaDict]:
        """
        Sample queries from voxel features.

        Args:
            pyramid_features: List of voxel features at different resolutions
            num_queries: Optional dict to override num_gaussian_queries per stream
            past_density: Optional density from previous frame [b, 1, d, h, w]
                         (after ego-motion compensation) for density-aware sampling

        Returns:
            Dict mapping stream names to query features and coordinates
        """
        if num_queries is None:
            num_queries = self.num_gaussian_queries

        # Compute sampling priorities per pyramid level
        priorities_per_level = self._compute_sampling_priorities(
            pyramid_features, past_density
        )

        # Sample queries
        if self.use_hierarchical_gaussians:
            # hierarchical sampling
            queries = self.sample_hierarchical_gaussian_queries(
                pyramid_features, num_queries, priorities_per_level
            )
        else:
            # single-stream sampling
            query_feats, query_coords, _ = self._sample_from_level(
                pyramid_features[0],
                self.voxel_coords,
                num_queries["default"],
                priorities=priorities_per_level[0],
            )

            queries = {
                "default": MetaDict(
                    {"query": query_feats, "query_coords": query_coords}
                )
            }

        # Initialize age, confidence, and instance_ids of new queries
        for _, data in queries.items():
            b, n = data.query.shape[:2]
            data.age = torch.zeros((b, n), dtype=torch.int64, device=data.query.device)
            data.confidence = torch.zeros(
                (b, n), dtype=torch.float32, device=data.query.device
            )

            # Assign sequential instance IDs from the runtime tracker counter
            start_id = self.runtime_tracker.gaussian_instance_id
            ids = torch.arange(start_id, start_id + n, device=data.query.device)
            data.instance_ids = ids.unsqueeze(0).expand(b, -1)
            self.runtime_tracker.gaussian_instance_id = start_id + n

        return queries

    def _compute_sampling_priorities(
        self,
        pyramid_features: list[torch.Tensor],
        past_density: torch.Tensor | None,
    ) -> dict[int, torch.Tensor]:
        """
        Compute sampling priorities for each pyramid level.

        Args:
            pyramid_features: List of voxel features at different resolutions
            past_density: Optional density from previous frame [b, 1, d, h, w]

        Returns:
            Dict mapping level index to priorities [b, d, h, w]
        """
        # Downsample past_density to each pyramid level if provided
        past_density_per_level: dict[int, torch.Tensor | None] = {}
        if past_density is not None:
            for level_idx, feats in enumerate(pyramid_features):
                _, _, d, h, w = feats.shape
                level_density = F.interpolate(
                    past_density,
                    size=(d, h, w),
                    mode="trilinear",
                    align_corners=False,
                )
                level_density = level_density.squeeze(1)  # [b, d, h, w]
                past_density_per_level[level_idx] = level_density

        # Compute priorities for each level using the priority computer
        priorities = {}
        for level_idx, feats in enumerate(pyramid_features):
            density = past_density_per_level.get(level_idx)
            priorities[level_idx] = self.sampling_priority_computer(feats, density)

        return priorities

    def _sample_from_level(
        self,
        voxel_feats: torch.Tensor,
        voxel_coords: torch.Tensor,
        num_queries: int,
        priorities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample queries from a single pyramid level based on priorities.

        Args:
            voxel_feats: Voxel features [b, c, d, h, w] at this pyramid level
            voxel_coords: Voxel coordinates [d, h, w, 3] in [0, 1]
            num_queries: Number of queries to sample
            priorities: Sampling priorities [b, d, h, w] - higher = more likely

        Returns:
            query_feats: Sampled features [b, num_queries, c]
            query_coords: Sampled coordinates [b, num_queries, 3] in [0, 1]
            topk_indexes: Indices of sampled voxels
        """
        if self.gaussian_query_nmspool > 1:
            # spatially pool features before sampling for NMS and better coverage
            priorities = priorities.unsqueeze(1)

            # pylint: disable-next=not-callable
            priorities, voxel_feats_idx = F.max_pool3d(
                priorities,
                kernel_size=self.gaussian_query_nmspool,
                stride=self.gaussian_query_nmspool,
                return_indices=True,
            )

            priorities = priorities.squeeze(1)
            voxel_feats_idx = voxel_feats_idx.squeeze(1)

        else:
            voxel_feats_idx = torch.arange(
                priorities[0, ...].numel(),
                device=priorities.device,
            )
            voxel_feats_idx = voxel_feats_idx.view(1, *priorities.shape[1:])
            voxel_feats_idx = voxel_feats_idx.expand(priorities.shape[0], -1, -1, -1)

        priorities = einops.rearrange(priorities, "b d h w -> b (d h w)")
        voxel_feats_idx = einops.rearrange(voxel_feats_idx, "b d h w -> b (d h w)")

        if self.gaussian_query_sampling == "topk":
            _, topk_indexes = torch.topk(
                priorities,
                num_queries,
                dim=-1,
                largest=True,
                sorted=False,
            )

        else:
            topk_indexes = torch.multinomial(
                priorities,
                num_queries,
                replacement=False,
            )

        # map to original indexes if pooled before
        topk_indexes = torch.gather(voxel_feats_idx, 1, topk_indexes)

        # get the sampled features
        b, c, _nz, _ny, _nx = voxel_feats.shape

        query_feats = voxel_feats.view(b, c, -1)
        query_feats = torch.gather(
            query_feats, 2, topk_indexes.unsqueeze(1).expand(-1, c, -1)
        )  # [b, c, num_queries]
        query_feats = query_feats.permute(0, 2, 1)  # [b, num_queries, c]

        # get coordinates of the sampled features
        query_coords = voxel_coords.view(1, -1, 3).expand(b, -1, -1)
        query_coords = torch.gather(
            query_coords, 1, topk_indexes.unsqueeze(-1).expand(-1, -1, 3)
        )  # [b, num_queries, 3] in [0, 1]

        return query_feats, query_coords, topk_indexes

    def sample_hierarchical_gaussian_queries_independent(
        self,
        pyramid_features: list[torch.Tensor],
        num_queries: dict[str, int],
        priorities_per_level: dict[int, torch.Tensor],
    ) -> dict[str, MetaDict]:
        """
        Sample gaussians independently from each pyramid level.

        This is Option A: Independent Sampling from the implementation plan.
        Each stream samples independently based on pre-computed priorities.

        Args:
            pyramid_features: List of voxel features at different resolutions.
                Expected format from voxel_backbone (finest to coarsest):
                    [0]: [b, c0, d0, h0, w0]  - finest
                    [1]: [b, c1, d1, h1, w1]  - medium
                    [2]: [b, c2, d2, h2, w2]  - coarsest
            num_queries: dict[str, int] specifying number of queries per stream
            priorities_per_level: Dict mapping level index to priorities [b, d, h, w]

        Returns:
            Dict mapping stream names to their query features and coordinates:
            {
                'coarse': {'query': [b, n_coarse, c], 'query_coords': [b, n_coarse, 3]},
                'medium': {'query': [b, n_medium, c], 'query_coords': [b, n_medium, 3]},
                'fine': {'query': [b, n_fine, c], 'query_coords': [b, n_fine, 3]},
            }
        """
        result = {}

        for stream, num_q in num_queries.items():
            # Get the pyramid level for this stream
            level_idx = self.gaussian_sample_source_level[stream]

            # Get features and coordinates at this pyramid level
            voxel_feats = pyramid_features[level_idx]
            voxel_coords = getattr(self, f"voxel_coords_level{level_idx}")

            # Sample queries using pre-computed priorities
            query_feats, query_coords, _ = self._sample_from_level(
                voxel_feats,
                voxel_coords,
                num_q,
                priorities=priorities_per_level[level_idx],
            )

            result[stream] = MetaDict(
                {
                    "query": query_feats,
                    "query_coords": query_coords,
                }
            )

        return result

    def sample_hierarchical_gaussian_queries_topdown(
        self,
        pyramid_features: list[torch.Tensor],
        num_queries: dict[str, int],
        priorities_per_level: dict[int, torch.Tensor],
    ) -> dict[str, MetaDict]:
        """
        Sample gaussians hierarchically using top-down approach.

        This is Option B: Hierarchical Top-Down Sampling from the implementation plan.
        - Sample coarse first for global coverage
        - Sample medium/fine near coarse/medium locations for local refinement

        Args:
            pyramid_features: List of voxel features at different resolutions.
            num_queries: dict[str, int] specifying number of queries per stream
            priorities_per_level: Dict mapping level index to priorities [b, d, h, w]

        Returns:
            Dict mapping stream names to their query features and coordinates.
        """
        # pylint: disable=too-many-locals

        result = {}
        b = pyramid_features[0].shape[0]

        # Use explicit stream order for top-down sampling
        stream_order = self.gaussian_sample_order

        # Sample first stream first (global coverage)
        stream = stream_order[0]
        num_queries_first = num_queries[stream]
        level_idx_first = self.gaussian_sample_source_level[stream]

        voxel_feats = pyramid_features[level_idx_first]
        voxel_coords = getattr(self, f"voxel_coords_level{level_idx_first}")
        query_feats, query_coords, _ = self._sample_from_level(
            voxel_feats,
            voxel_coords,
            num_queries_first,
            priorities=priorities_per_level[level_idx_first],
        )

        result[stream] = MetaDict(
            {
                "query": query_feats,
                "query_coords": query_coords,
            }
        )

        # For each subsequent level, sample near previous level
        prev_coords = query_coords  # [b, n_prev, 3] in [0, 1]

        for stream in stream_order[1:]:
            num_q = num_queries[stream]
            level_idx = self.gaussian_sample_source_level[stream]

            # Get radius for sampling around previous stream
            radius = self.gaussian_sample_radius[stream]

            # Get features at this level
            voxel_feats = pyramid_features[level_idx]
            b, c, _, _, _ = voxel_feats.shape

            # Get pre-computed priorities for this level
            level_priorities = priorities_per_level[level_idx]
            level_priorities_flat = einops.rearrange(
                level_priorities, "b d h w -> b (d h w)"
            )

            # Get all coordinates at this level
            voxel_coords = getattr(self, f"voxel_coords_level{level_idx}")
            voxel_coords = voxel_coords.view(1, -1, 3)
            voxel_coords = voxel_coords.expand(b, -1, -1)  # [b, n_voxels, 3]

            # Convert coordinates from [0, 1] to actual voxel range for distance computation
            vx_range = torch.as_tensor(self.voxel_range, device=prev_coords.device)
            prev_coords_real = (
                prev_coords * (vx_range[3:] - vx_range[:3]) + vx_range[:3]
            )
            level_coords_real = (
                voxel_coords * (vx_range[3:] - vx_range[:3]) + vx_range[:3]
            )

            # Find voxels within radius of any previous level query
            # [b, n_voxels, n_prev, 3]
            dists = torch.cdist(
                level_coords_real, prev_coords_real
            )  # [b, n_voxels, n_prev]
            min_dists = dists.min(dim=2).values  # [b, n_voxels]

            # Mask out voxels too far from any previous query
            within_radius = min_dists <= radius

            # Weight by priorities AND proximity
            proximity_weights = torch.exp(-min_dists / radius)  # Gaussian weighting
            sampling_weights = level_priorities_flat * proximity_weights
            sampling_weights = torch.where(
                within_radius, sampling_weights, torch.zeros_like(sampling_weights)
            )

            # Sample using weighted selection
            if self.gaussian_query_sampling == "topk":
                _, topk_indexes = torch.topk(
                    sampling_weights,
                    num_q,
                    dim=-1,
                    largest=True,
                    sorted=False,
                )
            else:
                # Add small epsilon to avoid zero probabilities
                sampling_weights = sampling_weights + 1e-6
                topk_indexes = torch.multinomial(
                    sampling_weights,
                    num_q,
                    replacement=False,
                )

            # Extract features and coordinates
            query_feats = voxel_feats.view(b, c, -1)
            query_feats = torch.gather(
                query_feats, 2, topk_indexes.unsqueeze(1).expand(-1, c, -1)
            )
            query_feats = query_feats.permute(0, 2, 1)  # [b, num_q, c]

            query_coords = torch.gather(
                voxel_coords, 1, topk_indexes.unsqueeze(-1).expand(-1, -1, 3)
            )

            result[stream] = MetaDict(
                {
                    "query": query_feats,
                    "query_coords": query_coords,
                }
            )

            # Update prev_coords for next level
            prev_coords = query_coords

        return result

    def sample_hierarchical_gaussian_queries(
        self,
        features: list[torch.Tensor],
        num_queries: dict[str, int],
        priorities_per_level: dict[int, torch.Tensor],
    ) -> dict[str, MetaDict]:
        """
        Sample gaussians from voxel pyramid for hierarchical processing.

        Dispatcher method that calls the appropriate sampling strategy based on config.

        Args:
            features: List of voxel features at different resolutions.
            num_queries: dict[str, int] specifying number of queries per stream
            priorities_per_level: Dict mapping level index to priorities [b, d, h, w]

        Returns:
            Dict mapping stream names to their query features and coordinates.
        """
        if self.gaussian_sample_mode == "independent":
            return self.sample_hierarchical_gaussian_queries_independent(
                features, num_queries, priorities_per_level
            )

        if self.gaussian_sample_mode == "top-down":
            return self.sample_hierarchical_gaussian_queries_topdown(
                features, num_queries, priorities_per_level
            )

        raise ValueError(f"Unknown gaussian_sample_mode: {self.gaussian_sample_mode}")

    def serialize_gaussian_queries(
        self,
        query_feats: torch.Tensor,
        query_coords: torch.Tensor,
        query_age: torch.Tensor,
        query_confidence: torch.Tensor,
        query_instance_ids: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Serialize gaussian queries using a space-filling curve.

        Args:
            query_feats: Query features [b, n, c]
            query_coords: Query coordinates [b, n, 3]
            query_age: Query ages [b, n]
            query_confidence: Query confidence scores [b, n]
            query_instance_ids: Query instance IDs [b, n]

        Returns:
            query_feats: Serialized query features [b, n, c]
            query_coords: Serialized query coordinates [b, n, 3]
            query_age: Serialized query ages [b, n]
            query_confidence: Serialized query confidence scores [b, n]
            query_instance_ids: Serialized query instance IDs [b, n]
            order: Permutation order [b, n] where order[i] = original index at
                   serialized position i. None if no serialization is applied.
        """
        if self.gaussian_serialization is None:
            return (
                query_feats,
                query_coords,
                query_age,
                query_confidence,
                query_instance_ids,
                None,
            )

        b, _, c = query_feats.shape

        # serialize/order keypoints using space-filling curve
        query_key = (query_coords * 2**self.gaussian_serialization.depth).long()
        query_key = query_key.flatten(end_dim=-2)
        query_key = self.gaussian_serialization.encode(query_key)
        query_key = query_key.view(b, -1)

        order = torch.argsort(query_key, dim=1)
        query_feats = torch.gather(query_feats, 1, order[..., None].expand(-1, -1, c))
        query_coords = torch.gather(query_coords, 1, order[..., None].expand(-1, -1, 3))
        query_age = torch.gather(query_age, 1, order)
        query_confidence = torch.gather(query_confidence, 1, order)
        query_instance_ids = torch.gather(query_instance_ids, 1, order)

        return (
            query_feats,
            query_coords,
            query_age,
            query_confidence,
            query_instance_ids,
            order,
        )

    def extract_volume_features(
        self,
        depth: torch.Tensor,
        depth_feats: torch.Tensor,
        image_feats: torch.Tensor | list[torch.Tensor],
        image_shape: torch.Size,
        image_padding: torch.Size,
        tx_project: torch.Tensor,
        tx_unproject: torch.Tensor,
        prev_temporal_state: TemporalState | None = None,
        ego_to_global_prev: torch.Tensor | None = None,  # [4, 4]
        ego_to_global_curr: torch.Tensor | None = None,  # [4, 4]
        sample: MetaDict | None = None,
        prev_labels: MetaDict | None = None,
    ) -> torch.Tensor:
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-branches

        # lift features to 3D
        voxel_feats = self.lift_and_pool(
            depth=depth,
            feats=depth_feats,
            img_shape=image_shape,
            unproject_tx=tx_unproject,
        )
        _b, _c, nz, ny, nx = voxel_feats.shape

        # voxel_feats: [1, 128, 16, 200, 200]

        # basic encoder
        voxel_feats = self.voxel_backbone(voxel_feats)

        # NOTE: we have pyramid features here
        #  [1, 32, 16, 200, 200]  <- finest (level 0)
        #  [1, 64,  8, 100, 100]  <- medium (level 1)
        #  [1, 96,  4,  50,  50]  <- coarsest (level 2)

        # Extract gaussians and density from temporal state
        prev_gaussians = (
            prev_temporal_state.gaussians if prev_temporal_state is not None else None
        )
        prev_density = (
            prev_temporal_state.density_map if prev_temporal_state is not None else None
        )

        # Pre-decode visibility and confidence (for transformer update gating)
        temporal_gate_visibility = {}
        temporal_gate_in_fov = {}
        temporal_gate_confidence = {}

        # Post-decode visibility and confidence (for temporal consistency loss)
        post_decode_visibility = {}
        post_decode_confidence = {}

        # Temporal gaussian aggregation: update previous gaussians if provided
        if self.use_gaussian_temporal and prev_gaussians is not None:
            # Increment age of previous gaussians
            for stream, data in prev_gaussians.streams.items():
                data.age = data.age + 1

            with torch.autocast("cuda", enabled=False):
                ego_to_global_curr = ego_to_global_curr.to(dtype=torch.float64)
                ego_to_global_prev = ego_to_global_prev.to(dtype=torch.float64)
                tx = torch.inverse(ego_to_global_curr) @ ego_to_global_prev

            # Apply ego-motion compensation
            prev_gaussians = self.ego_motion_compensation(
                gaussians=prev_gaussians,
                transform=tx.to(dtype=torch.float32),
            )

            # Apply pruning
            for op in self.gaussian_pruning:
                prev_gaussians = op(prev_gaussians)

            # Compute visibility for temporal gaussians (after EMC and pruning)
            with torch.no_grad():
                pre_vis_context = self.temporal_gate_visibility.prepare(
                    depth_dist=depth.detach(),
                    tx_project=tx_project,
                    img_shape=image_shape,
                )

            for stream, data in prev_gaussians.streams.items():
                with torch.no_grad():
                    vis, in_fov = self.temporal_gate_visibility(
                        centers=data.centers.detach(),
                        scales=data.scales.detach(),
                        rotations=data.rotations.detach(),
                        tx_project=tx_project,
                        img_shape=image_shape,
                        **pre_vis_context,
                    )
                temporal_gate_visibility[stream] = vis
                temporal_gate_in_fov[stream] = in_fov

                # Store previous frame's confidence for gating
                # (actual confidence update happens post-decode)
                temporal_gate_confidence[stream] = data.confidence

            # Transform density for density-aware sampling
            if self.density_transform is not None and prev_density is not None:
                prev_density = self.density_transform(
                    prev_density,
                    tx.to(dtype=torch.float32),
                )
        elif self.use_gaussian_temporal and prev_gaussians is None:
            # HACK: Reference all parameters in gaussian temporal modules to avoid
            #       autograd errors when no previous gaussians are provided
            params_emc = self.ego_motion_compensation.parameters()
            params_prune = self.gaussian_pruning.parameters()

            params_dt = []
            if self.density_transform is not None:
                params_dt = self.density_transform.parameters()

            params = list(params_emc) + list(params_prune) + list(params_dt)
            params = [p.mean() for p in params]
            voxel_feats[0] = voxel_feats[0] + sum(params) * 0.0

        # Track number of temporal gaussians per stream (needed for gating and losses)
        n_temporal_per_stream = {}
        if self.use_gaussian_temporal and prev_gaussians is not None:
            n_temporal_per_stream = {
                stream: data.centers.shape[1]
                for stream, data in prev_gaussians.streams.items()
            }

        # Store EMC-transformed targets for temporal consistency loss
        # These are detached since we want to match the EMC output, not change it
        temporal_targets = None
        matched_instance_ids = None
        if (
            self.use_gaussian_temporal
            and self.temporal_consistency_loss is not None
            and prev_gaussians is not None
        ):
            temporal_targets = {
                stream: MetaDict(
                    {
                        "centers": data.centers.clone().detach(),
                        "scales": data.scales.clone().detach(),
                        "rotations": data.rotations.clone().detach(),
                        "opacities": data.opacities.clone().detach(),
                        "logits": data.logits.clone().detach(),
                    }
                )
                for stream, data in prev_gaussians.streams.items()
            }

            # Compute dynamic object targets if previous frame labels available
            # This replaces EMC targets with object-motion targets for matched dynamics
            if self.training and self.dynamic_supervision_enabled:
                # if we have prev_gaussians we should also have prev_labels
                assert prev_labels is not None

                dynamic_targets, matched_instance_ids = compute_dynamic_targets(
                    temporal_gaussians=prev_gaussians.streams,
                    prev_labels=prev_labels,
                    curr_labels=sample.labels,
                    dynamic_class_ids=self.temporal_consistency_loss.dynamic_class_indices,
                    ego_transform=tx.to(dtype=torch.float32),
                    max_match_distance=self.dynamic_max_match_distance,
                )

                # Merge: use object-motion targets for matched dynamics
                for stream in temporal_targets:
                    if stream not in matched_instance_ids:
                        continue

                    mask = matched_instance_ids[stream] >= 0  # [b, n]
                    for key in ["centers", "rotations"]:
                        temporal_targets[stream][key] = torch.where(
                            mask[..., None],
                            dynamic_targets[stream][key],
                            temporal_targets[stream][key],
                        )

        # Compute the number of new queries to sample
        num_queries_to_sample = self.num_gaussian_queries.copy()
        if (
            self.use_gaussian_temporal
            and prev_gaussians is not None
            and self.gaussian_temporal_sampling == "adaptive"
        ):
            for stream, data in prev_gaussians.streams.items():
                n_prev = data.query.shape[1]
                n_total = self.num_gaussian_queries[stream]

                num_queries_to_sample[stream] = max(n_total - n_prev, 0)

        # Sample new initial queries with density-aware priorities
        queries = self.sample_gaussian_queries(
            voxel_feats, num_queries_to_sample, past_density=prev_density
        )

        # Project queries from pyramid channels to embed_dim
        for stream, data in queries.items():
            # Project: [b, n, pyramid_channels] -> [b, n, embed_dim]
            data.query = self.gaussian_pyramid_projs[stream](data.query)

        # Merge previous gaussians and newly sampled queries
        if self.use_gaussian_temporal and prev_gaussians is not None:
            for stream, data in queries.items():
                if stream not in prev_gaussians.streams:
                    continue

                prev = prev_gaussians.streams[stream]

                # Concatenate [temporal, fresh] along query dimension (dim=1)
                # Temporal queries: age > 0, already transformed and pruned
                # Fresh queries: age = 0, just sampled
                data.query = torch.cat([prev.query, data.query], dim=1)
                data.query_coords = torch.cat(
                    [prev.query_coords, data.query_coords], dim=1
                )
                data.age = torch.cat([prev.age, data.age], dim=1)
                data.confidence = torch.cat([prev.confidence, data.confidence], dim=1)
                data.instance_ids = torch.cat(
                    [prev.instance_ids, data.instance_ids], dim=1
                )

        # Apply age embeddings to provide temporal signal to transformer
        if self.use_gaussian_temporal and self.gaussian_age_embedding is not None:
            for stream, data in queries.items():
                age_embed = self.gaussian_age_embedding(data.age)  # [b, n, embed_dim]
                data.query = data.query + age_embed

        # serialize per-stream if configured
        # Also track serialization order for temporal consistency loss alignment
        serialization_order = {}
        if self.gaussian_serialization is not None:
            for stream, inputs in queries.items():
                (
                    inputs.query,
                    inputs.query_coords,
                    inputs.age,
                    inputs.confidence,
                    inputs.instance_ids,
                    order,
                ) = self.serialize_gaussian_queries(
                    query_feats=inputs.query,
                    query_coords=inputs.query_coords,
                    query_age=inputs.age,
                    query_confidence=inputs.confidence,
                    query_instance_ids=inputs.instance_ids,
                )
                serialization_order[stream] = order

        # Store pre-transformer queries for visibility gating
        pre_transformer_queries = {}
        if self.temporal_gate is not None:
            for stream, data in queries.items():
                pre_transformer_queries[stream] = data.query.clone()

        # refine queries with transformer
        if self.voxel_transformer is not None:
            if self.use_hierarchical_gaussians:
                outputs = self.voxel_transformer(
                    image_feats=image_feats,
                    queries=queries,
                    tx_project=tx_project,
                    tx_unproject=tx_unproject,
                    image_shape=image_shape,
                    image_padding=image_padding,
                )

                # outputs is dict of dicts: {stream: {'query': [...], 'query_coords': [...]}}
                # Update queries with refined outputs
                for stream, output in outputs.items():
                    queries[stream].query = output["query"]
                    queries[stream].query_coords = output["query_coords"]

            else:
                stream = queries["default"]

                stream.query, stream.query_coords = self.voxel_transformer(
                    image_feats=image_feats,
                    keypoint_feats=stream.query,
                    keypoint_coords=stream.query_coords,
                    tx_project=tx_project,
                    tx_unproject=tx_unproject,
                    image_shape=image_shape,
                    image_padding=image_padding,
                )

        else:
            # No transformer - queries dict already populated from sampling
            # Add layer dimension for consistency with transformer output
            for _stream, data in queries.items():
                data.query = data.query.unsqueeze(1)  # [b, 1, n, c]
                data.query_coords = data.query_coords.unsqueeze(1)

        # Apply visibility-based gating to temporal queries
        # This preserves temporal queries when visibility is low (occluded/out of view)
        gated = None
        if self.temporal_gate is not None and pre_transformer_queries:
            for stream, data in queries.items():
                n_temporal = n_temporal_per_stream.get(stream, 0)
                if n_temporal == 0 or stream not in pre_transformer_queries:
                    continue

                # Pre-transformer needs layer dimension to match post-transformer
                pre_tx = pre_transformer_queries[stream]
                if pre_tx.dim() == 3:
                    pre_tx = pre_tx.unsqueeze(1).expand_as(data.query)

                _b, num_layers, n_total, c = data.query.shape

                # Find where temporal queries ended up after serialization
                # order[i] = original index at serialized position i
                # inverse_order[j] = serialized position of original index j
                order = serialization_order[stream]  # [b, n_total]
                inverse_order = torch.argsort(order, dim=1)
                temporal_positions = inverse_order[:, :n_temporal]  # [b, n_temporal]

                # Gather pre/post at temporal positions
                gather_idx = temporal_positions[:, None, :, None].expand(
                    -1, num_layers, -1, c
                )
                pre_temporal = pre_tx.gather(2, gather_idx)
                post_temporal = data.query.gather(2, gather_idx)

                # Apply gating
                # Note: confidence scores are from the previous frame
                gated, _gate_vals = self.temporal_gate(
                    pre_transformer=pre_temporal,
                    post_transformer=post_temporal,
                    visibility=temporal_gate_visibility[stream],
                    in_fov=temporal_gate_in_fov[stream],
                    confidence=temporal_gate_confidence[stream],
                )

                # Scatter gated values back to their positions
                data.query = data.query.scatter(2, gather_idx, gated)

        if self.temporal_gate is not None and gated is None:
            # HACK: Reference all parameters in visibility_gate modules to avoid
            #       autograd errors when no previous gaussians are provided
            params = self.temporal_gate.parameters()
            params = [p.mean() for p in params]
            voxel_feats[0] = voxel_feats[0] + sum(params) * 0.0
        del gated

        # Decode gaussians (only for streams in decode_gaussian_streams)
        gaussians = {}
        gaussians_agg = {}
        query_feats = {}
        query_coords = {}

        for stream, data in queries.items():
            # Drop old queries if we only use them for past context only
            if self.gaussian_temporal_mode == "context":
                keep = data.age.squeeze(0) == 0

                data.query = data.query[:, :, keep, :]
                data.query_coords = data.query_coords[:, :, keep, :]
                data.age = data.age[:, keep]
                data.instance_ids = data.instance_ids[:, keep]

            # Only keep the last layer of features
            stream_query_feats = data.query[:, -1, :, :]  # [b, n_queries, c]
            stream_query_coords = data.query_coords[:, -1, :, :]

            query_feats[stream] = stream_query_feats
            query_coords[stream] = stream_query_coords

            # Skip decoding for streams not in decode_gaussian_streams
            if stream not in self.decode_gaussian_streams:
                continue

            # Decode this stream with per-stream decoder and scale range
            stream_gaussian = self.decode_gaussian_queries(
                query=data.query,  # [b, num_layers, n_queries, c]
                query_coords=data.query_coords,  # [b, num_layers, n_queries, 3]
                stream_name=stream,
            )
            stream_gaussian.query = data.query
            stream_gaussian.query_coords = data.query_coords
            stream_gaussian.age = data.age
            stream_gaussian.confidence = data.confidence
            stream_gaussian.instance_ids = data.instance_ids

            # Further process keypoint features for aggregation
            stream_query_feats = self.emb_decoder_gauss[stream](stream_query_feats)

            # Determine opacity for 3D feature aggregation based on mode
            if self.aggregation_opacity_mode == "fixed":
                # Fixed opacity=1.0: every gaussian contributes equally
                # Decouples 3D aggregation from depth-supervised opacity
                agg_opacities = torch.ones_like(stream_gaussian.opacities[:, -1, :])
            elif self.aggregation_opacity_mode == "confidence":
                # Use max(opacity, confidence): well-observed gaussians stay active
                # - Fresh gaussians (confidence=0) use their learned opacity
                # - Temporal gaussians use whichever is higher, so high-confidence
                #   gaussians contribute even if current opacity is low
                opacity = stream_gaussian.opacities[:, -1, :]
                confidence = data.confidence
                agg_opacities = torch.maximum(opacity, confidence)
            elif self.aggregation_opacity_mode == "local":
                # Separate learned opacity decoder
                agg_opacities = torch.sigmoid(
                    self.local_opacity_decoder[stream](data.query[:, -1]).squeeze(-1)
                )
            else:
                # Default: use decoded opacity (coupled with depth loss)
                agg_opacities = stream_gaussian.opacities[:, -1, :]

            # Apply opacity threshold: zero out gaussians below threshold
            if self.aggregation_opacity_threshold > 0.0:
                agg_opacities = agg_opacities * (
                    agg_opacities >= self.aggregation_opacity_threshold
                )

            stream_gaussian_agg = {
                "features": stream_query_feats,
                "centers": stream_gaussian.centers[:, -1, :, :],
                "scales": stream_gaussian.scales[:, -1, :, :],
                "rotations": stream_gaussian.rotations[:, -1, :, :],
                "opacities": agg_opacities,
            }

            # Store results
            gaussians[stream] = stream_gaussian
            gaussians_agg[stream] = MetaDict(stream_gaussian_agg)

            if self.tracking_head_projected_keypoints:
                query_feats[stream] = stream_query_feats
                query_coords[stream] = stream_query_coords

        # Post-decode confidence update for ALL gaussians (temporal + fresh)
        if self.confidence_tracker is not None:
            # Prepare visibility context once using all decoded gaussians
            with torch.no_grad():
                conf_vis_context = self.confidence_visibility.prepare(
                    depth_dist=depth.detach(),
                    gaussians=gaussians,
                    tx_project=tx_project,
                    img_shape=image_shape,
                )

            # Update confidence for each stream
            for stream, stream_gaussian in gaussians.items():
                decoded_centers = stream_gaussian.centers[:, -1, :, :]
                decoded_scales = stream_gaussian.scales[:, -1, :, :]
                decoded_rotations = stream_gaussian.rotations[:, -1, :, :]
                decoded_opacities = stream_gaussian.opacities[:, -1, :]

                # Compute visibility using decoded properties
                with torch.no_grad():
                    vis, in_fov = self.confidence_visibility(
                        centers=decoded_centers.detach(),
                        scales=decoded_scales.detach(),
                        rotations=decoded_rotations.detach(),
                        tx_project=tx_project,
                        img_shape=image_shape,
                        **conf_vis_context,
                    )

                # Update confidence using configured tracker
                stream_gaussian.confidence = self.confidence_tracker(
                    confidence=stream_gaussian.confidence,
                    centers=decoded_centers,
                    visibility=vis,
                    in_fov=in_fov,
                    opacity=decoded_opacities,
                    query=stream_gaussian.query[:, -1, :, :],
                    logits=stream_gaussian.logits[:, -1, :, :],
                )

                # Store post-decode visibility and confidence for temporal consistency loss
                post_decode_visibility[stream] = vis
                post_decode_confidence[stream] = stream_gaussian.confidence

        # Aggregate gaussians
        if (
            self.use_hierarchical_gaussians
            and self.gaussian_aggregation_mode == "hierarchical"
        ):
            # Per-stream aggregation with multi-scale fusion
            gaussian_occ, bin_scores, _, prob_sum = self.gaussian_aggregator(
                gaussians_per_stream=gaussians_agg,
                mask=None,
            )

            # Convert to spatial grid format
            gaussian_occ = einops.rearrange(
                gaussian_occ, "b (d h w) c -> b c d h w", d=nz, h=ny, w=nx
            )
            bin_scores = einops.rearrange(
                bin_scores, "b (d h w) -> b 1 d h w", d=nz, h=ny, w=nx
            )
            prob_sum = einops.rearrange(
                prob_sum, "b (d h w) -> b 1 d h w", d=nz, h=ny, w=nx
            )

        elif (
            self.use_hierarchical_gaussians
            and self.gaussian_aggregation_mode == "merged"
        ):
            # Merged aggregation: concatenate all streams, then aggregate once
            # Concatenate gaussians from all streams
            stream_names = list(gaussians_agg.keys())
            merged_features = torch.cat(
                [gaussians_agg[s].features for s in stream_names], dim=1
            )
            merged_centers = torch.cat(
                [gaussians_agg[s].centers for s in stream_names], dim=1
            )
            merged_scales = torch.cat(
                [gaussians_agg[s].scales for s in stream_names], dim=1
            )
            merged_rotations = torch.cat(
                [gaussians_agg[s].rotations for s in stream_names], dim=1
            )
            merged_opacities = torch.cat(
                [gaussians_agg[s].opacities for s in stream_names], dim=1
            )

            # Single aggregation of merged gaussians
            gaussian_occ, bin_scores, _, prob_sum = self.gaussian_aggregator(
                features=merged_features,
                centers=merged_centers,
                scales=merged_scales,
                rotations=merged_rotations,
                opacities=merged_opacities,
            )

            # Convert to spatial grid format
            gaussian_occ = einops.rearrange(
                gaussian_occ, "b (d h w) c -> b c d h w", d=nz, h=ny, w=nx
            )
            bin_scores = einops.rearrange(
                bin_scores, "b (d h w) -> b 1 d h w", d=nz, h=ny, w=nx
            )
            prob_sum = einops.rearrange(
                prob_sum, "b (d h w) -> b 1 d h w", d=nz, h=ny, w=nx
            )

        else:
            # Single-stream (non-hierarchical) aggregation
            gaussian_occ = gaussians_agg["default"]
            gaussian_occ, bin_scores, _, prob_sum = self.gaussian_aggregator(
                features=gaussian_occ.features,
                centers=gaussian_occ.centers,
                scales=gaussian_occ.scales,
                rotations=gaussian_occ.rotations,
                opacities=gaussian_occ.opacities,
            )
            gaussian_occ = einops.rearrange(
                gaussian_occ, "b (d h w) c -> b c d h w", d=nz, h=ny, w=nx
            )
            bin_scores = einops.rearrange(
                bin_scores, "b (d h w) -> b 1 d h w", d=nz, h=ny, w=nx
            )
            prob_sum = einops.rearrange(
                prob_sum, "b (d h w) -> b 1 d h w", d=nz, h=ny, w=nx
            )

        # upsample and aggregate to the output level
        fullres_occ = self.voxel_aggregate(voxel_feats)

        # fuse gaussian features and volume features
        fullres_occ = self.voxel_fuse(torch.cat((fullres_occ, gaussian_occ), dim=1))

        # Store current temporal state for next frame (gaussians + density)
        current_temporal_state = None
        if self.use_gaussian_temporal:
            current_gaussians = GaussianInstances(streams={})

            for stream, data in gaussians.items():
                # Optionally reconcile query_coords with decoded centers
                # This ensures EMC feature adaptation uses actual gaussian displacement
                # and transformer positional encoding reflects actual gaussian position
                if self.gaussian_temporal_reconcile_coords:
                    stream_query_coords = self._normalize_centers_to_coords(
                        data.centers[:, -1, :, :]
                    )
                else:
                    stream_query_coords = data.query_coords[:, -1, :, :]

                inst = GaussianStreamInstances(
                    query=data.query[:, -1, :, :],
                    query_coords=stream_query_coords,
                    logits=data.logits[:, -1, :, :],
                    centers=data.centers[:, -1, :, :],
                    scales=data.scales[:, -1, :, :],
                    rotations=data.rotations[:, -1, :, :],
                    opacities=data.opacities[:, -1, :],
                    age=data.age,
                    confidence=data.confidence,
                    instance_ids=data.instance_ids,
                )

                current_gaussians.streams[stream] = inst

            # Append previous gaussians as context
            if self.gaussian_temporal_mode == "context" and prev_gaussians is not None:
                current_gaussians = GaussianInstances.cat(
                    (prev_gaussians, current_gaussians), dim=1
                )

            # Bundle gaussians with opacity-weighted density for next frame
            current_temporal_state = TemporalState(
                gaussians=current_gaussians,
                density_map=(
                    prob_sum.detach() if self.density_transform is not None else None
                ),
            )

        # Compute temporal consistency loss if enabled
        gaussian_losses = {}
        if (
            self.temporal_consistency_loss is not None
            and temporal_targets is not None
            and self.gaussian_temporal_mode == "track"  # Only for track mode
        ):
            for stream, data in gaussians.items():
                if stream not in temporal_targets:
                    continue

                n_temporal = n_temporal_per_stream.get(stream, 0)
                if n_temporal == 0:
                    continue

                targets = temporal_targets[stream]

                # Get predictions aligned with targets (in original pre-serialization order)
                # Before serialization: temporal queries were at indices [0, n_temporal)
                # After serialization: they moved to positions given by inverse_order
                order = serialization_order[stream]  # [b, n_total]
                inverse_order = torch.argsort(order, dim=1)
                temporal_positions = inverse_order[:, :n_temporal]  # [b, n_temporal]

                # Gather predictions at positions corresponding to temporal targets
                num_classes = data.logits.shape[-1]
                preds = MetaDict(
                    {
                        "centers": data.centers[:, -1].gather(
                            1, temporal_positions[..., None].expand(-1, -1, 3)
                        ),
                        "scales": data.scales[:, -1].gather(
                            1, temporal_positions[..., None].expand(-1, -1, 3)
                        ),
                        "rotations": data.rotations[:, -1].gather(
                            1, temporal_positions[..., None].expand(-1, -1, 4)
                        ),
                        "opacities": data.opacities[:, -1].gather(
                            1, temporal_positions
                        ),
                        "logits": data.logits[:, -1].gather(
                            1,
                            temporal_positions[..., None].expand(-1, -1, num_classes),
                        ),
                    }
                )

                # Gather visibility and confidence at temporal positions
                stream_vis = post_decode_visibility.get(stream)
                if stream_vis is not None:
                    stream_vis = stream_vis.gather(1, temporal_positions)

                stream_conf = post_decode_confidence.get(stream)
                if stream_conf is not None:
                    stream_conf = stream_conf.gather(1, temporal_positions)

                # Get matched instance IDs for dynamic supervision
                stream_matched_ids = None
                if matched_instance_ids is not None and stream in matched_instance_ids:
                    stream_matched_ids = matched_instance_ids[stream]

                stream_loss = self.temporal_consistency_loss(
                    preds,
                    targets,
                    stream_vis,
                    stream_conf,
                    matched_instance_ids=stream_matched_ids,
                )
                for k, v in stream_loss.items():
                    gaussian_losses[f"{k}_{stream}"] = v

        # Compute fresh gaussian visibility losses if enabled
        gaussian_fov_losses = {}
        if self.loss_gaussian_fov is not None:
            # Prepare visibility context once for all streams
            with torch.no_grad():
                vis_context = self.loss_gaussian_fov_vis.prepare(
                    depth_dist=depth.detach(),
                    gaussians=gaussians,
                    tx_project=tx_project,
                    img_shape=image_shape,
                )

            for stream, data in gaussians.items():
                # In context mode, temporal queries were dropped - all remaining are fresh
                # In track mode, need to extract fresh queries using serialization order
                if (
                    not self.use_gaussian_temporal
                    or self.gaussian_temporal_mode == "context"
                ):
                    # All queries are fresh (temporal were dropped before decoding)
                    fresh_centers = data.centers[:, -1]
                    fresh_scales = data.scales[:, -1]
                    fresh_rotations = data.rotations[:, -1]
                    fresh_opacities = data.opacities[:, -1]
                else:
                    # Track mode: extract fresh queries at their serialized positions
                    n_temporal = n_temporal_per_stream.get(stream, 0)
                    n_total = data.centers.shape[2]
                    n_fresh = n_total - n_temporal

                    if n_fresh == 0:
                        continue

                    # Find where fresh queries ended up after serialization
                    # Fresh queries were originally at positions [n_temporal, n_total)
                    order = serialization_order[stream]  # [b, n_total]
                    inverse_order = torch.argsort(order, dim=1)
                    fresh_positions = inverse_order[:, n_temporal:]  # [b, n_fresh]

                    # Gather fresh gaussian properties at their serialized positions
                    fresh_centers = data.centers[:, -1].gather(
                        1, fresh_positions[..., None].expand(-1, -1, 3)
                    )
                    fresh_scales = data.scales[:, -1].gather(
                        1, fresh_positions[..., None].expand(-1, -1, 3)
                    )
                    fresh_rotations = data.rotations[:, -1].gather(
                        1, fresh_positions[..., None].expand(-1, -1, 4)
                    )
                    fresh_opacities = data.opacities[:, -1].gather(1, fresh_positions)

                # Compute visibility for fresh gaussians
                with torch.no_grad():
                    fresh_vis, fresh_in_fov = self.loss_gaussian_fov_vis(
                        centers=fresh_centers.detach(),
                        scales=fresh_scales.detach(),
                        rotations=fresh_rotations.detach(),
                        tx_project=tx_project,
                        img_shape=image_shape,
                        **vis_context,
                    )

                # Compute losses
                stream_losses = self.loss_gaussian_fov(
                    opacities=fresh_opacities,
                    visibility=fresh_vis,
                    in_fov=fresh_in_fov,
                )

                for k, v in stream_losses.items():
                    gaussian_fov_losses[f"fresh_{k}_{stream}"] = v

        # Merge fresh gaussian losses into temporal consistency losses
        gaussian_losses.update(gaussian_fov_losses)

        return (
            fullres_occ,  # [b, c, d, h, w] - post-fusion features
            query_feats,  # dict stream -> [b, n, c]
            query_coords,  # dict stream -> [b, n, 3] in [0, 1]
            gaussians,  # dict stream -> MetaDict of gaussian predictions
            gaussian_occ,  # [b, c, d, h, w] - pre-fusion gaussian features
            bin_scores,  # [b, 1, d, h, w] - binary occupancy scores
            current_temporal_state,  # TemporalState | None - for next frame
            gaussian_losses,  # dict of loss tensors
        )

    def decode_gaussian_queries(
        self,
        query: torch.Tensor,
        query_coords: torch.Tensor,
        stream_name: str = "default",
    ) -> MetaDict:
        # pylint: disable=too-many-locals

        if not self.supervise_intermediate_gaussians and query.shape[1] > 1:
            query = query[:, -2:-1, :, :]
            query_coords = query_coords[:, -2:-1, :, :]

        # Get decoder and scale range for this stream
        reg_decoder = self.reg_decoder_gauss[stream_name]
        scale_range = self.gaussian_scale_range[stream_name]

        # decode class scores [b, num_layers, n_queries, num_classes]
        # Only decode if cls_decoder exists (needed for semantic_gauss loss)
        if self.cls_decoder_gauss is not None:
            cls_decoder = self.cls_decoder_gauss[stream_name]
            class_scores = cls_decoder(query)
        else:
            class_scores = None

        # decode gaussian properties
        gauss_preds = reg_decoder(query)  # [b, num_layers, n_queries, 11]

        # decode centers
        if self.gaussian_center_decode:
            # decode center delta and add reference positions
            k = 3

            reference = inverse_sigmoid(query_coords.clone())

            centers = gauss_preds[..., 0:3]  # [b, num_layers, n_queries, 3]
            centers = torch.sigmoid(centers + reference)

        else:
            # take the reference positions as centers
            k = 0

            centers = query_coords.clone()

        # bring coordinates from [0, 1] back to full range
        vx_range = torch.as_tensor(self.voxel_range, device=centers.device)
        centers = centers * (vx_range[3:] - vx_range[:3]) + vx_range[:3]

        # decode scales
        scale_min, scale_max = scale_range
        scales = gauss_preds[..., k : k + 3]  # [b, num_layers, n_queries, 3]
        scales = torch.sigmoid(scales)
        scales = scales * (scale_max - scale_min) + scale_min

        # decode rotations
        rotations = gauss_preds[..., k + 3 : k + 7]  # [b, num_layers, n_queries, 4]
        rotations = F.normalize(rotations, dim=-1)

        # decode opacities
        opacities = gauss_preds[..., k + 7]  # [b, num_layers, n_queries, 1]
        opacities = torch.sigmoid(opacities)

        # collect
        gaussians = MetaDict()
        if class_scores is not None:
            gaussians.logits = class_scores
        gaussians.centers = centers
        gaussians.scales = scales
        gaussians.rotations = rotations
        gaussians.opacities = opacities

        return gaussians

    @staticmethod
    def _gather_voxels(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        Select along the (flattened) trailing voxel axis with a shared 1D index.

        Args:
            tensor: ``[..., z, y, x]`` (the last 3 dims are the voxel grid).
            index: ``[k]`` long indices into the flattened grid.

        Returns:
            ``[..., k]`` with the trailing grid dims replaced by the selection.
        """
        flat = tensor.reshape(*tensor.shape[:-3], -1)  # [..., V]
        return flat.index_select(-1, index)

    def _subsample_match_voxels(self, preds: MetaDict, targets: MetaDict):
        # pylint: disable=too-many-locals,too-many-statements
        """
        Class-balanced voxel subsampling for instance occupancy matching/loss.

        Caps the number of voxels fed into the mask cost/loss to
        ``fraction * grid`` by keeping (a budget share of) the occupied
        (positive) voxels and uniformly sampling the dominant free/background
        voxels, attaching per-voxel inverse-probability ``voxel_weights`` so the
        downstream (re-weighted) mask loss stays unbiased. Returns the inputs
        unchanged when disabled, the mask is missing, or the sample is already
        below the cap. Operates on copies; ``labels.occupancy`` is never mutated.
        """
        frac = self.match_voxel_fraction
        if frac is None or frac >= 1.0:
            return preds, targets

        mask_name = self.conf.occupancy_mask
        mask_name = mask_name if mask_name != "none" else None
        occ = targets.occupancy
        if (
            mask_name is None
            or "masks" not in occ
            or mask_name not in occ.masks
            or "instance_masks" not in occ
            or "instance_scores" not in preds.occupancy
        ):
            return preds, targets

        masks = occ.masks
        inst = occ.instance_masks  # PackedTensor or dense [b, T, z, y, x]
        scores = preds.occupancy.instance_scores  # dense [b, ..., z, y, x]

        m = masks[mask_name]  # dense [b, z, y, x] bool
        b = m.shape[0]
        v_total = m[0].numel()
        cap = round(frac * v_total)
        if cap <= 0 or cap >= v_total:
            return preds, targets

        m_flat = m.reshape(b, v_total)
        n_valid = m_flat.sum(dim=1)

        # Conservative for b > 1: only subsample when every element exceeds the
        # cap, so the kept count is uniformly ``cap`` and tensors stay rectangular.
        if bool((n_valid <= cap).any()):
            return preds, targets

        def instance_masks_for(bi: int) -> torch.Tensor:
            return inst.get(bi) if isinstance(inst, PackedTensor) else inst[bi]

        device = m.device
        pos_share = self.match_voxel_pos_share

        idx_per_sample: list[torch.Tensor] = []
        weights = torch.empty((b, cap), dtype=torch.float32, device=device)
        for bi in range(b):
            # positive = occupied by any instance, within the valid mask
            pos_map = instance_masks_for(bi).any(dim=0).reshape(v_total) & m_flat[bi]
            valid_idx = m_flat[bi].nonzero(as_tuple=True)[0]
            is_pos = pos_map[valid_idx]
            pos_idx = valid_idx[is_pos]
            neg_idx = valid_idx[~is_pos]
            n_pos, n_neg = pos_idx.numel(), neg_idx.numel()

            keep_pos = min(n_pos, int(round(pos_share * cap)))
            keep_neg = cap - keep_pos
            if keep_neg > n_neg:  # not enough negatives -> spend budget on positives
                keep_pos += keep_neg - n_neg
                keep_neg = n_neg

            sel_pos = pos_idx[torch.randperm(n_pos, device=device)[:keep_pos]]
            sel_neg = neg_idx[torch.randperm(n_neg, device=device)[:keep_neg]]
            idx_per_sample.append(torch.cat([sel_pos, sel_neg]))
            # inverse keep-probability per group (Horvitz-Thompson weights)
            weights[bi, :keep_pos] = n_pos / max(keep_pos, 1)
            weights[bi, keep_pos:] = n_neg / max(keep_neg, 1)

        preds = preds.copy()
        preds.occupancy = preds.occupancy.copy()
        preds.occupancy.instance_scores = torch.stack(
            [self._gather_voxels(scores[bi], idx_per_sample[bi]) for bi in range(b)],
            dim=0,
        )

        targets = targets.copy()
        targets.occupancy = targets.occupancy.copy()

        new_masks = masks.copy()
        for k in new_masks:
            mk = new_masks[k]
            new_masks[k] = torch.stack(
                [self._gather_voxels(mk[bi], idx_per_sample[bi]) for bi in range(b)],
                dim=0,
            )
        targets.occupancy.masks = new_masks

        inst_data = torch.cat(
            [
                self._gather_voxels(instance_masks_for(bi), idx_per_sample[bi])
                for bi in range(b)
            ],
            dim=0,
        )
        if isinstance(inst, PackedTensor):
            targets.occupancy.instance_masks = PackedTensor(inst_data, inst.offsets)
        else:
            targets.occupancy.instance_masks = inst_data.reshape(b, -1, cap)

        targets.occupancy.voxel_weights = weights

        return preds, targets

    def match_instances(self, out, labels):
        preds = {
            "boxes": out["all_bbox_preds"],
            "class_scores": out["all_cls_scores"],
            "instance_ids": out["track_instances"].obj_idxes.unsqueeze(dim=0),
            "occupancy": MetaDict(
                {
                    "instance_scores": out["all_mask_preds"],
                }
            ),
        }
        preds = MetaDict(preds)

        targets = {
            "boxes": labels.boxes,
            "class_ids": labels.class_ids,
            "instance_ids": labels.instance_ids,
            "occupancy": labels.occupancy,
        }
        targets = MetaDict(targets)

        # cap voxels fed into the mask cost + loss (no-op unless configured)
        preds, targets = self._subsample_match_voxels(preds, targets)

        assignments = self.matcher(preds, targets)

        return preds, targets, assignments

    def match_semantic_masks(self, out, labels):
        # note: we treat everything as instance

        # map occupancy classes to the semantic class subset used for training
        targets_semantic_cls = labels.occupancy.semantic_ids
        targets_semantic_cls = self.semantic_class_map[targets_semantic_cls]

        targets_semantic_mask = labels.occupancy.semantic_masks

        # filter out invalid semantic classes
        valid = targets_semantic_cls >= 0

        targets_semantic_cls = targets_semantic_cls[valid]
        targets_semantic_mask = targets_semantic_mask[valid]

        # prepare the predictions and targets for the matcher and loss
        preds = {
            "class_scores": out["semantic_cls_scores"],
            "occupancy": MetaDict(
                {
                    "instance_scores": out["semantic_mask_preds"],
                }
            ),
        }
        preds = MetaDict(preds)

        targets = {
            "class_ids": targets_semantic_cls,
            "occupancy": MetaDict(
                {
                    "masks": labels.occupancy.masks,
                    "instance_masks": targets_semantic_mask,
                }
            ),
        }
        targets = MetaDict(targets)

        # perform matching
        assignments = self.semantic_mask_matcher(preds, targets)

        return preds, targets, assignments

    def compute_latent_semantic_gauss_loss(self, gaussians, labels):
        # Map occupancy classes to the gaussian class subset used for training
        targets_semantics = labels.semantics
        targets_semantics = self.gaussian_class_map[targets_semantics]

        valid = targets_semantics >= 0

        # Convert invalid classes to the last class ("free"/"unmodeled")
        targets_semantics = torch.where(
            valid, targets_semantics, self.num_gaussian_classes
        )

        # Single-scale loss: concatenate per-stream gaussians
        if not self.semantic_gauss_loss_is_hierarchical:
            # Gaussians is dict of per-stream MetaDicts
            all_gaussians = MetaDict()
            all_gaussians.query = torch.cat(
                [gaussians[s].query for s in gaussians.keys()], dim=2
            )
            all_gaussians.logits = torch.cat(
                [gaussians[s].logits for s in gaussians.keys()], dim=2
            )
            all_gaussians.centers = torch.cat(
                [gaussians[s].centers for s in gaussians.keys()], dim=2
            )
            all_gaussians.scales = torch.cat(
                [gaussians[s].scales for s in gaussians.keys()], dim=2
            )
            all_gaussians.rotations = torch.cat(
                [gaussians[s].rotations for s in gaussians.keys()], dim=2
            )
            all_gaussians.opacities = torch.cat(
                [gaussians[s].opacities for s in gaussians.keys()], dim=2
            )
            all_gaussians.confidence = torch.cat(
                [gaussians[s].confidence for s in gaussians.keys()], dim=1
            )
            gaussians = all_gaussians

        targets = {
            "semantics": targets_semantics,
        }
        # The distance-field regularizer is the only consumer of the (large)
        # distance volume. When it is disabled (distance_weight == center_weight
        # == 0, the default) the data pipeline may omit labels.distance_field
        # entirely, so only reference it when the loss will actually use it.
        if (
            getattr(self.semantic_gauss_loss, "distance_weight", 0.0) > 0.0
            or getattr(self.semantic_gauss_loss, "center_weight", 0.0) > 0.0
        ):
            targets["distance_field"] = labels.distance_field
        targets = MetaDict(targets)

        return self.semantic_gauss_loss(
            preds=gaussians,
            targets=targets,
            masks=labels.masks,
        )

    def get_sample_data_for_frame(self, sample: Sample, frame_index: int) -> MetaDict:
        """Extract sample data for a single frame from the multi-frame sample"""

        out = MetaDict()

        # ego transforms
        out.points = MetaDict()
        out.points.meta = MetaDict()
        out.points.meta.transforms = MetaDict()
        out.points.meta.transforms.ego_to_global = sample.points[
            frame_index
        ].meta.transforms.ego_to_global

        # images
        out.images = MetaDict()
        out.images.data = sample.images.data[:, frame_index, ...]

        out.images.meta = MetaDict()
        out.images.meta.shape = [s[1:] for s in sample.images.meta.shape]
        out.images.meta.padding = sample.images.meta.padding

        img_transforms = sample.images.meta.transforms
        out.images.meta.transforms = MetaDict(
            {
                "intrinsic": img_transforms.intrinsic[:, frame_index, ...],
                "extrinsic": img_transforms.extrinsic[:, frame_index, ...],
                "image_to_ego": img_transforms.image_to_ego[:, frame_index, ...],
                "ego_to_image": img_transforms.ego_to_image[:, frame_index, ...],
            }
        )

        # depth
        out.depth = MetaDict()
        out.depth.data = sample.depth.data[:, frame_index, ...]
        out.depth.mask = sample.depth.mask[:, frame_index, ...]

        # labels
        out.labels = sample.labels[frame_index]

        return out

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "use trainign_step(), validation_step(), or predict_step() instead"
        )

    def warmup_step_single(
        self,
        sample: MetaDict,
        prev_temporal_state: TemporalState | None = None,
        ego_to_global_prev: torch.Tensor | None = None,
    ) -> TemporalState | None:
        """Inference-only step to build temporal gaussian state without losses.

        This is used during warmup frames to build up temporal context before
        the actual training frames. It runs the feature extraction and volume
        feature computation, but skips the tracking head and all loss computation.

        Args:
            sample: Single-frame sample data
            prev_temporal_state: Temporal state from previous frame
            ego_to_global_prev: Ego-to-global transform from previous frame

        Returns:
            Current temporal state for propagation to next frame
        """
        # pylint: disable=too-many-locals

        # Extract image features
        feats = self.extract_image_features(sample.images.data)

        # Get current ego transform for temporal gaussian tracking
        ego_to_global_curr = sample.points.meta.transforms.ego_to_global[0]

        # Compute depth and context features (handles mono and stereo)
        depth, context = self._compute_depth(
            image_feats=feats,
            img_meta=sample.images.meta,
            ego_to_global=ego_to_global_curr,
            ego_to_image=sample.images.meta.transforms.ego_to_image,
            image_to_ego=sample.images.meta.transforms.image_to_ego,
        )

        # Compute volume features (includes temporal state propagation)
        (
            _fullres_occ,
            _keypoint_queries,
            _keypoint_coords,
            _gaussians,
            _gaussian_occ,
            _bin_scores,
            current_temporal_state,
            _temporal_consistency_losses,
        ) = self.extract_volume_features(
            depth=depth,
            depth_feats=context,
            image_feats=feats,
            image_shape=sample.images.meta.shape[0],
            image_padding=sample.images.meta.padding[0],
            tx_project=sample.images.meta.transforms.ego_to_image,
            tx_unproject=sample.images.meta.transforms.image_to_ego,
            prev_temporal_state=prev_temporal_state,
            ego_to_global_prev=ego_to_global_prev,
            ego_to_global_curr=ego_to_global_curr,
            sample=sample,
        )

        return current_temporal_state

    def training_step_single(
        self,
        sample: MetaDict,
        track_instances: Instances,
        prev_temporal_state: TemporalState | None = None,
        ego_to_global_prev: torch.Tensor | None = None,
        prev_labels: MetaDict | None = None,
    ) -> tuple[dict[str, torch.Tensor], Instances, TemporalState | None]:
        # pylint: disable=too-many-locals,too-many-statements,too-many-branches
        losses = {}

        # extract image features
        feats = self.extract_image_features(sample.images.data)

        # Get current ego transform for temporal gaussian tracking
        ego_to_global_curr = sample.points.meta.transforms.ego_to_global[0]

        # compute depth and context features (handles mono and stereo)
        depth, context = self._compute_depth(
            image_feats=feats,
            img_meta=sample.images.meta,
            ego_to_global=ego_to_global_curr,
            ego_to_image=sample.images.meta.transforms.ego_to_image,
            image_to_ego=sample.images.meta.transforms.image_to_ego,
        )

        # depth shape: [B=1, num_cams=6, num_bins=92, H, W]

        if self.depth_loss is not None:
            depth_loss = self.depth_loss(depth, sample.depth.data, sample.depth.mask)
            losses["depth_loss"] = depth_loss

        # compute volume features
        (
            fullres_occ,
            keypoint_queries,
            keypoint_coords,
            gaussians,
            gaussian_occ,
            bin_scores,
            current_temporal_state,
            temporal_consistency_losses,
        ) = self.extract_volume_features(
            depth=depth,
            depth_feats=context,
            image_feats=feats,  # Multi-level features from FPN
            image_shape=sample.images.meta.shape[0],
            image_padding=sample.images.meta.padding[0],
            tx_project=sample.images.meta.transforms.ego_to_image,
            tx_unproject=sample.images.meta.transforms.image_to_ego,
            prev_temporal_state=prev_temporal_state,
            ego_to_global_prev=ego_to_global_prev,
            ego_to_global_curr=ego_to_global_curr,
            sample=sample,
            prev_labels=prev_labels,
        )

        # Add temporal consistency losses
        if temporal_consistency_losses:
            losses |= {
                f"temporal/{k}": v for k, v in temporal_consistency_losses.items()
            }

        # optional: per-gaussian semantic loss
        if self.semantic_gauss_loss is not None:
            latent_loss = self.compute_latent_semantic_gauss_loss(
                gaussians, sample.labels.occupancy
            )
            losses |= {f"latent/{k}": v for k, v in latent_loss.items()}

        # optional: learned confidence loss
        if self.learned_confidence_loss is not None:
            gt_semantic = sample.labels.occupancy.semantics
            for stream, stream_gaussian in gaussians.items():
                # Get direct prediction (not EMA-updated) for loss
                predicted = self.confidence_tracker.predict(
                    stream_gaussian.query[:, -1]
                )

                # Compute correctness targets using class mapping
                is_correct, valid_mask = confidence_module.compute_correctness_targets(
                    logits=stream_gaussian.logits[:, -1],
                    centers=stream_gaussian.centers[:, -1],
                    gt_semantic=gt_semantic,
                    voxel_range=self.voxel_range,
                    class_map=self.gaussian_class_map,
                )

                # BCE loss
                loss = self.learned_confidence_loss(predicted, is_correct, valid_mask)
                losses[f"learned_confidence/{stream}"] = loss

        # optional: direct semantic occupancy predictions
        if self.voxel_loss is not None:
            voxel_preds = self.voxel_predictor(fullres_occ)

            voxel_loss = self.voxel_loss(
                preds=voxel_preds, targets=sample.labels.occupancy
            )
            losses["voxel_semantic_loss"] = voxel_loss

        # optional: dense voxel-based gaussian semantic loss
        if self.gaussian_voxel_loss is not None:
            # Select feature source (gaussian_occ or fullres_occ)
            if self.gaussian_voxel_source == "gaussian_occ":
                gauss_feats = gaussian_occ
            else:
                gauss_feats = fullres_occ

            # Project to semantic scores
            gauss_voxel_preds = self.gaussian_voxel_predictor(gauss_feats)

            # Map occupancy classes to gaussian_voxel semantic classes
            gaussian_voxel_targets = MetaDict(sample.labels.occupancy)
            gaussian_voxel_targets.semantics = self.gaussian_voxel_class_map[
                sample.labels.occupancy.semantics
            ]

            # Compute loss
            gauss_voxel_loss = self.gaussian_voxel_loss(
                preds=gauss_voxel_preds, targets=gaussian_voxel_targets
            )
            losses["gaussian_voxel_loss"] = gauss_voxel_loss

        # optional: binary occupancy loss on bin_scores
        if self.gaussian_occupancy_loss is not None:
            # Convert bin_scores to binary occupancy prediction
            # bin_scores is [b, 1, d, h, w], need [b, 2, d, h, w] for binary classification
            free_scores = 1.0 - bin_scores  # Probability of being free
            binary_occ_preds = torch.cat([free_scores, bin_scores], dim=1)

            # Map occupancy classes to free/occupied classes
            nonfree = self.gaussian_voxel_occ_map[sample.labels.occupancy.semantics]
            nonfree = torch.where(nonfree < 0, 0, 1)  # free=0, occupied=1

            occ_targets = MetaDict(sample.labels.occupancy)
            occ_targets.semantics = nonfree

            # Compute loss (SemanticOccupancyLoss expects same format as targets)
            gauss_occ_loss = self.gaussian_occupancy_loss(
                preds=binary_occ_preds, targets=occ_targets
            )
            losses["gaussian_occupancy_loss"] = gauss_occ_loss

        # optional: gaussian depth rendering loss (gsplat)
        if self.gaussian_depth_loss is not None:
            # Concatenate all decoded streams
            all_centers = torch.cat(
                [gs.centers[:, -1] for gs in gaussians.values()], dim=1
            )
            all_scales = torch.cat(
                [gs.scales[:, -1] for gs in gaussians.values()], dim=1
            )
            all_rotations = torch.cat(
                [gs.rotations[:, -1] for gs in gaussians.values()], dim=1
            )
            all_opacities = torch.cat(
                [gs.opacities[:, -1] for gs in gaussians.values()], dim=1
            )

            # Get image shape from sample metadata
            img_shape = sample.images.meta.shape[0][-2:]  # (h, w)

            gauss_depth_losses = self.gaussian_depth_loss(
                centers=all_centers,
                scales=all_scales,
                rotations=all_rotations,
                opacities=all_opacities,
                ego_to_image=sample.images.meta.transforms.ego_to_image,
                image_shape=img_shape,
                gt_depth=sample.depth.data,
                gt_mask=sample.depth.mask,
            )
            losses |= {f"gaussian_depth/{k}": v for k, v in gauss_depth_losses.items()}

        # Concatenate per-stream keypoint features and coords for tracking_head
        # tracking_head expects [b, total_queries, c], not dicts
        streams = [s for s in self.tracking_head_streams if s in keypoint_queries]
        keypoint_queries = torch.cat([keypoint_queries[s] for s in streams], dim=1)
        keypoint_coords = torch.cat([keypoint_coords[s] for s in streams], dim=1)

        # run detection head
        out = self.tracking_head(
            features=feats[self.image_base_feat_level],
            voxel_features=(fullres_occ,),
            img_meta=sample.images.meta,
            query_targets=track_instances.query_feats,
            query_embs=track_instances.query_embeds,
            reference_points=track_instances.reference_points,
            keypoint_features=keypoint_queries,
            keypoint_coords=keypoint_coords,
        )

        # record the detections into the track instances cache
        track_instances = self.load_detection_output_into_cache(track_instances, out)
        out["track_instances"] = track_instances

        # perform ground-truth matching and get updated assignments
        preds, targets, assignments = self.match_instances(out, sample.labels)

        # compute single-frame/detection loss
        losses |= self.single_frame_loss(preds, targets, assignments)

        # drop the per-layer assignment information and just keep the last
        assignments = assignments[:, -1]

        # update the matched object indices
        track_instances = self.update_assignments(
            track_instances, assignments, sample.labels
        )

        # semantic query matching
        preds, targets, sem_assignments = self.match_semantic_masks(out, sample.labels)

        losses |= self.semantic_mask_loss(preds, targets, sem_assignments)

        # spatio-temporal reasoning
        track_instances = self.spatial_temporal_reason(track_instances)

        if self.st_history_reasoning:
            losses |= self.mem_bank_loss(
                track_instances,
                sample.labels,
                assignments,
            )

        if self.st_future_reasoning:
            losses |= self.prediction_loss(
                track_instances.cache_motion_predictions.unsqueeze(0),
                sample.labels.trajectories,
                assignments,
            )

        # prepare for next frame
        track_instances = self.frame_summarization(track_instances, tracking=False)

        active_mask = assignments[0] >= 0
        track_instances.track_query_mask[active_mask] = True
        track_instances = track_instances[active_mask]

        return losses, track_instances, current_temporal_state

    def training_step_motion_update(
        self, batch: Sample, frame_index: int, track_instances: Instances
    ) -> Instances:
        if self.use_motion_prediction:
            timestamp = batch.timestamp
            time_delta = timestamp[frame_index + 1] - timestamp[frame_index]

            track_instances = self.st_reasoner.update_reference_points(
                track_instances,
                time_delta[0],
                use_prediction=self.use_motion_prediction_ref_update,
                tracking=False,
            )

        if self.use_ego_update:
            track_instances = self.st_reasoner.update_ego(
                track_instances,
                batch.points[frame_index].meta.transforms.ego_to_global[0],
                batch.points[frame_index + 1].meta.transforms.ego_to_global[0],
            )

        track_instances = self.sync_pos_embedding(track_instances)

        return track_instances

    def training_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        # pylint: disable=unused-argument,arguments-differ
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        losses = {}

        if self.optimization_mode == "detached":
            self.optimizer.zero_grad(self)

        num_frames = batch.images.data.shape[1]

        # Determine number of warmup frames (ensure at least 1 training frame)
        num_warmup = self.num_warmup_frames
        assert num_warmup < num_frames, "Need at least one training frame"

        self.runtime_tracker.reset()

        # === WARMUP PHASE: build temporal state without gradients ===
        if num_warmup > 0:
            # Skip duplicate frames at sequence start (padding)
            # Find offset where unique frames begin
            sample_ids = batch.frames.sample_ids[0]  # only works with batch_size=1
            for warmup_start in range(num_warmup):
                if sample_ids[warmup_start] != sample_ids[warmup_start + 1]:
                    break

            # Run warmup in eval mode (no BN stats updates, no dropout, no SyncBN syncs)
            self.eval()
            with torch.no_grad():
                for frame_index in range(warmup_start, num_warmup):
                    sample = self.get_sample_data_for_frame(batch, frame_index)

                    temporal_state = self.warmup_step_single(
                        sample=sample,
                        prev_temporal_state=self.runtime_tracker.temporal_state,
                        ego_to_global_prev=self.runtime_tracker.ego_to_global,
                    )

                    self.runtime_tracker.temporal_state = temporal_state
                    self.runtime_tracker.ego_to_global = batch.points[
                        frame_index
                    ].meta.transforms.ego_to_global[0]
            self.train()

            # Ensure clean detach before training phase
            if self.runtime_tracker.temporal_state is not None:
                self.runtime_tracker.temporal_state = (
                    self.runtime_tracker.temporal_state.detach()
                )
            if self.runtime_tracker.stereo_state is not None:
                self.runtime_tracker.stereo_state = (
                    self.runtime_tracker.stereo_state.detach()
                )

        # === TRAINING PHASE: normal per-frame training with losses ===
        # Fresh track_instances after warmup - temporal context is in gaussians
        track_instances = self.generate_empty_instance()

        for frame_index in range(num_warmup, num_frames):
            # reformat to single-frame image metadata
            sample = self.get_sample_data_for_frame(batch, frame_index)

            # Get previous frame labels for dynamic object supervision
            if frame_index > 0:
                prev_labels = batch.labels[frame_index - 1]
            else:
                prev_labels = None

            # per-frame training step
            frame_losses, track_instances, current_temporal_state = (
                self.training_step_single(
                    sample=sample,
                    track_instances=track_instances,
                    prev_temporal_state=self.runtime_tracker.temporal_state,
                    ego_to_global_prev=self.runtime_tracker.ego_to_global,
                    prev_labels=prev_labels,
                )
            )

            # manual/detached per-frame optimization
            if self.optimization_mode == "detached":
                self.manual_backward(sum(frame_losses.values()))

                frame_losses = {k: v.detach() for k, v in frame_losses.items()}
                track_instances = track_instances.detach()

                # Detach temporal state for next frame
                if current_temporal_state is not None:
                    current_temporal_state = current_temporal_state.detach()

                # Detach stereo state for next frame
                if self.runtime_tracker.stereo_state is not None:
                    self.runtime_tracker.stereo_state = (
                        self.runtime_tracker.stereo_state.detach()
                    )

            # perform motion updates and prepare instances for the next frame
            if frame_index < num_frames - 1:
                track_instances = self.training_step_motion_update(
                    batch=batch,
                    frame_index=frame_index,
                    track_instances=track_instances,
                )

                empty_instances = self.generate_empty_instance()
                track_instances = Instances.cat((empty_instances, track_instances))

            # Store temporal state and ego transform for next frame
            self.runtime_tracker.temporal_state = current_temporal_state
            self.runtime_tracker.ego_to_global = batch.points[
                frame_index
            ].meta.transforms.ego_to_global[0]

            # aggregate losses with relative training frame index (f0, f1, ...)
            train_idx = frame_index - num_warmup
            losses |= {f"f{train_idx}/{k}": v for k, v in frame_losses.items()}

        self.runtime_tracker.reset()

        # sum losses
        loss = sum(v for _, v in losses.items())

        # log losses
        self.log_dict(losses | {"loss": loss}, batch_size=batch.batch_size)

        if self.optimization_mode == "detached":
            self.optimizer.step(self)

        return loss

    def on_train_epoch_end(self) -> None:
        self.runtime_tracker.reset()

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> MetaDict:
        # pylint: disable=unused-argument,arguments-differ
        # pylint: disable=too-many-locals,too-many-statements
        # pylint: disable=too-many-branches

        assert batch.batch_size == 1

        # Parts of the code below assume a batch size of one. So if we have an
        # invalid sample, make things easier and just return None here.
        if not batch.is_valid[0]:
            return None

        # Check if we have a new sequence
        tracker = self.runtime_tracker

        new_seq = tracker.current_seq is None
        new_seq = new_seq or batch.meta[0].sequence_id != tracker.current_seq

        if new_seq or ((not self.tracking) and self.num_warmup_frames == 0):
            tracker.reset()
            tracker.current_seq = batch.meta[0].sequence_id
            tracker.timestamp = batch.timestamp[0]
        elif not self.tracking and self.num_warmup_frames > 0:
            tracker.track_instances = None

        tracker.time_delta = batch.timestamp[0] - tracker.timestamp
        tracker.timestamp = batch.timestamp[0]

        # Extract image features
        feats = self.extract_image_features(batch.images.data)

        # Compute depth and context features (handles mono and stereo)
        depth, context = self._compute_depth(
            image_feats=feats,
            img_meta=batch.images.meta,
            ego_to_global=batch.points.meta.transforms.ego_to_global[0],
            ego_to_image=batch.images.meta.transforms.ego_to_image,
            image_to_ego=batch.images.meta.transforms.image_to_ego,
        )

        # Compute volume features
        (
            fullres_occ,
            keypoint_queries,
            keypoint_coords,
            gaussians,
            _gaussian_occ,
            _bin_scores,
            current_temporal_state,
            _temporal_consistency_losses,
        ) = self.extract_volume_features(
            depth=depth,
            depth_feats=context,
            image_feats=feats,  # Multi-level features from FPN
            image_shape=batch.images.meta.shape[0],
            image_padding=batch.images.meta.padding[0],
            tx_project=batch.images.meta.transforms.ego_to_image,
            tx_unproject=batch.images.meta.transforms.image_to_ego,
            prev_temporal_state=tracker.temporal_state,
            ego_to_global_prev=tracker.ego_to_global,
            ego_to_global_curr=batch.points.meta.transforms.ego_to_global[0],
            sample=batch,
        )

        # 1. Update the information of previous active tracks
        if tracker.track_instances is None:
            track_instances = self.generate_empty_instance()

        else:
            track_instances = tracker.track_instances

            if self.use_motion_prediction:
                track_instances = self.st_reasoner.update_reference_points(
                    track_instances,
                    tracker.time_delta,
                    use_prediction=self.use_motion_prediction_ref_update,
                    tracking=True,
                )

            if self.use_ego_update:
                track_instances = self.st_reasoner.update_ego(
                    track_instances,
                    tracker.ego_to_global,
                    batch.points.meta.transforms.ego_to_global[0],
                )

            track_instances = self.sync_pos_embedding(track_instances)

            track_instances = Instances.cat(
                (self.generate_empty_instance(), track_instances)
            )

        tracker.ego_to_global = batch.points.meta.transforms.ego_to_global[0]

        # Concatenate per-stream keypoint features and coords for tracking_head
        # tracking_head expects [b, total_queries, c], not dicts
        streams = [s for s in self.tracking_head_streams if s in keypoint_queries]
        keypoint_queries = torch.cat([keypoint_queries[s] for s in streams], dim=1)
        keypoint_coords = torch.cat([keypoint_coords[s] for s in streams], dim=1)

        # 2. PETR detection head
        out = self.tracking_head(
            features=feats[self.image_base_feat_level],
            voxel_features=(fullres_occ,),
            img_meta=batch.images.meta,
            query_targets=track_instances.query_feats,
            query_embs=track_instances.query_embeds,
            reference_points=track_instances.reference_points,
            keypoint_features=keypoint_queries,
            keypoint_coords=keypoint_coords,
        )

        # 3. Record the information into the track instances cache
        track_instances = self.load_detection_output_into_cache(track_instances, out)
        out["track_instances"] = track_instances

        # 4. Spatial-temporal Reasoning
        self.spatial_temporal_reason(track_instances)
        track_instances = self.frame_summarization(track_instances, self.tracking)
        out["all_cls_scores"][0, -1, :] = track_instances.logits
        out["all_bbox_preds"][0, -1, :] = track_instances.bboxes

        if self.st_future_reasoning:
            # motion forecasting has the shape of [num_query, T, 2]
            out["all_motion_forecasting"] = track_instances.motion_predictions.clone()
        else:
            out["all_motion_forecasting"] = None

        # 5. Track class filtering: before decoding bboxes, only leave the
        # objects under tracking categories
        if self.filter_track_classes is not None:
            max_cat = torch.argmax(out["all_cls_scores"][0, -1, :].sigmoid(), dim=-1)
            mask = torch.isin(max_cat, self.filter_track_classes)

            track_instances = track_instances[mask]
            out["all_cls_scores"] = out["all_cls_scores"][:, :, mask, :]
            out["all_bbox_preds"] = out["all_bbox_preds"][:, :, mask, :]
            out["all_mask_preds"] = out["all_mask_preds"][:, :, mask, :]
            if out["all_motion_forecasting"] is not None:
                out["all_motion_forecasting"] = out["all_motion_forecasting"][mask, ...]

            out["track_instances"] = track_instances

        # 6. assign ids
        if self.tracking:
            active_mask = track_instances.scores > self.runtime_tracker.threshold

            for i in range(len(track_instances)):
                if track_instances.obj_idxes[i] < 0:
                    track_instances.obj_idxes[i] = self.runtime_tracker.current_id
                    self.runtime_tracker.current_id += 1

                    if active_mask[i]:
                        track_instances.track_query_mask[i] = True

            out["track_instances"] = track_instances

        # 7. Prepare for the next frame
        if self.tracking:
            score_mask = track_instances.scores > self.runtime_tracker.output_threshold
        else:
            score_mask = torch.ones_like(track_instances.scores, dtype=torch.bool)

        out["all_masks"] = score_mask

        # 8. Decode output
        if self.voxel_predictor is not None:
            semantics = self.voxel_predictor(fullres_occ)
        else:
            semantics = None

        if self.tracking:
            instance_ids = track_instances.obj_idxes.clone().unsqueeze(0)
        else:
            instance_ids = None

        preds = self.tracking_head.get_bboxes(out)
        preds.occupancy = self.occupancy_predictor(
            volume_semantics=semantics,
            instance_ids=instance_ids,
            instance_class_scores=out["all_cls_scores"][:, -1],
            instance_mask_scores=out["all_mask_preds"][:, -1],
            instance_valid=score_mask[None, :],
            semantic_class_scores=out["semantic_cls_scores"][:, -1],
            semantic_mask_scores=out["semantic_mask_preds"][:, -1],
        )

        # Expose the decoded gaussians (dict stream -> MetaDict of final-layer
        # predictions) so prediction writers can persist them alongside the
        # panoptic occupancy grid.
        preds.gaussians = gaussians

        # 9. Update tracks for next frame
        if self.tracking:
            self.runtime_tracker.update_active_tracks(track_instances, active_mask)

        # Store temporal state for next frame (temporal aggregation)
        if self.use_gaussian_temporal:
            self.runtime_tracker.temporal_state = current_temporal_state

        return preds

    def on_validation_epoch_end(self) -> None:
        self.runtime_tracker.reset()

    # pylint: disable-next=unused-argument,arguments-differ
    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> MetaDict:
        return self.validation_step(batch, batch_idx, dataloader_idx)

    def on_predict_epoch_end(self) -> None:
        self.runtime_tracker.reset()


class Optimizer:
    """
    A simple optimizer wrapper for manual optimization in PyTorch Lightning.
    """

    def __init__(
        self,
        gradient_clip_algorithm: str | None = None,
        gradient_clip_val: float | None = None,
    ):
        self.gradient_clip_algorithm = gradient_clip_algorithm
        self.gradient_clip_val = gradient_clip_val

    def zero_grad(self, model: L.LightningModule) -> None:
        trainer = model.trainer
        optimizer = model.optimizers()

        # NOTE: This is a workaround for the fact that `on_before_zero_grad` is
        # not called by the Trainer when using manual optimization.
        # pylint: disable=protected-access
        call._call_callback_hooks(trainer, "on_before_zero_grad", optimizer)
        call._call_lightning_module_hook(trainer, "on_before_zero_grad", optimizer)

        # zero gradients
        optimizer.zero_grad()

    def step(self, model: L.LightningModule) -> None:
        optimizer = model.optimizers()
        scheduler = model.lr_schedulers()

        # perform gradient clipping, if specified
        if self.gradient_clip_algorithm is not None:
            model.clip_gradients(
                optimizer,
                gradient_clip_algorithm=self.gradient_clip_algorithm,
                gradient_clip_val=self.gradient_clip_val,
            )

        # step the optimizer and LR-scheduler
        optimizer.step()
        scheduler.step()
