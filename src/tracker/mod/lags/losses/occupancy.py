# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import einops
import torch
from mmdet.registry import MODELS
from omegaconf import OmegaConf
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F

from ....utils.torch import amp
from ....utils.torch.dist import all_reduce
from ....utils.types import MetaDict
from ..head import QueryDecoder
from ..occupancy.gaussian import GaussianSemanticAggregator
from . import utils


class SemanticOccupancyLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mask: str | None,
        cls_loss: OmegaConf,
        ignore_index: int | None = None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.mask = mask if mask != "none" else None
        self.ignore_index = ignore_index

        cls_loss = OmegaConf.to_container(cls_loss, resolve=True)
        self.loss_cls = MODELS.build(cls_loss)

    def forward(self, preds: torch.Tensor, targets: MetaDict) -> torch.Tensor:
        preds = einops.rearrange(preds, "b c d h w -> b d h w c")
        labels = targets.semantics

        preds = preds.flatten(end_dim=-2)  # [n, c]
        labels = labels.flatten()  # [n]

        mask = None
        if self.mask is not None:
            mask = targets.masks[self.mask].flatten()

        if self.ignore_index is not None:
            mask = mask if mask is not None else True
            mask = mask & (labels != self.ignore_index)

        if mask is not None:
            preds = preds[mask]
            labels = labels[mask]

        return self.loss_cls(preds, labels.long(), avg_factor=labels.numel())


class SemanticMaskLoss(nn.Module):
    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        num_classes: int,
        cls_loss: OmegaConf,
        base_mask_loss: OmegaConf,
        dice_mask_loss: OmegaConf,
        bg_cls_weight: float = 0.0,
        sync_cls_avg_factor: bool = False,
        interm_loss: bool = True,
        occupancy_mask: str | None = "valid",
        mask_pos_weight: float = 1.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.interm_loss = interm_loss
        self.bg_cls_weight = bg_cls_weight
        self.sync_cls_avg_factor = sync_cls_avg_factor

        cls_loss = OmegaConf.to_container(cls_loss, resolve=True)
        dice_mask_loss = OmegaConf.to_container(dice_mask_loss, resolve=True)
        base_mask_loss = OmegaConf.to_container(base_mask_loss, resolve=True)

        self.loss_cls = MODELS.build(cls_loss)
        self.loss_mask_dice = MODELS.build(dice_mask_loss)
        self.base_mask_loss = MODELS.build(base_mask_loss)
        self.occupancy_mask = occupancy_mask if occupancy_mask != "none" else None
        self.mask_pos_weight = mask_pos_weight

    def classification_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
        num_pos: int,
        num_neg: int,
    ) -> torch.Tensor:
        """
        Compute the classification loss.

        Args:
            preds (torch.Tensor): The predicted class scores. Shape: [b, num_preds, num_classes].
            targets (torch.Tensor): The target class labels. Shape: [b, num_preds].
            weights (torch.Tensor): The weights for the target classes. Shape: [b, num_preds].
            num_pos (int): The number of positive samples.
            num_neg (int): The number of negative samples.

        Returns:
            torch.Tensor: The classification loss.
        """
        # flatten the tensors across batch dimensions
        preds = preds.flatten(end_dim=-2)
        targets = targets.flatten()
        weights = weights.flatten()

        # construct weighted avg_factor to match with the official DETR repo
        avg_factor = num_pos * 1.0 + num_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            avg_factor = preds.new_tensor([avg_factor])
            avg_factor = all_reduce(avg_factor, op=dist.ReduceOp.AVG)

        avg_factor = max(avg_factor, 1)

        # NOTE: MMCV's FocalLoss does not support bfloat16 inputs, so we need
        # to upcast the predictions and weights to float32.
        preds = amp.upcast(preds, dtype=torch.float32)
        weights = amp.upcast(weights, dtype=torch.float32)

        # compute the actual loss
        loss = self.loss_cls(preds, targets, weights, avg_factor=avg_factor)
        loss = torch.nan_to_num(loss)

        return loss

    @torch.autocast("cuda", enabled=False)
    def mask_loss(
        self,
        preds: torch.Tensor,  # [num_pos, z, y, x]
        targets: torch.Tensor,  # [num_pos, z, y, x]
        mask: torch.Tensor | None,  # [num_pos, z, y, x]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=too-many-locals

        # prepare tensors
        preds = preds.flatten(start_dim=1).to(dtype=torch.float32)
        targets = targets.flatten(start_dim=1)
        mask = mask.flatten(start_dim=1) if mask is not None else True

        # filter out predictions that have no occupied voxels
        valid = (targets & mask).sum(dim=-1) > 0
        mask = mask & valid[:, None]
        mask = mask.expand_as(targets)

        # compute averages for normalization accross all gpus
        num_pos = valid.sum().float()
        num_pos = torch.clamp(all_reduce(num_pos, op=dist.ReduceOp.AVG), min=1)

        num_voxels = mask.sum().float()
        num_voxels = torch.clamp(all_reduce(num_voxels, op=dist.ReduceOp.AVG), min=1)

        # compute per-voxel weights
        weight = torch.where(targets, self.mask_pos_weight, 1.0)
        weight = weight * mask

        # compute focal loss
        loss_base = self.base_mask_loss(
            preds,
            targets,
            weight=weight,
            avg_factor=num_voxels,
        )

        # compute dice loss
        loss_dice = self.loss_mask_dice(
            pred=torch.where(mask, preds, 0) if mask is not None else preds,
            target=torch.where(mask, targets, 0) if mask is not None else targets,
            avg_factor=num_pos,
        )

        return loss_base, loss_dice

    def layer_loss(
        self,
        preds: MetaDict,
        targets: MetaDict,
        assignments: torch.Tensor,
        layer: int,
    ) -> dict[str, torch.Tensor]:
        # pylint: disable=too-many-locals
        """
        Compute the loss for a specific decoder layer.

        Args:
            preds (MetaDict): The predictions for the current layer.
            targets (MetaDict): The targets for the current layer.
            assignments (torch.Tensor): The assignments for the current layer.
            layer (int): The index of the current decoder layer.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the loss values.
        """
        n_classes = self.num_classes
        _b, _n_layers, n_preds = assignments.shape

        # get targets for the current layer
        assign = assignments[:, layer]  # [b, n_preds]

        valid = assign >= 0

        target_class = targets.class_ids.unbind()
        target_class = utils.map_targets(assign, target_class, n_classes, mask=valid)
        target_class_mask = torch.ones_like(target_class, dtype=torch.float)

        target_masks = targets.occupancy.instance_masks.unbind()
        target_masks = utils.map_targets(assign, target_masks, False, mask=valid)

        # get predictions for the current layer
        pred_class = preds.class_scores[:, layer]  # [b, n_preds, n_classes]
        pred_masks = preds.occupancy.instance_scores[:, layer]  # [b, n_preds, z, y, x]

        # get mask for occupancy
        mask_occ = self.occupancy_mask
        if mask_occ is not None:
            mask_occ = targets.occupancy.masks[mask_occ]  # [b, z, y, x]
            mask_occ = einops.repeat(mask_occ, "b z y x -> b n z y x", n=n_preds)

        # get number of positive and negative samples
        num_pos = valid.sum().item()
        num_neg = valid.numel() - num_pos

        # compute losses
        loss_cls = self.classification_loss(
            pred_class,
            target_class,
            target_class_mask,
            num_pos,
            num_neg,
        )

        loss_mask_base, loss_mask_dice = self.mask_loss(
            pred_masks[valid],  # [num_pos, z, y, x]
            target_masks[valid],  # [num_pos, z, y, x]
            mask=mask_occ[valid] if mask_occ is not None else None,
        )

        return {
            "loss_semantic_cls": loss_cls,
            "loss_semantic_base": loss_mask_base,
            "loss_semantic_dice": loss_mask_dice,
        }

    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
        assignments: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _batch_size, num_layers, _num_preds = assignments.shape

        first = 0 if self.interm_loss else num_layers - 1

        losses = {}
        for layer in range(first, num_layers):
            loss = self.layer_loss(preds, targets, assignments, layer)
            losses |= {f"d{layer}/{k}": v for k, v in loss.items()}

        return losses


class SemanticGaussianLoss(nn.Module):
    # pylint: disable=too-many-instance-attributes

    class_weights: torch.Tensor | None

    def __init__(
        self,
        mask: str | None,
        weight: float = 1.0,
        distance_weight: float = 0.0,
        center_weight: float = 0.0,
        volume_weight: float = 0.0,
        min_volume: float = 0.125,
        voxel_range: list[float] = (-40.0, -40.0, -1.0, 40.0, 40.0, 10.0),
        voxel_size: list[float] = (0.4, 0.4, 0.4),
        scale_multiplier: float = 5.0,
        class_weights: tuple[float] | None = None,
        mask_weights: float = 0.0,
        aggregation_opacity_mode: str = "decoded",
        aggregation_opacity_threshold: float = 0.0,
        query_dim: int = 256,
        opacity_head_layers: int = 1,
    ):
        # pylint: disable=too-many-locals
        super().__init__()

        self.weight = weight
        self.distance_weight = distance_weight
        self.center_weight = center_weight
        self.volume_weight = volume_weight
        self.min_volume = min_volume
        self.mask = mask if mask != "none" else None
        self.mask_weights = mask_weights

        assert aggregation_opacity_mode in [
            "decoded",
            "fixed",
            "confidence",
            "local",
        ], (
            f"Invalid aggregation_opacity_mode: {aggregation_opacity_mode}. "
            "Must be 'decoded', 'fixed', 'confidence', or 'local'"
        )
        self.aggregation_opacity_mode = aggregation_opacity_mode
        self.aggregation_opacity_threshold = aggregation_opacity_threshold

        self.aggregator = GaussianSemanticAggregator(
            voxel_size=voxel_size,
            voxel_range=voxel_range,
            scale_multiplier=scale_multiplier,
        )

        # Store voxel configuration for distance field sampling
        self.voxel_range = torch.tensor(voxel_range, dtype=torch.float32)
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)

        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights, persistent=False)
        else:
            self.class_weights = None

        # Local opacity decoder (only needed for "local" mode)
        if aggregation_opacity_mode == "local":
            self.opacity_decoder = QueryDecoder(
                embed_dim=query_dim,
                output_dim=1,
                num_layers=opacity_head_layers,
            )
        else:
            self.opacity_decoder = None

    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
        masks: MetaDict,
    ) -> dict[str, torch.Tensor]:
        loss = {}

        # Compute classification loss (always)
        loss.update(self._compute_classification_loss(preds, targets.semantics, masks))

        # Compute distance field regularization loss (if enabled)
        if self.distance_weight > 0.0 or self.center_weight > 0.0:
            loss.update(self._compute_distance_field_loss(preds, targets))

        # Compute volume regularization loss (if enabled)
        if self.volume_weight > 0.0:
            loss.update(self._compute_volume_regularization_loss(preds))

        return loss

    @torch.compile(fullgraph=True, mode="reduce-overhead")
    def _compute_classification_loss_internal(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | float,
    ) -> dict[str, torch.Tensor]:
        num_layers = preds.shape[1]

        loss = {}
        for layer in range(num_layers):
            layer_preds = preds[:, layer].flatten(end_dim=-2)

            layer_loss = F.nll_loss(
                layer_preds,
                targets,
                weight=self.class_weights,
                reduction="none",
            )
            layer_loss = torch.mean(layer_loss * weights)

            loss[f"d{layer}/ce"] = layer_loss * self.weight

        return loss

    def _compute_classification_loss(
        self,
        preds: MetaDict,
        targets: torch.Tensor,  # [b, z, y, x]
        masks: MetaDict,
    ) -> dict[str, torch.Tensor]:
        # pylint: disable=unused-argument, too-many-locals
        mask = masks[self.mask] if self.mask is not None else None

        if self.mask_weights != 0.0 and mask is not None:
            binary_mask = None
            weights = torch.where(mask, 1.0, self.mask_weights).flatten()
        elif mask is not None:
            binary_mask = mask
            weights = 1.0
        else:
            binary_mask = None
            weights = 1.0

        # Determine opacity for semantic splatting based on mode
        if self.aggregation_opacity_mode == "fixed":
            # Fixed opacity=1.0: every gaussian contributes equally
            opacities = torch.ones_like(preds.opacities)
        elif self.aggregation_opacity_mode == "confidence":
            # Use max(opacity, confidence): well-observed gaussians stay active
            # preds.opacities: [b, l, n_gaussians]
            # preds.confidence: [b, n_gaussians] - expand to match
            confidence = preds.confidence.unsqueeze(1).expand_as(preds.opacities)
            opacities = torch.maximum(preds.opacities, confidence)
        elif self.aggregation_opacity_mode == "local":
            # Separate learned opacity decoder
            # preds.query: [b, l, n_gaussians, query_dim]
            opacities = torch.sigmoid(self.opacity_decoder(preds.query).squeeze(-1))
        else:
            # Default: use decoded opacity (coupled with depth loss)
            opacities = preds.opacities

        # Apply opacity threshold: zero out gaussians below threshold
        if self.aggregation_opacity_threshold > 0.0:
            opacities = opacities * (opacities >= self.aggregation_opacity_threshold)

        # aggregate the predictions
        aggregated_preds, _, _, _ = self.aggregator(
            logits=preds.logits,
            centers=preds.centers,
            scales=preds.scales,
            rotations=preds.rotations,
            opacities=opacities,
            mask=binary_mask,
        )  # [b, l, n, c]

        # get the target semantics
        if binary_mask is not None:
            targets = targets[binary_mask]

        targets = targets.flatten()  # [n_unmasked]

        # main loss: cross entropy
        with torch.autocast("cuda", enabled=False):
            preds_upcast = amp.upcast(aggregated_preds, dtype=torch.float32)
            preds_upcast = torch.clamp(preds_upcast, min=1e-6, max=1 - 1e-6)
            preds_upcast = torch.log(preds_upcast)

            loss = self._compute_classification_loss_internal(
                preds_upcast, targets, weights
            )

        return {f"semantic_gauss/{k}": v.clone() for k, v in loss.items()}

    def _world_to_grid_coords(self, centers: torch.Tensor) -> torch.Tensor:
        """
        Convert gaussian centers from world coordinates to grid sample coordinates [-1, 1].

        Args:
            centers: [b, l, n_gaussians, 3] in world coordinates

        Returns:
            grid_coords: [b, l, n_gaussians, 3] in [-1, 1] range for F.grid_sample
        """
        # Ensure voxel_range is on the same device
        voxel_range = self.voxel_range.to(centers.device)

        # Normalize to [0, 1]
        normalized = (centers - voxel_range[:3]) / (voxel_range[3:] - voxel_range[:3])

        # Convert to grid_sample coordinates [-1, 1]
        grid_coords = normalized * 2.0 - 1.0

        return grid_coords

    def _get_voxel_grid_center(self, device: torch.device) -> torch.Tensor:
        """Get the center of the voxel grid in world coordinates."""
        voxel_range = self.voxel_range.to(device)
        return (voxel_range[:3] + voxel_range[3:]) / 2.0

    def _sample_distance_field(
        self,
        centers: torch.Tensor,  # [b, l, n_gaussians, 3]
        distance_field: torch.Tensor,  # [b, z, y, x]
        vectors_field: torch.Tensor,  # [b, z, y, x, 3]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample distance and vector fields at gaussian center locations.

        Args:
            centers: Gaussian centers in world coordinates
            distance_field: Distance field tensor
            vectors_field: Vector field tensor

        Returns:
            sampled_distances: [b, l, n_gaussians]
            sampled_vectors: [b, l, n_gaussians, 3]
        """
        # pylint: disable=too-many-locals
        b, l, n, _ = centers.shape

        # Convert to grid coordinates
        grid_coords = self._world_to_grid_coords(centers)

        # Detect out-of-bounds gaussians
        out_of_bounds = (torch.abs(grid_coords) > 1.0).any(
            dim=-1
        )  # [b, l, n_gaussians]

        # Reshape for grid_sample: need [B, D, H, W, 3] format
        # But we have single samples, so we'll reshape to [b*l*n, 1, 1, 1, 3]
        sampling_coords = grid_coords.view(b * l * n, 1, 1, 1, 3)

        # Sample distances
        # distance_field: [b, z, y, x] -> [1, 1, z, y, x] for grid_sample
        distance_field_5d = distance_field[
            :, None, None, None, ...
        ]  # [b, 1, 1, 1, z, y, x]
        distance_field_5d = distance_field_5d.expand(-1, l, n, -1, -1, -1, -1)

        sampled_distances = F.grid_sample(
            distance_field_5d.flatten(0, 2),  # [b*l*n, z, y, x]
            sampling_coords,
            mode="bilinear",  # trilinear for 3D
            padding_mode="border",
            align_corners=False,
        )  # [b*l*n, 1, 1, 1, 1]

        # Reshape back to [b, l, n_gaussians]
        sampled_distances = sampled_distances.view(b, l, n)

        # Sample vectors
        # vectors_field: [b, z, y, x, 3] -> [b, 3, z, y, x] for grid_sample
        vectors_field_5d = vectors_field.permute(0, 4, 1, 2, 3)
        vectors_field_5d = vectors_field_5d[:, None, None, ...]  # [b, 1, 1, 3, z, y, x]
        vectors_field_5d = vectors_field_5d.expand(-1, l, n, -1, -1, -1, -1)

        sampled_vectors = F.grid_sample(
            vectors_field_5d.flatten(0, 2),  # [b*l*n, 3, z, y, x]
            sampling_coords,
            mode="bilinear",  # trilinear for 3D
            padding_mode="border",
            align_corners=False,
        )  # [b*l*n, 3, 1, 1, 1]

        # Reshape back to [b, l, n_gaussians, 3]
        sampled_vectors = (
            sampled_vectors.squeeze(-1).squeeze(-1).squeeze(-1)
        )  # [b*l*n, 3]
        sampled_vectors = sampled_vectors.view(b, l, n, 3)

        # Handle out-of-bounds gaussians with center-pointing vectors
        grid_center = self._get_voxel_grid_center(centers.device)  # [3]
        center_directions = grid_center - centers  # [b, l, n_gaussians, 3]
        center_directions = F.normalize(center_directions, dim=-1)

        # Replace vectors for out-of-bounds gaussians
        sampled_vectors = torch.where(
            out_of_bounds.unsqueeze(-1),  # [b, l, n_gaussians, 1]
            center_directions,
            sampled_vectors,
        )

        return sampled_distances, sampled_vectors

    def _compute_distance_field_loss(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> dict[str, torch.Tensor]:
        """
        Compute distance field regularization loss for gaussian centers.

        Args:
            preds: Gaussian predictions with centers [b, l, n_gaussians, 3]
            targets: Target dict containing distance_field.{distances, vectors}

        Returns:
            Dictionary of distance field losses per layer
        """
        # pylint: disable=too-many-locals
        distance_field = targets.distance_field.distances  # [b, z, y, x]
        vectors_field = targets.distance_field.vectors  # [b, z, y, x, 3]
        centers = preds.centers  # [b, l, n_gaussians, 3]

        _, num_layers, _, _ = centers.shape

        # Sample distance field at gaussian centers
        sampled_distances, sampled_vectors = self._sample_distance_field(
            centers, distance_field, vectors_field
        )

        # Handle saddle points: replace weak vectors with random ones
        vector_magnitudes = torch.norm(sampled_vectors, dim=-1)  # [b, l, n_gaussians]

        # Thresholds for saddle point detection
        vector_threshold = 1e-3
        distance_threshold = (
            0.1  # Only consider it a saddle if we're not too close to surface
        )

        # Detect saddle points: weak vectors but non-zero distance
        is_saddle = (vector_magnitudes < vector_threshold) & (
            sampled_distances > distance_threshold
        )

        # Generate random unit vectors for saddle points
        random_vectors = torch.randn_like(sampled_vectors)
        random_vectors = F.normalize(random_vectors, dim=-1)

        # Use random vectors for saddle points, sampled vectors elsewhere
        final_vectors = torch.where(
            is_saddle.unsqueeze(-1),  # [b, l, n_gaussians, 1]
            random_vectors,
            sampled_vectors,
        )

        # Ensure all vectors are normalized
        final_vectors = F.normalize(final_vectors, dim=-1)

        # Create pseudo-targets and compute explicit regression loss per layer
        loss = {}
        for layer in range(num_layers):
            layer_centers = centers[:, layer]  # [b, n_gaussians, 3]
            layer_vectors = final_vectors[:, layer]  # [b, n_gaussians, 3]
            layer_distances = sampled_distances[:, layer]  # [b, n_gaussians]

            # Create pseudo-targets: current center + vector * distance (detached from gradients)
            pseudo_targets = (layer_centers + layer_vectors).detach()

            # Explicit L2 regression loss: minimize distance between center and pseudo-target
            center_loss = F.mse_loss(layer_centers, pseudo_targets, reduction="mean")

            # Direct distance sampling loss: minimize sampled distance to occupied surfaces
            distance_loss = torch.mean(layer_distances)

            loss[f"d{layer}/distance"] = distance_loss * self.distance_weight
            loss[f"d{layer}/center"] = center_loss * self.center_weight

        return {f"distance_field/{k}": v for k, v in loss.items()}

    def _compute_volume_regularization_loss(
        self,
        preds: MetaDict,
    ) -> dict[str, torch.Tensor]:
        """
        Compute volume regularization loss to prevent tiny gaussians.

        Penalizes gaussians whose volume (product of scales) is below min_volume.
        This encourages the model to use opacity for "turning off" gaussians
        rather than shrinking them to near-zero size.

        Args:
            preds: Gaussian predictions with scales [b, l, n_gaussians, 3]

        Returns:
            Dictionary of volume regularization losses per layer
        """
        scales = preds.scales  # [b, l, n_gaussians, 3]
        _, num_layers, _, _ = scales.shape

        loss = {}
        for layer in range(num_layers):
            layer_scales = scales[:, layer]  # [b, n_gaussians, 3]

            # Compute volume as product of scales
            volume = torch.prod(layer_scales, dim=-1)  # [b, n_gaussians]

            # Hinge loss: penalize volumes below min_volume
            volume_loss = F.relu(self.min_volume - volume).mean()

            loss[f"d{layer}/volume"] = volume_loss * self.volume_weight

        return {f"volume_reg/{k}": v for k, v in loss.items()}
