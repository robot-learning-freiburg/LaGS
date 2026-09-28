# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Hierarchical multi-resolution occupancy loss for hierarchical gaussian processing.

This module provides multi-resolution supervision for hierarchical gaussian streams,
where each stream (coarse/medium/fine) is supervised at its natural resolution with
appropriately downsampled ground truth targets.

Provides both soft label (class distribution) and hard label (majority voting) variants.
"""

import torch
from torch import nn
from torch.nn import functional as F

from ....utils.torch import amp
from ....utils.types import MetaDict
from ..occupancy.gaussian import GaussianSemanticAggregator


class MultiResolutionOccupancyTargetSoft(nn.Module):
    """
    Downsample occupancy ground truth to multiple resolutions for hierarchical
    supervision using soft labels (class distributions).

    Uses class distribution targets (soft labels) for richer supervision signal.
    Preserves information about class uncertainty and boundary ambiguity.

    For hard label variant using majority voting, see MultiResolutionOccupancyTargetHard.
    """

    def __init__(
        self,
        target_resolutions: dict[str, list[int]],
        ignore_index: int = 255,
    ):
        """
        Initialize multi-resolution target generator.

        Args:
            target_resolutions: Dict mapping stream names to target resolutions.
                Example: {
                    'coarse': [50, 50, 4],
                    'medium': [100, 100, 8],
                    'fine': [200, 200, 16],
                }
            ignore_index: Index to ignore in ground truth (e.g., 255 for unlabeled).
        """
        super().__init__()

        self.target_resolutions = target_resolutions
        self.ignore_index = ignore_index

    def _downsample_to_class_distribution(
        self,
        semantics: torch.Tensor,  # [b, z, y, x]
        target_shape: tuple[int, int, int],  # (z_target, y_target, x_target)
        num_classes: int,
        mask: torch.Tensor | None = None,  # [b, z, y, x]
    ) -> torch.Tensor:
        """
        Downsample semantic labels to class distribution targets.

        Each target voxel contains the class distribution within its spatial region,
        providing richer supervision than hard labels.

        Args:
            semantics: Input semantic labels [b, z, y, x]
            target_shape: Target resolution (z, y, x)
            num_classes: Number of semantic classes (excluding background/free)
            mask: Optional validity mask [b, z, y, x]. Only valid voxels contribute.

        Returns:
            class_distributions: [b, z_target, y_target, x_target, num_classes]
                Soft probability distributions over classes
        """
        # pylint: disable=too-many-locals

        b, z_src, y_src, x_src = semantics.shape
        z_tgt, y_tgt, x_tgt = target_shape

        # Compute downsampling factors
        z_factor = z_src // z_tgt
        y_factor = y_src // y_tgt
        x_factor = x_src // x_tgt

        assert z_src == z_tgt * z_factor, "Source Z must be divisible by target Z"
        assert y_src == y_tgt * y_factor, "Source Y must be divisible by target Y"
        assert x_src == x_tgt * x_factor, "Source X must be divisible by target X"

        # Reshape to group voxels that map to same target voxel
        # [b, z_tgt, z_factor, y_tgt, y_factor, x_tgt, x_factor]
        semantics = semantics.view(b, z_tgt, z_factor, y_tgt, y_factor, x_tgt, x_factor)

        # Flatten the factors to get all source voxels per target voxel
        # [b, z_tgt, y_tgt, x_tgt, z_factor*y_factor*x_factor]
        semantics = semantics.permute(0, 1, 3, 5, 2, 4, 6).flatten(start_dim=4)

        # Also reshape mask if provided
        if mask is not None:
            mask = mask.view(b, z_tgt, z_factor, y_tgt, y_factor, x_tgt, x_factor)
            mask = mask.permute(0, 1, 3, 5, 2, 4, 6).flatten(start_dim=4)

        # Vectorized count of class occurrences (excluding ignore_index AND invalid voxels)
        # semantics: [b, z_tgt, y_tgt, x_tgt, grouped_size]

        # Create class indices tensor: [num_classes]
        class_indices = torch.arange(num_classes, device=semantics.device)
        # Reshape for broadcasting: [1, 1, 1, 1, 1, num_classes]
        class_indices = class_indices.view(1, 1, 1, 1, 1, num_classes)

        # Compare all classes at once: [b, z_tgt, y_tgt, x_tgt, grouped_size, num_classes]
        class_mask = semantics.unsqueeze(-1) == class_indices

        # Exclude ignore_index: [b, z_tgt, y_tgt, x_tgt, grouped_size, 1]
        class_mask = class_mask & (semantics != self.ignore_index).unsqueeze(-1)

        # Also exclude invalid voxels if mask provided
        if mask is not None:
            class_mask = class_mask & mask.unsqueeze(-1)

        # Sum over grouped_size dimension to get class counts
        # -> [b, z_tgt, y_tgt, x_tgt, num_classes]
        class_counts = class_mask.float().sum(dim=4)

        # Count valid voxels per target voxel (non-ignored AND mask-valid)
        valid_counts = class_counts.sum(dim=-1)  # [b, z_tgt, y_tgt, x_tgt]

        # Normalize to get probabilities (add epsilon to avoid division by zero)
        # [b, z_tgt, y_tgt, x_tgt, num_classes] / [b, z_tgt, y_tgt, x_tgt, 1]
        class_distributions = class_counts / (valid_counts.unsqueeze(-1) + 1e-8)

        return class_distributions

    def _downsample_mask_max_pool(
        self,
        mask: torch.Tensor,  # [b, z, y, x]
        target_shape: tuple[int, int, int],  # (z_target, y_target, x_target)
    ) -> torch.Tensor:
        """
        Downsample binary mask using max pooling.

        If ANY fine voxel in a region is valid, the coarse voxel is valid.
        This preserves supervision in regions with partial labels.

        Args:
            mask: Input binary mask [b, z, y, x]
            target_shape: Target resolution (z, y, x)

        Returns:
            downsampled_mask: [b, z_target, y_target, x_target]
        """
        # pylint: disable=too-many-locals

        if mask is None:
            return None

        b, z_src, y_src, x_src = mask.shape
        z_tgt, y_tgt, x_tgt = target_shape

        # Compute downsampling factors
        z_factor = z_src // z_tgt
        y_factor = y_src // y_tgt
        x_factor = x_src // x_tgt

        # Reshape and max pool
        # [b, z_tgt, z_factor, y_tgt, y_factor, x_tgt, x_factor]
        mask = mask.view(b, z_tgt, z_factor, y_tgt, y_factor, x_tgt, x_factor)

        # Permute and flatten: [b, z_tgt, y_tgt, x_tgt, z_factor*y_factor*x_factor]
        mask = mask.permute(0, 1, 3, 5, 2, 4, 6).flatten(start_dim=4)

        # Max pool: if any voxel in region is True, result is True
        mask = mask.any(dim=-1)  # [b, z_tgt, y_tgt, x_tgt]

        return mask

    def forward(
        self,
        semantics: torch.Tensor,  # [b, z, y, x]
        mask: torch.Tensor | None,  # [b, z, y, x]
        num_classes: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Generate multi-resolution soft targets for all streams.

        Args:
            semantics: Ground truth semantic labels at full resolution
            masks: Dict of binary masks at full resolution
            num_classes: Number of semantic classes

        Returns:
            Dict mapping stream names to their targets:
            {
                'coarse': {
                    'labels': [b, z, y, x, num_classes],
                    'masks': {mask_name: [b, z, y, x]},
                },
                'medium': {...},
                'fine': {...},
            }
        """
        targets_per_stream = {}

        for stream_name, target_resolution in self.target_resolutions.items():
            # Check if this is full resolution (no downsampling needed)
            if list(semantics.shape[1:]) == target_resolution:
                # Full resolution - use original targets
                # Convert to one-hot distribution
                # pylint: disable-next=not-callable
                labels = F.one_hot(semantics.long(), num_classes=num_classes).float()

                # Handle ignore_index AND invalid voxels by zeroing out
                ignore_mask = semantics == self.ignore_index
                labels[ignore_mask] = 0.0

                if mask is not None:
                    labels[~mask] = 0.0

                downsampled_mask = mask
            else:
                # Downsample to target resolution using validity mask
                labels = self._downsample_to_class_distribution(
                    semantics,
                    tuple(target_resolution),
                    num_classes,
                    mask=mask,
                )

                # Downsample masks
                downsampled_mask = self._downsample_mask_max_pool(
                    mask, tuple(target_resolution)
                )

            targets_per_stream[stream_name] = {
                "labels": labels,
                "mask": downsampled_mask,
            }

        return targets_per_stream


class HierarchicalSemanticGaussianLossSoft(nn.Module):
    """
    Per-stream gaussian loss with multi-resolution supervision.

    Each stream (e.g., coarse/medium/fine) has:
    - Its own target resolution (downsampled appropriately)
    - Its own scale multiplier for aggregation
    - Its own loss weight
    """

    def __init__(
        self,
        voxel_range: list[float],
        voxel_size: list[float],
        mask: str | None,
        streams: dict[str, dict],
        ignore_index: int = 255,
        class_weights: tuple[float] | None = None,
    ):
        """
        Initialize hierarchical gaussian loss.

        Args:
            voxel_range: Full voxel coordinate range
            voxel_size: Base voxel size (for finest resolution)
            mask: Name of mask to use for supervision (e.g., 'valid')
            streams: Dict mapping stream names to their configs:
                {
                    'coarse': {
                        'target_resolution': [50, 50, 4],
                        'weight': 0.3,
                        'scale_multiplier': 8.0,
                    },
                    'medium': {
                        'target_resolution': [100, 100, 8],
                        'weight': 0.5,
                        'scale_multiplier': 4.0,
                    },
                    'fine': {
                        'target_resolution': [200, 200, 16],
                        'weight': 1.0,
                        'scale_multiplier': 2.0,
                    },
                }
            ignore_index: Index to ignore in ground truth (e.g., 255 for unlabeled)
            class_weights: Optional per-class weights for loss
        """
        super().__init__()

        self.mask = mask if mask != "none" else None
        self.streams = streams

        # Create multi-resolution target generator
        target_resolutions = {
            name: config["target_resolution"] for name, config in streams.items()
        }
        self.target_generator = MultiResolutionOccupancyTargetSoft(
            target_resolutions=target_resolutions,
            ignore_index=ignore_index,
        )

        # Infer full resolution from voxel_range and voxel_size
        # voxel_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        # voxel_size: [x_size, y_size, z_size]
        # full_res: [z, y, x] to match target_resolution convention
        full_res = [
            int((voxel_range[5] - voxel_range[2]) / voxel_size[2]),  # nz
            int((voxel_range[4] - voxel_range[1]) / voxel_size[1]),  # ny
            int((voxel_range[3] - voxel_range[0]) / voxel_size[0]),  # nx
        ]

        # Create per-stream aggregators with appropriate scale multipliers
        self.aggregators = nn.ModuleDict()
        for stream_name, stream_config in streams.items():
            target_res = stream_config["target_resolution"]  # [z, y, x]
            scale_mult = stream_config["scale_multiplier"]

            # Compute voxel size for this resolution
            # target_res is [z, y, x], so scale_factors are [z, y, x]
            scale_factors = [full_res[i] / target_res[i] for i in range(3)]
            # voxel_size is [x, y, z], so we need to reorder scale_factors
            stream_voxel_size = [
                voxel_size[0] * scale_factors[2],  # x
                voxel_size[1] * scale_factors[1],  # y
                voxel_size[2] * scale_factors[0],  # z
            ]

            self.aggregators[stream_name] = GaussianSemanticAggregator(
                voxel_size=stream_voxel_size,
                voxel_range=voxel_range,
                scale_multiplier=scale_mult,
            )

        # Store class weights
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights, persistent=False)
        else:
            self.class_weights = None

    def _compute_soft_cross_entropy(
        self,
        preds: torch.Tensor,  # [b, l, n, num_classes] - aggregated predictions
        targets: torch.Tensor,  # [b, z, y, x, num_classes]
        mask: torch.Tensor | None,  # [b, z, y, x]
    ) -> torch.Tensor:
        """
        Compute soft cross-entropy loss with class distribution targets.

        Args:
            preds: Aggregated predictions over all classes (including free space).
                If mask was used during aggregation, shape is [b, l, n_valid, num_classes]
                Otherwise shape is [b, l, n_all, num_classes]
            targets: Class distribution targets (soft labels) over all classes
            mask: Optional binary mask for valid voxels. If provided, targets are filtered.
                NOTE: If mask was used during aggregation, preds are already filtered.

        Returns:
            loss: Scalar loss value
        """
        # Apply mask to targets if provided
        # NOTE: preds are already filtered if mask was used during aggregation
        if mask is not None:
            targets = targets[mask]

        # Flatten
        preds = preds.flatten(end_dim=-2)  # [b*l*n, num_classes]
        targets = targets.flatten(end_dim=-2)  # [n_valid, num_classes]

        # Ensure predictions sum to 1 and are in log space
        log_preds = torch.log(preds + 1e-8)

        # Soft cross-entropy: -sum(target * log(pred))
        loss = -(targets * log_preds).sum(dim=-1)

        return loss.mean()

    def forward(
        self,
        preds: dict[str, MetaDict],
        targets: MetaDict,
        masks: MetaDict,
    ) -> dict[str, torch.Tensor]:
        """
        Compute hierarchical gaussian loss across all streams.

        Args:
            preds: Dict mapping stream names to their predictions:
                {
                    'coarse': MetaDict with (logits, centers, scales, rotations, opacities),
                    'medium': ...,
                    'fine': ...,
                }
            targets: Ground truth targets (at full resolution)
            masks: Dict of binary masks at full resolution

        Returns:
            Dict of losses per stream and layer
        """
        # pylint: disable=too-many-locals

        # Get ground truth at full resolution
        semantics = targets.semantics  # [b, z, y, x]
        mask = masks[self.mask] if self.mask is not None else None

        # Infer number of classes from first stream's predictions
        first_stream_preds = next(iter(preds.values()))
        num_classes = first_stream_preds.logits.shape[-1] + 1  # +1 for free space

        # Generate multi-resolution targets
        targets_per_stream = self.target_generator(
            semantics=semantics,
            mask=mask,
            num_classes=num_classes,
        )

        # Compute loss per stream
        losses = {}

        for stream_name, stream_preds in preds.items():
            stream_config = self.streams[stream_name]
            stream_weight = stream_config["weight"]
            stream_targets = targets_per_stream[stream_name]

            # Get mask for this stream (downsampled to match aggregator resolution)
            stream_mask = stream_targets["mask"]

            # Aggregate predictions for this stream
            # Pass mask to aggregator for efficiency (skip invalid voxels)
            aggregated_preds, _, _ = self.aggregators[stream_name](
                logits=stream_preds.logits,
                centers=stream_preds.centers,
                scales=stream_preds.scales,
                rotations=stream_preds.rotations,
                opacities=stream_preds.opacities,
                mask=stream_mask,
            )  # [b, l, n_valid, num_classes] if mask provided, else [b, l, n_all, num_classes]

            # Compute loss per layer
            num_layers = aggregated_preds.shape[1]
            for layer in range(num_layers):
                layer_preds = aggregated_preds[:, layer]  # [b, n, num_classes]

                with torch.autocast("cuda", enabled=False):
                    # Upcast to float32 for numerical stability
                    layer_preds = amp.upcast(layer_preds, dtype=torch.float32)
                    layer_preds = torch.clamp(layer_preds, min=1e-6, max=1 - 1e-6)

                    # Compute soft cross-entropy
                    layer_loss = self._compute_soft_cross_entropy(
                        preds=layer_preds.unsqueeze(1),  # Add layer dim back
                        targets=stream_targets["labels"],
                        mask=stream_mask,
                    )

                # Store loss with stream and layer prefix
                loss_key = f"{stream_name}/d{layer}/ce"
                losses[loss_key] = layer_loss * stream_weight

        return losses


class MultiResolutionOccupancyTargetHard(nn.Module):
    """
    Downsample occupancy ground truth using majority voting (hard labels).

    Each downsampled voxel contains the most common class label within its spatial region.
    This is simpler than class distributions but loses information about uncertainty.
    """

    def __init__(
        self,
        target_resolutions: dict[str, list[int]],
        ignore_index: int = 255,
        tie_breaker: str = "min",
    ):
        """
        Initialize multi-resolution target generator with majority voting.

        Args:
            target_resolutions: Dict mapping stream names to target resolutions.
            ignore_index: Index to ignore in ground truth (e.g., 255 for unlabeled).
            tie_breaker: How to handle ties in majority voting:
                - 'min': Take lowest class index (deterministic, default)
                - 'random': Randomly select among tied classes
                - 'mask': Mark as ignore_index (no supervision for ambiguous regions)
        """
        super().__init__()

        self.target_resolutions = target_resolutions
        self.ignore_index = ignore_index
        self.tie_breaker = tie_breaker

        if tie_breaker not in ["min", "random", "mask"]:
            raise ValueError(
                f"tie_breaker must be 'min', 'random', or 'mask', got {tie_breaker}"
            )

    def _downsample_to_majority_vote(
        self,
        semantics: torch.Tensor,  # [b, z, y, x]
        target_shape: tuple[int, int, int],  # (z_target, y_target, x_target)
        num_classes: int,
        mask: torch.Tensor | None = None,  # [b, z, y, x]
    ) -> torch.Tensor:
        """
        Downsample semantic labels using majority voting.

        Each target voxel gets the most common class label within its spatial region.

        Args:
            semantics: Input semantic labels [b, z, y, x]
            target_shape: Target resolution (z, y, x)
            num_classes: Number of semantic classes (excluding background/free)
            mask: Optional validity mask [b, z, y, x]. Only valid voxels contribute.

        Returns:
            labels: [b, z_target, y_target, x_target] - Hard class labels
        """
        # pylint: disable=too-many-locals

        b, z_src, y_src, x_src = semantics.shape
        z_tgt, y_tgt, x_tgt = target_shape

        # Compute downsampling factors
        z_factor = z_src // z_tgt
        y_factor = y_src // y_tgt
        x_factor = x_src // x_tgt

        assert z_src == z_tgt * z_factor, "Source Z must be divisible by target Z"
        assert y_src == y_tgt * y_factor, "Source Y must be divisible by target Y"
        assert x_src == x_tgt * x_factor, "Source X must be divisible by target X"

        # Reshape to group voxels that map to same target voxel
        # [b, z_tgt, z_factor, y_tgt, y_factor, x_tgt, x_factor]
        semantics = semantics.view(b, z_tgt, z_factor, y_tgt, y_factor, x_tgt, x_factor)

        # Flatten the factors to get all source voxels per target voxel
        # [b, z_tgt, y_tgt, x_tgt, z_factor*y_factor*x_factor]
        semantics = semantics.permute(0, 1, 3, 5, 2, 4, 6).flatten(start_dim=4)

        # Also reshape mask if provided
        if mask is not None:
            mask = mask.view(b, z_tgt, z_factor, y_tgt, y_factor, x_tgt, x_factor)
            mask = mask.permute(0, 1, 3, 5, 2, 4, 6).flatten(start_dim=4)

        # Vectorized count of class occurrences (excluding ignore_index AND invalid voxels)
        # semantics: [b, z_tgt, y_tgt, x_tgt, grouped_size]

        # Create class indices tensor: [num_classes]
        class_indices = torch.arange(num_classes, device=semantics.device)
        # Reshape for broadcasting: [1, 1, 1, 1, 1, num_classes]
        class_indices = class_indices.view(1, 1, 1, 1, 1, num_classes)

        # Compare all classes at once: [b, z_tgt, y_tgt, x_tgt, grouped_size, num_classes]
        class_mask = semantics.unsqueeze(-1) == class_indices

        # Exclude ignore_index: [b, z_tgt, y_tgt, x_tgt, grouped_size, 1]
        class_mask = class_mask & (semantics != self.ignore_index).unsqueeze(-1)

        # Also exclude invalid voxels if mask provided
        if mask is not None:
            class_mask = class_mask & mask.unsqueeze(-1)

        # Sum over grouped_size dimension to get class counts
        # -> [b, z_tgt, y_tgt, x_tgt, num_classes]
        class_counts = class_mask.long().sum(dim=4)

        # Find majority class
        max_counts, majority_classes = class_counts.max(
            dim=-1
        )  # [b, z_tgt, y_tgt, x_tgt]

        # Handle ties based on tie_breaker strategy
        if self.tie_breaker == "random":
            # Find positions with ties
            ties_mask = (class_counts == max_counts.unsqueeze(-1)).sum(dim=-1) > 1

            if ties_mask.any():
                # Vectorized random tie breaking: add small random noise to counts
                # and recompute argmax for tied positions
                # Noise is small enough (< 1) to only affect ties, not change actual counts
                noise = torch.rand_like(class_counts.float()) * 0.9
                noisy_counts = class_counts.float() + noise

                # Re-compute argmax for tied positions only
                _, new_classes = noisy_counts.max(dim=-1)
                majority_classes[ties_mask] = new_classes[ties_mask]

        elif self.tie_breaker == "mask":
            # Mark ties as ignore_index
            ties_mask = (class_counts == max_counts.unsqueeze(-1)).sum(dim=-1) > 1
            majority_classes[ties_mask] = self.ignore_index

        # If tie_breaker == "min", argmax already returns the first (minimum) index

        # Mark voxels with no valid samples as ignore_index
        majority_classes[max_counts == 0] = self.ignore_index

        return majority_classes

    def _downsample_mask_max_pool(
        self,
        mask: torch.Tensor,  # [b, z, y, x]
        target_shape: tuple[int, int, int],  # (z_target, y_target, x_target)
    ) -> torch.Tensor:
        """
        Downsample binary mask using max pooling.

        If ANY fine voxel in a region is valid, the coarse voxel is valid.
        This preserves supervision in regions with partial labels.

        Args:
            mask: Input binary mask [b, z, y, x]
            target_shape: Target resolution (z, y, x)

        Returns:
            downsampled_mask: [b, z_target, y_target, x_target]
        """
        # pylint: disable=too-many-locals

        if mask is None:
            return None

        b, z_src, y_src, x_src = mask.shape
        z_tgt, y_tgt, x_tgt = target_shape

        # Compute downsampling factors
        z_factor = z_src // z_tgt
        y_factor = y_src // y_tgt
        x_factor = x_src // x_tgt

        # Reshape and max pool
        # [b, z_tgt, z_factor, y_tgt, y_factor, x_tgt, x_factor]
        mask = mask.view(b, z_tgt, z_factor, y_tgt, y_factor, x_tgt, x_factor)

        # Permute and flatten: [b, z_tgt, y_tgt, x_tgt, z_factor*y_factor*x_factor]
        mask = mask.permute(0, 1, 3, 5, 2, 4, 6).flatten(start_dim=4)

        # Max pool: if any voxel in region is True, result is True
        mask = mask.any(dim=-1)  # [b, z_tgt, y_tgt, x_tgt]

        return mask

    def forward(
        self,
        semantics: torch.Tensor,  # [b, z, y, x]
        mask: torch.Tensor | None,  # [b, z, y, x]
        num_classes: int,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """
        Generate multi-resolution hard label targets for all streams.

        Args:
            semantics: Ground truth semantic labels at full resolution
            masks: Dict of binary masks at full resolution
            num_classes: Number of semantic classes

        Returns:
            Dict mapping stream names to their targets:
            {
                'coarse': {
                    'labels': [b, z, y, x],
                    'masks': {mask_name: [b, z, y, x]},
                },
                'medium': {...},
                'fine': {...},
            }
        """
        targets_per_stream = {}

        for stream_name, target_resolution in self.target_resolutions.items():
            # Check if this is full resolution (no downsampling needed)
            if list(semantics.shape[1:]) == target_resolution:
                # Full resolution - use original targets
                labels = semantics.clone()

                downsampled_mask = mask
            else:
                # Downsample to target resolution using majority voting
                labels = self._downsample_to_majority_vote(
                    semantics,
                    tuple(target_resolution),
                    num_classes,
                    mask=mask,
                )

                # Downsample masks
                downsampled_mask = self._downsample_mask_max_pool(
                    mask, tuple(target_resolution)
                )

            targets_per_stream[stream_name] = {
                "labels": labels,
                "mask": downsampled_mask,
            }

        return targets_per_stream


class HierarchicalSemanticGaussianLossHard(nn.Module):
    """
    Per-stream gaussian loss with multi-resolution supervision using hard labels.

    Uses majority voting for downsampling and standard cross-entropy loss.
    Simpler than soft labels but loses information about class uncertainty.
    """

    def __init__(
        self,
        voxel_range: list[float],
        voxel_size: list[float],
        mask: str | None,
        streams: dict[str, dict],
        ignore_index: int = 255,
        tie_breaker: str = "min",
        class_weights: tuple[float] | None = None,
    ):
        """
        Initialize hierarchical gaussian loss with hard labels.

        Args:
            voxel_range: Full voxel coordinate range
            voxel_size: Base voxel size (for finest resolution)
            mask: Name of mask to use for supervision (e.g., 'valid')
            streams: Dict mapping stream names to their configs
            ignore_index: Index to ignore in ground truth (e.g., 255 for unlabeled)
            tie_breaker: How to handle ties in majority voting ('min', 'random', 'mask')
            class_weights: Optional per-class weights for loss
        """
        # pylint: disable=too-many-locals
        super().__init__()

        self.mask = mask if mask != "none" else None
        self.streams = streams

        # Create multi-resolution target generator with majority voting
        target_resolutions = {
            name: config["target_resolution"] for name, config in streams.items()
        }
        self.target_generator = MultiResolutionOccupancyTargetHard(
            target_resolutions=target_resolutions,
            ignore_index=ignore_index,
            tie_breaker=tie_breaker,
        )

        # Infer full resolution from voxel_range and voxel_size
        # voxel_range: [x_min, y_min, z_min, x_max, y_max, z_max]
        # voxel_size: [x_size, y_size, z_size]
        # full_res: [z, y, x] to match target_resolution convention
        full_res = [
            int((voxel_range[5] - voxel_range[2]) / voxel_size[2]),  # nz
            int((voxel_range[4] - voxel_range[1]) / voxel_size[1]),  # ny
            int((voxel_range[3] - voxel_range[0]) / voxel_size[0]),  # nx
        ]

        # Create per-stream aggregators with appropriate scale multipliers
        self.aggregators = nn.ModuleDict()
        for stream_name, stream_config in streams.items():
            target_res = stream_config["target_resolution"]  # [z, y, x]
            scale_mult = stream_config["scale_multiplier"]

            # Compute voxel size for this resolution
            # target_res is [z, y, x], so scale_factors are [z, y, x]
            scale_factors = [full_res[i] / target_res[i] for i in range(3)]
            # voxel_size is [x, y, z], so we need to reorder scale_factors
            stream_voxel_size = [
                voxel_size[0] * scale_factors[2],  # x
                voxel_size[1] * scale_factors[1],  # y
                voxel_size[2] * scale_factors[0],  # z
            ]

            self.aggregators[stream_name] = GaussianSemanticAggregator(
                voxel_size=stream_voxel_size,
                voxel_range=voxel_range,
                scale_multiplier=scale_mult,
            )

        # Store class weights
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", class_weights, persistent=False)
        else:
            self.class_weights = None

    def _compute_cross_entropy(
        self,
        preds: torch.Tensor,  # [b, l, n, num_classes] - aggregated predictions
        targets: torch.Tensor,  # [b, z, y, x]
        mask: torch.Tensor | None,  # [b, z, y, x]
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss with hard labels.

        Args:
            preds: Aggregated predictions over all classes (including free space).
                If mask was used during aggregation, shape is [b, l, n_valid, num_classes]
                Otherwise shape is [b, l, n_all, num_classes]
            targets: Hard class labels (indices into num_classes)
            mask: Optional binary mask for valid voxels. If provided, targets are filtered.
                NOTE: If mask was used during aggregation, preds are already filtered.

        Returns:
            loss: Scalar loss value
        """
        # Apply mask to targets if provided
        # NOTE: preds are already filtered if mask was used during aggregation
        if mask is not None:
            targets = targets[mask]

        # Flatten
        preds = preds.flatten(end_dim=-2)  # [b*l*n, num_classes]
        targets = targets.flatten()  # [n_valid]

        # Compute cross-entropy (ignore_index is handled automatically)
        return F.cross_entropy(
            preds,
            targets.long(),
            weight=self.class_weights,
            ignore_index=self.target_generator.ignore_index,
            reduction="mean",
        )

    def forward(
        self,
        preds: dict[str, MetaDict],
        targets: MetaDict,
        masks: MetaDict,
    ) -> dict[str, torch.Tensor]:
        """
        Compute hierarchical gaussian loss across all streams using hard labels.

        Args:
            preds: Dict mapping stream names to their predictions
            targets: Ground truth targets (at full resolution)
            masks: Dict of binary masks at full resolution

        Returns:
            Dict of losses per stream and layer
        """
        # pylint: disable=too-many-locals

        # Get ground truth at full resolution
        semantics = targets.semantics  # [b, z, y, x]
        mask = masks[self.mask] if self.mask is not None else None

        # Infer number of classes from first stream's predictions
        first_stream_preds = next(iter(preds.values()))
        num_classes = first_stream_preds.logits.shape[-1] + 1  # +1 for free space

        # Generate multi-resolution targets
        targets_per_stream = self.target_generator(
            semantics=semantics,
            mask=mask,
            num_classes=num_classes,
        )

        # Compute loss per stream
        losses = {}

        for stream_name, stream_preds in preds.items():
            stream_config = self.streams[stream_name]
            stream_weight = stream_config["weight"]
            stream_targets = targets_per_stream[stream_name]

            # Get mask for this stream (downsampled to match aggregator resolution)
            stream_mask = stream_targets["mask"]

            # Aggregate predictions for this stream
            # Pass mask to aggregator for efficiency (skip invalid voxels)
            aggregated_preds, _, _ = self.aggregators[stream_name](
                logits=stream_preds.logits,
                centers=stream_preds.centers,
                scales=stream_preds.scales,
                rotations=stream_preds.rotations,
                opacities=stream_preds.opacities,
                mask=stream_mask,
            )  # [b, l, n_valid, num_classes] if mask provided, else [b, l, n_all, num_classes]

            # Compute loss per layer
            num_layers = aggregated_preds.shape[1]
            for layer in range(num_layers):
                layer_preds = aggregated_preds[:, layer]  # [b, n, num_classes]

                with torch.autocast("cuda", enabled=False):
                    # Upcast to float32 for numerical stability
                    layer_preds = amp.upcast(layer_preds, dtype=torch.float32)
                    layer_preds = torch.clamp(layer_preds, min=1e-6, max=1 - 1e-6)

                    # Compute cross-entropy
                    layer_loss = self._compute_cross_entropy(
                        preds=layer_preds.unsqueeze(1),  # Add layer dim back
                        targets=stream_targets["labels"],
                        mask=stream_mask,
                    )

                # Store loss with stream and layer prefix
                loss_key = f"{stream_name}/d{layer}/ce"
                losses[loss_key] = layer_loss * stream_weight

        return losses
