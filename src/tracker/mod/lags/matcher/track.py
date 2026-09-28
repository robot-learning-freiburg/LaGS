# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping, Self

import torch
from omegaconf import OmegaConf

from .... import config
from ....config.registry import FromConfigMixin
from ....utils.types import MetaDict
from .. import assigners
from ..assigners import Assigner
from . import utils
from .base import INDEX_DISCONTINUED, INDEX_UNMATCHED, Matcher, registry


@registry.register
class TrackMatcher(Matcher, FromConfigMixin):
    """
    Matcher for end-to-end tracking.

    This matcher is used to match predictions and targets in a tracking
    scenario. It uses a given assignment algorithm (e.g., Hungarian) to find
    the best matches between predictions and targets. Decoder layers are
    matched independently.

    The matcher is designed to be used in a tracking context, where
    predictions and targets are associated with instance IDs that
    are unique across frames. Previous assignments are kept across future
    frames. The matcher updates the instance IDs of the predictions based on
    the matches found in the current frame.
    """

    @classmethod
    def from_config(cls, conf: OmegaConf | Mapping[str, Any], *args, **kwargs) -> Self:
        conf = config.utils.copy(conf, readonly=False)

        assigner = conf.pop("assigner")
        assigner = assigners.build(assigner)

        conf_kwargs = config.utils.get_kwargs(conf)
        return cls(*args, **kwargs, **conf_kwargs, assigner=assigner)

    def __init__(self, assigner: Assigner) -> None:
        super().__init__()

        self.assigner = assigner

    @torch.no_grad()
    def propagate_tracked(
        self,
        pred_instance_ids: torch.Tensor,
        target_instance_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update tracked instance IDs from the previous frame to the current frame.

        Args:
            pred_instance_ids (torch.Tensor): Instance IDs from the previous
                frame. Assigns each prediction (index) to a ground truth
                instance ID if it has been matched previously. Negative values
                indicate that no match has been found yet for that specific
                prediction. Tensor of shape [num_preds].
            target_instance_ids (torch.Tensor): Ground truth instance IDs for the
                current frame. Assigns each frame-local ground truth index to
                an instance ID that is unique across frames. Tensor of shape
                [num_targets].

        Returns:
            torch.Tensor: Frame-local ground-truth indices for all predictions.
                Each prediction (index) is assigned to the (frame-local) index
                of the ground truth label it has been matched to, based on the
                instance IDs in pred_instance_ids. Negative values indicate
                that no match has been found for that specific prediction.
                Specifically: Predictions that have not been matched according
                to pred_instance_ids are assigned the value `UNMATCHED` (-1).
                Predictions that have been matched to a ground instance
                peviously, but are no longer present in the current frame, are
                assigned the value `DISCONTINUED` (-2).

                Tensor of shape [num_preds].
        """
        # build an assignment matrix, ensure that only valid targets are matched
        match = pred_instance_ids[:, None] == target_instance_ids[None, :]

        # get the indices of the matched targets
        valid, indices = torch.max(match, dim=-1)

        # set unmatched indices to UNMATCHED, discontinued indices to DISCONTINUED
        indices = torch.where(valid, indices, INDEX_DISCONTINUED)
        indices = torch.where(pred_instance_ids >= 0, indices, INDEX_UNMATCHED)

        return indices

    @torch.no_grad()
    def get_unmatched_preds_mask(
        self, assigned_gt_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the mask of the unmatched predictions.

        Args:
            assigned_gt_indices (torch.Tensor): Indices of the matched targets
                for each prediction. Tensor of shape [num_preds].

        Returns:
            torch.Tensor: Mask of the unmatched predictions. Entries are True
                if the corresponding prediction is unmatched, and False
                otherwise. Tensor of shape [num_preds].
        """
        return utils.get_unmatched_preds_mask(assigned_gt_indices)

    @torch.no_grad()
    def get_unmatched_targets_mask(
        self,
        assigned_gt_indices: torch.Tensor,
        num_targets: int,
    ) -> torch.Tensor:
        """
        Get the mask of the unmatched targets.

        Args:
            assigned_gt_indices (torch.Tensor): Indices of the matched targets
                for each prediction. Tensor of shape [num_preds].
            num_targets (int): Number of targets in the current frame.
                Note that assignet_gt_indices < num_targets must hold.

        Returns:
            torch.Tensor: Mask of the unmatched targets. Entries are True
                if the corresponding target is unmatched, and False otherwise.
                Tensor of shape [num_targets].
        """
        return utils.get_unmatched_targets_mask(assigned_gt_indices, num_targets)

    @torch.no_grad()
    def get_unmatched_preds(
        self,
        preds: MetaDict,
        unmatched: torch.Tensor,
    ) -> MetaDict:
        """
        Get the unmatched predictions.

        Args:
            preds (MetaDict): The predictions to filter. Note: predictions are
                not batched.
            unmatched (torch.Tensor): Mask of the unmatched predictions. Entries
                are True if the corresponding prediction is unmatched, and False
                otherwise. Tensor of shape [num_preds].

        Returns:
            MetaDict: The unmatched predictions.
        """
        return utils.filter_preds(preds, unmatched)

    @torch.no_grad()
    def get_unmatched_targets(
        self,
        targets: MetaDict,
        unmatched: torch.Tensor,
    ) -> MetaDict:
        """
        Get the unmatched targets.

        Args:
            targets (MetaDict): The targets to filter. Note: targets are not
                batched.
            unmatched (torch.Tensor): Mask of the unmatched targets. Entries
                are True if the corresponding target is unmatched, and False
                otherwise. Tensor of shape [num_targets].

        Returns:
            MetaDict: The unmatched targets.
        """
        return utils.filter_targets(targets, unmatched)

    @torch.no_grad()
    def split_layer_preds(
        self,
        preds: MetaDict,
    ) -> list[MetaDict]:
        """
        Split the predictions into a list of predictions per layer.

        Args:
            preds (MetaDict): The predictions to split. Note: predictions are
                not batched.

        Returns:
            list[MetaDict]: A list of predictions per layer.
        """
        return utils.split_layer_preds(preds)

    @torch.no_grad()
    def match_sample(self, preds: MetaDict, targets: MetaDict) -> torch.Tensor:
        num_targets = targets.instance_ids.shape[0]

        # propagate the tracked instance IDs from the previous frame and map
        # them to local-frame ground truth indices (shape: [num_preds])
        assigned_idx = self.propagate_tracked(
            pred_instance_ids=preds.instance_ids,
            target_instance_ids=targets.instance_ids,
        )

        # get predictions per layer
        preds = self.split_layer_preds(preds)
        num_layers = len(preds)

        # get mask for the unmatched predictions and targets
        preds_mask = self.get_unmatched_preds_mask(assigned_idx)
        targets_mask = self.get_unmatched_targets_mask(assigned_idx, num_targets)

        # return if there are no unmatched targets or preds remaining
        if not preds_mask.any() or not targets_mask.any():
            return assigned_idx.expand(num_layers, -1).clone()

        # get the actual unmatched predictions and targets
        unmatched_preds = [self.get_unmatched_preds(p, preds_mask) for p in preds]
        unmatched_targets = self.get_unmatched_targets(targets, targets_mask)

        # get the indices of the unmatched targets
        unmatched_target_indices = torch.arange(num_targets, device=targets_mask.device)
        unmatched_target_indices = unmatched_target_indices[targets_mask]

        # perform assignment independently for each layer
        assigned_idx = assigned_idx.expand(num_layers, -1)
        assigned_idx = assigned_idx.clone()

        for layer, layer_preds in enumerate(unmatched_preds):
            layer_assignment = self.assigner(layer_preds, unmatched_targets)

            # map the assigned indices to the full set of targets
            valid = layer_assignment >= 0
            layer_assignment[valid] = unmatched_target_indices[layer_assignment[valid]]
            layer_assignment[~valid] = INDEX_UNMATCHED

            # update the assigned indices
            assigned_idx[layer, preds_mask] = layer_assignment

        return assigned_idx  # [num_layers, num_preds]


@registry.register
class ConsistentTrackMatcher(TrackMatcher):
    """
    Layer-consistent matcher for end-to-end tracking.

    This matcher is used to match predictions and targets in a tracking
    scenario. It uses a Hungarian algorithm to find the best matches
    between predictions and targets based on the provided assigner.

    This is an extension of TrackMatcher. Specifically, ConsistentTrackMatcher
    performs assignment from the last decoder layer to the first, and retains
    found assignments across layers. In contrast, TrackMatcher performs
    assignments independently for each layer, which might lead to conflicting
    assignments across layers.
    """

    @torch.no_grad()
    def match_sample(self, preds: MetaDict, targets: MetaDict) -> torch.Tensor:
        num_tgt = targets.instance_ids.shape[0]

        # propagate the tracked instance IDs from the previous frame and map
        # them to local-frame ground truth indices (shape: [num_preds])
        assigned_idx = self.propagate_tracked(
            pred_instance_ids=preds.instance_ids,
            target_instance_ids=targets.instance_ids,
        )

        # get predictions per layer
        preds = self.split_layer_preds(preds)
        num_layers = len(preds)

        # perform assignment from last to first layer
        for layer_preds in reversed(preds):
            # get mask for the unmatched predictions and targets
            preds_mask = self.get_unmatched_preds_mask(assigned_idx)
            targets_mask = self.get_unmatched_targets_mask(assigned_idx, num_tgt)

            # return if there are no unmatched targets or preds remaining
            if not preds_mask.any() or not targets_mask.any():
                break

            # get the actual unmatched predictions and targets
            unmatched_preds = self.get_unmatched_preds(layer_preds, preds_mask)
            unmatched_targets = self.get_unmatched_targets(targets, targets_mask)

            # get the indices of the unmatched targets
            unmatched_target_indices = torch.arange(num_tgt, device=targets_mask.device)
            unmatched_target_indices = unmatched_target_indices[targets_mask]

            layer_assignment = self.assigner(unmatched_preds, unmatched_targets)

            # map the assigned indices to the full set of targets
            valid = layer_assignment >= 0
            layer_assignment[valid] = unmatched_target_indices[layer_assignment[valid]]
            layer_assignment[~valid] = INDEX_UNMATCHED

            # update the assigned indices
            assigned_idx[preds_mask] = layer_assignment

        # expand the assigned indices to all layers
        return assigned_idx.expand(num_layers, -1).clone()  # [num_layers, num_preds]
