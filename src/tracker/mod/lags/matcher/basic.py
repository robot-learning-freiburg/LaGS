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
from .base import INDEX_UNMATCHED, Matcher, registry


@registry.register
class BasicMatcher(Matcher, FromConfigMixin):
    """
    A basic matcher for 3D object detection.

    It uses a given assignment algorithm (e.g., Hungarian) to find the best
    matches between predictions and targets. Decoder layers are matched
    independently.
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
    def assign(self, preds: MetaDict, targets: MetaDict) -> torch.Tensor:
        assignment = self.assigner(preds, targets)
        assignment[assignment < 0] = INDEX_UNMATCHED

        return assignment

    @torch.no_grad()
    def match_sample(self, preds: MetaDict, targets: MetaDict) -> torch.Tensor:
        # get predictions per layer
        preds = self.split_layer_preds(preds)

        # perform assignment independently for each layer
        assignments = [self.assign(p, targets) for p in preds]
        assignments = torch.stack(assignments, dim=0)

        return assignments  # [num_layers, num_preds]


@registry.register
class ConsistentBasicMatcher(BasicMatcher):
    """
    A basic matcher for 3D object detection.

    It uses a given assignment algorithm (e.g., Hungarian) to find the best
    matches between predictions and targets.

    In contrast to BasicMatcher, decoder layers are matched from last to first,
    updating assingments to ensure they are consistent across all layers. This
    is done to avoid potentially conflicting assignments across different layers.
    """

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
    def match_sample(self, preds: MetaDict, targets: MetaDict) -> torch.Tensor:
        num_tgt = utils.infer_num_targets(targets)

        # get predictions per layer
        preds = self.split_layer_preds(preds)
        num_layers = len(preds)

        # We explicitly perform assignment for the last layer first, so that we
        # don't have to infer the shape of the predictions ourselves here (and
        # avoid a small bit of unnecessary work).
        assignment = self.assign(preds[-1], targets)

        # perform assignment from second-to-last to first layer
        for layer_preds in reversed(preds[:-1]):
            # get mask for the unmatched predictions and targets
            preds_mask = self.get_unmatched_preds_mask(assignment)
            targets_mask = self.get_unmatched_targets_mask(assignment, num_tgt)

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
            assignment[preds_mask] = layer_assignment

        # expand the assigned indices to all layers
        return assignment.expand(num_layers, -1).clone()  # [num_layers, num_preds]
