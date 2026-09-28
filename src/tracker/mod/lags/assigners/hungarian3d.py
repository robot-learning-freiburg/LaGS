# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping, Self, Sequence

import scipy
import torch
from omegaconf import OmegaConf
from torch import nn

from .... import config
from ....config.registry import FromConfigMixin
from ....utils.types import MetaDict, infer_device
from ..utils import denormalize_bbox
from . import costs
from .base import Assigner, registry
from .costs import MatchingCost


def infer_num_preds(preds: MetaDict) -> int | None:
    if "boxes" in preds:
        return preds.boxes.shape[0]

    if "class_scores" in preds:
        return preds.class_scores.shape[0]

    raise ValueError("Could not infer number of predictions.")


@registry.register
class HungarianAssigner3D(Assigner, FromConfigMixin):
    @classmethod
    def from_config(cls, conf: OmegaConf | Mapping[str, Any], *args, **kwargs) -> Self:
        conf_kwargs = config.utils.get_kwargs(conf)

        cost_fns = conf_kwargs.pop("costs")
        cost_fns = [costs.build(cfg) for cfg in cost_fns]

        return cls(*args, cost_fns=cost_fns, **kwargs, **conf_kwargs)

    def __init__(self, cost_fns: Sequence[MatchingCost]) -> None:
        super().__init__()

        self.cost_fns = nn.ModuleList(cost_fns)

    @torch.no_grad()
    def forward(
        self,
        preds: MetaDict,
        targets: MetaDict,
    ) -> torch.Tensor:
        device = infer_device(preds)
        num_pred = infer_num_preds(preds)

        # initialize to "no assignment"
        indices = torch.full((num_pred,), -1, device=device, dtype=torch.long)

        # denormalize the predicted boxes
        if "boxes" in preds:
            preds = preds.copy()
            preds.boxes = denormalize_bbox(preds.boxes)

        # compute matching costs
        cost = 0.0
        for cost_fn in self.cost_fns:
            cost += cost_fn(preds, targets)

        if isinstance(cost, float) or cost.numel() == 0:
            return indices

        # ensure the cost is finite
        assert torch.isfinite(cost).all(), "Cost matrix contains non-finite values."

        # perform Hungarian matching on CPU using linear_sum_assignment from scipy
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost.cpu().numpy())
        row_ind = torch.from_numpy(row_ind).to(device=device)
        col_ind = torch.from_numpy(col_ind).to(device=device)

        # assign matches
        indices[row_ind] = col_ind

        return indices
