# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Loading unified-format predictions written by ``occupancy.PanopticOccupancy``.

Predictions live at ``<pred_root>/<sequence_id>/<sample_id>.npz`` and store
``pano_sem``/``pano_inst`` in ``[Z, Y, X]`` order with ``-1`` = no instance
(identical to the GT convention, so no transpose/reindex). When the run had
``store_gaussians=true`` the same npz also carries the ``gaussian_*`` arrays,
which are handled in the sibling ``gaussians`` module.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Prediction path layout (<pred_root>/<sequence_id>/<sample_id>.npz) is shared with
# the eval tooling; re-export it so callers can use ocr.predictions.prediction_path.
from scripts.common.datasets import prediction_path
from tracker.utils.types import MetaDict

__all__ = ["prediction_path", "load_prediction", "build_pred_occupancy"]


def load_prediction(path: str | Path) -> Optional[dict]:
    """Load a prediction npz, or ``None`` if it does not exist.

    Returns a dict with ``semantics``/``instance_ids`` ([Z, Y, X] long tensors)
    and the raw ``npz`` handle (for the optional gaussian arrays).
    """
    path = Path(path)
    if not path.exists():
        return None

    data = np.load(path)
    return {
        "semantics": torch.from_numpy(data["pano_sem"]).long(),
        "instance_ids": torch.from_numpy(data["pano_inst"]).long(),
        "npz": data,
    }


def build_pred_occupancy(pred: dict, gt_sample) -> MetaDict:
    """Wrap a prediction as an occupancy mapping the renderer understands.

    Reuses the GT sample's observation masks so predictions can be gated by the
    same region (``mask_type`` / visual filter) as ground truth.
    """
    occ = MetaDict()
    occ.semantics = pred["semantics"]
    occ.instance_ids = pred["instance_ids"]

    labels = getattr(gt_sample, "labels", None)
    if labels is not None and "occupancy" in labels and "masks" in labels.occupancy:
        occ.masks = labels.occupancy.masks

    return occ
