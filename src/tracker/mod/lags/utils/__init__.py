# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch

from .bbox import denormalize_bbox, normalize_bbox
from .instances import Instances
from .labels import build_class_list, build_label_map
from .projection import unproject_image_rays


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return torch.log(x.clamp(min=eps, max=1) / (1 - x).clamp(min=eps, max=1))
