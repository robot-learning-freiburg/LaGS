# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping

from ...utils.types import Sample
from .registry import registry as transform
from .transform import Transform


# pylint: disable=too-few-public-methods
@transform.register
class Identity(Transform):
    """
    Identity transform. Does literally nothing.
    """

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {"type": f"{self.__class__.__name__}"}

    def apply(self, sample: Sample) -> Sample:
        return sample
