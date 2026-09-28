# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from abc import abstractmethod
from typing import Any, Dict

from torchmetrics import Metric

from ...utils.types import Sample


class ValidationMetrics(Metric):
    @abstractmethod
    # pylint: disable-next=arguments-differ
    def update(self, sample: Sample, preds: Any, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        pass


class EmptyValidationMetrics(ValidationMetrics):
    # pylint: disable=useless-parent-delegation
    def __init__(self):
        super().__init__()

    # pylint: disable-next=arguments-differ
    def update(self, sample: Sample, preds: Any, *args, **kwargs) -> None:
        pass

    def compute(self) -> Any:
        return {}
