# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import random
from typing import Any, Callable, Mapping, Self

from omegaconf import OmegaConf

from ... import config
from ...config.registry import RegistryBaseType
from ...utils import expr
from ...utils.types import Sample
from ..transform.collection import _build_transforms
from .registry import registry as transform
from .transform import Transform


class Condition:
    """
    Helper class to evaluate a condition expression string based on a sample.
    """

    def __init__(self, condition: str) -> None:
        self.condition = condition

    def __call__(self, sample: Sample) -> bool:
        args = {
            "epoch": sample.meta.epoch,
            # 0-based frame position within a multi-frame window, set by
            # ``LoadMultiFrameData``; defaults to 0 for single-frame / post-collate
            # samples so conditions that don't reference ``{frame}`` are unaffected.
            "frame": sample.meta.get("frame", 0),
        }

        return bool(expr.evaluate(self.condition, args))

    def __str__(self) -> str:
        return self.condition


@transform.register
class Conditional(Transform, RegistryBaseType):
    """
    Conditional transform that applies a list of transforms based on a
    condition.
    """

    @classmethod
    # pylint: disable=arguments-differ
    def from_config(cls, conf: OmegaConf | Mapping[str, Any], *args, **kwargs) -> Self:
        conf_kwargs = config.utils.get_kwargs(conf)

        transforms = conf_kwargs.pop("transforms")
        transforms = _build_transforms(transforms)

        condition = conf_kwargs.pop("condition")
        condition = Condition(condition)

        return cls(
            *args,
            condition=condition,
            transforms=transforms,
            **conf_kwargs,
            **kwargs,
        )

    def __init__(self, condition: Callable, transforms: list[Transform]) -> None:
        self.condition = condition
        self.transforms = transforms

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"transform.{self.__class__.__name__}",
            "condition": str(self.condition),
            "transforms": [t.hparams for t in self.transforms],
        }

    def apply(self, sample: Sample) -> Sample:
        if not self.condition(sample):
            return sample

        for tx in self.transforms:
            sample = tx(sample)

        return sample


@transform.register
class Random(Transform, RegistryBaseType):
    """
    Probabilistic transform that applies a list of transforms based on a given probability.
    """

    @classmethod
    # pylint: disable=arguments-differ
    def from_config(cls, conf: OmegaConf | Mapping[str, Any], *args, **kwargs) -> Self:
        conf_kwargs = config.utils.get_kwargs(conf)

        transforms = conf_kwargs.pop("transforms")
        transforms = _build_transforms(transforms)

        probability = conf_kwargs.pop("probability", 1.0)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1.")

        return cls(
            *args,
            probability=probability,
            transforms=transforms,
            **conf_kwargs,
            **kwargs,
        )

    def __init__(self, probability: float, transforms: list[Transform]) -> None:
        self.probability = probability
        self.transforms = transforms

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"transform.{self.__class__.__name__}",
            "probability": self.probability,
            "transforms": [t.hparams for t in self.transforms],
        }

    def apply(self, sample: Sample) -> Sample:
        if random.random() > self.probability:
            return sample

        for tx in self.transforms:
            sample = tx(sample)

        return sample
