# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, List, Mapping, Self

from omegaconf import OmegaConf

from ...config import utils
from ...config.registry import RegistryBaseType
from ...utils.types import Sample
from .registry import registry as transform
from .transform import Transform


def _build_transforms(transforms: List[OmegaConf]) -> List[Transform]:
    if transforms is None:
        return []

    return [transform.from_config(c) for c in transforms]


@transform.register
class Collection(Transform, RegistryBaseType):
    """
    Collection/list of transforms to allow grouping
    """

    def __init__(self, transforms: List[Transform]) -> None:
        self.transforms = transforms

    @classmethod
    # pylint: disable=arguments-differ
    def from_config(cls, conf: OmegaConf | Mapping[str, Any], *args, **kwargs) -> Self:
        conf_kwargs = utils.get_kwargs(conf)

        transforms = conf_kwargs.pop("transforms", None)
        transforms = _build_transforms(transforms)

        return cls(*args, transforms=transforms, **kwargs)

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "transforms": [t.hparams for t in self.transforms],
        }

    def apply(self, sample: Sample) -> Sample:
        for tx in self.transforms:
            sample = tx(sample)

        return sample
