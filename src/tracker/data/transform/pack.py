# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Mapping

import torch
from omegaconf import OmegaConf

from ...utils.types import MetaArray, MetaDict, Sample, TensorArray
from .registry import registry as transform
from .transform import Transform


@transform.register
class CollectMultiFrameImages(Transform):
    """
    Collect images from a multi-frame sample and stack them together.
    """

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {"type": f"{self.__class__.__name__}"}

    def apply(self, sample: Sample) -> Sample:
        imgs = [img.data for img in sample.images]

        transforms = sample.images[0].meta.transforms.keys()
        transforms = {
            k: TensorArray.stack([img.meta.transforms[k] for img in sample.images])
            for k in transforms
        }

        norm = {
            "mean": torch.stack([img.meta.norm.mean for img in sample.images]),
            "std": torch.stack([img.meta.norm.std for img in sample.images]),
        }

        meta = {
            "channel": MetaArray.stack([img.meta.channel for img in sample.images]),
            "timestamp": torch.stack([img.meta.timestamp for img in sample.images]),
            "transforms": MetaDict(transforms),
            "norm": MetaDict(norm),
        }

        images = {
            "data": torch.stack(imgs, dim=0),
            "meta": MetaDict(meta),
        }

        sample.images = MetaDict(images)

        return sample


@transform.register
class Drop(Transform):
    """
    Drop the given keys from the sample.
    """

    def __init__(self, keys: list[str]) -> None:
        super().__init__()

        if OmegaConf.is_config(keys):
            keys = OmegaConf.to_container(keys, resolve=True)

        self.keys = keys

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "keys": self.keys,
        }

    def apply(self, sample: Sample) -> Sample:

        def _drop(d: Any, key: str):
            # if the key points to a list, apply this to all elements in the list
            if isinstance(d, list):
                for elem in d:
                    _drop(elem, key)

                return

            key = key.split(".", maxsplit=1)

            # if the key points to a local element, delete it
            if len(key) == 1:
                if key[0] in d:
                    del d[key[0]]

            # if the key points to a nested element, recurse into it
            elif key[0] in d:
                _drop(d[key[0]], key[1])

        # drop the keys from the sample
        for key in self.keys:
            _drop(sample, key)

        return sample
