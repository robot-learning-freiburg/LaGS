# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, List, Mapping, Union

import lightning as L
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from .. import config, utils
from ..utils.types import RefCache
from . import dataset

log = utils.log.get_logger(__name__)


class DataModule(L.LightningDataModule):
    @classmethod
    def from_config(cls, conf: OmegaConf | Mapping[str, Any]):
        return DataModule(config.utils.as_config(conf))

    def __init__(self, conf: OmegaConf):
        super().__init__()

        self.config = conf
        self.loader_args = conf.loader

        # Note: The batch_size needs to be modifiable property here or in
        #       hparams for some internal pytorch lightning magic... So set it
        #       here and get/override it when creating the train dataloader
        #       below.
        self.batch_size = self.loader_args.train.batch_size

        self.datasets = RefCache()

    def _load_dataset(self, stage: str) -> Union[Dataset, List[Dataset]]:
        log.info("building dataset for stage '%s'", stage)

        conf = self.config.source.get(stage, None)
        if conf is None:
            return None

        if isinstance(conf, list):
            return [dataset.build(c) for c in conf]

        return dataset.build(conf)

    def setup(self, stage: str):
        if stage == "fit":
            self.datasets.acquire_or_store("train", self._load_dataset, "train")
            self.datasets.acquire_or_store("val", self._load_dataset, "val")
        elif stage == "validate":
            self.datasets.acquire_or_store("val", self._load_dataset, "val")
        elif stage in ["test", "predict"]:
            self.datasets.acquire_or_store(stage, self._load_dataset, stage)
        else:
            raise ValueError(f"unknown stage '{stage}'")

    def teardown(self, stage: str):
        if stage == "fit":
            self.datasets.release("train")
            self.datasets.release("val")
        elif stage == "validate":
            self.datasets.release("val")
        elif stage in ["test", "predict"]:
            self.datasets.release(stage)
        else:
            raise ValueError(f"unknown stage '{stage}'")

    def _build_dataloader(self, data, **kwargs):
        def _build(data, **kwargs):
            data.set_epoch(self.trainer.current_epoch)
            return StatefulDataLoader(data, **kwargs)

        if isinstance(data, Dataset):
            return _build(data, **kwargs)

        return [_build(x, **kwargs) for x in data]

    def train_dataloader(self):
        kwargs = {"batch_size": self.batch_size}
        kwargs = OmegaConf.merge(self.loader_args.train, kwargs)

        return self._build_dataloader(self.datasets["train"], **kwargs)

    def val_dataloader(self):
        return self._build_dataloader(self.datasets["val"], **self.loader_args.val)

    def test_dataloader(self):
        return self._build_dataloader(self.datasets["test"], **self.loader_args.test)

    def predict_dataloader(self):
        return self._build_dataloader(
            self.datasets["predict"], **self.loader_args.predict
        )


def build(conf: OmegaConf):
    return DataModule.from_config(conf)
