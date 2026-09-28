# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import torch.distributed as dist
from lightning import LightningModule, Trainer
from omegaconf import OmegaConf

from ... import utils
from ...utils.types.metadict import MetaDict
from ...utils.types.sample import Sample
from .. import dataset
from ..dataset.nuscenes import _parse_version_and_split
from ..validation.nuscenes import DetectionBox, Prediction, TrackingBox, _get_label_maps
from .registry import registry
from .writer import PredictionWriter

log = utils.log.get_logger(__name__)


CLASSES_DETECTION = (
    "car",
    "truck",
    "bus",
    "trailer",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "construction_vehicle",
    "traffic_cone",
    "barrier",
)

CLASSES_TRACKING = (
    "car",
    "truck",
    "bus",
    "trailer",
    "motorcycle",
    "bicycle",
    "pedestrian",
)


@registry.register(namespace="nuscenes")
# pylint: disable=too-many-instance-attributes
class Detection(PredictionWriter):
    preds: List[Prediction]

    def __init__(
        self,
        root: str | Path,
        split: str,
        labels,
        meta: Dict[str, Any] | OmegaConf,
        output_path: str | Path,
        ref_channel: str = "LIDAR_TOP",
        ref_frame: Literal["sensor", "ego"] = "sensor",
        classes: Sequence[str] = CLASSES_DETECTION,
    ):
        super().__init__()

        self.output_path = Path(output_path)
        self.ref_channel = ref_channel
        self.ref_frame = ref_frame
        self.classes = classes

        if OmegaConf.is_dict(meta):
            self.meta = OmegaConf.to_container(meta, resolve=True)
        else:
            self.meta = meta

        # parse NuScenes dataset version and split
        self.version, self.split = _parse_version_and_split(split)

        # load/build label maps
        self.label_maps = _get_label_maps(labels)

        # we only actually compute the metric on rank 0, so skip loading stuff
        # that we don't need
        if utils.mp.rank == 0:
            # load data object: we need that for the sample transforms
            self.data = dataset.nuscenes.acquire(self.version, root)

        self.preds = []

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: MetaDict,
        batch: Sample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for b in range(batch.batch_size):
            # filter out invalid samples/padding
            if not batch.is_valid[b]:
                continue

            # extract sample and move it to the cpu
            token = batch.meta[b].sample_id
            boxes = outputs.boxes.get(b).detach().to(device="cpu", copy=True)
            labels = outputs.labels.get(b).detach().to(device="cpu", copy=True)
            scores = outputs.scores.get(b).detach().to(device="cpu", copy=True)

            # store predictions for this sample
            pred = Prediction(token, boxes, labels, scores, None)
            self.preds.append(pred)

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # gather predictions from all processes
        preds = self._sync_dist_preds()

        # clear the list, we now have them locally on rank 0
        self.preds = []

        # if predictions are available (rank 0 or not distributed), process and store them
        if preds is not None:
            self._serialize_preds(preds)

    def _sync_dist_preds(self, dst_rank=0) -> List[Prediction] | None:
        use_dist = dist.is_available() and dist.is_initialized()

        # if we are not running distributedly, just return our predictions
        if not use_dist:
            return self.preds

        # gather predictions across processes
        rank, world_size = dist.get_rank(), dist.get_world_size()
        synced = [[] for _ in range(world_size)] if rank == dst_rank else None

        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            # Note: This will emit a warning that pickle is unsafe... we take
            # not of that and ignore it.
            dist.gather_object(self.preds, synced, dst=dst_rank)

        # we only have predictions on the target rank, return None for all others
        if rank != dst_rank:
            return None

        # reduce to a simple list
        return [pred for preds in synced for pred in preds]

    def _serialize_preds(self, preds: List[Prediction]):
        log.info("post-processing predictions for %s samples", len(preds))

        # convert predictions into boxes and apply required transforms
        results = self._extract_boxes(preds)

        # set up submission structure
        submission = {
            "meta": self.meta,
            "results": results,
        }

        # save submission data
        log.info("saving predictions to '%s'", self.output_path)
        with open(self.output_path, "w", encoding="utf-8") as fd:
            json.dump(submission, fd)

    def _extract_boxes(self, preds: List[Prediction]) -> Dict[str, Dict[str, Any]]:
        results = defaultdict(list)

        for pred in preds:
            # extract boxes from the predictions
            boxes = DetectionBox.from_preds(
                preds=pred,
                data=self.data,
                label_maps=self.label_maps,
                ref_channel=self.ref_channel,
                ref_frame=self.ref_frame,
            )

            # filter boxes by classes in nuscenes evaluation config
            boxes = [b for b in boxes if b.class_name in self.classes]

            # append to results
            results[pred.token] += [b.to_dict() for b in boxes]

        return dict(results)


@registry.register(namespace="nuscenes")
# pylint: disable=too-many-instance-attributes
class Tracking(Detection):
    def __init__(
        self,
        root: str | Path,
        split: str,
        labels,
        meta: Dict[str, Any] | OmegaConf,
        output_path: str | Path,
        ref_channel: str = "LIDAR_TOP",
        ref_frame: Literal["sensor", "ego"] = "sensor",
        classes: Sequence[str] = CLASSES_TRACKING,
    ):
        super().__init__(
            root=root,
            split=split,
            labels=labels,
            meta=meta,
            output_path=output_path,
            ref_channel=ref_channel,
            ref_frame=ref_frame,
            classes=classes,
        )

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: MetaDict,
        batch: Sample,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for b in range(batch.batch_size):
            # filter out invalid samples/padding
            if not batch.is_valid[b]:
                continue

            # extract sample and move it to the cpu
            token = batch.meta[b].sample_id
            boxes = outputs.boxes.get(b).detach().to(device="cpu", copy=True)
            labels = outputs.labels.get(b).detach().to(device="cpu", copy=True)
            scores = outputs.scores.get(b).detach().to(device="cpu", copy=True)
            instances = outputs.instances.get(b).detach().to(device="cpu", copy=True)

            # store predictions for this sample
            pred = Prediction(token, boxes, labels, scores, instances)
            self.preds.append(pred)

    def _extract_boxes(self, preds: List[Prediction]) -> Dict[str, Dict[str, Any]]:
        results = defaultdict(list)

        for pred in preds:
            # extract boxes from the predictions
            boxes = TrackingBox.from_preds(
                preds=pred,
                data=self.data,
                label_maps=self.label_maps,
                ref_channel=self.ref_channel,
                ref_frame=self.ref_frame,
            )

            # ensure unique tracking IDs per scene
            sample = self.data.get("sample", pred.token)
            scene_index = self.data.getind("scene", sample["scene_token"])

            # filter boxes by classes in nuscenes evaluation config
            boxes = [b for b in boxes if b.class_name in self.classes]

            for b in boxes:
                b.instance_id = f"{scene_index}-{b.instance_id}"

            # append to results
            results[pred.token] += [b.to_dict() for b in boxes]

        return dict(results)
