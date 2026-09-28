# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * nuScenes devkit (https://github.com/nutonomy/nuscenes-devkit), Copyright (c) 2018 nuTonomy, licensed under Apache-2.0 (DetectionEval/TrackingEval metric aggregation).
# See the LICENSES/ directory for full license texts.

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Self, Tuple, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import add_center_dist, filter_eval_boxes, load_gt
from nuscenes.eval.detection import algo as detection_algo
from nuscenes.eval.detection.constants import TP_METRICS as DETECTION_TP_METRICS
from nuscenes.eval.detection.data_classes import DetectionBox as NscDetectionBox
from nuscenes.eval.detection.data_classes import (
    DetectionMetricDataList,
    DetectionMetrics,
)
from nuscenes.eval.tracking import algo as tracking_algo
from nuscenes.eval.tracking import loaders as tracking_loaders
from nuscenes.eval.tracking.constants import AVG_METRIC_MAP as TRACKING_AVG_METRIC_MAP
from nuscenes.eval.tracking.constants import MOT_METRIC_MAP as TRACKING_MOT_METRIC_MAP
from nuscenes.eval.tracking.data_classes import TrackingBox as NscTrackingBox
from nuscenes.eval.tracking.data_classes import (
    TrackingMetricData,
    TrackingMetricDataList,
    TrackingMetrics,
)
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from torchmetrics.utilities.distributed import gather_all_tensors

from ... import utils
from ...utils.types import MetaDict, Sample
from .. import dataset
from ..dataset.nuscenes import _parse_version_and_split
from .metric import ValidationMetrics
from .registry import registry as validation

log = utils.log.get_logger(__name__)

T = TypeVar("T")

# likeliest attribute for each class
DEFAULT_BOX_ATTR = {
    "pedestrian": "pedestrian.moving",
    "car": "vehicle.parked",
    "bus": "vehicle.moving",
    "construction_vehicle": "vehicle.parked",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "motorcycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
    "ignore": "",
}


def _get_label_maps(labels):
    # make sure that all classes are referenced in the order list
    assert (set(labels.map.values()) - {"ignore"}) == set(labels.order)

    # construct all the necessary maps
    label_maps = {
        "label_to_cls": labels.map,
        "cls_to_id": {k: i for i, k in enumerate(labels.order)},
        "id_to_cls": labels.order,
    }
    return MetaDict(label_maps)


def _load_sample_transforms(
    data: NuScenes,
    sample_token: str,
    ref_channel: str,
    ref_frame: Literal["sensor", "ego"] = "sensor",
) -> Tuple[Quaternion, np.ndarray, Quaternion, np.ndarray]:
    # get sample data
    sample = data.get("sample", sample_token)
    sample_data_token = sample["data"][ref_channel]
    sample_data = data.get("sample_data", sample_data_token)

    # get ego pose
    ego_pose_token = sample_data["ego_pose_token"]
    ego_pose = data.get("ego_pose", ego_pose_token)

    ego_rotation = Quaternion(ego_pose["rotation"])
    ego_translation = np.array(ego_pose["translation"])

    if ref_frame == "sensor":
        # get sensor pose
        sensor_token = sample_data["calibrated_sensor_token"]
        sensor = data.get("calibrated_sensor", sensor_token)

        sensor_rotation = Quaternion(sensor["rotation"])
        sensor_translation = np.array(sensor["translation"])

    else:
        # use ego vehicle frame
        sensor_rotation = Quaternion()
        sensor_translation = np.zeros(3)

    return sensor_rotation, sensor_translation, ego_rotation, ego_translation


def _preds_to_box(box: np.ndarray, label: int, score: float, transforms) -> Box:
    sensor_rot, sensor_trans, ego_rot, ego_trans = transforms

    # create box object for transformations
    box = Box(
        center=box[0:3],
        size=box[3:6],
        # Note: this may not work for boxes not in the lidar or global frame
        orientation=Quaternion(axis=(0, 0, 1), radians=box[6]),
        label=label,
        score=score,
        velocity=(*box[7:9], 0.0),
    )

    # transform: sensor to ego vehicle
    box.rotate(sensor_rot)
    box.translate(sensor_trans)

    # transform: ego vehicle to global
    box.rotate(ego_rot)
    box.translate(ego_trans)

    return box


def _compute_box_attrib(
    box: Box, class_name: str, motion_th: float = 0.2
) -> str | None:
    # Try to set sensible attributes using a heuristic based on the
    # object's motion-state and class. This is the standard nuScenes
    # attribute-assignment heuristic (as used e.g. in mmdetection3d).
    if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > motion_th:
        if class_name in ["car", "construction_vehicle", "bus", "truck", "trailer"]:
            return "vehicle.moving"
        if class_name in ["bicycle", "motorcycle"]:
            return "cycle.with_rider"
    else:
        if class_name in ["pedestrian"]:
            return "pedestrian.standing"
        if class_name in ["bus"]:
            return "vehicle.stopped"

    # If we cannot deduce any attribute via the heuristica above: Choose
    # the one that is most probable based on its per-class frequency.
    return DEFAULT_BOX_ATTR.get(class_name)


def _translate_detection_metrics(
    metrics: Dict[str, Any], prefix: str = ""
) -> Dict[str, float]:
    values = {
        f"{prefix}summary/mAP": metrics["mean_ap"],
        f"{prefix}summary/mATE": metrics["tp_errors"]["trans_err"],
        f"{prefix}summary/mASE": metrics["tp_errors"]["scale_err"],
        f"{prefix}summary/mAOE": metrics["tp_errors"]["orient_err"],
        f"{prefix}summary/mAVE": metrics["tp_errors"]["vel_err"],
        f"{prefix}summary/mAAE": metrics["tp_errors"]["attr_err"],
        f"{prefix}summary/NDS": metrics["nd_score"],
    }

    for class_name in sorted(metrics["mean_dist_aps"].keys()):
        aps = metrics["mean_dist_aps"][class_name]
        tps = metrics["label_tp_errors"][class_name]

        values |= {
            f"{prefix}class/{class_name}/AP": aps,
            f"{prefix}class/{class_name}/ATE": tps["trans_err"],
            f"{prefix}class/{class_name}/ASE": tps["scale_err"],
            f"{prefix}class/{class_name}/AOE": tps["orient_err"],
            f"{prefix}class/{class_name}/AVE": tps["vel_err"],
        }

    return values


def _precision_recall(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    return precision, recall


@dataclass
class Prediction:
    token: str
    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    instances: torch.Tensor | None

    def __hash__(self) -> int:
        return hash(self.token)

    def cpu(self) -> Self:
        return Prediction(
            self.token,
            self.boxes.cpu(),
            self.labels.cpu(),
            self.scores.cpu(),
            self.instances.cpu() if self.instances is not None else None,
        )


@dataclass
class DetectionBox:
    # pylint: disable=too-many-instance-attributes

    token: str
    translation: Tuple[float, float, float]
    size: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    velocity: Tuple[float, float]
    score: float
    class_name: str
    attribute_name: str

    @classmethod
    def from_preds(
        cls,
        preds: Prediction,
        data: NuScenes,
        label_maps: MetaDict,
        ref_channel: str = "LIDAR_TOP",
        ref_frame: Literal["sensor", "ego"] = "sensor",
    ) -> List[Self]:
        # load required transformations
        transforms = _load_sample_transforms(data, preds.token, ref_channel, ref_frame)

        # convert tensors to numpy arrays
        boxes = preds.boxes.numpy()
        labels = preds.labels.numpy()
        scores = preds.scores.numpy()

        # convert numpy arrays to generic nuscenes boxes in reference frame
        boxes = [
            _preds_to_box(box, label, score, transforms)
            for box, label, score in zip(boxes, labels, scores)
        ]

        # map class names
        class_names = [label_maps.id_to_cls[box.label] for box in boxes]

        # compute box attributes
        attribs = [_compute_box_attrib(b, c) for b, c in zip(boxes, class_names)]

        # convert to detection boxes
        return [
            DetectionBox(
                token=preds.token,
                translation=box.center.tolist(),
                size=box.wlh.tolist(),
                rotation=box.orientation.elements.tolist(),
                velocity=box.velocity[:2].tolist(),
                score=box.score,
                class_name=clsname,
                attribute_name=attrib or "",
            )
            for box, clsname, attrib in zip(boxes, class_names, attribs)
        ]

    def to_nuscenes(self) -> NscDetectionBox:
        return NscDetectionBox(
            sample_token=self.token,
            translation=self.translation,
            size=self.size,
            rotation=self.rotation,
            velocity=self.velocity,
            detection_name=self.class_name,
            detection_score=self.score,
            attribute_name=self.attribute_name,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_token": self.token,
            "translation": self.translation,
            "size": self.size,
            "rotation": self.rotation,
            "velocity": self.velocity,
            "detection_name": self.class_name,
            "detection_score": self.score,
            "attribute_name": self.attribute_name,
        }


@dataclass
class TrackingBox:
    # pylint: disable=too-many-instance-attributes

    token: str
    translation: Tuple[float, float, float]
    size: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    velocity: Tuple[float, float]
    score: float
    class_name: str
    instance_id: str

    @classmethod
    def from_preds(
        cls,
        preds: Prediction,
        data: NuScenes,
        label_maps: MetaDict,
        ref_channel: str = "LIDAR_TOP",
        ref_frame: Literal["sensor", "ego"] = "sensor",
    ) -> List[Self]:
        # load required transformations
        transforms = _load_sample_transforms(data, preds.token, ref_channel, ref_frame)

        # convert tensors to numpy arrays
        boxes = preds.boxes.numpy()
        labels = preds.labels.numpy()
        scores = preds.scores.numpy()
        instances = preds.instances.numpy()

        # convert numpy arrays to generic nuscenes boxes in reference frame
        boxes = [
            _preds_to_box(box, label, score, transforms)
            for box, label, score in zip(boxes, labels, scores)
        ]

        # convert to tracking boxes
        return [
            TrackingBox(
                token=preds.token,
                translation=box.center.tolist(),
                size=box.wlh.tolist(),
                rotation=box.orientation.elements.tolist(),
                velocity=box.velocity[:2].tolist(),
                score=box.score,
                class_name=label_maps.id_to_cls[box.label],
                instance_id=str(iid),
            )
            for box, iid in zip(boxes, instances)
        ]

    def to_nuscenes(self) -> NscTrackingBox:
        return NscTrackingBox(
            sample_token=self.token,
            translation=self.translation,
            size=self.size,
            rotation=self.rotation,
            velocity=self.velocity,
            tracking_name=self.class_name,
            tracking_score=self.score,
            tracking_id=self.instance_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_token": self.token,
            "translation": self.translation,
            "size": self.size,
            "rotation": self.rotation,
            "velocity": self.velocity,
            "tracking_name": self.class_name,
            "tracking_score": self.score,
            "tracking_id": self.instance_id,
        }


@validation.register(namespace="nuscenes")
# pylint: disable-next=too-many-instance-attributes
class Detection(ValidationMetrics):
    preds: List[Prediction]

    def __init__(
        self,
        root: str | Path,
        split: str,
        labels,
        prefix: str = "val/",
        conf: str = "detection_cvpr_2019",
        ref_channel: str = "LIDAR_TOP",
        ref_frame: Literal["sensor", "ego"] = "sensor",
    ):
        super().__init__()

        self.prefix = prefix
        self.ref_channel = ref_channel
        self.ref_frame = ref_frame

        # parse NuScenes dataset version and split
        self.version, self.split = _parse_version_and_split(split)

        # load/build label maps
        self.label_maps = _get_label_maps(labels)

        # initialize state
        self.add_state("preds", default=[], dist_reduce_fx=None)

        # we only actually compute the metric on rank 0, so skip loading stuff
        # that we don't need
        if utils.mp.rank == 0:
            # load data and config
            self.data = dataset.nuscenes.acquire(self.version, root)
            self.config = config_factory(conf)
            self.classes = set(self.config.class_range.keys())

            # load and filter ground-truth boxes
            boxes = load_gt(self.data, self.split, NscDetectionBox)
            boxes = add_center_dist(self.data, boxes)
            boxes = filter_eval_boxes(self.data, boxes, self.config.class_range)
            self.gt_boxes = boxes

    @property
    def is_differentiable(self):
        return False

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Any | None = None,
    ) -> None:
        # Note: Torchmetrics currently does not support states other than
        #       tensors or lists of tensors. So we need to do things manually
        #       here.

        # gather predictions across processes
        world_size = dist.get_world_size(process_group)
        synced = [[] for _ in range(world_size)]

        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            # Note: This will emit a warning that pickle is unsafe... we take
            # not of that and ignore it.
            dist.all_gather_object(synced, self.preds, group=process_group)

        # reduce to a simple list
        self.preds = [pred for preds in synced for pred in preds]

    # pylint: disable-next=arguments-differ
    def update(
        self,
        sample: Sample,
        preds: MetaDict,
    ) -> None:
        for b in range(sample.batch_size):
            # filter out invalid samples/padding
            if not sample.is_valid[b]:
                continue

            # extract sample and move it to the cpu
            token = sample.meta[b].sample_id
            boxes = preds.boxes.get(b).detach().to(device="cpu", copy=True)
            labels = preds.labels.get(b).detach().to(device="cpu", copy=True)
            scores = preds.scores.get(b).detach().to(device="cpu", copy=True)

            # store predictions for this sample
            pred = Prediction(token, boxes, labels, scores, None)
            self.preds.append(pred)

    def _get_predicted_boxes(self) -> EvalBoxes:
        pred_boxes = EvalBoxes()

        for pred in self.preds:
            # extract boxes from the predictions
            boxes = DetectionBox.from_preds(
                preds=pred,
                data=self.data,
                ref_channel=self.ref_channel,
                ref_frame=self.ref_frame,
                label_maps=self.label_maps,
            )

            # filter boxes by classes in nuscenes evaluation config
            boxes = [b for b in boxes if b.class_name in self.classes]

            # store the boxes (if there are any)
            if boxes:
                boxes = [b.to_nuscenes() for b in boxes]
                pred_boxes.add_boxes(pred.token, boxes)

        return pred_boxes

    def _evaluate(
        self, pred_boxes: EvalBoxes
    ) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """Evaluation, based on NuScenes DetectionEval.evaluate()"""
        config = self.config

        start_time = time.time()

        # step 1: accumulate metric data for all classes and distance thresholds
        metric_data_list = DetectionMetricDataList()
        for class_name in config.class_names:
            for dist_th in config.dist_ths:
                md = detection_algo.accumulate(
                    self.gt_boxes,
                    pred_boxes,
                    class_name,
                    config.dist_fcn_callable,
                    dist_th,
                )
                metric_data_list.set(class_name, dist_th, md)

        # step 2: compute metrics from data
        metrics = DetectionMetrics(config)
        for class_name in config.class_names:
            # compute APs
            for dist_th in config.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = detection_algo.calc_ap(
                    metric_data, config.min_recall, config.min_precision
                )
                metrics.add_label_ap(class_name, dist_th, ap)

            # compute TP metrics
            for metric_name in DETECTION_TP_METRICS:
                metric_data = metric_data_list[(class_name, config.dist_th_tp)]
                if class_name in ["traffic_cone"] and metric_name in [
                    "attr_err",
                    "vel_err",
                    "orient_err",
                ]:
                    tp = np.nan
                elif class_name in ["barrier"] and metric_name in [
                    "attr_err",
                    "vel_err",
                ]:
                    tp = np.nan
                else:
                    tp = detection_algo.calc_tp(
                        metric_data, config.min_recall, metric_name
                    )
                metrics.add_label_tp(class_name, metric_name, tp)

        # store evaluation time
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def _evaluate_extended(
        self, pred_boxes: EvalBoxes
    ) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Extended evaluation, based on original NuScenes evaluation.

        Unfortunately, the original evaluation code does not store true/false
        positives and false negatives. So we need to re-implement part of the
        evaluation to get these values.
        """
        metrics = {}

        # store number of ground-truth and predicted boxes (after filtering)
        metrics["num_gt"] = len(self.gt_boxes.all)
        metrics["num_pred"] = len(pred_boxes.all)

        # compute true/false positives and false negatives for each class and threshold
        for dist_th in self.config.dist_ths:
            total_tp, total_fp, total_fn = 0, 0, 0

            for class_name in self.config.class_names:
                tp, fp, fn = self._accumulate(
                    pred_boxes, class_name, self.config.dist_fcn_callable, dist_th
                )

                # accumulate total values
                total_tp += tp
                total_fp += fp
                total_fn += fn

                # compute precision and recall
                precision, recall = _precision_recall(tp, fp, fn)

                # store the values
                metrics[f"{class_name}/d{dist_th}/TP"] = tp
                metrics[f"{class_name}/d{dist_th}/FP"] = fp
                metrics[f"{class_name}/d{dist_th}/FN"] = fn
                metrics[f"{class_name}/d{dist_th}/precision"] = precision
                metrics[f"{class_name}/d{dist_th}/recall"] = recall

            # compute precision and recall over all classes
            precision, recall = _precision_recall(total_tp, total_fp, total_fn)

            # store the values
            metrics[f"d{dist_th}/TP"] = total_tp
            metrics[f"d{dist_th}/FP"] = total_fp
            metrics[f"d{dist_th}/FN"] = total_fn
            metrics[f"d{dist_th}/precision"] = precision
            metrics[f"d{dist_th}/recall"] = recall

        return metrics

    def _accumulate(
        self,
        pred_boxes: EvalBoxes,
        class_name: str,
        dist_fcn: Callable,
        dist_th: float,
    ) -> tuple[int, int, int]:
        """
        Compute true/false positives and false negatives for a single class and
        distance threshold.

        Adapted from the original NuScenes evaluation code.
        """
        # pylint: disable=too-many-locals

        gt_boxes = self.gt_boxes

        # count the ground-truth boxes for the class, return if there are none
        npos = len([box for box in gt_boxes.all if box.detection_name == class_name])
        if npos == 0:
            return 0, 0, 0

        # get the predicted boxes for the class
        pred_boxes = [box for box in pred_boxes.all if box.detection_name == class_name]

        # sort predicted boxes by confidence
        pred_boxes = sorted(pred_boxes, key=lambda x: x.detection_score, reverse=True)

        # do the actual matching
        tp, fp = 0, 0

        taken = set()
        for pred_box in pred_boxes:
            min_dist = np.inf
            match_gt_idx = None

            # find the closest match among the remaining GT boxes
            for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
                if gt_box.detection_name != class_name:
                    continue

                if (pred_box.sample_token, gt_idx) in taken:
                    continue

                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

            # check if the closest match is close enough
            is_match = min_dist < dist_th

            # update the taken set
            if is_match:
                taken.add((pred_box.sample_token, match_gt_idx))

            # update the counters
            if is_match:
                tp += 1
            else:
                fp += 1

        # compute false negatives
        fn = npos - tp

        return tp, fp, fn

    def _compute(self) -> Dict[str, float]:
        # convert predicted boxes to NuScenes boxes
        pred_boxes = self._get_predicted_boxes()

        # filter predicted boxes
        if len(pred_boxes):
            pred_boxes = add_center_dist(self.data, pred_boxes)
            pred_boxes = filter_eval_boxes(
                self.data, pred_boxes, self.config.class_range
            )

        # compute metrics
        metrics, _metric_data_list = self._evaluate(pred_boxes)
        metrics = _translate_detection_metrics(metrics.serialize(), prefix=self.prefix)

        # compute extended metrics
        ext_metrics = self._evaluate_extended(pred_boxes)
        ext_metrics = {f"{self.prefix}{k}": v for k, v in ext_metrics.items()}
        metrics |= ext_metrics

        return metrics

    def compute(self) -> Dict[str, Any]:
        log.debug("performing NuScenes detection evaluation")

        # The NuScenes eval is a bit heavier to compute than your regular
        # metrics. So to avoid doing redundant work, we process everything on
        # rank 0 only.
        use_dist = dist.is_available() and dist.is_initialized()

        if not use_dist or dist.get_rank() == 0:
            metrics = self._compute()
        else:
            metrics = {}

        # Broadcast metrics from rank 0 to all other processes.
        if use_dist:
            metrics = [metrics]
            dist.broadcast_object_list(metrics, src=0)
            metrics = metrics[0]

        return metrics


@validation.register(namespace="nuscenes")
# pylint: disable-next=too-many-instance-attributes
class Tracking(ValidationMetrics):
    preds: List[Prediction]

    def __init__(
        self,
        root: str | Path,
        split: str,
        labels,
        prefix: str = "val/",
        conf: str = "tracking_nips_2019",
        ref_channel: str = "LIDAR_TOP",
        ref_frame: Literal["sensor", "ego"] = "sensor",
    ):
        super().__init__()

        self.prefix = prefix
        self.ref_channel = ref_channel
        self.ref_frame = ref_frame

        # parse NuScenes dataset version and split
        self.version, self.split = _parse_version_and_split(split)

        # load/build label maps
        self.label_maps = _get_label_maps(labels)

        # initialize state
        self.add_state("preds", default=[], dist_reduce_fx=None)

        # we only actually compute the metric on rank 0, so skip loading stuff
        # that we don't need
        if utils.mp.rank == 0:
            # load data and config
            self.data = dataset.nuscenes.acquire(self.version, root)
            self.config = config_factory(conf)
            self.classes = set(self.config.class_range.keys())

            # load and filter ground-truth boxes
            boxes = load_gt(self.data, self.split, NscTrackingBox)
            boxes = add_center_dist(self.data, boxes)
            boxes = filter_eval_boxes(self.data, boxes, self.config.class_range)
            self.gt_boxes = boxes

            # convert to tracking format
            self.gt_tracks = tracking_loaders.create_tracks(
                self.gt_boxes, self.data, self.split, gt=True
            )

    @property
    def is_differentiable(self):
        return False

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Any | None = None,
    ) -> None:
        # Note: Torchmetrics currently does not support states other than
        #       tensors or lists of tensors. So we need to do things manually
        #       here.

        # gather predictions across processes
        world_size = dist.get_world_size(process_group)
        synced = [[] for _ in range(world_size)]

        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            # Note: This will emit a warning that pickle is unsafe... we take
            # not of that and ignore it.
            dist.all_gather_object(synced, self.preds, group=process_group)

        # reduce to a simple list
        self.preds = [pred for preds in synced for pred in preds]

    # pylint: disable-next=arguments-differ
    def update(
        self,
        sample: Sample,
        preds: MetaDict,
    ) -> None:
        for b in range(sample.batch_size):
            # filter out invalid samples/padding
            if not sample.is_valid[b]:
                continue

            # extract sample and move it to the cpu
            token = sample.meta[b].sample_id
            boxes = preds.boxes.get(b).detach().to(device="cpu", copy=True)
            labels = preds.labels.get(b).detach().to(device="cpu", copy=True)
            scores = preds.scores.get(b).detach().to(device="cpu", copy=True)
            instances = preds.instances.get(b).detach().to(device="cpu", copy=True)

            # store predictions for this sample
            pred = Prediction(token, boxes, labels, scores, instances)
            self.preds.append(pred)

    def _get_predicted_boxes(self) -> EvalBoxes:
        pred_boxes = EvalBoxes()

        for pred in self.preds:
            # extract boxes from the predictions
            boxes = TrackingBox.from_preds(
                preds=pred,
                data=self.data,
                ref_channel=self.ref_channel,
                ref_frame=self.ref_frame,
                label_maps=self.label_maps,
            )

            # ensure unique tracking IDs per scene
            sample = self.data.get("sample", pred.token)
            scene_index = self.data.getind("scene", sample["scene_token"])

            for b in boxes:
                b.instance_id = f"{scene_index}-{b.instance_id}"

            # filter boxes by classes in nuscenes evaluation config
            boxes = [b for b in boxes if b.class_name in self.classes]

            # store the boxes (if there are any)
            if boxes:
                boxes = [b.to_nuscenes() for b in boxes]
                pred_boxes.add_boxes(pred.token, boxes)

        return pred_boxes

    def _evaluate(
        self, pred_tracks: Dict[str, Dict[int, List[TrackingBox]]]
    ) -> Tuple[TrackingMetrics, TrackingMetricDataList]:
        config = self.config

        start_time = time.time()

        # step 1: accumulate metric data for all classes and distance thresholds
        metric_data_list = TrackingMetricDataList()

        def accumulate_class(class_name):
            md = tracking_algo.TrackingEvaluation(
                tracks_gt=self.gt_tracks,
                tracks_pred=pred_tracks,
                class_name=class_name,
                dist_fcn=config.dist_fcn_callable,
                dist_th_tp=config.dist_th_tp,
                min_recall=config.min_recall,
                num_thresholds=TrackingMetricData.nelem,
                metric_worst=config.metric_worst,
                verbose=False,
            ).accumulate()

            metric_data_list.set(class_name, md)

        for class_name in config.class_names:
            accumulate_class(class_name)

        # step 2: aggregate metrics from the metric data
        metrics = TrackingMetrics(config)

        for class_name in config.class_names:
            # Find best MOTA to determine threshold to pick for traditional
            # metrics. If multiple thresholds have the same value, pick the one
            # with the highest recall.
            md = metric_data_list[class_name]
            if np.all(np.isnan(md.mota)):
                best_thresh_idx = None
            else:
                best_thresh_idx = np.nanargmax(md.mota)

            # Pick best value for traditional metrics.
            if best_thresh_idx is not None:
                for metric_name in TRACKING_MOT_METRIC_MAP.values():
                    if metric_name == "":
                        continue
                    value = md.get_metric(metric_name)[best_thresh_idx]
                    metrics.add_label_metric(metric_name, class_name, value)

            # Compute AMOTA / AMOTP.
            for metric_name, individual_name in TRACKING_AVG_METRIC_MAP.items():
                values = np.array(md.get_metric(individual_name))
                assert len(values) == TrackingMetricData.nelem

                if np.all(np.isnan(values)):
                    # If no GT exists, set to nan.
                    value = np.nan
                else:
                    # Overwrite any nan value with the worst possible value.
                    np.all(values[np.logical_not(np.isnan(values))] >= 0)
                    values[np.isnan(values)] = config.metric_worst[metric_name]
                    value = float(np.nanmean(values))
                metrics.add_label_metric(metric_name, class_name, value)

        # store evaluation time
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def _compute(self) -> Dict[str, float]:
        # convert predicted boxes to NuScenes boxes
        pred_boxes = self._get_predicted_boxes()

        # filter predicted boxes
        if len(pred_boxes):
            pred_boxes = add_center_dist(self.data, pred_boxes)
            pred_boxes = filter_eval_boxes(
                self.data, pred_boxes, self.config.class_range
            )

        # convert to tracking format
        pred_tracks = tracking_loaders.create_tracks(
            pred_boxes, self.data, self.split, gt=False
        )

        # compute metrics
        metrics, _metric_data_list = self._evaluate(pred_tracks)
        metrics = metrics.serialize()

        # translate metrics
        values = {}
        for metric_name in metrics["label_metrics"].keys():
            values[f"{self.prefix}summary/{metric_name}"] = metrics[metric_name]

        for metric_name, metric_values in metrics["label_metrics"].items():
            for class_name, value in metric_values.items():
                values[f"{self.prefix}class/{class_name}/{metric_name}"] = value

        return values

    def compute(self) -> Dict[str, Any]:
        log.debug("performing NuScenes tracking evaluation")

        # The NuScenes eval is a bit heavier to compute than your regular
        # metrics. So to avoid doing redundant work, we process everything on
        # rank 0 only.
        use_dist = dist.is_available() and dist.is_initialized()

        if not use_dist or dist.get_rank() == 0:
            metrics = self._compute()
        else:
            metrics = {}

        # Broadcast metrics from rank 0 to all other processes.
        if use_dist:
            metrics = [metrics]
            dist.broadcast_object_list(metrics, src=0)
            metrics = metrics[0]

        return metrics
