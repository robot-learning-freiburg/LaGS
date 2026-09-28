# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, Collection, Literal, Sequence

import torch
import torch.distributed
from omegaconf import OmegaConf

from ... import config
from ...utils.types import MetaDict, Sample
from .base_metrics import AssociationQuality as AssociationQualityMetric
from .base_metrics import PanopticQuality as PanopticQualityMetric
from .base_metrics import SegmentationTrackingQuality as StqMetric
from .base_metrics import SemanticQuality as SemanticQualityMetric
from .base_metrics.association_quality import Result as AssociationQualityResult
from .base_metrics.semantic_quality import Result as SemanticQualityResult
from .metric import ValidationMetrics
from .registry import registry as validation


@validation.register(namespace="occupancy")
class SemanticQuality(ValidationMetrics):
    """
    Semantic quality metrics for semantic occupancy prediction / semantic scene
    completion.

    Follows the SemanitcKITTI evaluation protocol, computing per-class IoU,
    precision, and recall, as well as mean IoU over all non-free classes and
    thing/stuff subsets.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        labels: OmegaConf,
        mask: str | Collection[str] = "valid",
        prefix: str = "val/",
    ) -> None:
        """
        Initialize the metric.

        Args:
            labels (OmegaConf):
                The label configuration for the dataset.
            mask (str or Collection[str], optional):
                The mask used to identify valid voxels for evaluation. Defaults
                to "valid".
            prefix (str, optional):
                The prefix to use for the metric names. Defaults to "val/".
        """
        super().__init__()

        self.prefix = prefix
        self.mask = [mask] if isinstance(mask, str) else mask
        self.ignore_index = labels.ignore_index

        self.classes = config.utils.to_primitive(labels.all)
        classes_semantic = config.utils.to_primitive(labels.semantic)
        classes_free = config.utils.to_primitive(labels.free)
        classes_thing = config.utils.to_primitive(labels.thing)
        classes_stuff = config.utils.to_primitive(labels.stuff)

        self.num_classes = len(self.classes)

        # classes to be considered for mIoU
        cls_mean = _class_mask(self.classes, classes_semantic, invert=False)
        self.register_buffer("cls_mean", cls_mean, persistent=False)

        # classes to be considered "non-free"/occupied
        cls_occ = _class_mask(self.classes, classes_free, invert=True)
        self.register_buffer("cls_occ", cls_occ, persistent=False)

        # classes to be considered "thing"
        cls_thing = _class_mask(self.classes, classes_thing)
        self.register_buffer("cls_thing", cls_thing, persistent=False)

        # classes to be considered "stuff"
        cls_stuff = _class_mask(self.classes, classes_stuff)
        self.register_buffer("cls_stuff", cls_stuff, persistent=False)

        # base metric
        self.base = SemanticQualityMetric(self.num_classes)

    def reset(self) -> None:
        super().reset()
        self.base.reset()

    # pylint: disable-next=arguments-differ
    def update(
        self,
        sample: Sample,
        preds: MetaDict,
    ) -> None:
        # if there are no valid samples in the batch, skip update
        if not sample.is_valid.any():
            return

        # get mask for valid samples in batch
        valid = sample.is_valid

        # build mask for valid voxels
        mask = sample.labels.occupancy.masks
        mask = [v for k, v in mask.items() if k in self.mask]
        mask = torch.stack(mask, dim=1).all(dim=1)
        mask = mask[valid, ...]

        # filter ground truth
        target = sample.labels.occupancy.semantics
        target = target[valid, ...]

        # filter predictions
        preds = preds.occupancy.semantics.detach()
        preds = preds[valid, ...]

        # update mask for ignore_index
        mask = mask & (target != self.ignore_index)

        # update confusion matrix
        self.base.update(preds, target, mask)

    def compute(self) -> dict[str, Any]:
        # pylint: disable=too-many-locals

        # compute base metric
        result = self.base.compute()

        # get true positives, false positives, false negatives
        tp, fp, fn = result.tp, result.fp, result.fn

        # compute per-class IoU, precision, recall
        sem_iou, sem_precision, sem_recall = result.iou, result.precision, result.recall

        # compute mean IoU (mIoU) over all non-free classes and thing/stuff subsets
        sem_miou, _, _ = result.mean_iou(self.cls_mean)
        sem_miou_thing, _, _ = result.mean_iou(self.cls_thing)
        sem_miou_stuff, _, _ = result.mean_iou(self.cls_stuff)

        # compute true positives, false positives, false negatives for binary occupancy
        occ_tp, occ_fp, occ_fn = result.binary_counts(self.cls_occ)

        # compute occupancy IoU, precision, recall
        occ_iou, occ_precision, occ_recall = result.binary_iou(self.cls_occ)

        # collect output
        metrics = {
            "summary/mIoU": sem_miou.item(),
            "subset/thing/mIoU": sem_miou_thing.item(),
            "subset/stuff/mIoU": sem_miou_stuff.item(),
            "occupancy/IoU": occ_iou.item(),
            "occupancy/precision": occ_precision.item(),
            "occupancy/recall": occ_recall.item(),
            "occupancy/TP": occ_tp.item(),
            "occupancy/FP": occ_fp.item(),
            "occupancy/FN": occ_fn.item(),
        }

        for i, name in enumerate(self.classes):
            metrics[f"class/{name}/IoU"] = sem_iou[i].item()
            metrics[f"class/{name}/precision"] = sem_precision[i].item()
            metrics[f"class/{name}/recall"] = sem_recall[i].item()
            metrics[f"class/{name}/TP"] = tp[i].item()
            metrics[f"class/{name}/FP"] = fp[i].item()
            metrics[f"class/{name}/FN"] = fn[i].item()

        return {self.prefix + k: v for k, v in metrics.items()}


@validation.register(namespace="occupancy")
class PanopticQuality(ValidationMetrics):
    """
    Panoptic quality metrics for panoptic occupancy prediction.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        labels: OmegaConf,
        things: Sequence[str] | None = None,
        mask: str | Collection[str] = "valid",
        prefix: str = "val/",
    ) -> None:
        """
        Initialize the metric.

        Args:
            labels (OmegaConf):
                The label configuration for the dataset.
            things (Sequence[str] | None, optional):
                The classes to consider as "thing" classes. If None, all
                classes with instance segmentation ground truth are considered.
                Defaults to None.
            mask (str or Collection[str], optional):
                The mask used to identify valid voxels for evaluation. Defaults
                to "valid".
            prefix (str, optional):
                The prefix to use for the metric names. Defaults to "val/".
        """
        super().__init__()

        self.prefix = prefix
        self.mask = [mask] if isinstance(mask, str) else mask
        self.classes = config.utils.to_primitive(labels.all)
        self.ignore_index = labels.ignore_index

        if things is None:
            things = config.utils.to_primitive(labels.instance)

        # ensure that all requested classes are present in the dataset
        assert all(n in self.classes for n in things)

        # divide classes into instance and semantic classes
        classes_thing = set(things)
        classes_stuff = set(self.classes) - classes_thing

        class_map = {n: i for i, n in enumerate(self.classes)}

        # classes to be considered for mean metrics
        cls_mean = config.utils.to_primitive(labels.semantic)
        cls_mean = _class_mask(self.classes, cls_mean, invert=False)
        self.register_buffer("cls_mean", cls_mean, persistent=False)

        # base metric
        self.base = PanopticQualityMetric(
            num_classes=len(self.classes),
            classes_thing=[class_map[n] for n in classes_thing],
            classes_stuff=[class_map[n] for n in classes_stuff],
        )

    def reset(self) -> None:
        super().reset()
        self.base.reset()

    # pylint: disable-next=arguments-differ
    def update(
        self,
        sample: Sample,
        preds: MetaDict,
    ) -> None:
        # if there are no valid samples in the batch, skip update
        if not sample.is_valid.any():
            return

        # get mask for valid samples in batch
        valid = sample.is_valid

        # build mask for valid voxels
        mask = sample.labels.occupancy.masks
        mask = [v for k, v in mask.items() if k in self.mask]
        mask = torch.stack(mask, dim=1).all(dim=1)
        mask = mask[valid, ...]

        # filter ground truth
        target_sem = sample.labels.occupancy.semantics
        target_sem = target_sem[valid, ...]

        target_iid = sample.labels.occupancy.instance_ids
        target_iid = target_iid[valid, ...]

        target = torch.stack([target_sem, target_iid], dim=-1)

        # filter predictions
        preds_sem = preds.occupancy.semantics.detach()
        preds_sem = preds_sem[valid, ...]

        preds_iid = preds.occupancy.instance_ids.detach()
        preds_iid = preds_iid[valid, ...]

        preds = torch.stack([preds_sem, preds_iid], dim=-1)

        # update mask for ignore_index
        mask = mask & (target_sem != self.ignore_index)

        # update metrics
        self.base.update(preds, target, mask)

    def compute(self) -> dict[str, Any]:
        result = self.base.compute()

        mean_pq, mean_sq, mean_rq = result.mean_pq(self.cls_mean)
        mean_mod_pq, mean_aiou = result.mean_mod_pq(self.cls_mean)

        metrics = {
            "summary/PQ": mean_pq,
            "summary/SQ": mean_sq,
            "summary/RQ": mean_rq,
            "summary/PQ†": mean_mod_pq,
            "summary/maIoU": mean_aiou,
            "summary/GT": result.gt.sum().item(),
            "summary/TP": result.tp.sum().item(),
            "summary/FP": result.fp.sum().item(),
            "summary/FN": result.fn.sum().item(),
        }

        for i, name in enumerate(self.classes):
            metrics[f"class/{name}/PQ"] = result.pq[i].item()
            metrics[f"class/{name}/SQ"] = result.sq[i].item()
            metrics[f"class/{name}/RQ"] = result.rq[i].item()
            metrics[f"class/{name}/PQ†"] = result.mod_pq[i].item()
            metrics[f"class/{name}/aIoU"] = result.aiou[i].item()
            metrics[f"class/{name}/GT"] = result.gt[i].item()
            metrics[f"class/{name}/TP"] = result.tp[i].item()
            metrics[f"class/{name}/FP"] = result.fp[i].item()
            metrics[f"class/{name}/FN"] = result.fn[i].item()

        return {self.prefix + k: v for k, v in metrics.items()}


@validation.register(namespace="occupancy")
class AssociationQuality(ValidationMetrics):
    """
    Association quality metrics for panoptic occupancy prediction.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        labels: OmegaConf,
        classes: Sequence[str] | None = None,
        mask: str | Collection[str] = "valid",
        per_frame: bool = False,
        prefix: str = "val/",
        show_per_class: bool = True,
        show_per_sequence: bool = False,
        allow_invalid_instances: Literal["disallow", "restrict", "ignore"] = "disallow",
    ) -> None:
        """
        Initialize the metric.

        Args:
            labels (OmegaConf):
                The label configuration for the dataset.
            classes (Sequence[str] | None, optional):
                The classes to consider for the association quality metric.
                If None, all classes with instance segmentation ground truth
                are considered. Defaults to None.
            mask (str or Collection[str], optional):
                The mask used to identify valid voxels for evaluation. Defaults
                to "valid".
            prefix (str, optional):
                The prefix to use for the metric names. Defaults to "val/".
            show_per_class (bool, optional):
                Whether to show per-class metrics. Defaults to True.
            show_per_sequence (bool, optional):
                Whether to show per-sequence metrics. Defaults to True.
            allow_invalid_instances (str, optional):
                How to handle invalid instance IDs in the ground-truth data.
                Options are:
                - "disallow": Raise an error if invalid instance IDs are found
                  for instance/thing classes.
                - "restrict": Treat invalid instance IDs for instance/thing
                  classes as non-instance (i.e., stuff).
                - "ignore": Ignore invalid instance IDs for instance/thing
                  classes, meaning the respective points/voxels are masked out
                  and not considered for evaluation.
        """
        super().__init__()

        self.prefix = prefix
        self.mask = [mask] if isinstance(mask, str) else mask
        self.classes = config.utils.to_primitive(labels.all)
        self.ignore_index = labels.ignore_index
        self.show_per_class = show_per_class
        self.show_per_sequence = show_per_sequence

        # build the set of included classes
        class_map = {n: i for i, n in enumerate(self.classes)}

        include = classes
        if include is None:
            include = config.utils.to_primitive(labels.instance)

        assert all(n in self.classes for n in include)

        include = set(include)
        include = {class_map[n] for n in include}

        # base metric
        self.base = AssociationQualityMetric(
            num_classes=len(self.classes),
            class_subset=include,
            per_frame=per_frame,
            allow_invalid_instances=allow_invalid_instances,
        )

    def reset(self) -> None:
        super().reset()
        self.base.reset()

    # pylint: disable-next=arguments-differ
    def update(
        self,
        sample: Sample,
        preds: MetaDict,
    ) -> None:
        # if there are no valid samples in the batch, skip update
        if not sample.is_valid.any():
            return

        # get mask for valid samples in batch
        valid = sample.is_valid

        # build mask for valid voxels
        mask = sample.labels.occupancy.masks
        mask = [v for k, v in mask.items() if k in self.mask]
        mask = torch.stack(mask, dim=1).all(dim=1)
        mask = mask[valid, ...]

        # filter ground truth
        target_sem = sample.labels.occupancy.semantics
        target_sem = target_sem[valid, ...]

        target_iid = sample.labels.occupancy.instance_ids
        target_iid = target_iid[valid, ...]

        target = torch.stack([target_sem, target_iid], dim=-1)

        # filter predictions
        preds_sem = preds.occupancy.semantics.detach()
        preds_sem = preds_sem[valid, ...]

        preds_iid = preds.occupancy.instance_ids.detach()
        preds_iid = preds_iid[valid, ...]

        preds = torch.stack([preds_sem, preds_iid], dim=-1)

        # filter sequence IDs
        sequence_id = [m.sequence_id for m, v in zip(sample.meta, valid) if v]

        # update mask for ignore_index
        mask = mask & (target_sem != self.ignore_index)

        # update metrics
        self.base.update(preds, target, sequence_id, mask)

    def compute(self) -> dict[str, Any]:
        result = self.base.compute()

        metrics = {
            "summary/AQ": result.total.aq,
            "summary/AQ_FPR": result.total.fpr,
            "summary/AQ_FNR": result.total.fnr,
        }

        if self.show_per_sequence:
            for seq, aq in result.total.aq_per_seq.items():
                metrics[f"seq/{seq}/AQ"] = aq

            for seq, fpr in result.total.fpr_per_seq.items():
                metrics[f"seq/{seq}/AQ_FPR"] = fpr

            for seq, fnr in result.total.fnr_per_seq.items():
                metrics[f"seq/{seq}/AQ_FNR"] = fnr

        if self.show_per_class:
            for cls, res in result.per_class.items():
                name = self.classes[cls]

                metrics[f"class/{name}/AQ"] = res.aq
                metrics[f"class/{name}/AQ_FPR"] = res.fpr
                metrics[f"class/{name}/AQ_FNR"] = res.fnr

        return {self.prefix + k: v for k, v in metrics.items()}


@validation.register(namespace="occupancy")
class SegmentationTrackingQuality(ValidationMetrics):
    """
    Segmentation and Tracking Quality (STQ) metrics for panoptic occupancy
    prediction and tracking.
    """

    # pylint: disable=too-many-instance-attributes

    is_differentiable: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        labels: OmegaConf,
        semantic_classes: Sequence[str] | None = None,
        track_classes: Sequence[str] | None = None,
        mask: str | Collection[str] = "valid",
        per_frame: bool = False,
        prefix: str = "val/",
        show_per_class: bool = True,
        allow_invalid_instances: Literal["disallow", "restrict", "ignore"] = "disallow",
    ) -> None:
        """
        Initialize the metric.

        Args:
            labels (OmegaConf):
                The label configuration for the dataset.
            semantic_classes (Sequence[str] | None, optional):
                The classes to consider for the semantic quality metric. If
                None, all classes except free space will be considered.
                Defaults to None.
            track_classes (Sequence[str] | None, optional):
                The classes to consider for the association quality metric.
                If None, all classes with instance segmentation ground truth
                are considered. Defaults to None.
            mask (str or Collection[str], optional):
                The mask used to identify valid voxels for evaluation. Defaults
                to "valid".
            prefix (str, optional):
                The prefix to use for the metric names. Defaults to "val/".
            show_per_class (bool, optional):
                Whether to show per-class metrics. Defaults to True.
            allow_invalid_instances (str, optional):
                How to handle invalid instance IDs in the ground-truth data.
                Options are:
                - "disallow": Raise an error if invalid instance IDs are found
                  for instance/thing classes.
                - "restrict": Treat invalid instance IDs for instance/thing
                  classes as non-instance (i.e., stuff).
                - "ignore": Ignore invalid instance IDs for instance/thing
                  classes, meaning the respective points/voxels are masked out
                  and not considered for evaluation.
        """
        super().__init__()

        self.prefix = prefix
        self.mask = [mask] if isinstance(mask, str) else mask
        self.classes = config.utils.to_primitive(labels.all)
        self.ignore_index = labels.ignore_index
        self.show_per_class = show_per_class

        # build the class-name to index mapping
        class_map = {n: i for i, n in enumerate(self.classes)}

        # build the set of semantic classes
        if semantic_classes is None:
            semantic_classes = config.utils.to_primitive(labels.semantic)
            semantic_classes = set(semantic_classes)

        semantic_classes = set(semantic_classes)
        semantic_classes = {class_map[n] for n in semantic_classes}

        # build the set of tracked classes
        if track_classes is None:
            track_classes = config.utils.to_primitive(labels.instance)

        track_classes = set(track_classes)
        track_classes = {class_map[n] for n in track_classes}

        # build the set of occupancy classes
        classes_free = config.utils.to_primitive(labels.free)
        classes_occ = set(self.classes) - set(classes_free)
        classes_occ = {class_map[n] for n in classes_occ}

        # build the set of thing classes
        classes_thing = config.utils.to_primitive(labels.thing)
        classes_thing = {class_map[n] for n in classes_thing}

        # build the set of stuff classes
        classes_stuff = config.utils.to_primitive(labels.stuff)
        classes_stuff = {class_map[n] for n in classes_stuff}

        # set class subsets
        self.classes_semantic = semantic_classes
        self.classes_track = track_classes
        self.classes_background = semantic_classes - track_classes
        self.classes_occupancy = classes_occ
        self.classes_thing = classes_thing
        self.classes_stuff = classes_stuff

        # base metric
        self.base = StqMetric(
            num_classes=len(self.classes),
            semantic_classes=semantic_classes,
            track_classes=track_classes,
            per_frame=per_frame,
            allow_invalid_instances=allow_invalid_instances,
        )

    def reset(self) -> None:
        super().reset()
        self.base.reset()

    # pylint: disable-next=arguments-differ
    def update(
        self,
        sample: Sample,
        preds: MetaDict,
    ) -> None:
        # if there are no valid samples in the batch, skip update
        if not sample.is_valid.any():
            return

        # get mask for valid samples in batch
        valid = sample.is_valid

        # build mask for valid voxels
        mask = sample.labels.occupancy.masks
        mask = [v for k, v in mask.items() if k in self.mask]
        mask = torch.stack(mask, dim=1).all(dim=1)
        mask = mask[valid, ...]

        # filter ground truth
        target_sem = sample.labels.occupancy.semantics
        target_sem = target_sem[valid, ...]

        target_iid = sample.labels.occupancy.instance_ids
        target_iid = target_iid[valid, ...]

        target = torch.stack([target_sem, target_iid], dim=-1)

        # filter predictions
        preds_sem = preds.occupancy.semantics.detach()
        preds_sem = preds_sem[valid, ...]

        preds_iid = preds.occupancy.instance_ids.detach()
        preds_iid = preds_iid[valid, ...]

        preds = torch.stack([preds_sem, preds_iid], dim=-1)

        # filter sequence IDs
        sequence_id = [m.sequence_id for m, v in zip(sample.meta, valid) if v]

        # update mask for ignore_index
        mask = mask & (target_sem != self.ignore_index)

        # update metrics
        self.base.update(preds, target, sequence_id, mask)

    def _compute_semantic(self, result: SemanticQualityResult) -> dict[str, Any]:
        # pylint: disable=too-many-locals
        metrics = {}

        # subsets
        thing_miou, _, _ = result.mean_iou(self.classes_thing)
        stuff_miou, _, _ = result.mean_iou(self.classes_stuff)
        track_miou, _, _ = result.mean_iou(self.classes_track)
        background_miou, _, _ = result.mean_iou(self.classes_background)
        metrics |= {
            "subset/thing/mIoU": thing_miou,
            "subset/stuff/mIoU": stuff_miou,
            "subset/track/mIoU": track_miou,
            "subset/background/mIoU": background_miou,
        }

        # binary occupancy
        occ_tp, occ_fp, occ_fn = result.binary_counts(self.classes_occupancy)
        occ_iou, occ_precision, occ_recall = result.binary_iou(self.classes_occupancy)

        metrics |= {
            "occupancy/IoU": occ_iou.item(),
            "occupancy/precision": occ_precision.item(),
            "occupancy/recall": occ_recall.item(),
            "occupancy/TP": occ_tp.item(),
            "occupancy/FP": occ_fp.item(),
            "occupancy/FN": occ_fn.item(),
        }

        # per-class semantics
        if self.show_per_class:
            tp, fp, fn = result.tp, result.fp, result.fn
            iou, prec, recall = (result.iou, result.precision, result.recall)

            for i, name in enumerate(self.classes):
                metrics[f"class/{name}/sq/IoU"] = iou[i].item()
                metrics[f"class/{name}/sq/precision"] = prec[i].item()
                metrics[f"class/{name}/sq/recall"] = recall[i].item()
                metrics[f"class/{name}/sq/TP"] = tp[i].item()
                metrics[f"class/{name}/sq/FP"] = fp[i].item()
                metrics[f"class/{name}/sq/FN"] = fn[i].item()

        return metrics

    def _compute_association(self, result: AssociationQualityResult) -> dict[str, Any]:
        # base AQ stats
        metrics = {
            "summary/AQ_FPR": result.total.fpr,
            "summary/AQ_FNR": result.total.fnr,
        }

        if self.show_per_class:
            for cls, res in result.per_class.items():
                name = self.classes[cls]

                metrics[f"class/{name}/aq/AQ"] = res.aq
                metrics[f"class/{name}/aq/FPR"] = res.fpr
                metrics[f"class/{name}/aq/FNR"] = res.fnr

        return metrics

    def compute(self) -> dict[str, Any]:
        result = self.base.compute()

        # base STQ metrics
        metrics = {
            "summary/STQ": result.stq,
            "summary/mIoU": result.sq,
            "summary/AQ": result.aq,
        }

        # per-class STQ metrics
        if self.show_per_class:
            for c, v in result.stq_per_class.items():
                metrics[f"class/{self.classes[c]}/STQ"] = v

        # SQ and AQ sub-stats
        metrics |= self._compute_semantic(result.semantic)
        metrics |= self._compute_association(result.association)

        return {self.prefix + k: v for k, v in metrics.items()}


def _class_mask(
    classes: Sequence[str], subset: Collection[str], invert: bool = False
) -> torch.Tensor:
    mask = [n in subset for n in classes]
    mask = torch.tensor(mask, dtype=torch.bool)

    if invert:
        mask = ~mask

    return mask
