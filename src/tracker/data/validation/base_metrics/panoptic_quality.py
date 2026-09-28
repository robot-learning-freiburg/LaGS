# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Collection, Literal, Self

import torch
from torchmetrics import Metric

# Segment "color" (batch, class, and instance IDs)
_Color = tuple[int, int, int]


class Result:
    """
    Proxy object to compute panoptic quality metrics, such as PQ and PQ†,
    from intermediate results.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        gt: torch.Tensor,
        tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor,
        iou_gt: torch.Tensor,
        iou_tp: torch.Tensor,
        thing_mask: torch.Tensor,
    ) -> None:
        """
        Initialize a new PanopticQualityResult instance.

        Args:
            gt (torch.Tensor [num_classes]):
                The number of ground truth segments per class.
            tp (torch.Tensor [num_classes]):
                The number of true positives per class.
            fp (torch.Tensor [num_classes]):
                The number of false positives per class.
            fn (torch.Tensor [num_classes]):
                The number of false negatives per class.
            iou_gt (torch.Tensor [num_classes]):
                The sum of all IoU scores of all segments, per class.
            iou_tp (torch.Tensor [num_classes]):
                The sum of all IoU scores of all true positives, per class.
        """

        self._gt = gt
        self._tp = tp
        self._fp = fp
        self._fn = fn
        self._iou_gt = iou_gt
        self._iou_tp = iou_tp
        self._thing_mask = thing_mask

        self._pq_valid = None
        self._pq = None
        self._sq = None
        self._rq = None

        self._mod_pq_valid = None
        self._mod_pq = None
        self._aiou = None

    def _compute_pq(self) -> None:
        """
        Compute Panoptic Quality (PQ) metrics, including SQ (Segmentation Quality)
        and RQ (Recognition Quality).
        """
        if self._pq_valid is not None:
            return

        denom = self._tp + 0.5 * self._fp + 0.5 * self._fn
        sq = torch.where(self._tp > 0.0, self._iou_tp / self._tp, 0.0)
        rq = torch.where(denom > 0.0, self._tp / denom, 0.0)
        pq = sq * rq

        self._pq_valid = denom > 0.0
        self._pq = pq
        self._sq = sq
        self._rq = rq

    def _compute_mod_pq(self) -> None:
        """
        Compute modified Panoptic Quality (PQ†) metrics, including aIoU (average IoU).
        """

        if self._mod_pq_valid is not None:
            return

        self._compute_pq()

        aiou = torch.where(self.gt > 0.0, self._iou_gt / self._gt, 0.0)
        mod_pq = torch.where(self._thing_mask, self._pq, aiou)
        valid = torch.where(self._thing_mask, self._pq_valid, self._gt > 0.0)

        self._mod_pq_valid = valid
        self._mod_pq = mod_pq
        self._aiou = aiou

    @property
    def gt(self) -> torch.Tensor:
        """
        Number of ground truth segments per class.
        """
        return self._gt

    @property
    def tp(self) -> torch.Tensor:
        """
        Number of true positives per class.
        """
        return self._tp

    @property
    def fp(self) -> torch.Tensor:
        """
        Number of false positives per class.
        """
        return self._fp

    @property
    def fn(self) -> torch.Tensor:
        """
        Number of false negatives per class.
        """
        return self._fn

    @property
    def pq(self) -> torch.Tensor:
        """
        Per-class Panoptic Quality (PQ).
        """
        self._compute_pq()
        return self._pq

    @property
    def sq(self) -> torch.Tensor:
        """
        Per-class Segmentation Quality (SQ).
        """
        self._compute_pq()
        return self._sq

    @property
    def rq(self) -> torch.Tensor:
        """
        Per-class Recognition Quality (RQ).
        """
        self._compute_pq()
        return self._rq

    @property
    def mod_pq(self) -> torch.Tensor:
        """
        Per-class modified Panoptic Quality (PQ†).
        """
        self._compute_mod_pq()
        return self._mod_pq

    @property
    def aiou(self) -> torch.Tensor:
        """
        Per-class average IoU (aIoU).
        """
        self._compute_mod_pq()
        return self._aiou

    def mean_pq(
        self, classes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mean Panoptic Quality (SQ) over all classes.

        Args:
            classes (torch.Tensor [num_classes] optional):
                The classes to consider for the mean. If None, all classes are
                considered. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The mean Panoptic Quality (PQ), Segmentation Quality (SQ), and
                Recognition Quality (RQ).
        """
        self._compute_pq()

        mask = self._pq_valid
        mask = _class_mask(mask, classes)

        pq = torch.mean(self._pq[mask])
        sq = torch.mean(self._sq[mask])
        rq = torch.mean(self._rq[mask])

        return pq, sq, rq

    def mean_mod_pq(
        self, classes: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mean modified Panoptic Quality (PQ†) over all classes.

        Args:
            classes (torch.Tensor [num_classes] optional):
                The classes to consider for the mean. If None, all classes are
                considered. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                The mean modified Panoptic Quality (PQ†) and average IoU (aIoU).
        """
        self._compute_mod_pq()

        mask = self._mod_pq_valid
        mask = _class_mask(mask, classes)

        pqm = torch.mean(self._mod_pq[mask])
        aiou = torch.mean(self._aiou[mask])

        return pqm, aiou

    def to(self, *args, **kwargs) -> Self:
        return Result(
            gt=self._gt.to(*args, **kwargs),
            tp=self._tp.to(*args, **kwargs),
            fp=self._fp.to(*args, **kwargs),
            fn=self._fn.to(*args, **kwargs),
            iou_gt=self._iou_gt.to(*args, **kwargs),
            iou_tp=self._iou_tp.to(*args, **kwargs),
            thing_mask=self._thing_mask.to(*args, **kwargs),
        )


class PanopticQuality(Metric):
    """
    Panoptic Quality (PQ) metric for panoptic segmentation.

    Adapted from the torchmetrics implementation. Compared to the torchmetrics
    implementation, this version computes both standard PQ and modified PQ
    (PQ†) metrics and provides extended statistics.

    Refer to the following papers for more information:
    - PQ, SQ, RQ: https://arxiv.org/abs/1801.00868
    - PQ†: https://arxiv.org/abs/1905.01220
    """

    # pylint: disable=too-many-instance-attributes

    is_differentiable: bool = False
    full_state_update: bool = False

    # buffers
    classes_thing: torch.Tensor
    classes_stuff: torch.Tensor

    # state
    iou_sum_gt: torch.Tensor
    iou_sum_tp: torch.Tensor
    num_gt: torch.Tensor
    num_tp: torch.Tensor
    num_fp: torch.Tensor
    num_fn: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        classes_thing: Collection[int],
        classes_stuff: Collection[int],
        allow_unknown_preds: bool = True,
        allow_invalid_instances: Literal["disallow", "ignore", "merge"] = "disallow",
    ):
        """
        Initialize a new PanopticQuality instance.

        Note: Any class that is not in classes_thing or classes_stuff is
        considered unlabelled/invalid and ignored for evaluation.

        Args:
            num_classes (int):
                The total number of classes in the dataset.
            classes_thing (Collection[int]):
                The class IDs of "thing" classes (countable objects).
            classes_stuff (Collection[int]):
                The class IDs of "stuff" classes (amorphous regions).
            allow_unknown (bool):
                Whether to allow unknown classes (classes that are not in
                classes_thing or classes_stuff). If False, an error is raised
                if any class ID is not in either set. Defaults to True.
            allow_invalid_instances (str):
                How to handle target thing-class pixels with a negative instance
                ID. "disallow" (default) raises an error. "ignore" excludes those
                pixels from all computations (treated as void). "merge" keeps the
                legacy behaviour: all invalid-instance pixels of the same class
                form one combined segment that participates in normal matching.
        """
        super().__init__()

        self.num_classes = num_classes
        self.allow_unknown_preds = allow_unknown_preds
        self.allow_invalid_instances = allow_invalid_instances
        assert allow_invalid_instances in {"disallow", "ignore", "merge"}

        classes_thing = set(classes_thing)
        classes_stuff = set(classes_stuff)
        classes = classes_thing | classes_stuff

        self.void_category = _get_void_category(classes_thing, classes_stuff)

        # make sure that thing and stuff classes are disjoint
        assert not classes_thing & classes_stuff

        # make sure that we have at least one class
        assert self.num_classes > 0

        # make sure that all classes are within the valid range
        assert max(classes_thing) < self.num_classes
        assert max(classes_stuff) < self.num_classes

        # Strides for integer encoding of (batch, class, instance) into a single int64.
        # void_category ≤ num_classes, so num_classes + 1 covers the full class range.
        self._instance_stride = 1 << 25
        self._class_stride = num_classes + 1

        # Boolean LUTs (size num_classes + 2, class IDs shifted by +1 at lookup).
        # Index 0 and num_classes+1 are sentinels; out-of-range IDs clamp there.
        is_thing_lut = _make_lut(num_classes, classes_thing)
        self.register_buffer("is_thing_lut", is_thing_lut, persistent=False)

        is_stuff_lut = _make_lut(num_classes, classes_stuff)
        self.register_buffer("is_stuff_lut", is_stuff_lut, persistent=False)

        # set up class tensors
        classes_thing = torch.tensor(sorted(classes_thing), dtype=torch.long)
        self.register_buffer("classes_thing", classes_thing, persistent=False)

        classes_stuff = torch.tensor(sorted(classes_stuff), dtype=torch.long)
        self.register_buffer("classes_stuff", classes_stuff, persistent=False)

        classes = torch.tensor(sorted(classes), dtype=torch.long)
        self.register_buffer("classes", classes, persistent=False)

        # set up state tensors
        iou = torch.zeros(self.num_classes, dtype=torch.double)
        self.add_state("iou_sum_gt", default=iou.clone(), dist_reduce_fx="sum")
        self.add_state("iou_sum_tp", default=iou.clone(), dist_reduce_fx="sum")

        count = torch.zeros(self.num_classes, dtype=torch.long)
        self.add_state("num_gt", default=count.clone(), dist_reduce_fx="sum")
        self.add_state("num_tp", default=count.clone(), dist_reduce_fx="sum")
        self.add_state("num_fp", default=count.clone(), dist_reduce_fx="sum")
        self.add_state("num_fn", default=count.clone(), dist_reduce_fx="sum")

    def _sanitize_invalid_instances(
        self,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Update mask to handle thing-class target pixels with negative instance IDs."""

        if self.allow_invalid_instances == "merge":
            # legacy behaviour: all invalid-instance pixels of the same class
            # will be merged into one segment via the encoding step below, so
            # no change to the mask is needed
            return mask

        # identify invalid-instance pixels: thing-class pixels where instance ID < 0
        n = len(self.is_thing_lut) - 1
        safe_cls = (target[..., 0] + 1).clamp(0, n)
        is_thing = self.is_thing_lut[safe_cls]
        is_invalid = is_thing & (target[..., 1] < 0)

        # "disallow": assert that there are no invalid-instance pixels in the valid set
        if self.allow_invalid_instances == "disallow":
            checked = is_invalid if mask is None else (mask & is_invalid)
            assert not checked.any()
            return mask

        # "ignore": exclude invalid-instance pixels from all computations
        if mask is None:
            return ~is_invalid

        return mask & ~is_invalid

    def _prepare_input(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        allow_unknown: bool,
    ) -> torch.Tensor:
        """
        Prepare input tensor for processing.

        Args:
            x (torch.Tensor [B, *D, 2]):
                The input tensor, where x[..., 0] is the class ID and x[..., 1] the
                instance ID.
            mask (torch.Tensor [B, *D] optional):
                An optional mask to filter valid samples in the batch. Only samples
                where mask is True are considered for evaluation.
            allow_unknown (bool):
                Whether to allow unknown classes (classes that are not in
                classes_thing or classes_stuff). If False, an error is raised
                if any class ID is not in either set.

        Returns:
            torch.Tensor [N, 3]:
                The prepared input tensor, where x[..., 0] is the batch ID, x[..., 1]
                is the class ID, and x[..., 2] is the instance ID.
        """
        batch_size, *_, c = x.shape
        assert c == 2

        # detach input to avoid side effects
        x = x.detach().flatten(1, -2)

        # add batch ID to all elements
        batch_id = torch.arange(batch_size, device=x.device)
        batch_id = batch_id.view(batch_size, 1, 1).expand(-1, x.shape[1], 1)
        x = torch.cat([batch_id, x], dim=-1)

        # apply mask and flatten input to [N, 3]
        if mask is not None:
            x = x[mask.flatten(1, -1), :]
        else:
            x = x.flatten(0, -2)

        # generate class masks via O(1) LUT lookup (replaces torch.isin).
        # Shift by +1 and clamp so out-of-range IDs map to the sentinel slots
        # at index 0 (negative) or index num_classes+1 (too large), both False.
        n = len(self.is_thing_lut) - 1
        safe_ids = (x[..., 1] + 1).clamp(0, n)
        is_thing = self.is_thing_lut[safe_ids]
        is_stuff = self.is_stuff_lut[safe_ids]

        # make sure that we do not have any unknown classes
        if allow_unknown:
            x[~(is_thing | is_stuff), 1] = self.void_category
            x[~(is_thing | is_stuff), 2] = 0
        else:
            assert torch.all(is_thing | is_stuff)

        # reset/sanitize instance IDs for stuff classes
        x[is_stuff, 2] = 0

        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode [N, 3] = (batch, class, instance) into a single int64 per row.

        Class and instance are shifted by +1 so ``instance=-1`` remains
        representable without aliasing into a different class bucket. Valid
        classes ``[0, num_classes-1]`` map to ``[1, num_classes]`` and
        ``void_category`` maps to ``num_classes + 1``.
        """
        batch = x[:, 0]
        cls = x[:, 1] + 1
        inst = x[:, 2] + 1

        return (batch * self._class_stride + cls) * self._instance_stride + inst

    def _decode(self, keys: torch.Tensor) -> list[_Color]:
        """Decode int64 keys back to (batch, class, instance) color tuples."""
        inst = (keys % self._instance_stride - 1).tolist()
        rem = keys // self._instance_stride

        cls = (rem % self._class_stride - 1).tolist()
        batch = (rem // self._class_stride).tolist()

        return list(zip(batch, cls, inst))

    def _compute_intersections(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        pred_inv: torch.Tensor,
        tgt_inv: torch.Tensor,
        pred_colors: list[_Color],
        tgt_colors: list[_Color],
    ) -> dict[tuple[_Color, _Color], int]:
        """Compute pixel-level intersections between same-class pred/target segments.

        Only pairs where pred and target share a class (and neither is void) are
        included.  Reuses the inverses from the unique() calls so no extra sort
        is needed.

        inter_mask filters to pixels where pred and GT agree on class: cross-class
        overlaps are not intersections and are accounted for in FP/FN instead.
        """
        inter_mask = (preds[:, 1] == target[:, 1]) & (preds[:, 1] != self.void_category)

        n_tgt = len(tgt_colors)
        pair_keys = pred_inv[inter_mask].long() * n_tgt + tgt_inv[inter_mask].long()

        pair_unique, pair_counts = torch.unique(pair_keys, return_counts=True)
        pi_arr = (pair_unique // n_tgt).tolist()
        ti_arr = (pair_unique % n_tgt).tolist()

        return {
            (pred_colors[pi], tgt_colors[ti]): int(cnt)
            for pi, ti, cnt in zip(pi_arr, ti_arr, pair_counts.tolist())
        }

    def _compute_void_areas(
        self,
        pred_inv: torch.Tensor,
        pred_colors: list[_Color],
        target: torch.Tensor,
    ) -> dict[_Color, int]:
        """Return the number of void-target pixels covered by each predicted segment."""
        void_mask = target[:, 1] == self.void_category
        if not void_mask.any():
            return {}

        void_unique, void_counts = torch.unique(pred_inv[void_mask], return_counts=True)

        return {
            pred_colors[int(idx)]: int(cnt)
            for idx, cnt in zip(void_unique.tolist(), void_counts.tolist())
        }

    def _count_tp_fp_fn(
        self,
        intersections: dict[tuple[_Color, _Color], int],
        pred_void_areas: dict[_Color, int],
        pred_area: dict[_Color, int],
        target_area: dict[_Color, int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Count TP/FP/FN and accumulate IoU sums per class.

        Returns (num_gt, num_tp, num_fp, num_fn, iou_sum_gt, iou_sum_tp).
        """
        # pylint: disable=too-many-locals

        num_gt = torch.zeros(self.num_classes, dtype=torch.long)
        num_tp = torch.zeros(self.num_classes, dtype=torch.long)
        num_fp = torch.zeros(self.num_classes, dtype=torch.long)
        num_fn = torch.zeros(self.num_classes, dtype=torch.long)
        iou_sum_gt = torch.zeros(self.num_classes, dtype=torch.double)
        iou_sum_tp = torch.zeros(self.num_classes, dtype=torch.double)

        pred_matched: set[_Color] = set()
        target_matched: set[_Color] = set()

        for (pred_color, target_color), intersect_area in intersections.items():
            assert pred_color[:2] == target_color[:2]

            class_id = target_color[1]
            pred_void_area = pred_void_areas.get(pred_color, 0)

            # Exclude void-overlap from the union (following panopticapi / EOPSN):
            # pred pixels that land on void GT should not count against the prediction,
            # so we subtract them from the pred's contribution to the union area.
            union = (
                pred_area[pred_color]
                - pred_void_area
                + target_area[target_color]
                - intersect_area
            )

            assert union > 0 and union >= intersect_area

            iou = intersect_area / union
            assert iou > 0.0

            iou_sum_gt[class_id] += iou

            if iou > 0.5:
                num_tp[class_id] += 1
                iou_sum_tp[class_id] += iou
                pred_matched.add(pred_color)
                target_matched.add(target_color)

        for _batch, class_id, _inst in target_area:
            if class_id != self.void_category:
                num_gt[class_id] += 1

        # Predictions whose majority (> 50%) overlaps void GT are suppressed as FP:
        # they shouldn't be penalised for predicting in regions the GT left unlabelled.
        for pred_color in pred_area.keys() - pred_matched:
            _batch, class_id, _inst = pred_color
            if class_id == self.void_category:
                continue
            void_area = pred_void_areas.get(pred_color, 0)
            if void_area / pred_area[pred_color] <= 0.5:
                num_fp[class_id] += 1

        for _batch, class_id, _inst in target_area.keys() - target_matched:
            if class_id != self.void_category:
                num_fn[class_id] += 1

        return num_gt, num_tp, num_fp, num_fn, iou_sum_gt, iou_sum_tp

    # pylint: disable-next=arguments-differ
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            preds (torch.Tensor [B, *D 2]):
                The predicted panoptic segmentation, where preds[..., 0] is the
                class ID and preds[..., 1] the instance ID.
            target (torch.Tensor [B, *D, 2]):
                The target panoptic segmentation, where target[..., 0] is the
                class ID and target[..., 1] the instance ID.
            mask (torch.Tensor [B, *D] optional):
                An optional mask to filter valid samples in the batch. Only samples
                where mask is True are considered for evaluation.
        """
        # pylint: disable=too-many-locals

        mask = self._sanitize_invalid_instances(target, mask)

        # sanitize and flatten inputs to [N, 3]: (batch, class, instance)
        preds = self._prepare_input(preds, mask, self.allow_unknown_preds)
        target = self._prepare_input(target, mask, True)

        # encode each pixel as a single int64 and find unique segments in one sort
        pred_keys = self._encode(preds)
        tgt_keys = self._encode(target)

        pred_unique, pred_inv, pred_counts = torch.unique(
            pred_keys, return_inverse=True, return_counts=True
        )
        tgt_unique, tgt_inv, tgt_counts = torch.unique(
            tgt_keys, return_inverse=True, return_counts=True
        )

        pred_colors = self._decode(pred_unique)
        tgt_colors = self._decode(tgt_unique)

        pred_area = dict(zip(pred_colors, pred_counts.tolist()))
        target_area = dict(zip(tgt_colors, tgt_counts.tolist()))

        intersections = self._compute_intersections(
            preds, target, pred_inv, tgt_inv, pred_colors, tgt_colors
        )
        pred_void_areas = self._compute_void_areas(pred_inv, pred_colors, target)

        num_gt, num_tp, num_fp, num_fn, iou_sum_gt, iou_sum_tp = self._count_tp_fp_fn(
            intersections, pred_void_areas, pred_area, target_area
        )

        # pylint: disable=no-member
        self.num_gt += num_gt.to(self.num_gt.device)
        self.num_tp += num_tp.to(self.num_tp.device)
        self.num_fp += num_fp.to(self.num_fp.device)
        self.num_fn += num_fn.to(self.num_fn.device)
        self.iou_sum_gt += iou_sum_gt.to(self.iou_sum_gt.device)
        self.iou_sum_tp += iou_sum_tp.to(self.iou_sum_tp.device)

    def compute(self) -> Result:
        """
        Compute the panoptic quality metrics.

        Returns:
            Result:
                A proxy object that allows to derive various metrics from the
                computed intermediate results as well as the Panoptic Quality
                (PQ) itself.
        """
        # pylint: disable=no-member
        device = self.iou_sum_gt.device

        thing_mask = torch.arange(self.num_classes, device=device)
        thing_mask = torch.isin(thing_mask, self.classes_thing)

        return Result(
            gt=self.num_gt.clone(),
            tp=self.num_tp.clone(),
            fp=self.num_fp.clone(),
            fn=self.num_fn.clone(),
            iou_gt=self.iou_sum_gt.clone(),
            iou_tp=self.iou_sum_tp.clone(),
            thing_mask=thing_mask,
        )


def _make_lut(
    num_classes: int,
    true_at: Collection[int],
    *,
    sentinel: bool = False,
) -> torch.Tensor:
    """Create a boolean look-up table for class IDs in ``[0, num_classes)``.

    The LUT has size ``num_classes + 2`` and uses a +1 shift so that:
    - index 0                → lower sentinel for ``cls < 0``
    - indices 1..num_classes → class IDs 0..num_classes-1
    - index num_classes+1    → upper sentinel for ``cls >= num_classes``

    Interior slots default to ``False``; slots for IDs in ``true_at`` are set
    to ``True``.  Both sentinel slots are set to ``sentinel``.

    Args:
        num_classes: Number of valid class IDs (``[0, num_classes)``).
        true_at: Class IDs whose slot should be ``True``.
        sentinel: Value assigned to both out-of-range sentinel slots.  Use
            ``False`` (default) for membership LUTs so out-of-range IDs are
            treated as non-members; use ``True`` for ignore LUTs so
            out-of-range IDs (e.g. ``-1``, ``255``) are treated as void.

    Look up a class ID tensor with::

        lut[(cls + 1).clamp(0, num_classes + 1)]
    """
    lut = torch.zeros(num_classes + 2, dtype=torch.bool)

    lut[0] = sentinel
    lut[num_classes + 1] = sentinel

    if true_at:
        indices = torch.tensor(sorted(true_at), dtype=torch.long) + 1
        lut[indices] = True

    return lut


def _class_mask(valid: torch.Tensor, classes: torch.Tensor | None) -> torch.Tensor:
    """
    Generate a class mask.

    Args:
        valid (torch.Tensor [num_classes]):
            The valid mask.

        classes (torch.Tensor [num_classes or N] optional):
            The classes to consider for the mask. Either a bool tensor or an
            index tensor. If None, all classes are considered. Defaults to
            None.

    Returns:
        torch.Tensor [num_classes]:
            The class mask.
    """
    if classes is None:
        return valid

    if not torch.is_tensor(classes):
        classes = list(classes)
        classes = torch.tensor(classes, device=valid.device, dtype=torch.int)

    if classes.dtype == torch.bool:
        return valid & classes

    num_classes = valid.shape[0]
    mask = torch.arange(num_classes, device=valid.device)
    mask = torch.isin(mask, classes)

    return valid & mask


def _get_void_category(things: set[int], stuffs: set[int]) -> tuple[int, int]:
    """Get an unused category ID.

    Args:
        things: The set of category IDs for things.
        stuffs: The set of category IDs for stuffs.

    Returns:
        A new category ID that does not belong to things nor stuffs.

    """
    return 1 + max([0, *list(things), *list(stuffs)])
