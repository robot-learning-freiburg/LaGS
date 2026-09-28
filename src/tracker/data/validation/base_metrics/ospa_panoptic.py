# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Optimal Subpattern Assignment (OSPA) Metric for Panoptic Segmentation and Tracking

References:
- Original OSPA metric: https://ieeexplore.ieee.org/abstract/document/4567674
- Adaptation for panoptic tracking (JRDB dataset): https://arxiv.org/abs/2404.01686
"""

import warnings
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Literal, Sequence

import numpy as np
import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors


@dataclass(unsafe_hash=True)
class _SegmentKey:
    frame: int
    sequence: Hashable
    category: int
    instance: int


@dataclass
class _Segment:
    key: _SegmentKey
    area: int


@dataclass(unsafe_hash=True)
class _IntersectionKey:
    pred: _SegmentKey
    target: _SegmentKey


@dataclass
class _Intersection:
    key: _IntersectionKey
    area: int


@dataclass
class _PerSeqAccum:
    """Accumulates per-sequence OSPA scores for one class (or the aggregate total)."""

    ospa: dict = field(default_factory=dict)
    loc: dict = field(default_factory=dict)
    card: dict = field(default_factory=dict)
    tp: dict = field(default_factory=dict)
    fp: dict = field(default_factory=dict)
    fn: dict = field(default_factory=dict)
    gt: dict = field(default_factory=dict)

    def add(
        self,
        seq: Any,
        ospa: float,
        loc: float,
        card: float,
        tp: int,
        fp: int,
        fn: int,
        gt_count: int,
    ) -> None:
        self.ospa[seq] = ospa
        self.loc[seq] = loc
        self.card[seq] = card
        self.tp[seq] = tp
        self.fp[seq] = fp
        self.fn[seq] = fn
        self.gt[seq] = gt_count

    def to_subset_result(self) -> "SubsetResult":
        return SubsetResult(
            self.ospa, self.loc, self.card, self.tp, self.fp, self.fn, self.gt
        )


class SubsetResult:
    """
    OSPA result for a single class (or the mean across classes for ``total``),
    with a per-sequence breakdown.

    The OSPA score decomposes as OSPA^p = OSPA_LOC^p + OSPA_CARD^p, where:
    - OSPA_LOC reflects the quality of matched pairs (localization error)
    - OSPA_CARD reflects the mismatch in instance counts (cardinality error)

    With the default c=1, p=1 this simplifies to OSPA = OSPA_LOC + OSPA_CARD.

    TP, FP, and FN are defined at the track level (no IoU threshold):
    - TP = min(m, n): matched GT-prediction pairs
    - FP = max(0, n - m): unmatched predicted tracks
    - FN = max(0, m - n): unmatched GT tracks
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        ospa_per_seq: dict[Any, float],
        loc_per_seq: dict[Any, float],
        card_per_seq: dict[Any, float],
        tp_per_seq: dict[Any, int],
        fp_per_seq: dict[Any, int],
        fn_per_seq: dict[Any, int],
        gt_per_seq: dict[Any, int],
    ) -> None:
        self._ospa_per_seq = ospa_per_seq
        self._loc_per_seq = loc_per_seq
        self._card_per_seq = card_per_seq
        self._tp_per_seq = tp_per_seq
        self._fp_per_seq = fp_per_seq
        self._fn_per_seq = fn_per_seq
        self._gt_per_seq = gt_per_seq

        # cached scalar values (lazy)
        self._ospa: float | None = None
        self._loc: float | None = None
        self._card: float | None = None

    def _compute_scalars(self) -> None:
        if self._ospa is not None:
            return

        n = len(self._ospa_per_seq)
        if n == 0:
            self._ospa = 0.0
            self._loc = 0.0
            self._card = 0.0
        else:
            self._ospa = sum(self._ospa_per_seq.values()) / n
            self._loc = sum(self._loc_per_seq.values()) / n
            self._card = sum(self._card_per_seq.values()) / n

    @property
    def ospa(self) -> float:
        """Mean OSPA score over evaluated sequences. Returns 0.0 if none exist."""
        self._compute_scalars()
        return self._ospa

    @property
    def loc(self) -> float:
        """Mean localization (LOC) component over evaluated sequences."""
        self._compute_scalars()
        return self._loc

    @property
    def card(self) -> float:
        """Mean cardinality (CARD) component over evaluated sequences."""
        self._compute_scalars()
        return self._card

    @property
    def ospa_per_seq(self) -> dict[Any, float]:
        """Per-sequence OSPA. Sequences with no GT and no predictions are excluded."""
        return self._ospa_per_seq

    @property
    def loc_per_seq(self) -> dict[Any, float]:
        """Per-sequence LOC. Sequences with no GT and no predictions are excluded."""
        return self._loc_per_seq

    @property
    def card_per_seq(self) -> dict[Any, float]:
        """Per-sequence CARD. Sequences with no GT and no predictions are excluded."""
        return self._card_per_seq

    @property
    def tp(self) -> int:
        """Total matched GT–prediction track pairs."""
        return sum(self._tp_per_seq.values())

    @property
    def fp(self) -> int:
        """Total unmatched predicted tracks."""
        return sum(self._fp_per_seq.values())

    @property
    def fn(self) -> int:
        """Total unmatched GT tracks."""
        return sum(self._fn_per_seq.values())

    @property
    def gt(self) -> int:
        """Total GT instance/track count."""
        return sum(self._gt_per_seq.values())

    @property
    def tp_per_seq(self) -> dict[Any, int]:
        """Matched GT–prediction pairs per sequence."""
        return self._tp_per_seq

    @property
    def fp_per_seq(self) -> dict[Any, int]:
        """Unmatched predicted tracks per sequence."""
        return self._fp_per_seq

    @property
    def fn_per_seq(self) -> dict[Any, int]:
        """Unmatched GT tracks per sequence."""
        return self._fn_per_seq

    @property
    def gt_per_seq(self) -> dict[Any, int]:
        """GT instance/track count per sequence."""
        return self._gt_per_seq


class Result:
    """
    Proxy object for Panoptic OSPA metrics.

    Exposes two views of the computed scores:

    - ``total``: a :class:`SubsetResult` that averages over all non-ignored
      classes. For each sequence the value is the mean over classes with any
      GT or any prediction in that sequence; the overall scalar is then the
      mean across sequences (equal weight per sequence).

    - ``per_class``: a ``dict[int, SubsetResult]`` mapping each non-ignored
      class ID to its per-sequence SubsetResult. Each scalar is the mean of
      the per-sequence values for that class.
    """

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        per_class: dict[int, SubsetResult],
        total: SubsetResult,
    ) -> None:
        self.per_class = per_class
        self.total = total

    def mean_ospa(
        self, classes: Collection[int] | None = None
    ) -> tuple[float, float, float]:
        """
        Compute mean OSPA, LOC, and CARD over a selected subset of classes.

        For each sequence, the mean is taken over the selected classes that
        have at least one GT or one prediction in that sequence; the overall
        scalar is then the mean across sequences (equal weight per sequence).

        Args:
            classes (Collection[int] | None):
                Classes to include. If None, all non-ignored classes are
                considered. Class IDs that are ignored or absent from
                ``per_class`` are silently skipped.

        Returns:
            tuple[float, float, float]:
                Mean OSPA, LOC, CARD. All zero if no sequence contributes.
        """
        if classes is None:
            cls_list = list(self.per_class.keys())
        else:
            cls_list = [c for c in classes if c in self.per_class]

        # collect every sequence touched by any selected class
        all_seqs: set = set()
        for cls in cls_list:
            all_seqs.update(self.per_class[cls].ospa_per_seq.keys())

        seq_ospas: list[float] = []
        seq_locs: list[float] = []
        seq_cards: list[float] = []
        for seq in all_seqs:
            present = [c for c in cls_list if seq in self.per_class[c].ospa_per_seq]
            if not present:
                continue
            n = len(present)
            seq_ospas.append(
                sum(self.per_class[c].ospa_per_seq[seq] for c in present) / n
            )
            seq_locs.append(
                sum(self.per_class[c].loc_per_seq[seq] for c in present) / n
            )
            seq_cards.append(
                sum(self.per_class[c].card_per_seq[seq] for c in present) / n
            )

        if not seq_ospas:
            return 0.0, 0.0, 0.0

        n = len(seq_ospas)
        return (
            sum(seq_ospas) / n,
            sum(seq_locs) / n,
            sum(seq_cards) / n,
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


class PanopticOspa(Metric):
    """
    Panoptic OSPA metric for panoptic segmentation and tracking.

    Computes the Optimal Subpattern Assignment (OSPA) metric adapted for
    panoptic segmentation, supporting both single-frame and tracking evaluation.

    In single-frame mode (per_frame=True), each frame is evaluated independently:
    segments within a frame are matched to minimize the total assignment cost.

    In tracking mode (per_frame=False), tracks are matched globally across all
    frames within each sequence. The cost between two tracks is the average
    IoU-based distance over all frames where at least one track exists
    (OSPA2 accumulation strategy).

    The OSPA score for a set of m GT tracks and n predicted tracks is:

        OSPA = ((optimal_cost + c^p * |m - n|) / max(m, n))^(1/p)

    where the cost matrix D[i, j] = min(avg_d[i, j], c)^p and
    avg_d[i, j] = 1 - iou_sum[i,j] / union_frame_count[i,j].

    Refer to https://arxiv.org/abs/2404.01686 for more details.
    """

    # pylint: disable=too-many-instance-attributes

    is_differentiable: bool = False
    full_state_update: bool = False

    segments_pred: list[_Segment]
    segments_target: list[_Segment]
    segments_intersect: list[_Intersection]

    def __init__(
        self,
        num_classes: int,
        ignored_classes: Collection[int] | None = None,
        per_frame: bool = False,
        c: float = 1.0,
        p: float = 1.0,
        allow_invalid_instances: Literal["disallow", "ignore", "merge"] = "disallow",
    ) -> None:
        """
        Initialize the metric.

        Args:
            num_classes (int):
                The number of classes in the dataset.
            ignored_classes (Collection[int] | None):
                Classes to ignore. GT segments of ignored classes are excluded
                (no FN), and GT pixels of ignored classes act as void for
                predicted segments (pred area in ignored regions is masked out).
                Defaults to None.
            per_frame (bool):
                If True, evaluate each frame independently (panoptic segmentation
                mode). If False, match tracks across frames within each sequence
                (panoptic tracking mode). Defaults to False.
            c (float):
                OSPA cutoff parameter. Distances are capped at c. Must be > 0.
                Defaults to 1.0.
            p (float):
                OSPA order parameter. Higher values penalize large errors more
                heavily. Defaults to 1.0 (linear).
            allow_invalid_instances (str):
                How to handle target pixels with a negative instance ID.
                "disallow" raises an error. "ignore" excludes those pixels from
                all computations (treated as void). "merge" keeps the legacy
                behaviour: all invalid-instance pixels of the same class form one
                combined segment that participates in normal matching.
        """
        super().__init__()

        assert c > 0.0, "Cutoff c must be positive."
        assert p > 0.0, "Order p must be positive."
        assert allow_invalid_instances in {"disallow", "ignore", "merge"}

        self.num_classes = num_classes
        self.ignored_classes = set(ignored_classes) if ignored_classes else set()
        self.per_frame = per_frame
        self.c = c
        self.p = p
        self.allow_invalid_instances = allow_invalid_instances

        self.frame_count = 0

        # Strides for integer encoding of (frame, seq, class, instance) into a single int64.
        # Class and instance are shifted by +1 so NO_INSTANCE (-1) maps to 0.
        # Valid classes [0, num_classes-1] map to [1, num_classes].
        self._instance_stride = 1 << 25
        self._class_stride = num_classes + 1

        ignored = torch.tensor(sorted(self.ignored_classes), dtype=torch.long)
        self.register_buffer("ignored_classes_tensor", ignored, persistent=False)

        # Boolean LUT (size num_classes + 2, class IDs shifted by +1 at lookup).
        # Index 0 and num_classes+1 are sentinels; out-of-range IDs clamp there.
        is_ignored_lut = _make_lut(num_classes, self.ignored_classes, sentinel=True)
        self.register_buffer("is_ignored_lut", is_ignored_lut, persistent=False)

        self.add_state("segments_pred", default=[], dist_reduce_fx=None)
        self.add_state("segments_target", default=[], dist_reduce_fx=None)
        self.add_state("segments_intersect", default=[], dist_reduce_fx=None)

    def reset(self) -> None:
        super().reset()

        self.frame_count = 0

    def _get_local_sequence_index(
        self, sequence_id: torch.Tensor | Sequence[Hashable]
    ) -> tuple[torch.Tensor, dict[int, Hashable]]:
        """
        Convert arbitrary sequence IDs to batch-local integer indices.

        Args:
            sequence_id (torch.Tensor [B] | Sequence[Hashable]):
                Sequence identifier for each sample in the batch.

        Returns:
            tuple[torch.Tensor, dict[int, Hashable]]:
                - local_indices [B]: integer index per sample, unique within
                  this batch.
                - index_to_id: mapping from local index back to original
                  sequence ID.
        """
        if torch.is_tensor(sequence_id):
            sequence_id = sequence_id.tolist()

        # dict.fromkeys preserves order and deduplicates
        unique_ids = list(dict.fromkeys(sequence_id))
        id_to_index = {sid: i for i, sid in enumerate(unique_ids)}
        index_to_id = {i: sid for sid, i in id_to_index.items()}

        local = torch.tensor(
            [id_to_index[sid] for sid in sequence_id], dtype=torch.long
        )
        return local, index_to_id

    def _prepare_input(
        self,
        x: torch.Tensor,
        sequence_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flatten input and prepend (frame_id, seq_id) to each pixel entry.

        Args:
            x (torch.Tensor [B, *D, 2]):
                Input tensor with (class, instance) per pixel.
            sequence_index (torch.Tensor [B]):
                Batch-local sequence index per sample.
            mask (torch.Tensor [B, *D]):
                Boolean validity mask.

        Returns:
            torch.Tensor [N, 4]:
                Flattened valid pixels: (frame_id, seq_id, class, instance).
        """
        batch_size = x.shape[0]

        x = x.detach().flatten(1, -2).to(dtype=torch.long)

        # always assign a unique frame ID per batch sample
        frame_id = torch.arange(batch_size, device=x.device)
        frame_id = frame_id.view(batch_size, 1, 1).expand(-1, x.shape[1], 1)

        if self.per_frame:
            # in per_frame mode each frame is its own sequence
            seq = frame_id
        else:
            seq = sequence_index.to(device=x.device)
            seq = seq.view(batch_size, 1, 1).expand(-1, x.shape[1], 1)

        x = torch.cat((frame_id, seq, x), dim=-1)
        return x[mask.flatten(1, -1), :]

    def _encode(self, x: torch.Tensor, seq_stride: int) -> torch.Tensor:
        """Encode [K, 4] = (frame, seq, class, instance) into a single int64 per row."""
        frame, seq = x[:, 0], x[:, 1]
        cls = x[:, 2] + 1  # shift: -1 → 0
        inst = x[:, 3] + 1  # shift: -1 → 0

        val = frame
        val = val * seq_stride + seq
        val = val * self._class_stride + cls
        val = val * self._instance_stride + inst

        return val

    def _decode(self, keys: torch.Tensor, seq_stride: int) -> list[tuple]:
        """Decode int64 keys back to (frame, seq, class, instance) tuples."""
        inst = ((keys % self._instance_stride) - 1).tolist()
        rem = keys // self._instance_stride

        cls = ((rem % self._class_stride) - 1).tolist()
        rem = rem // self._class_stride

        seq = (rem % seq_stride).tolist()
        frame = (rem // seq_stride).tolist()

        return list(zip(frame, seq, cls, inst))

    def _remap_to_global(
        self,
        pr_area: list[_Segment],
        gt_area: list[_Segment],
        isect: list[_Intersection],
        seq_idx_to_id: dict[int, Hashable],
    ) -> None:
        """Remap local frame/sequence indices to global IDs in-place.

        Adds ``self.frame_count`` to each frame ID (making frame IDs globally
        unique across batches) and either replaces the sequence ID with the
        global original (tracking mode) or with the remapped frame ID
        (per_frame mode, where each frame is its own independent sequence).
        """
        offset = self.frame_count

        for seg in pr_area:
            seg.key.frame += offset
            seg.key.sequence = (
                seg.key.frame if self.per_frame else seq_idx_to_id[seg.key.sequence]
            )

        for seg in gt_area:
            seg.key.frame += offset
            seg.key.sequence = (
                seg.key.frame if self.per_frame else seq_idx_to_id[seg.key.sequence]
            )

        for seg in isect:
            seg.key.pred.frame += offset
            seg.key.target.frame += offset
            if self.per_frame:
                seg.key.pred.sequence = seg.key.pred.frame
                seg.key.target.sequence = seg.key.target.frame
            else:
                seg.key.pred.sequence = seq_idx_to_id[seg.key.pred.sequence]
                seg.key.target.sequence = seq_idx_to_id[seg.key.target.sequence]

    # pylint: disable-next=arguments-differ
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sequence_id: torch.Tensor | Sequence[Hashable],
        mask: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric with new predictions and targets.

        Args:
            preds (torch.Tensor [B, *D, 2]):
                Predicted panoptic segmentation. preds[..., 0] is the class ID,
                preds[..., 1] the instance ID. All classes (stuff and thing) are
                included with their natural instance IDs (stuff: 0).
            target (torch.Tensor [B, *D, 2]):
                Target panoptic segmentation, same structure as preds.
            sequence_id (torch.Tensor [B] | Sequence[Hashable]):
                Sequence identifier for each sample in the batch. In per_frame
                mode this is only used for reporting; each sample is evaluated
                independently regardless of its sequence ID.
            mask (torch.Tensor [B, *D] | None):
                Optional boolean mask. Pixels where mask is False are excluded
                from all computations. Defaults to None (all pixels valid).
        """
        # pylint: disable=too-many-locals
        batch_size = preds.shape[0]

        local_seq_idx, seq_idx_to_id = self._get_local_sequence_index(sequence_id)

        # void mask via LUT (replaces torch.isin); out-of-range class IDs (e.g. -1, 255)
        # clamp to the True sentinels and are treated as void too
        n = len(self.is_ignored_lut) - 1
        is_void = self.is_ignored_lut[(target[..., 0] + 1).clamp(0, n)]

        valid = (~is_void) if mask is None else (mask & ~is_void)

        if self.allow_invalid_instances == "disallow":
            # instance IDs < 0 are invalid; no valid pixel can have an invalid instance ID
            assert not (valid & (target[..., 1] < 0)).any()
        elif self.allow_invalid_instances == "ignore":
            valid = valid & (target[..., 1] >= 0)
        elif self.allow_invalid_instances == "merge":
            # all invalid-instance pixels of the same class will be merged into
            # one segment via the encoding step below
            pass

        preds_flat = self._prepare_input(preds, local_seq_idx, valid)
        target_flat = self._prepare_input(target, local_seq_idx, valid)

        if preds_flat.shape[0] == 0:
            self.frame_count += batch_size
            return

        n_local_seqs = int(local_seq_idx.max().item()) + 1 if batch_size > 0 else 1
        # seq_stride must exceed both the max local sequence index and the max
        # batch-local frame index (which goes up to batch_size - 1), so that
        # encoding is unambiguous when all samples belong to a single sequence.
        seq_stride = max(n_local_seqs, batch_size)

        pred_keys = self._encode(preds_flat, seq_stride)
        tgt_keys = self._encode(target_flat, seq_stride)

        pred_unique, pred_inv, pred_counts = torch.unique(
            pred_keys, return_inverse=True, return_counts=True
        )
        tgt_unique, tgt_inv, tgt_counts = torch.unique(
            tgt_keys, return_inverse=True, return_counts=True
        )

        pred_keys_dec = self._decode(pred_unique, seq_stride)
        tgt_keys_dec = self._decode(tgt_unique, seq_stride)

        pr_area = [
            _Segment(_SegmentKey(*k), int(a))
            for k, a in zip(pred_keys_dec, pred_counts.tolist())
        ]
        gt_area = [
            _Segment(_SegmentKey(*k), int(a))
            for k, a in zip(tgt_keys_dec, tgt_counts.tolist())
        ]

        # Intersections via compact pair keys, reuse inverses from unique above
        n_tgt = len(tgt_unique)
        pair_keys = pred_inv.long() * n_tgt + tgt_inv.long()
        pair_unique, pair_cnts = torch.unique(pair_keys, return_counts=True)
        pi_arr = (pair_unique // n_tgt).tolist()
        ti_arr = (pair_unique % n_tgt).tolist()
        isect = [
            _Intersection(
                _IntersectionKey(
                    _SegmentKey(*pred_keys_dec[pi]), _SegmentKey(*tgt_keys_dec[ti])
                ),
                int(cnt),
            )
            for pi, ti, cnt in zip(pi_arr, ti_arr, pair_cnts.tolist())
        ]

        # only same-class pairs are meaningful for per-class OSPA
        isect = [s for s in isect if s.key.pred.category == s.key.target.category]

        self._remap_to_global(pr_area, gt_area, isect, seq_idx_to_id)
        self.frame_count += batch_size

        # pylint: disable=no-member
        self.segments_pred += pr_area
        self.segments_target += gt_area
        self.segments_intersect += isect

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Any | None = None,
    ) -> None:
        # pylint: disable=too-many-locals

        state = (self.segments_pred, self.segments_target, self.segments_intersect)
        world_size = dist.get_world_size(process_group)
        synced = [([], [], []) for _ in range(world_size)]

        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            dist.all_gather_object(synced, state, group=process_group)

        # always reindex frame IDs to avoid collisions between processes;
        # OSPA always stores per-frame data (unlike AQ which uses frame=0 in
        # tracking mode)
        n = 0
        for pred, target, intersect in synced:
            max_frame_p = max((x.key.frame for x in pred), default=-1)
            max_frame_t = max((x.key.frame for x in target), default=-1)
            max_frame = max(max_frame_p, max_frame_t)

            for p in pred:
                p.key.frame += n
                if self.per_frame:
                    p.key.sequence = p.key.frame

            for t in target:
                t.key.frame += n
                if self.per_frame:
                    t.key.sequence = t.key.frame

            for i in intersect:
                i.key.pred.frame += n
                i.key.target.frame += n
                if self.per_frame:
                    i.key.pred.sequence = i.key.pred.frame
                    i.key.target.sequence = i.key.target.frame

            n = n + max_frame + 1

        preds, targets, intersects = zip(*synced)
        preds = [x for proc in preds for x in proc]
        targets = [x for proc in targets for x in proc]
        intersects = [x for proc in intersects for x in proc]

        self.segments_pred = preds
        self.segments_target = targets
        self.segments_intersect = intersects

    def _build_index(
        self,
    ) -> tuple[
        dict[tuple, dict[int, dict[int, float]]],
        dict[tuple, dict[int, dict[int, float]]],
        dict[tuple, dict[tuple, float]],
        list[Any],
    ]:
        """
        Group accumulated state into per-(seq, cls) lookup structures.

        Returns:
            gt_segments:
                (seq, cls) → {instance: {frame: pixel_area}}
            pred_segments:
                (seq, cls) → {instance: {frame: pixel_area}}
            iou_sums:
                (seq, cls) → {(gt_inst, pred_inst): accumulated_iou}
                Only pairs with at least one frame of pixel overlap are present.
                Frames where both tracks appear but don't overlap contribute 0 to
                the numerator while still growing the union-frame denominator via
                the per-instance frame sets in gt_segments / pred_segments.
            all_sequences:
                All sequence IDs seen, in insertion order.
        """
        # pylint: disable=no-member, too-many-locals

        gt_segments = defaultdict(lambda: defaultdict(dict))
        seen_seqs: dict[Any, None] = {}

        for seg in self.segments_target:
            k, area = seg.key, seg.area

            gt_segments[(k.sequence, k.category)][k.instance][k.frame] = area
            seen_seqs[k.sequence] = None

        pred_segments = defaultdict(lambda: defaultdict(dict))

        for seg in self.segments_pred:
            k, area = seg.key, seg.area

            pred_segments[(k.sequence, k.category)][k.instance][k.frame] = area
            seen_seqs[k.sequence] = None

        # iou_sums[(seq, cls)][(gt_inst, pred_inst)] accumulates the sum of per-frame
        #
        # IoU values for each (GT, pred) pair that ever had pixel overlap. Only
        # non-zero contributions are stored here; the union-frame-count denominator
        # is computed separately in _compute_seq_cls from the per-instance frame sets,
        # so frames where both tracks appear but don't overlap still count in that
        # denominator even though they contribute 0 to the IoU sum here.
        iou_sums = defaultdict(lambda: defaultdict(float))
        for seg in self.segments_intersect:
            pk = seg.key.pred
            tk = seg.key.target

            assert tk.sequence == pk.sequence
            assert tk.category == pk.category
            assert tk.frame == pk.frame

            key = (tk.sequence, tk.category)
            frame = tk.frame

            gt_area = gt_segments[key][tk.instance][frame]
            pred_area = pred_segments[key][pk.instance][frame]

            inter_area = seg.area
            union = gt_area + pred_area - inter_area
            iou_sums[key][(tk.instance, pk.instance)] += inter_area / union

        return gt_segments, pred_segments, iou_sums, list(seen_seqs)

    def _compute_seq_cls(
        self,
        gt_seg: dict[int, dict[int, float]],
        pred_seg: dict[int, dict[int, float]],
        iou_sums: dict[tuple, float],
    ) -> tuple[float, float, float, int, int, int] | None:
        """
        Compute OSPA components for one (seq, cls) pair.

        Returns None when both gt_seg and pred_seg are empty (nothing to
        evaluate).  When exactly one side is empty the result is a pure
        cardinality error: OSPA = LOC = c, CARD = c, and all count in
        the appropriate FP or FN bucket.

        Args:
            gt_seg: instance → {frame: pixel_area} for GT tracks.
            pred_seg: instance → {frame: pixel_area} for predicted tracks.
            iou_sums: (gt_inst, pred_inst) → accumulated per-frame IoU.

        Returns:
            (ospa, loc, card, tp, fp, fn), or None if nothing to evaluate.
        """
        # pylint: disable=too-many-locals

        c = self.c
        p = self.p
        c_p = c**p

        gt_ids = sorted(gt_seg.keys())
        pred_ids = sorted(pred_seg.keys())
        m = len(gt_ids)
        n = len(pred_ids)

        if m == 0 and n == 0:
            return None

        if m == 0 or n == 0:
            # pure cardinality error: no matches possible
            return c, 0.0, c, 0, n if m == 0 else 0, m if n == 0 else 0

        # collect every frame where any track of this (seq, cls) appears
        all_frames: set[int] = set()
        for inst in gt_ids:
            all_frames.update(gt_seg[inst].keys())
        for inst in pred_ids:
            all_frames.update(pred_seg[inst].keys())
        frame_to_idx = {f: i for i, f in enumerate(sorted(all_frames))}
        f_count = len(frame_to_idx)

        # binary presence: gt_mask[f, i] == 1 iff GT track i appears in frame f
        gt_mask = np.zeros((f_count, m), dtype=np.int64)
        for i, inst in enumerate(gt_ids):
            for f in gt_seg[inst]:
                gt_mask[frame_to_idx[f], i] = 1

        pred_mask = np.zeros((f_count, n), dtype=np.int64)
        for j, inst in enumerate(pred_ids):
            for f in pred_seg[inst]:
                pred_mask[frame_to_idx[f], j] = 1

        # union_count[i, j] = |frames_i ∪ frames_j| via inclusion-exclusion:
        #   |A ∪ B| = |A| + |B| - |A ∩ B|
        # The inner product gt_mask.T @ pred_mask gives |frames_i ∩ frames_j| for
        # every (i, j) pair simultaneously.
        cooccur = gt_mask.T @ pred_mask  # [m, n]
        size_gt = gt_mask.sum(axis=0)  # [m]
        size_pred = pred_mask.sum(axis=0)  # [n]
        union_count = size_gt[:, None] + size_pred[None, :] - cooccur  # [m, n]

        # iou_total[i, j] = sum of per-frame IoU for GT track i and pred track j.
        # Dividing by union_count gives the mean IoU over all frames where either
        # track existed (OSPA2 temporal accumulation strategy).
        gt_id_to_idx = {gid: i for i, gid in enumerate(gt_ids)}
        pred_id_to_idx = {pid: j for j, pid in enumerate(pred_ids)}
        iou_total = np.zeros((m, n), dtype=np.float64)
        for (gt_inst, pred_inst), v in iou_sums.items():
            iou_total[gt_id_to_idx[gt_inst], pred_id_to_idx[pred_inst]] = v

        # avg_d[i, j] = 1 - mean_IoU(i, j): distance in [0, 1], 0 = perfect overlap.
        # Capped at c and raised to power p to form the OSPA cost matrix entry.
        avg_d = 1.0 - iou_total / union_count
        distance = np.minimum(avg_d, c) ** p

        # Optimal assignment minimises total cost; unmatched tracks contribute c^p each.
        match_gt, match_pred = linear_sum_assignment(distance)
        optimal_cost = distance[match_gt, match_pred].sum()

        # OSPA^p = (assignment_cost + c^p * |m - n|) / max(m, n)
        # Split into LOC (matched-pair quality) and CARD (count mismatch penalty),
        # then apply the p-th root to recover values in the original distance space.
        max_mn = max(m, n)
        loc_raw = optimal_cost / max_mn
        card_raw = c_p * abs(m - n) / max_mn
        ospa_raw = loc_raw + card_raw

        return (
            ospa_raw ** (1.0 / p),
            loc_raw ** (1.0 / p),
            card_raw ** (1.0 / p),
            min(m, n),
            max(0, n - m),
            max(0, m - n),
        )

    def compute(self) -> Result:
        """
        Compute the Panoptic OSPA metric from accumulated state.

        Returns:
            Result:
                Per-(sequence, class) OSPA values with ``total`` and
                ``per_class`` SubsetResults. (Sequence, class) pairs with no
                GT and no predictions are excluded from averages.
        """
        # pylint: disable=too-many-locals
        gt_segments, pred_segments, iou_sums, all_sequences = self._build_index()

        valid_cls = [
            c for c in range(self.num_classes) if c not in self.ignored_classes
        ]

        per_class: dict[int, _PerSeqAccum] = {cls: _PerSeqAccum() for cls in valid_cls}
        total = _PerSeqAccum()

        for seq in all_sequences:
            seq_ospas: list[float] = []
            seq_locs: list[float] = []
            seq_cards: list[float] = []
            seq_tp = seq_fp = seq_fn = seq_gt = 0

            for cls in valid_cls:
                sc = (seq, cls)
                gt_seg = gt_segments.get(sc, {})
                pred_seg = pred_segments.get(sc, {})

                result = self._compute_seq_cls(gt_seg, pred_seg, iou_sums.get(sc, {}))
                if result is None:
                    continue

                ospa, loc, card, tp, fp, fn = result
                m = len(gt_seg)

                per_class[cls].add(seq, ospa, loc, card, tp, fp, fn, m)

                seq_ospas.append(ospa)
                seq_locs.append(loc)
                seq_cards.append(card)
                seq_tp += tp
                seq_fp += fp
                seq_fn += fn
                seq_gt += m

            if seq_ospas:
                k = len(seq_ospas)
                total.add(
                    seq,
                    sum(seq_ospas) / k,
                    sum(seq_locs) / k,
                    sum(seq_cards) / k,
                    seq_tp,
                    seq_fp,
                    seq_fn,
                    seq_gt,
                )

        return Result(
            per_class={cls: acc.to_subset_result() for cls, acc in per_class.items()},
            total=total.to_subset_result(),
        )
