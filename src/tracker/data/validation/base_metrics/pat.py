# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Panoptic Tracking (PAT) metric and corresponding Tracking Quality (TQ) metric.

PAT is the harmonic mean of Panoptic Quality (PQ) and Tracking Quality (TQ). TQ
itself combines an association score and a track fragmentation component.

References:
- PAT and TQ metrics: https://arxiv.org/abs/2109.03805
"""

import math
import warnings
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Callable, Collection, Literal, Sequence

import torch
import torch.distributed as dist
from torchmetrics import Metric
from torchmetrics.utilities.distributed import gather_all_tensors

from .panoptic_quality import PanopticQuality
from .panoptic_quality import Result as PanopticQualityResult

NO_INSTANCE = -1  # sentinel for "not a valid instance" in input (class, instance) pairs
NO_MATCH = -1  # sentinel for "GT unmatched in this frame" in _GtMatch records

# Segment "color" key (batch, class, instance)
_Color = tuple[int, int, int]


@dataclass(unsafe_hash=True)
class _GtMatchKey:
    sequence: Hashable
    category: int
    gt_instance: int
    frame: int


@dataclass
class _GtMatch:
    key: _GtMatchKey
    pred_instance: int  # NO_MATCH if unmatched


@dataclass(unsafe_hash=True)
class _PredOccKey:
    sequence: Hashable
    pred_instance: int
    frame: int


@dataclass
class _PredOcc:
    key: _PredOccKey


class TrackingQualityResult:
    """
    Proxy object to compute Tracking Quality (TQ) and related per-track and
    per-class statistics from accumulated GT matches and pred occurrences.
    """

    def __init__(
        self,
        gt_matches: list[_GtMatch],
        pred_occurrences: list[_PredOcc],
        classes: Collection[int],
    ) -> None:
        """
        Initialize a new TrackingQualityResult instance.

        Args:
            gt_matches (list[_GtMatch]):
                One entry per (sequence, class, GT instance, frame): the
                predicted instance matched to the GT in that frame, or
                NO_MATCH if unmatched.
            pred_occurrences (list[_PredOcc]):
                One entry per (sequence, predicted instance, frame):
                represents that this predicted instance appeared in that
                frame. Pred occurrences are class-agnostic - the total
                occurrence count for a (sequence, predicted instance) pair
                spans all classes in which that pred ID was predicted.
            classes (Collection[int]):
                The set of tracked classes. Only GT tracks in these classes
                contribute to the aggregate TQ.
        """
        self._gt_matches = gt_matches
        self._pred_occurrences = pred_occurrences
        self._classes = set(classes)

        self._per_track: dict[tuple[Hashable, int, int], float] | None = None
        self._per_track_as: dict[tuple[Hashable, int, int], float] | None = None
        self._per_track_tf: dict[tuple[Hashable, int, int], float] | None = None

    @staticmethod
    def _association_score(
        pr_ids: list[int],
        length: int,
        pred_totals: dict[tuple[Hashable, int], int],
        sequence: Hashable,
    ) -> float:
        """Compute the association score for a single GT track.

        For each pred ID p that was ever matched to this GT track:
            count(p) = frames where GT was matched to p
            total(p) = total frames p appeared in this sequence (all classes)
            fp(p)    = total(p) - count(p)  (frames p existed but wasn't matched here)

            contribution = count(p)^2 / (L + fp(p))

        The score is the sum of contributions divided by L (GT track length).
        A perfect single-pred match with no extra occurrences scores 1.0; ID
        switches or FP occurrences reduce it toward 0.
        """
        matched = [p for p in pr_ids if p != NO_MATCH]
        if not matched:
            return 0.0

        counts: dict[int, int] = defaultdict(int)
        for p in matched:
            counts[p] += 1

        assoc = 0.0
        for pred_inst, count in counts.items():
            total = pred_totals.get((sequence, pred_inst), count)
            fp = total - count
            assoc += (count * count) / (length + fp)

        return assoc / length

    @staticmethod
    def _track_fragmentation(pr_ids: list[int]) -> float:
        """Compute the track fragmentation score (1 - IDS/(L-1)) for a single GT track.

        IDS (identity switches) counts transitions between consecutive frames where
        the matched pred changes, including transitions to/from NO_MATCH (missed frames).
        A track with no switches scores 1.0; one switch per consecutive pair scores 0.
        Returns 1.0 for tracks of length ≤ 1 (no transitions possible).
        """
        length = len(pr_ids)
        if length <= 1:
            return 1.0

        ids = 0
        prev: int | None = None
        for p in pr_ids:
            if prev is not None and (p != prev or prev == NO_MATCH):
                ids += 1
            prev = p

        return 1.0 - ids / (length - 1)

    def _compute(self) -> None:
        if self._per_track is not None:
            return

        # group GT matches by track (sequence, class, gt_instance) -> list of (frame, pred_inst)
        tracks: dict[tuple[Hashable, int, int], list[tuple[int, int]]] = defaultdict(
            list
        )
        for m in self._gt_matches:
            tracks[(m.key.sequence, m.key.category, m.key.gt_instance)].append(
                (m.key.frame, m.pred_instance)
            )

        # count pred occurrences per (sequence, pred_instance) — class-agnostic, matching the
        # nuScenes reference: the same pred ID's total frame count spans all classes in a sequence
        pred_totals: dict[tuple[Hashable, int], int] = defaultdict(int)
        for occ in self._pred_occurrences:
            pred_totals[(occ.key.sequence, occ.key.pred_instance)] += 1

        per_track: dict[tuple[Hashable, int, int], float] = {}
        per_track_as: dict[tuple[Hashable, int, int], float] = {}
        per_track_tf: dict[tuple[Hashable, int, int], float] = {}

        for track_key, entries in tracks.items():
            sequence, _, _ = track_key

            entries.sort(key=lambda x: x[0])
            pr_ids = [pi for _, pi in entries]
            length = len(pr_ids)

            assoc = self._association_score(pr_ids, length, pred_totals, sequence)
            frag = self._track_fragmentation(pr_ids)

            per_track_as[track_key] = assoc
            per_track_tf[track_key] = frag
            per_track[track_key] = math.sqrt(assoc * frag)

        self._per_track = per_track
        self._per_track_as = per_track_as
        self._per_track_tf = per_track_tf

    def _aggregate(
        self, values: dict[tuple[Hashable, int, int], float], classes: set[int] | None
    ) -> float:
        if not values:
            return 0.0

        if classes is None:
            selected = list(values.values())
        else:
            selected = [v for (_s, c, _i), v in values.items() if c in classes]

        if not selected:
            return 0.0

        return sum(selected) / len(selected)

    @property
    def tq(self) -> float:
        """
        Mean Tracking Quality (TQ) over all GT tracks in the configured
        tracked classes.
        """
        return self.mean_tq()

    @property
    def association_score(self) -> float:
        """
        Mean per-track association score over all GT tracks in the configured
        tracked classes.
        """
        return self.mean_association_score()

    @property
    def track_fragmentation(self) -> float:
        """
        Mean per-track track fragmentation score over all GT tracks in the
        configured tracked classes (1 - IDS/(L-1); higher = fewer identity
        switches).
        """
        return self.mean_track_fragmentation()

    def mean_tq(self, classes: Collection[int] | None = None) -> float:
        """
        Mean Tracking Quality (TQ) over all GT tracks in the given class
        subset.

        Args:
            classes (Collection[int], optional):
                The classes to aggregate over. If None, the classes configured
                on the metric are used. Defaults to None.
        """
        self._compute()
        selected = self._classes if classes is None else set(classes)
        return self._aggregate(self._per_track, selected)

    def mean_association_score(self, classes: Collection[int] | None = None) -> float:
        """
        Mean per-track association score over all GT tracks in the given
        class subset.

        Args:
            classes (Collection[int], optional):
                The classes to aggregate over. If None, the classes configured
                on the metric are used. Defaults to None.
        """
        self._compute()
        selected = self._classes if classes is None else set(classes)
        return self._aggregate(self._per_track_as, selected)

    def mean_track_fragmentation(self, classes: Collection[int] | None = None) -> float:
        """
        Mean per-track track fragmentation score over all GT tracks in the
        given class subset.

        Args:
            classes (Collection[int], optional):
                The classes to aggregate over. If None, the classes configured
                on the metric are used. Defaults to None.
        """
        self._compute()
        selected = self._classes if classes is None else set(classes)
        return self._aggregate(self._per_track_tf, selected)

    @property
    def tq_per_class(self) -> dict[int, float]:
        """
        Per-class mean TQ. Classes with no GT tracks are omitted.
        """
        return self._per_class_mean(self._track_values)

    @property
    def association_score_per_class(self) -> dict[int, float]:
        """
        Per-class mean per-track association score. Classes with no GT tracks
        are omitted.
        """
        return self._per_class_mean(self._track_as_values)

    @property
    def track_fragmentation_per_class(self) -> dict[int, float]:
        """
        Per-class mean per-track track fragmentation score. Classes with no
        GT tracks are omitted.
        """
        return self._per_class_mean(self._track_tf_values)

    @property
    def _track_values(self) -> dict[tuple[Hashable, int, int], float]:
        self._compute()
        return self._per_track

    @property
    def _track_as_values(self) -> dict[tuple[Hashable, int, int], float]:
        self._compute()
        return self._per_track_as

    @property
    def _track_tf_values(self) -> dict[tuple[Hashable, int, int], float]:
        self._compute()
        return self._per_track_tf

    def _per_class_mean(
        self, values: dict[tuple[Hashable, int, int], float]
    ) -> dict[int, float]:
        per_class: dict[int, list[float]] = defaultdict(list)
        for (_s, c, _i), v in values.items():
            if c in self._classes:
                per_class[c].append(v)

        return {c: sum(vs) / len(vs) for c, vs in per_class.items()}

    @property
    def num_tracks(self) -> int:
        """
        Number of GT tracks contributing to the aggregate TQ.
        """
        self._compute()
        return sum(1 for (_s, c, _i) in self._per_track if c in self._classes)


class TrackingQuality(Metric):
    """
    Tracking Quality (TQ) metric for panoptic tracking.

    Computes a per-track association score and track fragmentation score and
    combines them as TQ = sqrt(association_score * track_fragmentation), then
    takes the mean over all GT tracks in the tracked classes.

    The caller supplies an explicit `frame_id` for each sample in `update()`.
    Frame IDs must be unique within each sequence (across all batches and all
    processes) and sort into temporal order. Given this, sequences may be
    split across processes freely.

    Refer to https://arxiv.org/abs/2109.03805 for more details.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    # buffers
    track_classes: torch.Tensor
    ignored_classes: torch.Tensor

    # state (list of _GtMatch / _PredOcc, custom _sync_dist)
    gt_matches: list[_GtMatch]
    pred_occurrences: list[_PredOcc]

    def __init__(
        self,
        num_classes: int,
        track_classes: Collection[int],
        ignored_classes: Collection[int] | None = None,
        min_instance_count: int = 0,
        allow_invalid_instances: Literal["disallow", "restrict", "ignore"] = "disallow",
    ):
        """
        Initialize a new TrackingQuality instance.

        Input sanitization assumes methods only assign instance IDs to
        instance (thing) classes - on both sides, pixels whose class is not
        in `track_classes` are treated as non-instance regardless of their
        instance ID. A negative pred instance ID is also treated as
        non-instance as a safeguard.

        Args:
            num_classes (int):
                The total number of classes in the dataset.
            track_classes (Collection[int]):
                The class IDs to compute tracking quality for. Typically the
                "thing" classes.
            ignored_classes (Collection[int], optional):
                Classes to ignore. Regions with these GT labels are masked
                out, so predictions there do not count as false positives.
                Defaults to None.
            min_instance_count (int, optional):
                Minimum per-frame element count for a GT or predicted instance
                to be considered valid. Instances with fewer elements in a
                given frame are ignored for that frame (on both the GT and
                pred sides, matching the nuScenes reference). Defaults to 0.
            allow_invalid_instances (str, optional):
                How to handle target pixels whose class is in `track_classes`
                but whose instance ID is negative. "disallow" (default) asserts
                no such pixel exists. "restrict" treats them as non-instance.
                "ignore" drops them entirely (as if masked out). Defaults to
                "disallow".
        """
        super().__init__()

        self.num_classes = num_classes
        self.min_instance_count = min_instance_count
        self.allow_invalid_instances = allow_invalid_instances
        assert allow_invalid_instances in {"disallow", "restrict", "ignore"}

        track_classes_set = set(track_classes)
        track = torch.tensor(sorted(track_classes_set), dtype=torch.long)
        self.register_buffer("track_classes", track, persistent=False)

        ignored_set = set(ignored_classes) if ignored_classes else set()
        ignored = torch.tensor(sorted(ignored_set), dtype=torch.long)
        self.register_buffer("ignored_classes", ignored, persistent=False)

        # Strides for integer encoding of (batch, class, instance) into a single int64.
        # NO_INSTANCE = -1 is shifted to 0 by +1.
        # Strides for integer encoding of (batch, class, instance) into a single int64.
        # Class and instance are shifted by +1 so NO_INSTANCE (-1) maps to 0.
        # Valid classes [0, num_classes-1] map to [1, num_classes], so the stride
        # must be > num_classes → num_classes + 1 is tight-correct.
        self._instance_stride = 1 << 25
        self._class_stride = num_classes + 1

        # Boolean LUTs (size num_classes + 2, class IDs shifted by +1 at lookup).
        # Index 0 and num_classes+1 are sentinels; out-of-range IDs clamp there.
        is_track_lut = _make_lut(num_classes, track_classes_set)
        self.register_buffer("is_track_lut", is_track_lut, persistent=False)

        is_ignored_lut = _make_lut(num_classes, ignored_set, sentinel=True)
        self.register_buffer("is_ignored_lut", is_ignored_lut, persistent=False)

        self.add_state("gt_matches", default=[], dist_reduce_fx=None)
        self.add_state("pred_occurrences", default=[], dist_reduce_fx=None)

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Any | None = None,
    ) -> None:
        # Callers supply globally-unique frame IDs (unique within each sequence
        # across all batches and all ranks), so record keys never collide and
        # we just concatenate the gathered lists.

        state = (self.gt_matches, self.pred_occurrences)

        world_size = dist.get_world_size(process_group)
        synced: list[tuple[list[_GtMatch], list[_PredOcc]]] = [
            ([], []) for _ in range(world_size)
        ]

        with warnings.catch_warnings(action="ignore", category=FutureWarning):
            # dist.all_gather_object emits a pickle-is-unsafe warning; accept it
            # (mirrors association_quality.AssociationQuality).
            dist.all_gather_object(synced, state, group=process_group)

        gt_all: list[_GtMatch] = []
        pred_all: list[_PredOcc] = []
        for gts, preds in synced:
            gt_all.extend(gts)
            pred_all.extend(preds)

        self.gt_matches = gt_all
        self.pred_occurrences = pred_all

    # pylint: disable-next=arguments-differ,too-many-locals,too-many-branches
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sequence_id: torch.Tensor | Sequence[Hashable],
        frame_id: torch.Tensor | Sequence[int],
        mask: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            preds (torch.Tensor [B, *D, 2]):
                The predicted panoptic segmentation, where preds[..., 0] is
                the class ID and preds[..., 1] the instance ID.
            target (torch.Tensor [B, *D, 2]):
                The target panoptic segmentation, where target[..., 0] is the
                class ID and target[..., 1] the instance ID.
            sequence_id (torch.Tensor [B] | Sequence[Hashable]):
                The sequence ID for each sample in the batch.
            frame_id (torch.Tensor [B] | Sequence[int]):
                The temporal frame index for each sample. Must be unique
                within a sequence across all batches and all ranks, and sort
                into the intended temporal order.
            mask (torch.Tensor [B, *D], optional):
                An optional mask to filter valid samples in the batch. Only
                samples where mask is True are considered for evaluation.
        """
        batch_size = preds.shape[0]

        # materialize sequence_id / frame_id as Python lists
        sequence_ids = (
            sequence_id.tolist() if torch.is_tensor(sequence_id) else list(sequence_id)
        )
        frame_ids = frame_id.tolist() if torch.is_tensor(frame_id) else list(frame_id)
        assert len(sequence_ids) == batch_size
        assert len(frame_ids) == batch_size

        # filter out ignored GT classes via LUT (faster than torch.isin); out-of-range
        # class IDs (e.g. -1, 255) clamp to the True sentinels and are ignored too
        n = len(self.is_ignored_lut) - 1
        is_ignored = self.is_ignored_lut[(target[..., 0] + 1).clamp(0, n)]
        mask = (mask & ~is_ignored) if mask is not None else ~is_ignored

        # flatten inputs to [N, 3] = (batch, class, instance)
        preds_flat = self._prepare_input(preds, mask)
        target_flat = self._prepare_input(target, mask)
        preds_flat, target_flat = self._sanitize_instance_ids(preds_flat, target_flat)

        target_area, pred_area, intersections = self._compute_areas_and_intersections(
            preds_flat, target_flat
        )
        target_matches = self._find_target_matches(
            intersections, pred_area, target_area
        )

        self.gt_matches += self._emit_gt_matches(
            target_area, target_matches, sequence_ids, frame_ids
        )
        self.pred_occurrences += self._emit_pred_occurrences(
            pred_area, sequence_ids, frame_ids
        )

    # Two separate encodings are used for targets and predictions:
    # - _encode_tgt includes the class so GT segments can be grouped by (class, instance)
    #   to support per-class tracking metrics.
    # - _encode_pred is class-agnostic: the same pred ID accumulates its occurrence count
    #   across all classes in a sequence, matching the nuScenes reference behaviour where
    #   a predicted object "exists" regardless of which class it was assigned to.

    def _encode_tgt(self, x: torch.Tensor) -> torch.Tensor:
        """Encode [N, 3] = (batch, cls, inst) into a single int64 per row."""
        batch = x[:, 0]
        cls = x[:, 1] + 1  # shift: -1 → 0
        inst = x[:, 2] + 1  # shift: -1 → 0

        return (batch * self._class_stride + cls) * self._instance_stride + inst

    def _decode_tgt(self, keys: torch.Tensor) -> list[tuple[int, int, int]]:
        """Decode int64 keys back to (batch, cls, inst) tuples."""
        inst = ((keys % self._instance_stride) - 1).tolist()
        rem = keys // self._instance_stride

        cls = ((rem % self._class_stride) - 1).tolist()
        batch = (rem // self._class_stride).tolist()

        return list(zip(batch, cls, inst))

    def _encode_pred(self, x: torch.Tensor) -> torch.Tensor:
        """Encode [N, 3] = (batch, _, inst) into a single int64 per row (class-agnostic)."""
        batch = x[:, 0]
        inst = x[:, 2] + 1  # shift: -1 → 0

        return batch * self._instance_stride + inst

    def _decode_pred(self, keys: torch.Tensor) -> list[tuple[int, int]]:
        """Decode int64 keys back to (batch, inst) tuples."""
        inst = ((keys % self._instance_stride) - 1).tolist()
        batch = (keys // self._instance_stride).tolist()

        return list(zip(batch, inst))

    def _compute_areas_and_intersections(
        self,
        preds_flat: torch.Tensor,
        target_flat: torch.Tensor,
    ) -> tuple[dict, dict, dict]:
        """Encode segments and compute per-frame area and intersection maps.

        Returns:
            target_area:    {(batch, class, instance): pixel_count}
            pred_area:      {(batch, pred_instance): pixel_count}  (class-agnostic)
            intersections:  {((batch, pred_inst), (batch, class, gt_inst)): pixel_count}
        """
        # pylint: disable=too-many-locals

        tgt_keys = self._encode_tgt(target_flat)
        tgt_unique, tgt_inv, tgt_counts = torch.unique(
            tgt_keys, return_inverse=True, return_counts=True
        )
        tgt_dec = self._decode_tgt(tgt_unique)

        target_area = {
            (int(b), int(c), int(i)): int(cnt)
            for (b, c, i), cnt in zip(tgt_dec, tgt_counts.tolist())
        }

        # pred: class-agnostic — collapse class so pred IDs span all classes in a sequence
        pred_keys = self._encode_pred(preds_flat)
        pred_unique, pred_inv, pred_counts = torch.unique(
            pred_keys, return_inverse=True, return_counts=True
        )
        pred_dec = self._decode_pred(pred_unique)

        pred_area = {
            (int(b), int(pi)): int(cnt)
            for (b, pi), cnt in zip(pred_dec, pred_counts.tolist())
        }

        # Intersections via compact pair keys: encode (pred_idx, tgt_idx) as a single
        # integer so torch.unique counts co-occurrences (= intersection pixel area)
        # without any explicit loop.  Reuses the inverses from the unique() calls above.
        n_tgt = len(tgt_unique)
        pair_keys = pred_inv.long() * n_tgt + tgt_inv.long()
        pair_unique, pair_cnts = torch.unique(pair_keys, return_counts=True)
        pa = (pair_unique // n_tgt).tolist()
        ta = (pair_unique % n_tgt).tolist()
        intersections = {
            (
                (int(pred_dec[pi][0]), int(pred_dec[pi][1])),
                (int(tgt_dec[ti][0]), int(tgt_dec[ti][1]), int(tgt_dec[ti][2])),
            ): int(cnt)
            for pi, ti, cnt in zip(pa, ta, pair_cnts.tolist())
        }

        return target_area, pred_area, intersections

    def _find_target_matches(
        self,
        intersections: dict,
        pred_area: dict,
        target_area: dict,
    ) -> dict[_Color, int]:
        """For each GT target, find the pred instance (if any) with IoU > 0.5.

        IoU > 0.5 guarantees at most one GT can match any given pred and vice versa
        (two segments with IoU > 0.5 each cannot both overlap a third by > 0.5),
        so this greedy scan produces a valid injective matching without needing the
        Hungarian algorithm.
        """
        target_matches: dict[_Color, int] = {}
        for (pred_key, target_key), inter_area in intersections.items():
            _bp, pi = pred_key
            _bt, ct, ti = target_key

            if NO_INSTANCE in (pi, ct, ti):
                continue

            union = pred_area[pred_key] + target_area[target_key] - inter_area
            assert union > 0
            iou = inter_area / union

            if iou > 0.5:
                target_matches[target_key] = pi

        return target_matches

    def _emit_gt_matches(
        self,
        target_area: dict,
        target_matches: dict[_Color, int],
        sequence_ids: list,
        frame_ids: list,
    ) -> list[_GtMatch]:
        """Build _GtMatch records for every valid GT segment in a tracked class."""
        result: list[_GtMatch] = []
        for (b, c, i), area in target_area.items():
            if NO_INSTANCE in (c, i):
                continue
            if area <= self.min_instance_count:
                continue

            result.append(
                _GtMatch(
                    key=_GtMatchKey(
                        sequence=sequence_ids[b],
                        category=c,
                        gt_instance=i,
                        frame=frame_ids[b],
                    ),
                    pred_instance=target_matches.get((b, c, i), NO_MATCH),
                )
            )
        return result

    def _emit_pred_occurrences(
        self,
        pred_area: dict,
        sequence_ids: list,
        frame_ids: list,
    ) -> list[_PredOcc]:
        """Build _PredOcc records for every valid predicted instance.

        Small predictions are excluded so they do not inflate the FP denominator
        for matched preds in other frames.
        """
        result: list[_PredOcc] = []
        for (b, pi), area in pred_area.items():
            if pi == NO_INSTANCE:
                continue
            if area <= self.min_instance_count:
                continue

            result.append(
                _PredOcc(
                    key=_PredOccKey(
                        sequence=sequence_ids[b],
                        pred_instance=pi,
                        frame=frame_ids[b],
                    ),
                )
            )
        return result

    def _prepare_input(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, *_, c = x.shape
        assert c == 2

        x = x.detach().flatten(1, -2).to(dtype=torch.long)

        batch_id = torch.arange(batch_size, device=x.device)
        batch_id = batch_id.view(batch_size, 1, 1).expand(-1, x.shape[1], 1)
        x = torch.cat([batch_id, x], dim=-1)

        return x[mask.flatten(1, -1), :]

    def _sanitize_instance_ids(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mark non-instance pixels as NO_INSTANCE on both sides so they are
        uniformly filterable downstream.

        Inputs are the flattened `[N, 3] = (batch, class, instance)` tensors
        returned by `_prepare_input`. They are already freshly allocated, so
        in-place modifications do not leak back to the caller.

        Sanitization rules (assumes evaluated methods only assign instance IDs
        to instance classes, i.e. `track_classes`):
        - Pred pixels whose class is not in `track_classes` → (class,
          instance) = NO_INSTANCE. Additionally, a negative instance ID is
          treated as NO_INSTANCE as a safeguard.
        - Target pixels whose class is not in `track_classes` → (class,
          instance) = NO_INSTANCE.
        - Target pixels with class in `track_classes` but `instance < 0`:
          handled per `allow_invalid_instances` (assert, treat as
          non-instance, or drop entirely).
        """
        # sanitize predictions: class ∉ track_classes → NO_INSTANCE (also negative instance)
        n = len(self.is_track_lut) - 1
        preds_invalid = ~self.is_track_lut[(preds[..., 1] + 1).clamp(0, n)]
        preds_invalid = preds_invalid | (preds[..., 2] < 0)
        preds[preds_invalid, 1:] = NO_INSTANCE

        # sanitize target: class-based filter first (LUT replaces torch.isin)
        is_instance = self.is_track_lut[(target[..., 1] + 1).clamp(0, n)]

        if self.allow_invalid_instances == "disallow":
            assert torch.all(
                target[is_instance, 2] >= 0
            ), "Target instance IDs for track classes must be non-negative."

        elif self.allow_invalid_instances == "restrict":
            is_instance = is_instance & (target[..., 2] >= 0)

        elif self.allow_invalid_instances == "ignore":
            is_invalid = is_instance & (target[..., 2] < 0)
            is_instance = is_instance[~is_invalid]
            preds = preds[~is_invalid, :]
            target = target[~is_invalid, :]

        target[~is_instance, 1:] = NO_INSTANCE

        return preds, target

    def compute(self) -> TrackingQualityResult:
        """
        Compute the tracking quality metric.

        Returns:
            TrackingQualityResult:
                Proxy object exposing TQ and its per-class / per-track
                breakdowns.
        """
        return TrackingQualityResult(
            gt_matches=list(self.gt_matches),
            pred_occurrences=list(self.pred_occurrences),
            classes={int(c) for c in self.track_classes.tolist()},
        )


class Result:
    """
    Result object for Panoptic Tracking (PAT) metric.
    """

    def __init__(
        self,
        panoptic: PanopticQualityResult,
        tracking: TrackingQualityResult,
        panoptic_classes: torch.Tensor,
        use_mod_pq: bool = False,
    ) -> None:
        """
        Initialize a new PAT Result instance.

        Args:
            panoptic (PanopticQualityResult):
                The result of the PanopticQuality metric.
            tracking (TrackingQualityResult):
                The result of the TrackingQuality metric.
            panoptic_classes (torch.Tensor):
                The class subset over which the PQ component of PAT is
                averaged by default.
            use_mod_pq (bool, optional):
                If True, use the modified PQ (PQ†) instead of PQ when
                computing PAT. Defaults to False.
        """
        self.panoptic = panoptic
        self.tracking = tracking
        self._panoptic_classes = panoptic_classes
        self._use_mod_pq = use_mod_pq

    @property
    def pq(self) -> float:
        """
        Mean PQ (or PQ†, if configured) over the default panoptic class
        subset.
        """
        return self.mean_pq()

    @property
    def tq(self) -> float:
        """
        Mean Tracking Quality (TQ) over the default tracked class subset.
        """
        return self.mean_tq()

    @property
    def pat(self) -> float:
        """
        The Panoptic Tracking (PAT) metric as harmonic mean of PQ and TQ over
        the default class subsets.
        """
        return self.mean_pat()

    def mean_pq(self, classes: Collection[int] | None = None) -> float:
        """
        Mean PQ (or PQ†, if configured) over a class subset.

        Args:
            classes (Collection[int], optional):
                The classes to aggregate over. If None, the panoptic class
                subset configured on the metric is used. Defaults to None.
        """
        cls = self._panoptic_classes if classes is None else classes

        if self._use_mod_pq:
            pqm, _ = self.panoptic.mean_mod_pq(cls)
            return pqm.item()

        pq, _, _ = self.panoptic.mean_pq(cls)
        return pq.item()

    def mean_tq(self, classes: Collection[int] | None = None) -> float:
        """
        Mean TQ over a class subset.

        Args:
            classes (Collection[int], optional):
                The classes to aggregate over. If None, the tracked class
                subset configured on the metric is used. Defaults to None.
        """
        return self.tracking.mean_tq(classes)

    def mean_pat(
        self,
        panoptic_classes: Collection[int] | None = None,
        track_classes: Collection[int] | None = None,
    ) -> float:
        """
        Panoptic Tracking (PAT) metric over configurable class subsets.

        Args:
            panoptic_classes (Collection[int], optional):
                The classes to average PQ over. If None, the configured
                panoptic class subset is used. Defaults to None.
            track_classes (Collection[int], optional):
                The classes to average TQ over. If None, the configured
                tracked class subset is used. Defaults to None.
        """
        return _harmonic_mean(
            self.mean_pq(panoptic_classes), self.mean_tq(track_classes)
        )

    @property
    def pq_per_class(self) -> dict[int, float]:
        """
        Per-class PQ (or PQ†, if configured) for the configured panoptic
        class subset.
        """
        pq_tensor = self.panoptic.mod_pq if self._use_mod_pq else self.panoptic.pq
        return {
            int(c): float(pq_tensor[int(c)]) for c in self._panoptic_classes.tolist()
        }

    @property
    def tq_per_class(self) -> dict[int, float]:
        """
        Per-class TQ for the configured tracked class subset. Classes with no
        GT tracks are omitted.
        """
        return self.tracking.tq_per_class

    @property
    def pat_per_class(self) -> dict[int, float]:
        """
        Per-class PAT, defined as the harmonic mean of per-class PQ and
        per-class TQ. Restricted to classes that appear in both the panoptic
        and tracked class subsets (and have GT tracks on the TQ side).
        """
        pq_pc = self.pq_per_class
        tq_pc = self.tq_per_class

        return {
            c: _harmonic_mean(pq_pc[c], tq_pc[c]) for c in pq_pc.keys() & tq_pc.keys()
        }


class PanopticTrackingMetric(Metric):
    """
    Panoptic Tracking (PAT) metric.

    Composes a PanopticQuality and a TrackingQuality instance and combines
    their results as PAT = 2 * PQ * TQ / (PQ + TQ).

    The caller supplies an explicit `frame_id` for each sample in `update()`.
    Frame IDs must be unique within each sequence (across all batches and all
    processes) and sort into temporal order. Given this, sequences may be
    split across processes freely.

    Refer to https://arxiv.org/abs/2109.03805 for more details.
    """

    is_differentiable: bool = False
    full_state_update: bool = False

    _panoptic_classes: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        classes_thing: Collection[int],
        classes_stuff: Collection[int],
        track_classes: Collection[int] | None = None,
        panoptic_classes: Collection[int] | None = None,
        min_instance_count: int = 0,
        allow_unknown_preds: bool = True,
        allow_invalid_instances: Literal["disallow", "restrict", "ignore"] = "disallow",
        use_mod_pq: bool = False,
    ):
        """
        Initialize a new PanopticTrackingMetric instance.

        Any class in `range(num_classes)` not in `classes_thing ∪
        classes_stuff` is treated as an ignored/void class. PanopticQuality
        handles this internally via its `void_category` logic, and here we
        forward the same set of ignored classes to TrackingQuality so pred
        pixels over ignored GT regions do not inflate pred occurrence totals.

        Args:
            num_classes (int):
                Total number of classes in the dataset.
            classes_thing (Collection[int]):
                Class IDs of "thing" classes (countable objects).
            classes_stuff (Collection[int]):
                Class IDs of "stuff" classes (amorphous regions).
            track_classes (Collection[int], optional):
                Classes used for tracking quality. Defaults to classes_thing.
            panoptic_classes (Collection[int], optional):
                Classes over which PQ is averaged when computing PAT. Defaults
                to classes_thing ∪ classes_stuff (matches nuScenes' "include"
                convention).
            min_instance_count (int, optional):
                Minimum per-frame element count for a GT or predicted instance
                to be considered valid (passed to TrackingQuality). Defaults
                to 0.
            allow_unknown_preds (bool, optional):
                Forwarded to PanopticQuality. Defaults to True.
            allow_invalid_instances (str, optional):
                Forwarded to TrackingQuality. Defaults to "disallow".
            use_mod_pq (bool, optional):
                If True, PAT uses PQ† instead of PQ. Defaults to False.
        """
        super().__init__()

        self.num_classes = num_classes
        self._use_mod_pq = use_mod_pq

        self.pq = PanopticQuality(
            num_classes=num_classes,
            classes_thing=classes_thing,
            classes_stuff=classes_stuff,
            allow_unknown_preds=allow_unknown_preds,
        )

        ignored_classes = set(range(num_classes)) - (
            set(classes_thing) | set(classes_stuff)
        )

        self.tq = TrackingQuality(
            num_classes=num_classes,
            track_classes=track_classes if track_classes is not None else classes_thing,
            ignored_classes=ignored_classes,
            min_instance_count=min_instance_count,
            allow_invalid_instances=allow_invalid_instances,
        )

        if panoptic_classes is None:
            panoptic_classes = set(classes_thing) | set(classes_stuff)

        pc = torch.tensor(sorted(set(panoptic_classes)), dtype=torch.long)
        self.register_buffer("_panoptic_classes", pc, persistent=False)

    def reset(self) -> None:
        super().reset()
        self.pq.reset()
        self.tq.reset()

    # pylint: disable-next=arguments-differ
    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor,
        sequence_id: torch.Tensor | Sequence[Hashable],
        frame_id: torch.Tensor | Sequence[int],
        mask: torch.Tensor | None = None,
    ) -> None:
        """
        Update the metric state with new predictions and targets.

        Args:
            preds (torch.Tensor [B, *D, 2]):
                Predicted panoptic segmentation.
            target (torch.Tensor [B, *D, 2]):
                Target panoptic segmentation.
            sequence_id (torch.Tensor [B] | Sequence[Hashable]):
                Sequence ID for each sample in the batch.
            frame_id (torch.Tensor [B] | Sequence[int]):
                Temporal frame index for each sample. Must be unique within a
                sequence across all batches and all ranks, and sort into the
                intended temporal order.
            mask (torch.Tensor [B, *D], optional):
                Optional mask to filter valid samples. Defaults to None.
        """
        self.pq.update(preds=preds, target=target, mask=mask)
        self.tq.update(
            preds=preds,
            target=target,
            sequence_id=sequence_id,
            frame_id=frame_id,
            mask=mask,
        )

    def compute(self) -> Result:
        """
        Compute the Panoptic Tracking (PAT) metric.
        """
        return Result(
            panoptic=self.pq.compute(),
            tracking=self.tq.compute(),
            panoptic_classes=self._panoptic_classes,
            use_mod_pq=self._use_mod_pq,
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


def _harmonic_mean(a: float, b: float) -> float:
    if (a + b) == 0:
        return 0.0
    return (2 * a * b) / (a + b)
