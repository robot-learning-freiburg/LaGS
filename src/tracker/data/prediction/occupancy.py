# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from lightning import LightningModule, Trainer

from ... import utils
from ...utils.types import MetaDict, Sample
from .registry import registry
from .writer import PredictionWriter

log = utils.log.get_logger(__name__)


def _extract_gaussians(gaussians: Mapping[str, MetaDict], b: int) -> dict[str, object]:
    """Flatten a per-stream gaussian prediction dict for batch element ``b``.

    ``gaussians`` maps stream name to a ``MetaDict`` of decoded, layer-stacked
    predictions (see ``LatentGaussianOccupancyTracker.decode_gaussian_queries``);
    we keep the final decoder layer (index ``-1``) and concatenate all streams
    along the gaussian axis. ``gaussian_stream_counts`` (aligned with
    ``gaussian_streams``) records how many gaussians each stream contributed so
    stream identity is recoverable. Centers are in world (ego) meters, matching
    the model output.
    """
    streams = list(gaussians.keys())
    # ``logits`` is only decoded when a class head exists; it is present for all
    # streams or none (the class head is a global config choice).
    has_logits = all("logits" in gaussians[s] for s in streams)

    def col(name: str, layer_indexed: bool = True) -> list[np.ndarray]:
        out = []
        for s in streams:
            t = gaussians[s][name]
            t = t[b, -1] if layer_indexed else t[b]
            t = t.detach().cpu()

            # NumPy has no bfloat16/half; upcast floating tensors to float32
            # (integer ids are left untouched).
            if t.is_floating_point():
                t = t.float()

            out.append(t.numpy())
        return out

    payload: dict[str, object] = {
        "gaussian_center": np.concatenate(col("centers"), axis=0),
        "gaussian_scale": np.concatenate(col("scales"), axis=0),
        "gaussian_rotation": np.concatenate(col("rotations"), axis=0),
        "gaussian_opacity": np.concatenate(col("opacities"), axis=0),
        "gaussian_instance_id": np.concatenate(
            col("instance_ids", layer_indexed=False), axis=0
        ),
        "gaussian_streams": np.array(streams),
        "gaussian_stream_counts": np.array(
            [gaussians[s].centers.shape[2] for s in streams]
        ),
    }
    if has_logits:
        payload["gaussian_logits"] = np.concatenate(col("logits"), axis=0)

    return payload


def _sample_id_str(sample_id: object, pad: int = 0) -> str:
    """Path-safe string form of a ``meta.sample_id`` for use as a filename stem.

    Integer ids (e.g. Waymo's ``sample_idx``) are zero-padded to ``pad`` digits
    so lexical order matches numeric order; string ids (e.g. NuScenes tokens)
    are used verbatim. ``pad`` defaults to ``0``.
    """
    if isinstance(sample_id, (int, np.integer)):
        return f"{int(sample_id):0{pad}d}"
    return str(sample_id)


@registry.register(namespace="occupancy", key="PanopticOccupancy")
class PanopticOccupancy(PredictionWriter):
    """Dataset-agnostic prediction writer for panoptic occupancy.

    Keys every frame purely on ``meta.sample_id`` and ``meta.sequence_id`` (both
    populated by every dataset's ``__getitem__``), so no dataset-specific
    re-indexing is needed. One npz is written per frame, grouped into a
    per-sequence subdirectory:

        ``<output_path>/<sequence_id>/<sample_id>.npz``

    The payload uses the pipeline-native (Z, Y, X) axis order, matching the
    ground truth produced by the dataset transforms, so the two are directly
    comparable without any transpose or re-indexing:

    - pano_sem: semantic predictions, axis order (Z, Y, X)
    - pano_inst: instance predictions, axis order (Z, Y, X); ids >= 0 are valid
      instances, -1 means no instance
    - sample_id: the frame's ``meta.sample_id`` (token / sample_idx / UUID)
    - sequence_id: the frame's ``meta.sequence_id`` (scene token / context / name)

    When ``store_gaussians`` is set and the model exposes ``outputs.gaussians``,
    the final-layer gaussian parameters (all streams concatenated) are added to
    the same npz. Centers are in world (ego) meters, not the occupancy grid:

    - gaussian_center: (N, 3) gaussian centers in ego meters
    - gaussian_scale: (N, 3) per-axis scales
    - gaussian_rotation: (N, 4) rotation quaternions (w, x, y, z)
    - gaussian_opacity: (N,) opacities in [0, 1]
    - gaussian_instance_id: (N,) per-gaussian instance ids
    - gaussian_logits: (N, C) semantic class logits (only if a class head exists)
    - gaussian_streams: (S,) stream names, in concatenation order
    - gaussian_stream_counts: (S,) gaussian count per stream (sums to N)

    ``sample_id_pad`` zero-pads integer ``sample_id`` filename stems (default 7,
    so Waymo's ``sample_idx`` files sort numerically); it has no effect on string
    ids.
    """

    def __init__(
        self,
        output_path: str | Path,
        sample_id_pad: int = 7,
        store_gaussians: bool = False,
    ):
        super().__init__()
        self.output_path = Path(output_path)
        self.sample_id_pad = sample_id_pad
        self.store_gaussians = store_gaussians

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.output_path.mkdir(parents=True, exist_ok=True)
        log.info("Starting panoptic occupancy prediction...")

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
            if not batch.is_valid[b]:
                continue

            meta = batch.meta[b]
            sample_id = meta.sample_id
            sequence_id = meta.sequence_id

            # Extract occupancy predictions in pipeline-native (Z, Y, X) order.
            sem = outputs.occupancy.semantics[b].detach().cpu().numpy()
            inst = outputs.occupancy.instance_ids[b].detach().cpu().numpy()

            payload: dict[str, object] = {
                "pano_sem": sem,
                "pano_inst": inst,
                "sample_id": sample_id,
                "sequence_id": sequence_id,
            }

            # Optionally persist the decoded gaussians alongside the grid.
            if self.store_gaussians and "gaussians" in outputs:
                payload.update(_extract_gaussians(outputs.gaussians, b))

            # Save immediately, grouped per sequence.
            filepath = (
                self.output_path
                / str(sequence_id)
                / f"{_sample_id_str(sample_id, self.sample_id_pad)}.npz"
            )
            filepath.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(filepath, **payload)

    def on_predict_end(self, trainer, pl_module):
        log.info("Saved panoptic occupancy predictions to '%s'", self.output_path)
