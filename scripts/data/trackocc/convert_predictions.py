#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Convert TrackOcc baseline occupancy predictions to our unified format.

TrackOcc (``../baselines/TrackOcc``) writes one npz per frame under
``occupancy_pred/<sample_idx>.npz`` where ``sample_idx = sequence * 1000 + frame``
(see ``loaders/track_nuscenes_occ_dataset.py``). Those files differ from the
unified ``occupancy.PanopticOccupancy`` format in three ways:

* **Frame identity.** They are keyed by the integer ``sample_idx``; the unified
  format keys on ``(sequence_id, sample_id)``. How that mapping is recovered
  depends on ``--dataset``:

  - ``nuscenes``: the stock nuScenes infos pkl stores only ``sample_idx`` (no
    per-frame token), so the mapping is reconstructed straight from the nuScenes
    DB, replicating ``prepare_nusc.build_sample_index``
    (``sample_idx = scene_number * 1000 + frame_index``). This yields
    ``sequence_id = scene_token`` and ``sample_id = sample token``, matching the
    eval (see ``tracker.data.dataset.nuscenes``). The pkl is not needed.
  - ``waymo``: TrackOcc and the eval read the *same* Waymo infos pkl, where each
    frame carries a globally unique integer ``sample_idx`` and its
    ``context_name``. The mapping is a direct read of that pkl, yielding
    ``sequence_id = context_name`` and ``sample_id = sample_idx`` (an int, so the
    output filename is zero-padded via the eval's own path helper), matching the
    eval (see ``tracker.data.dataset.waymo_to``). The pkl is read from the data
    root.
* **Axis order.** ``pano_sem``/``pano_inst`` are ``(X, Y, Z)``; the unified
  format uses the pipeline-native ``(Z, Y, X)``.
* **Class ordering & instance sentinel.** For ``nuscenes`` / ``waymo`` Occ3D already
  orders ``free`` last, matching the eval vocabulary, so the semantic remap is the
  identity. Instances are 1-based with ``0`` = stuff/none (see ``trackocc.py``
  ``obj_idxes = active.obj_idxes + 1``); the unified format uses ``-1`` = none and
  ids >= 0, so a ``- 1`` shift restores that while preserving track identity.

TrackOcc's ``obj_embeddings`` / ``obj_idxes`` extras are dropped (the unified
format has no slot for them). Frames whose ``sample_idx`` is absent from the
index are skipped with a warning.

Example (nuScenes, the default; no --infos needed):
    python scripts/data/trackocc/convert_predictions.py \
        --input ../baselines/TrackOcc/test/.../occupancy_pred \
        --output output/trackocc_nusc_preds

Example (Waymo):
    python scripts/data/trackocc/convert_predictions.py \
        --dataset waymo \
        --input ../baselines/TrackOcc/test/.../occupancy_pred \
        --output output/trackocc_waymo_preds
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

from tracker import utils
from tracker.utils import progress

# Make the top-level ``scripts`` package importable when this file is run directly
# (`python scripts/data/trackocc/convert_predictions.py`), not only via
# ``python -m scripts.data.trackocc.convert_predictions``: running a file as a script
# puts only its own directory on sys.path, and ``scripts`` lives at the repo root (it
# is not part of the editable-installed ``tracker`` package). So add the repo root --
# the parent of the ``scripts`` dir -- ourselves; under ``-m`` it is already present.
_REPO_ROOT = next(
    p.parent for p in Path(__file__).resolve().parents if p.name == "scripts"
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Shared dataset helpers: the label / global-config helpers (so the class vocabulary
# matches the data-prep exactly) and the eval's prediction-path helper. Import the
# names directly -- like the prepare_* scripts -- so cross-module use of the
# (private) helpers is not flagged as protected-access.
from scripts.common.datasets import (  # noqa: E402  pylint: disable=wrong-import-position
    _DATA,
    _label_conf,
    _set_global_config,
    prediction_path,
)

log = logging.getLogger(__name__)


def _build_semantic_lut(dataset: str) -> np.ndarray:
    """LUT mapping a TrackOcc class index (``free`` last) to the eval vocabulary.

    ``lut[trackocc_idx] == native_idx``. For ``nuscenes`` / ``waymo`` the eval
    vocabulary already orders ``free`` last (Occ3D convention), so this reduces to
    the identity.
    """
    _set_global_config()
    native = list(_label_conf("occupancy", dataset).all)
    trackocc_order = [c for c in native if c != "free"] + ["free"]
    return np.array([native.index(name) for name in trackocc_order], dtype=np.int64)


def _load_sample_index_nuscenes(
    nuscenes_root: Path, version_and_split: str
) -> dict[int, tuple[str, str]]:
    """Map ``sample_idx`` -> ``(scene_token, sample_token)`` for nuScenes.

    The stock nuScenes infos pkl stores no per-frame ``token`` (only
    ``sample_idx``), so we instead reconstruct the mapping straight from the
    nuScenes DB, replicating ``prepare_nusc.build_sample_index``:
    ``sample_idx = scene_number * 1000 + frame_index`` where ``scene_number`` is
    the numeric suffix of the scene name and ``frame_index`` is the temporal
    order within the scene (the ``first_sample_token`` -> ``next`` chain). The
    keys therefore match ``tracker.data.dataset.nuscenes``
    (``sequence_id == scene_token``, ``sample_id == token``). No pkl required.
    """
    # Imported lazily so the waymo path never pays the nuscenes-devkit import; the
    # frame-index reconstruction also reaches into a few nuscenes dataset internals.
    # pylint: disable=import-outside-toplevel,protected-access
    from tracker.data.dataset import nuscenes as nusc_ds

    version, split = nusc_ds._parse_version_and_split(version_and_split)
    data = nusc_ds.acquire(version=version, dataroot=nuscenes_root)
    try:
        scenes = nusc_ds._get_scenes_for_split(data, split)
        index: dict[int, tuple[str, str]] = {}
        for scene in scenes:
            scene_number = int(scene["name"].split("-")[-1], 10)
            token = scene["first_sample_token"]
            frame_index = 0
            while token:
                assert 0 <= frame_index < 1000, "frame index must be in [0, 1000)"
                index[scene_number * 1000 + frame_index] = (scene["token"], token)
                token = data.get("sample", token)["next"]
                frame_index += 1
    finally:
        nusc_ds.release(data)
    return index


def _load_sample_index_waymo(
    waymo_root: Path, split: str
) -> dict[int, tuple[str, int]]:
    """Map ``sample_idx`` -> ``(context_name, sample_idx)`` for Waymo.

    TrackOcc's Waymo predictions and the unified eval read the *same* Waymo infos
    pkl, where each frame carries a globally unique integer ``sample_idx`` and its
    ``context_name``. The eval keys ``sequence_id`` on ``context_name`` and
    ``sample_id`` on that same ``sample_idx`` (see
    ``tracker.data.dataset.waymo_to``), so the mapping is a direct read of the pkl
    -- no reconstruction needed. ``_get_pkl_data`` applies the eval's
    ``sample_idx % 5 == 0`` keyframe filter, so only evaluated frames are indexed.
    """
    # Imported lazily so the other datasets never pay the waymo pkl import; the read
    # also reaches into a waymo dataset internal.
    # pylint: disable=import-outside-toplevel,protected-access
    from tracker.data.dataset import waymo_to as wto

    data = wto._get_pkl_data(waymo_root, split)
    return {
        int(frame.sample_idx): (str(frame.context_name), int(frame.sample_idx))
        for frame in data.data_list
    }


def _convert_array(npz, sem_lut: np.ndarray) -> dict:
    # (X, Y, Z) -> (Z, Y, X); remap TrackOcc class indices (free last) to the
    # eval vocabulary (identity for nuScenes).
    sem = sem_lut[np.transpose(npz["pano_sem"], (2, 1, 0)).astype(np.int64)]
    # (X, Y, Z) -> (Z, Y, X); 1-based with 0 = stuff/none -> -1 = none, ids >= 0.
    inst = np.transpose(npz["pano_inst"], (2, 1, 0)).astype(np.int64) - 1
    return {"pano_sem": sem, "pano_inst": inst}


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(["nuscenes", "waymo"]),
    default="nuscenes",
    show_default=True,
    help="Source dataset. Selects how frame identity is recovered and whether "
    "the semantic class remap is applied.",
)
@click.option(
    "--input",
    "input_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="TrackOcc prediction directory (searched recursively for "
    "<sample_idx>.npz). Point at a single run's 'occupancy_pred' directory.",
)
@click.option(
    "--nuscenes-root",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="nuScenes dataset root (for --dataset nuscenes; defaults to "
    "<data>/nuscenes).",
)
@click.option(
    "--nuscenes-split",
    default="v1.0-val",
    show_default=True,
    help="nuScenes version/split whose scenes are indexed (for --dataset " "nuscenes).",
)
@click.option(
    "--waymo-root",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Waymo infos root (for --dataset waymo; defaults to "
    "<data>/TrackOcc-waymo/kitti_format).",
)
@click.option(
    "--waymo-split",
    default="validation",
    show_default=True,
    help="Waymo split whose infos pkl is read (for --dataset waymo).",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Output directory; predictions are written as <output>/<sequence_id>/"
    "<sample_id>.npz in the unified format.",
)
def main(
    dataset,
    input_dir,
    nuscenes_root,
    nuscenes_split,
    waymo_root,
    waymo_split,
    output_dir,
):
    # pylint: disable=too-many-locals
    utils.log.initialize()

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if dataset == "nuscenes":
        root = Path(nuscenes_root) if nuscenes_root else _DATA / "nuscenes"
        sample_index = _load_sample_index_nuscenes(root, nuscenes_split)
    else:  # waymo
        root = (
            Path(waymo_root)
            if waymo_root
            else _DATA / "TrackOcc-waymo" / "kitti_format"
        )
        sample_index = _load_sample_index_waymo(root, waymo_split)
    sem_lut = _build_semantic_lut(dataset)

    files = sorted(input_dir.rglob("*.npz"))
    if not files:
        raise click.ClickException(f"no .npz files found under '{input_dir}'")

    log.info("converting %d files -> %s", len(files), output_dir)

    converted = 0
    missing = 0
    for path in progress.track(files, description="Converting"):
        try:
            sample_idx = int(path.stem)
        except ValueError:
            log.warning("skipping '%s' (stem is not an integer sample_idx)", path.name)
            missing += 1
            continue

        entry = sample_index.get(sample_idx)
        if entry is None:
            log.warning("skipping sample_idx %d (not in index)", sample_idx)
            missing += 1
            continue
        sequence_id, sample_id = entry

        with np.load(path) as npz:
            data = _convert_array(npz, sem_lut)
        data["sample_id"] = sample_id
        data["sequence_id"] = sequence_id

        # Use the eval's own path helper so the filename stem matches how it reads
        # predictions back (int sample_ids are zero-padded; strings pass through).
        out_path = prediction_path(output_dir, sequence_id, sample_id)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **data)
        converted += 1

    log.info("converted %d files (%d skipped)", converted, missing)


if __name__ == "__main__":
    main()
