# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scene selection on top of the shared occupancy pipeline.

Builds the occupancy ground-truth context (:mod:`scripts.common.datasets`) and
collects a single scene's frames, chosen by ordinal ``--scene`` or by
``--scene-name`` (a sequence id -- Waymo context / nuScenes token -- or a
human-readable nuScenes name like ``scene-0013``).
"""

import scripts.common.datasets as od


def _match_name(ctx, dataset, seq_ids, scene_name):
    """Resolve ``scene_name`` to one of ``seq_ids`` (the pipeline's sequence ids)."""
    if scene_name in seq_ids:
        return scene_name
    # nuScenes: allow the human-readable scene name (e.g. scene-0013) -> token.
    if dataset == "nuscenes":
        nusc = getattr(getattr(ctx.dataset, "source", None), "data", None)
        by_name = {s["name"]: s["token"] for s in getattr(nusc, "scene", [])}
        token = by_name.get(scene_name)
        if token in seq_ids:
            return token
    available = ", ".join(map(str, seq_ids))
    raise ValueError(
        f"scene '{scene_name}' not in the selected split. Available: {available}"
    )


def resolve_scene(dataset, split, scene, scene_name):
    """Return ``(ctx, sequence_id, [Frame, ...])`` for the chosen scene."""
    ctx = od.load_context(dataset, split=split, sequential=False)
    seq_ids = list(od.scene_index(ctx))
    if not seq_ids:
        raise ValueError("no scenes in the selected split")

    if scene_name is not None:
        key = _match_name(ctx, dataset, seq_ids, scene_name)
    elif 0 <= scene < len(seq_ids):
        key = seq_ids[scene]
    else:
        raise IndexError(f"scene index {scene} out of range ({len(seq_ids)} scenes)")

    frames = od.collect_scene(ctx, seq_ids.index(key))
    return ctx, key, frames
