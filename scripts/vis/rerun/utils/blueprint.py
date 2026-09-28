# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rerun blueprint: a 3D scene beside a per-dataset camera grid.

Copied from ``test/visualize_occupancy_panoptic_rerun.py`` (camera layout per
dataset) and extended for the prediction bundle: the opacity-coloured gaussians
start hidden so the semantic ones are visible first, and both can be toggled.
"""

import rerun.blueprint as rrb


def build_camera_grid(dataset_type: str):
    """Return the per-dataset camera ``rrb.Vertical`` grid."""
    if dataset_type == "nuscenes":
        names = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
        ]
        row1 = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"cameras/{names[1]}", name=names[1]),
            rrb.Spatial2DView(origin=f"cameras/{names[4]}", name=names[4]),
        )
        row2 = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"cameras/{names[0]}", name=names[0]),
            rrb.Spatial2DView(origin=f"cameras/{names[2]}", name=names[2]),
        )
        row3 = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"cameras/{names[3]}", name=names[3]),
            rrb.Spatial2DView(origin=f"cameras/{names[5]}", name=names[5]),
        )
        return rrb.Vertical(row1, row2, row3)

    if dataset_type == "waymo":
        names = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_SIDE_LEFT",
            "CAM_SIDE_RIGHT",
        ]
        row1 = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"cameras/{names[0]}", name=names[0]),
        )
        row2 = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"cameras/{names[1]}", name=names[1]),
            rrb.Spatial2DView(origin=f"cameras/{names[2]}", name=names[2]),
        )
        row3 = rrb.Horizontal(
            rrb.Spatial2DView(origin=f"cameras/{names[3]}", name=names[3]),
            rrb.Spatial2DView(origin=f"cameras/{names[4]}", name=names[4]),
        )
        return rrb.Vertical(row1, row2, row3)

    raise ValueError(f"Unknown dataset type: {dataset_type}")


def _prediction_overrides(method_names) -> dict:
    """Per-method visibility defaults for the ``prediction/<name>/`` sub-trees.

    Within each method the visual-filtered occupancy is shown and the unfiltered
    (raw) one plus the gaussians start hidden (the opacity gaussians stay hidden
    even when their parent is re-shown, so the semantic ones surface first). Only
    the first method is visible on load; the rest have their whole sub-tree hidden
    and can be toggled on in the UI.
    """
    hidden = rrb.EntityBehavior(visible=False)
    overrides: dict = {}
    for i, name in enumerate(method_names):
        root = f"prediction/{name}"
        if i != 0:
            overrides[root] = hidden
        overrides[f"{root}/gaussians"] = hidden
        overrides[f"{root}/gaussians/opacity"] = hidden
        overrides[f"{root}/occupancy/unfiltered"] = hidden
    return overrides


def build_blueprint(dataset_type: str, method_names) -> rrb.Blueprint:
    camera_grid = build_camera_grid(dataset_type)
    # Start the camera image grid hidden so the 3D scene fills the viewport; the
    # container can be re-shown from the blueprint panel.
    camera_grid.visible = False

    overrides = {
        # Boxes/trajectories start hidden so the occupancy reads clearly; toggle
        # in the UI.
        "annotations": rrb.EntityBehavior(visible=False),
        # Per-method prediction visibility (occupancy/gaussians).
        **_prediction_overrides(method_names),
    }
    # With predictions present, hide ground truth by default so the predicted
    # occupancy reads clearly; in ground-truth-only mode, leave it visible.
    if method_names:
        overrides["ground_truth"] = rrb.EntityBehavior(visible=False)

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(
                    origin="/",
                    name="3D Scene",
                    overrides=overrides,
                    eye_controls=rrb.EyeControls3D(
                        # Near top-down bird's-eye. eye_up is exactly +z so the
                        # orbit rotates around the world z axis. Rather than looking
                        # straight down (degenerate roll), the view is nudged a
                        # little forward via look_target ahead in +x: this defines
                        # the roll and puts the forward driving direction (+x) at the
                        # top of the screen. The nudge must stay well above Rerun's
                        # degeneracy threshold (a ~1-unit offset reverts to the
                        # default roll), so ~10 units (~5-6 deg tilt) is about as
                        # small as it can go.
                        position=(-1, 0.0, 105.0),
                        look_target=(1, 0.0, 0.0),
                        eye_up=(0.0, 0.0, 1.0),
                        kind=rrb.Eye3DKind.Orbital,
                    ),
                ),
                rrb.TimePanel(state="collapsed"),
            ),
            camera_grid,
        ),
        rrb.SelectionPanel(state="collapsed"),
    )
