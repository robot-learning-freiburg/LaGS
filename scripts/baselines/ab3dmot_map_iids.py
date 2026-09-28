#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Map pre-tracking instance IDs to AB3DMOT tracking IDs.

Matches bounding boxes from the original detections (boxes.json) to the
tracked boxes output by AB3DMOT based on 3D center distance in KITTI camera
coordinates.

Output: JSON file with mappings
[(sequence_id, sample_id, instance_id, track_id), ...].
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.kitti import KittiDB
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment


def box_to_kitti_center(xyz, rotation_yaw):
    """Convert box from nuScenes lidar to KITTI camera coordinates (center only)."""
    # KITTI transformation parameters
    velo_to_cam_rot = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    velo_to_cam_trans = np.array([0, 0, 0])
    r0_rect = Quaternion(axis=[1, 0, 0], angle=0)

    # Create box in lidar frame
    rotation_quat = Quaternion(axis=[0, 0, 1], angle=rotation_yaw)
    wlh = [1.0, 1.0, 1.0]  # dummy dimensions, we only need center
    box = Box(xyz, wlh, rotation_quat, name="dummy", token="")

    # Convert to KITTI camera frame
    box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
        box, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect
    )

    return box_cam_kitti.center


def load_boxes_by_scene(boxes_json_path):
    """
    Load boxes and organize by (scene_index, frame_index).

    ``scene_index`` / ``frame_index`` are the contiguous 0-based indices recorded
    at conversion time; ``scene-{scene_index:04d}`` is the AB3DMOT scene stem.

    Returns:
        dict: {scene_index: {frame_index:
                   [(sequence_id, sample_id, instance_id, center_kitti), ...]}}
    """
    print(f"Loading boxes from: {boxes_json_path}")
    with open(boxes_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    boxes_by_scene = {}
    for sample in data["samples"]:
        scene = boxes_by_scene.setdefault(sample["scene_index"], {})
        # Create the frame entry even if it has no boxes: AB3DMOT still emits an
        # (empty) output file for it, so frame indices must line up.
        frame = scene.setdefault(sample["frame_index"], [])

        for box in sample["boxes"]:
            center_kitti = box_to_kitti_center(np.array(box["center"]), box["rotation"])
            frame.append(
                (
                    sample["sequence_id"],
                    sample["sample_id"],
                    box["instance_id"],
                    center_kitti,
                )
            )

    print(f"Loaded {len(boxes_by_scene)} scenes")
    return boxes_by_scene


def load_tracking_results(tracking_dir):
    """
    Load AB3DMOT tracking results from trk_withid_0/ directory.

    Note: We use trk_withid_0/ instead of data_0/ because data_0/ only contains
    boxes above a score threshold, while trk_withid_0/ contains ALL tracked boxes.

    Returns:
        dict: {scene_index: {frame_index: [(track_id, center_kitti), ...]}}
    """
    print(f"Loading tracking results from: {tracking_dir}")
    tracking_dir = Path(tracking_dir)
    trk_dir = tracking_dir / "trk_withid_0"

    if not trk_dir.exists():
        raise ValueError(f"trk_withid_0 directory not found: {trk_dir}")

    tracking_by_scene = {}

    # Process each scene directory (named scene-{scene_index:04d} by AB3DMOT).
    for scene_dir in sorted(trk_dir.glob("scene-*")):
        if not scene_dir.is_dir():
            continue

        scene_index = int(scene_dir.name.split("-")[1])
        tracking_by_scene[scene_index] = {}

        # Process each frame file in this scene (sorted -> contiguous frame_index)
        for frame_index, frame_file in enumerate(sorted(scene_dir.glob("*.txt"))):
            assert frame_index not in tracking_by_scene[scene_index]
            tracking_by_scene[scene_index][frame_index] = []

            # Load tracking results for this frame
            with open(frame_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 17:
                        continue

                    # Format: Type -1 -1 alpha bbox2d(4) h w l x y z ry score track_id
                    # Indices: 0    1  2  3     4-7      8 9 10 11 12 13 14 15    16
                    track_id = int(parts[16])
                    x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                    center_kitti = np.array([x, y, z])

                    tracking_by_scene[scene_index][frame_index].append(
                        (track_id, center_kitti)
                    )

    print(f"Loaded tracking results for {len(tracking_by_scene)} scenes")
    return tracking_by_scene


def match_boxes_in_frame(detection_boxes, tracked_boxes):
    """
    Match detection boxes to tracked boxes using Hungarian algorithm.

    Args:
        detection_boxes: [(sequence_id, sample_id, instance_id, center_kitti), ...]
        tracked_boxes: [(track_id, center_kitti), ...]

    Returns:
        list: [(sequence_id, sample_id, instance_id, track_id), ...] matched pairs
    """
    # pylint: disable=too-many-locals

    if len(detection_boxes) == 0 or len(tracked_boxes) == 0:
        return []

    # Build cost matrix based on center distance
    n_det = len(detection_boxes)
    n_trk = len(tracked_boxes)
    cost_matrix = np.zeros((n_det, n_trk))

    for i, (*_, det_center) in enumerate(detection_boxes):
        for j, (_, trk_center) in enumerate(tracked_boxes):
            # Euclidean distance in KITTI camera coordinates
            dist = np.linalg.norm(det_center - trk_center)
            cost_matrix[i, j] = dist

    # Use Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Collect matches with distance threshold
    matches = []
    distance_threshold = 5.0  # meters in KITTI camera space

    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < distance_threshold:
            sequence_id, sample_id, instance_id, _ = detection_boxes[i]
            track_id, _ = tracked_boxes[j]
            matches.append((sequence_id, sample_id, instance_id, track_id))

    return matches


@click.command()
@click.option(
    "--boxes",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to boxes JSON file",
)
@click.option(
    "--tracking-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Path to AB3DMOT tracking results directory (e.g., results/nuScenes/occupancy_val_H1)",
)
@click.option(
    "--output",
    type=click.Path(file_okay=True, dir_okay=False),
    required=True,
    help="Output JSON file for instance-to-track mappings",
)
def main(boxes, tracking_dir, output):
    """Map pre-tracking instance IDs to AB3DMOT tracking IDs."""
    # pylint: disable=too-many-locals

    print("Starting instance-to-track ID mapping...")
    print()

    # Load data
    boxes_by_scene = load_boxes_by_scene(boxes)
    tracking_by_scene = load_tracking_results(tracking_dir)

    # Match boxes frame by frame
    all_mappings = []
    unmatched_detections = 0
    total_detections = 0
    total_tracked = 0
    unmatched_details = []

    print("\nMatching boxes...")
    for scene_index in sorted(boxes_by_scene.keys()):
        assert scene_index in tracking_by_scene

        sys.stdout.write(f"Processing scene {scene_index:04d}\r")
        sys.stdout.flush()

        for frame_index in sorted(boxes_by_scene[scene_index].keys()):
            detection_boxes = boxes_by_scene[scene_index][frame_index]
            tracked_boxes = tracking_by_scene[scene_index][frame_index]

            total_detections += len(detection_boxes)
            total_tracked += len(tracked_boxes)

            # Match boxes in this frame
            matches = match_boxes_in_frame(detection_boxes, tracked_boxes)
            all_mappings.extend(matches)

            num_unmatched = len(detection_boxes) - len(matches)
            unmatched_detections += num_unmatched

            # Track unmatched cases for debugging
            if num_unmatched > 0:
                unmatched_details.append(
                    {
                        "scene_index": scene_index,
                        "frame_index": frame_index,
                        "num_detections": len(detection_boxes),
                        "num_tracked": len(tracked_boxes),
                        "num_matched": len(matches),
                        "num_unmatched": num_unmatched,
                    }
                )

    print()
    print("\nMatching complete!")
    print(f"Total detections: {total_detections}")
    print(f"Total tracked: {total_tracked}")
    print(f"Matched: {len(all_mappings)}")
    print(f"Unmatched: {unmatched_detections}")
    print(f"Match rate: {len(all_mappings) / total_detections * 100:.1f}%")

    # Verify all boxes matched
    if unmatched_detections > 0:
        print(
            f"\nWARNING: {unmatched_detections} boxes from boxes.json did not receive a track ID!"
        )

    # Save mappings to JSON
    output_data = {
        "mappings": [
            {
                "sequence_id": sequence_id,
                "sample_id": sample_id,
                "instance_id": int(instance_id),
                "track_id": int(track_id),
            }
            for sequence_id, sample_id, instance_id, track_id in all_mappings
        ]
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nMappings saved to: {output}")


if __name__ == "__main__":
    main()
