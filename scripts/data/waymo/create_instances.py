#!/usr/bin/env -S uv run --script

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

# /// script
# requires-python = "==3.11.*"
# dependencies = [
#     "waymo-open-dataset-tf-2-12-0==1.6.7",  # latest waymo-open-dataset release
#     "click",
#     "tqdm",
# ]
#
# [[tool.uv.index]]
# # jaxlib==0.4.13, pinned by waymo-open-dataset 1.6.7, is not published on PyPI;
# # it lives on Google's jax release index (a find-links-style flat page).
# url = "https://storage.googleapis.com/jax-releases/jax_releases.html"
# format = "flat"
# ///
#
# This preprocessing script depends on tensorflow + waymo-open-dataset, whose
# pins (tensorflow 2.13, numpy 1.23.5, jaxlib 0.4.13, typing-extensions<4.6, ...)
# are fundamentally incompatible with the project's torch stack — they cannot
# share the project's environment. The PEP 723 metadata above lets uv run it in
# an isolated, ephemeral env instead of the project's .venv. (tensorflow/numpy
# are pulled in at waymo-open-dataset's pinned versions.)
#
# The `env -S uv run --script` shebang makes the file self-executing, so with
# uv on PATH you can run it directly (uv provisions the env on first run):
#
#     ./scripts/data/waymo/create_instances.py --pkl ... --occ ... --waymo ... -o ...
#
# or explicitly:  uv run --script scripts/data/waymo/create_instances.py ...
#
"""
Script to process all Waymo samples and create 'instances' data structure
by matching TFRecord bounding boxes with occupancy map instance IDs.

Waymo Object Classes:
  0: TYPE_UNKNOWN
  1: TYPE_VEHICLE
  2: TYPE_PEDESTRIAN
  3: TYPE_SIGN
  4: TYPE_CYCLIST
"""

import os
import pickle
from pathlib import Path

import click
import numpy as np
import tensorflow as tf  # pylint: disable=import-error  # isolated PEP 723 dep, see header
from tqdm import tqdm

# Disable GPU for TensorFlow to avoid CUDA OOM errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import Waymo Open Dataset protos (isolated PEP 723 dep, see header)
# pylint: disable-next=wrong-import-position,import-error
from waymo_open_dataset import dataset_pb2, label_pb2


def load_tfrecord_frame(tfrecord_path, frame_number, verbose=False):
    """Load a specific frame from a Waymo TFRecord file."""
    if not tfrecord_path.exists():
        if verbose:
            print(f"TFRecord not found: {tfrecord_path}")
        return None

    dataset = tf.data.TFRecordDataset(str(tfrecord_path), compression_type="")

    for idx, data in enumerate(dataset):
        if idx == frame_number:
            frame = dataset_pb2.Frame()  # pylint: disable=no-member
            frame.ParseFromString(data.numpy())
            return frame

    if verbose:
        print(f"Frame {frame_number} not found in TFRecord")
    return None


def bbox_to_voxel_indices(bbox_3d, voxel_size, voxel_range):
    """Convert 3D bounding box to voxel indices."""
    # pylint: disable=too-many-locals

    x, y, z, w, l, h, _ = bbox_3d  # yaw unused - using axis-aligned approximation
    vx, vy, vz = voxel_size
    x_min, y_min, z_min, x_max, y_max, z_max = voxel_range

    # Calculate grid dimensions
    nx = int((x_max - x_min) / vx)
    ny = int((y_max - y_min) / vy)
    nz = int((z_max - z_min) / vz)

    # Create axis-aligned bounding box (expand to account for rotation)
    max_dim = max(w, l)
    half_w = max_dim / 2
    half_l = max_dim / 2
    half_h = h / 2

    # Get bbox corners
    x_bbox_min = x - half_w
    x_bbox_max = x + half_w
    y_bbox_min = y - half_l
    y_bbox_max = y + half_l
    z_bbox_min = z - half_h
    z_bbox_max = z + half_h

    # Convert to voxel indices
    ix_min = max(0, int((x_bbox_min - x_min) / vx))
    ix_max = min(nx - 1, int((x_bbox_max - x_min) / vx))
    iy_min = max(0, int((y_bbox_min - y_min) / vy))
    iy_max = min(ny - 1, int((y_bbox_max - y_min) / vy))
    iz_min = max(0, int((z_bbox_min - z_min) / vz))
    iz_max = min(nz - 1, int((z_bbox_max - z_min) / vz))

    return (ix_min, ix_max, iy_min, iy_max, iz_min, iz_max)


def find_instance_id_for_bbox(bbox_3d, occ_instances, voxel_size, voxel_range):
    """Find the instance ID that occurs most frequently in the bbox region."""
    # pylint: disable=too-many-locals

    ix_min, ix_max, iy_min, iy_max, iz_min, iz_max = bbox_to_voxel_indices(
        bbox_3d, voxel_size, voxel_range
    )

    # Extract bbox region from occupancy map
    bbox_region = occ_instances[
        ix_min : ix_max + 1, iy_min : iy_max + 1, iz_min : iz_max + 1
    ]

    # Find unique instance IDs and their counts (excluding background 0)
    unique_ids, counts = np.unique(bbox_region, return_counts=True)

    # Filter out background (0)
    non_zero_mask = unique_ids > 0
    if not np.any(non_zero_mask):
        return None, 0

    unique_ids = unique_ids[non_zero_mask]
    counts = counts[non_zero_mask]

    # Get majority instance ID
    max_idx = np.argmax(counts)
    instance_id = unique_ids[max_idx]
    voxel_count = counts[max_idx]

    return int(instance_id), int(voxel_count)


def process_sample(
    sample,
    sample_idx_str,
    occ_data,
    original_waymo_path,
    voxel_size,
    voxel_range,
    verbose=False,
):
    """
    Process a single sample to create instances data structure.
    Ensures one-to-one mapping between bboxes and instance IDs.

    Note: Bounding boxes from frame.laser_labels are in the ego-vehicle frame.
    The voxel_range is also defined relative to the ego-vehicle frame, so the
    coordinate systems are aligned for matching.
    """
    # pylint: disable=too-many-locals

    frame_number = int(sample_idx_str[4:7])
    occ_instances = occ_data["instances"]

    # Load TFRecord
    tfrecord_path = (
        original_waymo_path
        / f"segment-{sample['context_name']}_with_camera_labels.tfrecord"
    )
    frame = load_tfrecord_frame(tfrecord_path, frame_number, verbose=verbose)

    if frame is None:
        return [], {}

    # Get all unique instance IDs from occupancy map (excluding background)
    unique_occ_ids = np.unique(occ_instances)
    available_instance_ids = set(unique_occ_ids[unique_occ_ids > 0])

    # First pass: collect all bbox-instance matches with scores
    bbox_candidates = []

    for laser_label in frame.laser_labels:
        box = laser_label.box

        # Bounding box coordinates are in the ego-vehicle frame
        bbox_3d = np.array(
            [
                box.center_x,
                box.center_y,
                box.center_z,
                box.width,
                box.length,
                box.height,
                box.heading,
            ]
        )

        instance_id, voxel_count = find_instance_id_for_bbox(
            bbox_3d, occ_instances, voxel_size, voxel_range
        )

        # Extract velocity from metadata
        velocity = [0.0, 0.0, 0.0]
        if hasattr(laser_label, "metadata") and laser_label.metadata is not None:
            velocity = [laser_label.metadata.speed_x, laser_label.metadata.speed_y, 0.0]

        bbox_candidates.append(
            {
                "bbox_3d": bbox_3d.tolist(),
                "instance_id": instance_id,
                "voxel_count": voxel_count,
                "bbox_label": laser_label.type,
                "velocity": velocity,
                "original_waymo_id": laser_label.id,
            }
        )

    # Second pass: resolve conflicts (multiple bboxes with same instance_id)
    # Strategy: assign each instance_id to the bbox with highest voxel_count
    assigned_instance_ids = set()
    instances = {}  # Changed from list to dict with original_waymo_id as key
    removed_bboxes = []

    # Sort by voxel_count (descending) to prioritize better matches
    bbox_candidates_sorted = sorted(
        bbox_candidates, key=lambda x: x["voxel_count"], reverse=True
    )

    for candidate in bbox_candidates_sorted:
        instance_id = candidate["instance_id"]

        # Check if bbox has no matching instance
        if instance_id is None or instance_id == 0:
            removed_bboxes.append(
                {
                    "reason": "no_instance_match",
                    "bbox_label": candidate["bbox_label"],
                    "original_waymo_id": candidate["original_waymo_id"],
                }
            )
            continue

        # Check if instance_id already assigned to another bbox
        if instance_id in assigned_instance_ids:
            removed_bboxes.append(
                {
                    "reason": "duplicate_instance",
                    "bbox_label": candidate["bbox_label"],
                    "original_waymo_id": candidate["original_waymo_id"],
                    "duplicate_instance_id": instance_id,
                }
            )
            continue

        # Valid assignment - use original_waymo_id as key
        waymo_id = candidate["original_waymo_id"]
        instances[waymo_id] = {
            "bbox_3d": candidate["bbox_3d"],
            "instance_id": instance_id,
            "bbox_label": candidate["bbox_label"],
            "velocity": candidate["velocity"],
            "original_waymo_id": candidate["original_waymo_id"],
            "num_lidar_pts": candidate["voxel_count"],
        }
        assigned_instance_ids.add(instance_id)

    # Find unassigned instances in occupancy map
    unassigned_instances = available_instance_ids - assigned_instance_ids

    stats = {
        "total_bboxes": len(bbox_candidates),
        "matched_bboxes": len(instances),
        "removed_bboxes": len(removed_bboxes),
        "removed_details": removed_bboxes,
        "total_occ_instances": len(available_instance_ids),
        "assigned_instances": len(assigned_instance_ids),
        "unassigned_instances": list(unassigned_instances),
    }

    if verbose and removed_bboxes:
        print(f"\n  Removed {len(removed_bboxes)} bboxes:")
        for rb in removed_bboxes:
            class_name = label_pb2.Label.Type.Name(  # pylint: disable=no-member
                rb["bbox_label"]
            )
            print(
                f"    - Class {rb['bbox_label']} ({class_name}) "
                f"(Waymo ID: {rb['original_waymo_id']}): {rb['reason']}"
            )

    return instances, stats


@click.command()
@click.option(
    "--pkl",
    "pkl_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to input pickle file (e.g., waymo_infos_train_jpg.pkl)",
)
@click.option(
    "--occ",
    "occ_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to occupancy ground truth directory (e.g., pano_voxel04/training)",
)
@click.option(
    "--waymo",
    "waymo_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to original Waymo dataset directory (e.g., waymo_open_dataset_v_1_4_3/training)",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output pickle file (e.g., waymo_infos_train_jpg_with_instances.pkl)",
)
@click.option(
    "--voxel-size",
    type=float,
    nargs=3,
    default=[0.4, 0.4, 0.4],
    help="Voxel size as three floats [x, y, z]",
)
@click.option(
    "--voxel-range",
    type=float,
    nargs=6,
    default=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
    help="Voxel range as six floats [x_min, y_min, z_min, x_max, y_max, z_max]",
)
def main(pkl_path, occ_path, waymo_path, output_path, voxel_size, voxel_range):
    """Main function to process all Waymo samples and create instances."""
    # pylint: disable=too-many-locals,too-many-statements

    # Convert tuples to lists for voxel parameters
    voxel_size = list(voxel_size)
    voxel_range = list(voxel_range)

    print(f"Loading pickle file: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    data_list = data["data_list"]
    print(f"Total samples: {len(data_list)}")

    # Process all samples with occupancy data (sample_idx % 5 == 0)
    processed_count = 0
    skipped_count = 0
    all_stats = []

    for sample in tqdm(data_list, desc="Processing samples"):
        sample_idx = sample["sample_idx"]

        # Only process samples with occupancy GT
        if sample_idx % 5 != 0:
            continue

        # Get sample_idx_str and derive paths
        sample_idx_str = str(sample_idx).zfill(7)
        scene_id = sample_idx_str[1:4]
        frame_number = sample_idx_str[4:7]

        occ_file_path = occ_path / scene_id / f"{frame_number}_04.npz"

        if not occ_file_path.exists():
            skipped_count += 1
            continue

        # Load occupancy data
        occ_data = np.load(occ_file_path)

        # Process sample and create instances
        instances, stats = process_sample(
            sample,
            sample_idx_str,
            occ_data,
            waymo_path,
            voxel_size,
            voxel_range,
            verbose=(processed_count < 5),  # Show details for first 5 samples
        )

        # Add instances to sample
        sample["instances"] = instances
        all_stats.append(stats)
        processed_count += 1

    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Samples processed: {processed_count}")
    print(f"Samples skipped: {skipped_count}")

    # Aggregate statistics
    if all_stats:
        total_bboxes = sum(s["total_bboxes"] for s in all_stats)
        total_matched = sum(s["matched_bboxes"] for s in all_stats)
        total_removed = sum(s["removed_bboxes"] for s in all_stats)
        total_occ_instances = sum(s["total_occ_instances"] for s in all_stats)
        total_assigned = sum(s["assigned_instances"] for s in all_stats)

        print(f"\n{'='*80}")
        print("STATISTICS")
        print(f"{'='*80}")
        print(f"Total bounding boxes: {total_bboxes}")
        print(f"Matched bboxes: {total_matched}")
        print(
            f"Removed bboxes: {total_removed} ({100*total_removed/total_bboxes:.1f}%)"
        )
        print(f"\nTotal occupancy instances: {total_occ_instances}")
        print(
            f"Assigned instances: {total_assigned} ({100*total_assigned/total_occ_instances:.1f}%)"
        )
        print(f"Unassigned instances: {total_occ_instances - total_assigned}")

        # Analyze removed bboxes by reason
        removed_by_reason = {}
        removed_by_class = {}

        for stats in all_stats:
            for rb in stats["removed_details"]:
                reason = rb["reason"]
                class_label = rb["bbox_label"]

                removed_by_reason[reason] = removed_by_reason.get(reason, 0) + 1
                removed_by_class[class_label] = removed_by_class.get(class_label, 0) + 1

        print(f"\n{'='*80}")
        print("REMOVED BBOXES BREAKDOWN")
        print(f"{'='*80}")
        print("\nBy reason:")
        for reason, count in sorted(removed_by_reason.items()):
            print(f"  {reason}: {count}")

        print("\nBy class label:")
        for class_label, count in sorted(
            removed_by_class.items(), key=lambda x: x[1], reverse=True
        ):
            class_name = label_pb2.Label.Type.Name(  # pylint: disable=no-member
                class_label
            )
            print(f"  Class {class_label} ({class_name}): {count}")

        # Show unassigned instances summary
        total_unassigned = sum(len(s["unassigned_instances"]) for s in all_stats)
        print(f"\nTotal unassigned instances in occupancy maps: {total_unassigned}")

    # Save updated pickle file
    print(f"\nSaving updated pickle to: {output_path}")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print("✓ Done!")

    # Show statistics
    total_instances = sum(len(sample.get("instances", {})) for sample in data_list)
    print(f"\nTotal instances created: {total_instances}")

    # Show sample
    for sample in data_list:
        if "instances" in sample and len(sample["instances"]) > 0:
            print(f"\nExample sample (sample_idx={sample['sample_idx']}):")
            print(f"  Number of instances: {len(sample['instances'])}")
            print(f"\n  First instance (key: {list(sample['instances'].keys())[0]}):")
            first_key = list(sample["instances"].keys())[0]
            for key, value in sample["instances"][first_key].items():
                print(f"    {key}: {value}")
            break


if __name__ == "__main__":
    main()
