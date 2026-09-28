#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Convert occupancy boxes JSON directly to AB3DMOT input format.

Output: per-sequence files in AB3DMOT tracking input format.

AB3DMOT input format (one line per detection):
frame class_id type truncated occluded alpha bbox_2d(x1,y1,x2,y2)
dimensions_3d(h,w,l) location_3d(x,y,z) rotation_y score
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
from nuscenes.utils.data_classes import Box
from nuscenes.utils.kitti import KittiDB
from pyquaternion import Quaternion

# ========== NuScenes Mappings ==========
# Occ3D class ID to name mapping (from config/data/labels/occupancy/nuscenes.yaml)
NUSCENES_OCC3D_ID_TO_NAME = {
    1: "barrier",
    2: "bicycle",
    3: "bus",
    4: "car",
    5: "construction_vehicle",
    6: "motorcycle",
    7: "pedestrian",
    8: "traffic_cone",
    9: "trailer",
    10: "truck",
}

# AB3DMOT class name to ID mapping for NuScenes
# From AB3DMOT_libs/utils.py det_id2str
NUSCENES_AB3DMOT_NAME_TO_ID = {
    "Pedestrian": 1,
    "Car": 2,
    "Bicycle": 3,
    "Motorcycle": 4,
    "Bus": 5,
    "Trailer": 6,
    "Truck": 7,
    "Construction_vehicle": 8,
    "Barrier": 9,
    "Traffic_cone": 10,
}

# ========== Waymo Mappings ==========
# Occ3D class ID to name mapping (from config/data/labels/occupancy/waymo.yaml)
WAYMO_OCC3D_ID_TO_NAME = {
    1: "vehicle",
    2: "pedestrian",
    4: "cyclist",
}

# AB3DMOT class name to ID mapping for Waymo
# Using subset of AB3DMOT classes that match Waymo
WAYMO_AB3DMOT_NAME_TO_ID = {
    "Vehicle": 1,
    "Pedestrian": 2,
    "Cyclist": 4,
}


@click.command()
@click.option(
    "--boxes",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="Path to boxes JSON file",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Output directory for AB3DMOT input files",
)
@click.option(
    "--dataset",
    type=click.Choice(["nuscenes", "waymo"]),
    default="nuscenes",
    help="Dataset type (nuscenes or waymo)",
)
@click.option(
    "--det-name",
    type=str,
    default="occupancy",
    help="Detection method name (for folder naming)",
)
@click.option(
    "--split",
    type=str,
    default="val",
    help="Split name (train/val/test)",
)
@click.option(
    "--min-voxels",
    type=int,
    default=1,
    help="Minimum number of voxels for a box to be included",
)
def main(boxes, output_dir, dataset, det_name, split, min_voxels):
    """Convert occupancy boxes JSON to AB3DMOT input format."""
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    # Select mappings based on dataset
    if dataset == "nuscenes":
        occ3d_id_to_name = NUSCENES_OCC3D_ID_TO_NAME
        ab3dmot_name_to_id = NUSCENES_AB3DMOT_NAME_TO_ID
    elif dataset == "waymo":
        occ3d_id_to_name = WAYMO_OCC3D_ID_TO_NAME
        ab3dmot_name_to_id = WAYMO_AB3DMOT_NAME_TO_ID
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Dataset: {dataset}")
    print(f"Loading boxes from: {boxes}")
    print(f"Output directory: {output_dir}")
    print(f"Minimum voxels: {min_voxels}")
    print("Converting from Occ3D format to AB3DMOT format")
    print()

    # Load boxes
    with open(boxes, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # KITTI transformation parameters
    velo_to_cam_rot = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    velo_to_cam_trans = np.array([0, 0, 0])
    r0_rect = Quaternion(axis=[1, 0, 0], angle=0)
    p_left_kitti = np.zeros((3, 4))
    p_left_kitti[:3, :3] = np.eye(3)
    imsize = (1600, 900)

    # Group samples by scene (scene_index / frame_index recorded at conversion).
    sequences = {}
    for sample in data["samples"]:
        sequences.setdefault(sample["scene_index"], []).append(sample)

    print(f"Found {len(sequences)} sequences")
    print("Converting to AB3DMOT input format...")

    # Get all unique categories from data and map from Occ3D to AB3DMOT
    all_categories = set()
    for sample in data["samples"]:
        for box in sample["boxes"]:
            # Parse Occ3D class ID from class_name field (format: "N")
            occ3d_id = int(box["class_name"])
            occ3d_name = occ3d_id_to_name[occ3d_id]
            # Convert to AB3DMOT format (capitalized)
            ab3dmot_name = occ3d_name.capitalize()
            all_categories.add(ab3dmot_name)

    # Create output folders for each category and 'all'
    category_files = {}
    for category in list(all_categories) + ["all"]:
        # Path format: {output_dir}/{det_name}_{category}_{split}/
        cat_dir = output_path / f"{det_name}_{category}_{split}"
        cat_dir.mkdir(parents=True, exist_ok=True)
        category_files[category] = cat_dir

    # Process each sequence
    for scene_index, samples in sorted(sequences.items()):
        # Frame order within the scene.
        samples = sorted(samples, key=lambda x: x["frame_index"])

        # Scene file stem the external AB3DMOT reads/writes back (kept identical
        # to the historic format); scene_index -> sequence_id lives in boxes.json.
        seq_name = f"scene-{scene_index:04d}"

        sys.stdout.write(f"Processing sequence {seq_name}: {len(samples)} frames\r")
        sys.stdout.flush()

        # Open files for each category for this sequence
        seq_files = {}
        for category, cat_dir in category_files.items():
            seq_file = cat_dir / f"{seq_name}.txt"

            # pylint: disable-next=consider-using-with
            seq_files[category] = open(seq_file, "w", encoding="utf-8")

        # Process frames
        for sample in samples:
            frame_id = sample["frame_index"]
            boxes_list = sample["boxes"]

            for box_dict in boxes_list:
                # Filter by minimum voxels
                if box_dict["num_voxels"] < min_voxels:
                    continue

                # Parse Occ3D class ID from class_name field (format: "N")
                occ3d_id = int(box_dict["class_name"])

                # Map from Occ3D to AB3DMOT format
                occ3d_name = occ3d_id_to_name[occ3d_id]

                # Convert to AB3DMOT format (capitalized)
                ab3dmot_name = occ3d_name.capitalize()
                ab3dmot_id = ab3dmot_name_to_id[ab3dmot_name]

                # Create nuScenes box (already in lidar frame)
                xyz = box_dict["center"]
                wlh = box_dict["dimensions"]
                rotation_yaw = box_dict["rotation"]

                # Create quaternion from yaw
                rotation_quat = Quaternion(axis=[0, 0, 1], angle=rotation_yaw)

                # Create box in lidar frame
                box = Box(xyz, wlh, rotation_quat, name=ab3dmot_name, token="")

                # Convert from nuScenes lidar to KITTI camera format
                box_cam_kitti = KittiDB.box_nuscenes_to_kitti(
                    box, Quaternion(matrix=velo_to_cam_rot), velo_to_cam_trans, r0_rect
                )

                # Project 3d box to 2d box in image
                bbox_2d = KittiDB.project_kitti_box_to_image(
                    box_cam_kitti, p_left_kitti, imsize=imsize
                )
                if bbox_2d is None:
                    bbox_2d = (-1, -1, -1, -1)

                score = min(box_dict["num_voxels"] / 100.0, 1.0)

                # Compute alpha (observation angle)
                # alpha = ry - arctan2(x, z) where x, z are in camera frame
                ry = box_cam_kitti.orientation.radians
                alpha = ry - np.arctan2(
                    box_cam_kitti.center[0], box_cam_kitti.center[2]
                )

                # Normalize alpha to [-pi, pi]
                while alpha > np.pi:
                    alpha -= 2 * np.pi
                while alpha < -np.pi:
                    alpha += 2 * np.pi

                # AB3DMOT input format (comma-separated):
                # frame,type_id,xmin,ymin,xmax,ymax,score,h,w,l,x,y,z,ry,alpha
                line = f"{frame_id:d},{ab3dmot_id:d},"
                line += f"{bbox_2d[0]:.2f},{bbox_2d[1]:.2f},"
                line += f"{bbox_2d[2]:.2f},{bbox_2d[3]:.2f},"
                line += f"{score:.2f},"
                line += f"{box_cam_kitti.wlh[2]:.2f},"  # h
                line += f"{box_cam_kitti.wlh[0]:.2f},"  # w
                line += f"{box_cam_kitti.wlh[1]:.2f},"  # l
                line += f"{box_cam_kitti.center[0]:.2f},"  # x
                line += f"{box_cam_kitti.center[1]:.2f},"  # y
                line += f"{box_cam_kitti.center[2]:.2f},"  # z
                line += f"{ry:.2f},"  # ry
                line += f"{alpha:.2f}"  # alpha

                # Write to category-specific file and 'all' file
                seq_files[ab3dmot_name].write(line + "\n")
                seq_files["all"].write(line + "\n")

        # Close all files for this sequence
        for f in seq_files.values():
            f.close()

        print(f"Sequence {seq_name}: {len(samples)} frames")

    print(f"\nResults saved to: {output_dir}")
    print(f"Total sequences: {len(sequences)}")
    print("\nYou can now run AB3DMOT with these input files.")


if __name__ == "__main__":
    main()
