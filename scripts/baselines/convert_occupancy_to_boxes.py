#!/usr/bin/env python3

# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Convert occupancy instance predictions to 3D bounding boxes.

First stage of the AB3DMOT box-tracking baseline. Takes per-frame panoptic
occupancy predictions and fits a PCA-based 3D oriented box to each instance
(DBSCAN clustering + quality scoring). Writes a ``boxes.json`` keyed by
``(sequence_id, sample_id)``, with a per-sequence ``scene_index`` / ``frame_index``
so the downstream AB3DMOT scripts can reconstruct scene/frame order.

Example:
    ./scripts/baselines/convert_occupancy_to_boxes.py \\
        --dataset nuscenes --preds runs/.../predictions --output boxes.json
"""

import json
import sys
from pathlib import Path

import click
import numpy as np
from sklearn.cluster import DBSCAN

from tracker.utils import progress

# Make the top-level ``scripts`` package importable when run directly (see the
# eval scripts for the rationale): add the repo root -- the parent of ``scripts``.
_REPO_ROOT = next(
    p.parent for p in Path(__file__).resolve().parents if p.name == "scripts"
)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# pylint: disable=wrong-import-position
import scripts.common.datasets as od  # noqa: E402
from scripts.baselines import common as bl  # noqa: E402

# NuScenes occupancy parameters
# (from config/experiment/lags/data/common.yaml)
VOXEL_SIZE = np.array([0.4, 0.4, 0.4])  # [x, y, z] in meters
VOXEL_RANGE = np.array(
    [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
)  # [x_min, y_min, z_min, x_max, y_max, z_max]
GRID_SHAPE = (200, 200, 16)  # (X, Y, Z)


def voxel_to_world(voxel_indices):
    """
    Convert voxel indices to world coordinates (ego-vehicle frame).

    Args:
        voxel_indices: Array of shape [N, 3] with voxel indices [x, y, z]

    Returns:
        Array of shape [N, 3] with world coordinates in meters
    """
    # voxel center = range_min + (voxel_index + 0.5) * voxel_size
    world_coords = VOXEL_RANGE[:3] + (voxel_indices + 0.5) * VOXEL_SIZE
    return world_coords


def cluster_instance_voxels(voxel_indices, eps=2.5, min_samples=3):
    """
    Cluster voxels within a single instance using DBSCAN.

    This separates dense object clusters from scattered artifacts/outliers.

    Args:
        voxel_indices: Array of shape [N, 3] with voxel indices [x, y, z]
        eps: DBSCAN radius in voxel units (default 2.5 = ~1m with 0.4m voxels)
        min_samples: Minimum voxels to form a cluster

    Returns:
        List of cluster voxel arrays (one per cluster, excluding noise)
        Dict with diagnostic info
    """
    if len(voxel_indices) < min_samples:
        return [voxel_indices], {"num_clusters": 1, "num_noise": 0}

    # Run DBSCAN in voxel space
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = clustering.fit_predict(voxel_indices)

    # Separate clusters (label >= 0) from noise (label == -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    clusters = []
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_voxels = voxel_indices[cluster_mask]
        if len(cluster_voxels) >= min_samples:
            clusters.append(cluster_voxels)

    num_noise = np.sum(labels == -1)

    diagnostics = {
        "num_clusters": len(clusters),
        "num_noise": num_noise,
    }

    return clusters, diagnostics


def score_box_quality(bbox_dict, voxel_indices):
    """
    Score the quality of a fitted bounding box.

    Higher score = better quality (more likely to be a real object).

    Metrics:
    - Number of voxels: prefer larger clusters
    - Size reasonableness: penalize very small or very large boxes
    - Aspect ratio: penalize extreme elongations
    - Voxel count: absolute number matters

    Args:
        bbox_dict: Dict from fit_bbox_pca with center, dimensions, rotation
        voxel_indices: Array of voxels used for fitting

    Returns:
        float: quality score (higher is better)
    """
    # pylint: disable=too-many-locals

    dims = bbox_dict["dimensions"]  # [width, length, height]
    w, l, h = dims

    # Box volume
    box_volume = w * l * h
    if box_volume <= 0:
        return 0.0

    num_voxels = len(voxel_indices)

    # 1. Voxel count score
    # Prefer clusters with more voxels (more evidence = better)
    # Use log scale to not overly favor huge clusters
    # Typical: pedestrian ~10-50 voxels, car ~50-500 voxels
    voxel_score = np.log1p(num_voxels) / 10.0  # log1p(100) ≈ 4.6
    voxel_score = min(1.0, voxel_score)

    # 2. Size reasonableness score
    # Prefer boxes in reasonable size range
    # Small objects (pedestrians): 0.5-2m
    # Medium (cars): 2-6m
    # Large (trucks): 6-15m
    max_dim = max(w, l, h)
    if max_dim < 0.3:  # Too small (likely noise)
        size_score = max_dim / 0.3
    elif max_dim > 20:  # Too large (likely artifact)
        size_score = max(0.0, 1.0 - (max_dim - 20) / 20.0)
    else:
        size_score = 1.0

    # 3. Aspect ratio score
    # Penalize extreme elongations (likely artifacts)
    min_dim = min(w, l, h)
    if min_dim > 0:
        aspect_ratio = max_dim / min_dim
        # Reasonable objects: aspect ratio < 10
        aspect_score = max(0.0, 1.0 - (aspect_ratio - 1) / 20.0)
    else:
        aspect_score = 0.0

    # 4. Volume score
    # Prefer reasonably sized boxes
    # Penalize tiny boxes (likely noise) and huge boxes (likely artifacts)
    if box_volume < 0.1:  # < 0.1 m³
        volume_score = box_volume / 0.1
    elif box_volume > 100:  # > 100 m³
        volume_score = max(0.0, 1.0 - (box_volume - 100) / 100.0)
    else:
        volume_score = 1.0

    # Weighted combination
    # Emphasize voxel count (evidence) and size reasonableness
    total_score = (
        voxel_score * 0.5  # Most important: more voxels = more evidence
        + size_score * 0.2  # Reasonable dimensions
        + aspect_score * 0.15  # Not too elongated
        + volume_score * 0.15  # Reasonable volume
    )

    return total_score


def remove_outliers_iterative(points, percentile=95, iterations=2):
    """
    Iteratively remove outlier points based on distance from the cluster center.

    Args:
        points: Array of shape [N, 3] with 3D coordinates
        percentile: Keep points within this percentile of distances (e.g., 95)
        iterations: Number of iterations to perform

    Returns:
        Filtered points array of shape [M, 3] where M <= N
    """
    if len(points) <= 3:
        return points

    filtered_points = points.copy()

    for _ in range(iterations):
        if len(filtered_points) <= 3:
            break

        # Compute center
        center = np.mean(filtered_points, axis=0)

        # Compute distances from center
        distances = np.linalg.norm(filtered_points - center, axis=1)

        # Keep points within the percentile threshold
        threshold = np.percentile(distances, percentile)
        mask = distances <= threshold

        # Ensure we keep at least a few points
        if np.sum(mask) < 3:
            break

        filtered_points = filtered_points[mask]

    return filtered_points


def fit_bbox_pca(points, apply_outlier_removal=True):
    """
    Fit an oriented bounding box to a set of 3D points using PCA.
    Optionally applies iterative outlier removal before fitting.

    Args:
        points: Array of shape [N, 3] with 3D coordinates
        apply_outlier_removal: Whether to apply iterative outlier removal

    Returns:
        dict with keys:
            - center: [x, y, z] center of the box
            - dimensions: [w, l, h] width, length, height
            - rotation: yaw angle around z-axis in radians
            - num_points_used: number of points used for fitting (after filtering)
    """
    # pylint: disable=too-many-locals

    if len(points) == 0:
        return None

    # Apply outlier removal if requested
    if apply_outlier_removal and len(points) > 3:
        filtered_points = remove_outliers_iterative(points, percentile=95, iterations=2)
        if len(filtered_points) == 0:
            filtered_points = points
    else:
        filtered_points = points

    # Compute center
    center = np.mean(filtered_points, axis=0)

    # For rotation, we only care about XY plane (PCA on 2D)
    # Use filtered points for PCA computation
    filtered_points_xy = filtered_points[:, :2]
    center_xy = center[:2]

    # Center the points
    centered_xy = filtered_points_xy - center_xy

    # Compute covariance matrix
    if len(filtered_points) > 1:
        cov = np.cov(centered_xy.T)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort by eigenvalue (largest first)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]

        # The first eigenvector gives us the primary direction (length)
        # The second eigenvector gives us the secondary direction (width)
        length_vec = eigenvectors[:, 0]

        # Compute rotation angle (yaw) from the length direction
        # In NuScenes: yaw=0 means facing +X, yaw=pi/2 means facing +Y
        rotation = np.arctan2(length_vec[1], length_vec[0])

        # Create rotation matrix
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

        # Transform points to aligned coordinate system
        aligned_xy = centered_xy @ rot_matrix.T
    else:
        # Single point - no meaningful rotation
        rotation = 0.0
        aligned_xy = centered_xy

    # Compute dimensions using filtered points
    if len(filtered_points) > 1:
        # In aligned frame, get min/max along each axis
        min_aligned = np.min(aligned_xy, axis=0)
        max_aligned = np.max(aligned_xy, axis=0)

        # After rotating to aligned frame:
        # aligned_xy[:, 0] is along the length direction (first principal component)
        # aligned_xy[:, 1] is along the width direction (second principal component)
        # In NuScenes convention: dimensions = [width, length, height]
        length = max_aligned[0] - min_aligned[0]
        width = max_aligned[1] - min_aligned[1]

        # Height from Z axis
        min_z = np.min(filtered_points[:, 2])
        max_z = np.max(filtered_points[:, 2])
        height = max_z - min_z
    else:
        # Single point - use minimum box size
        width = VOXEL_SIZE[0]
        length = VOXEL_SIZE[1]
        height = VOXEL_SIZE[2]

    # Ensure minimum dimensions (at least one voxel)
    width = max(width, VOXEL_SIZE[0])
    length = max(length, VOXEL_SIZE[1])
    height = max(height, VOXEL_SIZE[2])

    return {
        "center": center,
        "dimensions": np.array([width, length, height]),
        "rotation": rotation,
        "num_points_used": len(filtered_points),
    }


def process_sample(
    semantics,
    instance_ids,
    dbscan_eps=2.5,
    dbscan_min_samples=3,
    apply_outlier_removal=True,
):
    """
    Process a single frame's occupancy and extract bounding boxes.

    Strategy:
    1. For each instance ID, use DBSCAN to find dense clusters
    2. Fit a box to each cluster
    3. Score each box by quality metrics
    4. Keep the best box for each instance

    Args:
        semantics: semantic labels [X, Y, Z]
        instance_ids: instance IDs [X, Y, Z], 0-indexed (-1 = no instance)
        dbscan_eps: DBSCAN radius in voxel units
        dbscan_min_samples: Minimum voxels to form a cluster
        apply_outlier_removal: Whether to apply iterative outlier removal

    Returns:
        dict with:
            - num_instances: number of boxes kept
            - boxes: list of box dicts
            - filtering_stats: dict with filtering statistics
    """
    # pylint: disable=too-many-locals

    # Get unique instance IDs (excluding -1 which is background/no instance)
    unique_ids = np.unique(instance_ids)
    unique_ids = unique_ids[unique_ids >= 0]

    boxes = []
    filtering_stats = {
        "no_clusters": 0,
        "degenerate": 0,
        "multi_cluster": 0,
        "total_clusters": 0,
        "noise_voxels": 0,
    }

    for inst_id in unique_ids:
        # Get voxel indices for this instance
        mask = instance_ids == inst_id
        voxel_indices = np.argwhere(mask)  # Shape: [N, 3] with [x, y, z]

        if len(voxel_indices) == 0:
            continue

        # Cluster voxels using DBSCAN
        clusters, diagnostics = cluster_instance_voxels(
            voxel_indices, eps=dbscan_eps, min_samples=dbscan_min_samples
        )

        filtering_stats["noise_voxels"] += diagnostics["num_noise"]
        filtering_stats["total_clusters"] += diagnostics["num_clusters"]

        if len(clusters) == 0:
            filtering_stats["no_clusters"] += 1
            continue

        if len(clusters) > 1:
            filtering_stats["multi_cluster"] += 1

        # Fit box to each cluster and score them
        candidate_boxes = []

        for cluster_voxels in clusters:
            # Get semantic class (majority vote) from cluster voxels
            cluster_mask = np.zeros_like(mask)
            cluster_mask[
                cluster_voxels[:, 0],
                cluster_voxels[:, 1],
                cluster_voxels[:, 2],
            ] = True
            sem_labels = semantics[cluster_mask]
            semantic_id = int(np.bincount(sem_labels).argmax())

            # Convert voxel indices to world coordinates
            world_coords = voxel_to_world(cluster_voxels)

            # Fit bounding box with outlier removal
            bbox = fit_bbox_pca(
                world_coords, apply_outlier_removal=apply_outlier_removal
            )

            if bbox is None:
                continue

            # Filter degenerate boxes (too small)
            if (
                bbox["dimensions"][0] <= 0.1
                or bbox["dimensions"][1] <= 0.1
                or bbox["dimensions"][2] <= 0.1
            ):
                continue

            # Score the box
            quality_score = score_box_quality(bbox, cluster_voxels)

            # Store candidate
            candidate_boxes.append(
                {
                    "bbox": bbox,
                    "semantic_id": semantic_id,
                    "cluster_voxels": cluster_voxels,
                    "score": quality_score,
                }
            )

        # Select best box based on score
        if len(candidate_boxes) == 0:
            filtering_stats["degenerate"] += 1
            continue

        best_candidate = max(candidate_boxes, key=lambda x: x["score"])

        # Store box with metadata
        box_dict = {
            "instance_id": int(inst_id),
            "semantic_id": int(best_candidate["semantic_id"]),
            "class_name": f"{best_candidate['semantic_id']}",
            "num_voxels": int(len(voxel_indices)),  # Original count
            "num_voxels_in_cluster": int(len(best_candidate["cluster_voxels"])),
            "num_points_used": int(best_candidate["bbox"]["num_points_used"]),
            "num_clusters": int(len(clusters)),
            "quality_score": float(best_candidate["score"]),
            "center": best_candidate["bbox"]["center"].tolist(),
            "dimensions": best_candidate["bbox"]["dimensions"].tolist(),
            "rotation": float(best_candidate["bbox"]["rotation"]),
            "velocity": [0.0, 0.0],  # No velocity from occupancy
        }

        boxes.append(box_dict)

    return {
        "num_instances": int(len(boxes)),
        "boxes": boxes,
        "filtering_stats": {k: int(v) for k, v in filtering_stats.items()},
    }


@click.command()
@click.option(
    "--dataset",
    type=click.Choice(od.DATASETS, case_sensitive=False),
    default="nuscenes",
    show_default=True,
    help="Dataset whose pipeline provides frame grouping/order.",
)
@click.option(
    "--preds",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="Directory of predictions (<sequence_id>/<sample_id>.npz).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default="occupancy_boxes.json",
    help="Output JSON file for bounding boxes",
)
@click.option("--split", default=None, help="Override the default (val) split.")
@click.option(
    "--dbscan-eps",
    type=float,
    default=2.5,
    help="DBSCAN radius in voxel units (default 2.5 ≈ 1m with 0.4m voxels)",
)
@click.option(
    "--dbscan-min-samples",
    type=int,
    default=3,
    help="Minimum voxels to form a DBSCAN cluster",
)
@click.option(
    "--disable-outlier-removal",
    is_flag=True,
    help="Disable iterative outlier removal during box fitting",
)
@click.option(
    "--min-quality-score",
    type=float,
    default=0.0,
    help="Minimum quality score to keep a box (0.0-1.0, higher = stricter)",
)
def main(
    dataset,
    preds,
    output,
    split,
    dbscan_eps,
    dbscan_min_samples,
    disable_outlier_removal,
    min_quality_score,
):
    """
    Convert occupancy instance predictions to 3D bounding boxes.

    DBSCAN-based clustering approach:
    - For each instance ID, use DBSCAN to find dense clusters
    - Fit boxes to all clusters
    - Score boxes by quality metrics (density, compactness, size, aspect ratio)
    - Select best box per instance
    """
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-arguments

    apply_outlier_removal = not disable_outlier_removal

    print(f"Dataset: {dataset}")
    print(f"Predictions path: {preds}")
    print(f"Output file: {output}")
    print(f"DBSCAN eps: {dbscan_eps} voxels (~{dbscan_eps * 0.4:.2f}m)")
    print(f"DBSCAN min samples: {dbscan_min_samples} voxels")
    print(f"Outlier removal: {'enabled' if apply_outlier_removal else 'disabled'}")
    print(f"Min quality score: {min_quality_score}")
    print(f"Voxel size: {VOXEL_SIZE}")
    print(f"Voxel range: {VOXEL_RANGE}")
    print()

    ctx = od.load_context(dataset.lower(), split=split)

    # Process all predictions
    all_results = []
    scenes: dict[int, str] = {}  # scene_index -> sequence_id
    scene_of: dict[str, int] = {}  # sequence_id -> scene_index
    total_boxes = 0
    total_instances_in_occupancy = 0
    total_filtering_stats = {
        "no_clusters": 0,
        "degenerate": 0,
        "multi_cluster": 0,
        "total_clusters": 0,
        "noise_voxels": 0,
        "low_quality": 0,
    }

    stream = bl.stream_predictions(ctx, preds)
    for seq_id, frame_index, sample_id, path in progress.track(
        stream, "Converting to boxes...", total=od.frame_count(ctx)
    ):
        if seq_id not in scene_of:
            scene_of[seq_id] = len(scenes)
            scenes[len(scenes)] = seq_id

        # Predictions are stored [Z, Y, X]; transpose to [X, Y, Z] for the box
        # geometry (ids stay in the stored -1 = none convention).
        with np.load(path) as data:
            semantics = np.asarray(data["pano_sem"]).transpose(2, 1, 0)
            instance_ids = np.asarray(data["pano_inst"]).transpose(2, 1, 0).copy()

        # HACK: ignore the first 15 x-slices to avoid ego-vehicle artifacts.
        instance_ids[:15, :, :] = -1

        unique_ids = np.unique(instance_ids)
        total_instances_in_occupancy += len(unique_ids[unique_ids >= 0])

        result = process_sample(
            semantics,
            instance_ids,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            apply_outlier_removal=apply_outlier_removal,
        )

        # Accumulate filtering stats
        for key in total_filtering_stats:
            if key in result["filtering_stats"]:
                total_filtering_stats[key] += result["filtering_stats"][key]

        # Filter by quality score
        filtered_boxes = []
        for box in result["boxes"]:
            if box["quality_score"] < min_quality_score:
                total_filtering_stats["low_quality"] += 1
                continue

            filtered_boxes.append(box)

        result["sequence_id"] = seq_id
        result["sample_id"] = sample_id
        result["scene_index"] = scene_of[seq_id]
        result["frame_index"] = frame_index
        result["boxes"] = filtered_boxes
        result["num_instances"] = len(filtered_boxes)

        all_results.append(result)
        total_boxes += result["num_instances"]

    if not all_results:
        print(f"No predictions found under {preds}")
        return

    # Save results
    total_filtered = (
        total_filtering_stats["no_clusters"]
        + total_filtering_stats["degenerate"]
        + total_filtering_stats["low_quality"]
    )

    output_data = {
        "metadata": {
            "voxel_size": VOXEL_SIZE.tolist(),
            "voxel_range": VOXEL_RANGE.tolist(),
            "grid_shape": list(GRID_SHAPE),
            "dbscan_eps": float(dbscan_eps),
            "dbscan_min_samples": int(dbscan_min_samples),
            "outlier_removal_enabled": bool(apply_outlier_removal),
            "min_quality_score": float(min_quality_score),
            "num_samples": int(len(all_results)),
            "total_instances_in_occupancy": int(total_instances_in_occupancy),
            "total_instances_filtered": int(total_filtered),
            "total_boxes": int(total_boxes),
            "filtering_stats": {k: int(v) for k, v in total_filtering_stats.items()},
        },
        "scenes": {str(idx): seq for idx, seq in scenes.items()},
        "samples": all_results,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # pylint: disable=line-too-long
    print(f"\nResults saved to {output}")
    print()
    print("=" * 70)
    print("INSTANCE FILTERING SUMMARY")
    print("=" * 70)
    print(f"Total instances in occupancy predictions: {total_instances_in_occupancy}")
    print(
        f"Total instances filtered out: {total_filtered} ({total_filtered / total_instances_in_occupancy * 100:.1f}%)"
    )
    print()
    print("DBSCAN clustering stats:")
    print(
        f"  Instances with multiple clusters: {total_filtering_stats['multi_cluster']}"
    )
    print(f"  Total clusters found: {total_filtering_stats['total_clusters']}")
    print(f"  Noise voxels removed: {total_filtering_stats['noise_voxels']}")
    print()
    print("Instances filtered out by reason:")
    print(f"  No valid clusters: {total_filtering_stats['no_clusters']}")
    print(f"  Degenerate (box fitting failed): {total_filtering_stats['degenerate']}")
    print(
        f"  Low quality score (< {min_quality_score}): {total_filtering_stats['low_quality']}"
    )
    print()
    print(
        f"Final boxes output: {total_boxes} ({total_boxes / total_instances_in_occupancy * 100:.1f}%)"
    )
    print()
    print(f"Total samples: {len(all_results)}")
    print(f"Average boxes per sample: {total_boxes / len(all_results):.2f}")
    print(
        f"Average instances per sample (original): {total_instances_in_occupancy / len(all_results):.2f}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
