# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Dynamic object temporal consistency supervision.

Extends the temporal consistency loss to supervise dynamic/thing-class gaussians
using GT object trajectories. For gaussians matched to GT objects, computes
object-motion targets instead of EMC targets.
"""

import torch

from ....utils.math import distance_points_box, intersect_points_box
from ....utils.types import MetaDict
from ..occupancy.gaussian.utils import (
    quaternion_multiply,
    rotation_matrix_to_quaternion,
)


def compute_object_transforms(
    prev_boxes: torch.Tensor,
    prev_instance_ids: torch.Tensor,
    curr_boxes: torch.Tensor,
    curr_instance_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-object rigid transforms between frames.

    For an object rotating around its center:
        transform = T(center_curr) @ R_z(delta_yaw) @ T(-center_prev)

    Note: the produced transforms are ego-relative. Previous and current boxes
    are both relative to their respecitve ego-frame, which is not the same.
    Meaning returned transforms are relative to the ego vehicle and its motion.

    Args:
        prev_boxes: Previous frame boxes [M, 10] = center(3), wlh(3), yaw(1), vel(2)
        prev_instance_ids: Previous frame instance IDs [M]
        curr_boxes: Current frame boxes [N, 10]
        curr_instance_ids: Current frame instance IDs [N]

    Returns:
        object_ids: [K] instance IDs with valid transforms
        transforms: [K, 4, 4] rigid transforms (prev -> curr)
    """
    # pylint: disable=too-many-locals

    # Find matching objects across frames
    # Match by instance ID: prev_instance_ids[i] == curr_instance_ids[j]
    match_matrix = prev_instance_ids[:, None] == curr_instance_ids[None, :]  # [M, N]

    # Get valid matches (each prev can match at most one curr)
    # filter out negative IDs (invalid objects)
    valid_prev = prev_instance_ids >= 0
    valid_curr = curr_instance_ids >= 0
    match_matrix = match_matrix & valid_prev[:, None] & valid_curr[None, :]

    # Get indices of matched pairs
    prev_matched, curr_matched = torch.where(match_matrix)

    if len(prev_matched) == 0:
        # No matched objects
        device = prev_boxes.device
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(
            0, 4, 4, device=device
        )

    # Extract matched box data
    prev_centers = prev_boxes[prev_matched, :3]  # [K, 3]
    prev_yaws = prev_boxes[prev_matched, 6]  # [K]

    curr_centers = curr_boxes[curr_matched, :3]  # [K, 3]
    curr_yaws = curr_boxes[curr_matched, 6]  # [K]

    # Compute transforms: T(curr_center) @ R_z(delta_yaw) @ T(-prev_center)
    delta_yaws = curr_yaws - prev_yaws  # [K]

    # Build 4x4 transforms
    k = len(prev_matched)
    device = prev_boxes.device
    dtype = prev_boxes.dtype

    transforms = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(k, -1, -1)
    transforms = transforms.clone()

    # Rotation around Z-axis
    cos_yaw = torch.cos(delta_yaws)
    sin_yaw = torch.sin(delta_yaws)

    # Rotation matrix (Z-axis)
    # [[cos, -sin, 0],
    #  [sin,  cos, 0],
    #  [0,    0,   1]]
    transforms[:, 0, 0] = cos_yaw
    transforms[:, 0, 1] = -sin_yaw
    transforms[:, 1, 0] = sin_yaw
    transforms[:, 1, 1] = cos_yaw

    # Apply T(curr_center) @ R @ T(-prev_center)
    # For a point p: p' = T(curr_center) @ R @ T(-prev_center) @ p
    # = R @ (p - prev_center) + curr_center
    # = R @ p - R @ prev_center + curr_center

    # Translation: -R @ prev_center + curr_center
    rotated_prev = torch.stack(
        [
            cos_yaw * prev_centers[:, 0] - sin_yaw * prev_centers[:, 1],
            sin_yaw * prev_centers[:, 0] + cos_yaw * prev_centers[:, 1],
            prev_centers[:, 2],
        ],
        dim=-1,
    )  # [K, 3]

    transforms[:, :3, 3] = curr_centers - rotated_prev

    object_ids = prev_instance_ids[prev_matched]

    return object_ids, transforms


def match_gaussians_to_objects(
    gaussian_centers: torch.Tensor,
    box_centers: torch.Tensor,
    box_dims: torch.Tensor,
    box_rots: torch.Tensor,
    box_instance_ids: torch.Tensor,
    max_distance: float = 2.0,
) -> torch.Tensor:
    """
    Match gaussian centers to GT boxes using voxel assignment pattern.

    Strategy:
    1. intersect_points_box() - check if inside any box
    2. count == 1: direct assignment
    3. count == 0: distance_points_box() -> nearest box IF within max_distance
    4. count > 1: Use center distance for tie-breaking

    Args:
        gaussian_centers: Gaussian centers [b, n, 3]
        box_centers: Box centers [M, 3]
        box_dims: Box dimensions [M, 3]
        box_rots: Box yaw angles [M]
        box_instance_ids: Box instance IDs [M]
        max_distance: Max distance for fallback matching

    Returns:
        matched_ids: [b, n] instance ID per gaussian (-1 = unmatched)
    """
    # pylint: disable=too-many-locals

    b, n, _ = gaussian_centers.shape
    device = gaussian_centers.device
    dtype = gaussian_centers.dtype

    if box_centers.shape[0] == 0:
        # No boxes - all unmatched
        return torch.full((b, n), -1, dtype=torch.long, device=device)

    # Flatten batch dimension for matching
    centers_flat = gaussian_centers.view(b * n, 3)  # [b*n, 3]

    # Check which points are inside which boxes
    # intersect_points_box returns [M, b*n] where True = inside
    inside = intersect_points_box(
        centers_flat, box_centers, box_dims, box_rots
    )  # [M, b*n]

    # Count how many boxes each point is inside
    inside_count = inside.sum(dim=0)  # [b*n]

    # Initialize result with -1 (unmatched)
    matched_ids = torch.full((b * n,), -1, dtype=torch.long, device=device)

    # Case 1: Points inside exactly one box -> direct assignment
    single_box_mask = inside_count == 1
    if single_box_mask.any():
        # Get the box index for each point
        box_idx = inside[:, single_box_mask].long().argmax(dim=0)  # [num_single]
        matched_ids[single_box_mask] = box_instance_ids[box_idx]

    # Case 2: Points inside multiple boxes -> tie-break by center distance
    multi_box_mask = inside_count > 1
    if multi_box_mask.any():
        multi_points = centers_flat[multi_box_mask]  # [num_multi, 3]
        # Compute distances to all box centers
        dists = torch.cdist(multi_points, box_centers)  # [num_multi, M]
        # Mask out boxes that don't contain the point
        inside_multi = inside[:, multi_box_mask].t()  # [num_multi, M]
        dists = torch.where(
            inside_multi, dists, torch.tensor(float("inf"), device=device, dtype=dtype)
        )
        # Select nearest containing box
        nearest_box = dists.argmin(dim=1)  # [num_multi]
        matched_ids[multi_box_mask] = box_instance_ids[nearest_box]

    # Case 3: Points not inside any box -> nearest box if within max_distance
    no_box_mask = inside_count == 0
    if no_box_mask.any():
        no_points = centers_flat[no_box_mask]  # [num_none, 3]
        # Compute distance to box surfaces
        dists = distance_points_box(
            no_points, box_centers, box_dims, box_rots
        )  # [M, num_none]
        dists = dists.t()  # [num_none, M]

        # Find nearest box for each point
        min_dists, nearest_box = dists.min(dim=1)  # [num_none], [num_none]

        # Only assign if within max_distance
        close_enough = min_dists < max_distance
        if close_enough.any():
            no_box_indices = torch.where(no_box_mask)[0]
            matched_ids[no_box_indices[close_enough]] = box_instance_ids[
                nearest_box[close_enough]
            ]

    return matched_ids.view(b, n)


def _transform_centers(
    centers: torch.Tensor,
    transform: torch.Tensor,
) -> torch.Tensor:
    """
    Transform centers using 4x4 transform matrix.

    Args:
        centers: Centers [b, n, 3]
        transform: 4x4 transformation matrix

    Returns:
        Transformed centers [b, n, 3]
    """
    b, n, _ = centers.shape

    # Convert to homogeneous coordinates
    ones = torch.ones(b, n, 1, device=centers.device, dtype=centers.dtype)
    centers_h = torch.cat([centers, ones], dim=-1)  # [b, n, 4]

    # Apply transformation
    centers_transformed = torch.einsum("ij,bnj->bni", transform, centers_h)

    return centers_transformed[..., :3]


def _quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Compute quaternion conjugate (inverse for unit quaternions).

    Args:
        q: Quaternion [..., 4] in [w, x, y, z] format

    Returns:
        Conjugate quaternion [..., 4]
    """
    # conjugate = [w, -x, -y, -z]
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)


def compute_dynamic_targets(
    temporal_gaussians: dict[str, MetaDict],
    prev_labels: MetaDict,
    curr_labels: MetaDict,
    dynamic_class_ids: torch.Tensor,
    ego_transform: torch.Tensor,
    max_match_distance: float = 2.0,
) -> tuple[dict[str, MetaDict], dict[str, torch.Tensor]]:
    """
    Compute object-motion targets for dynamic gaussians.

    For gaussians matched to GT objects, computes where they should be
    based on object motion. For static gaussians or unmatched dynamics,
    falls back to EMC targets (handled by caller).

    Args:
        temporal_gaussians: Previous frame gaussians (already EMC-transformed,
            so centers are in curr_ego frame)
            dict[stream] -> MetaDict(centers, rotations, ...)
        prev_labels: Previous frame GT labels (boxes in prev_ego frame)
        curr_labels: Current frame GT labels (boxes in curr_ego frame)
        dynamic_class_ids: Tensor of class IDs considered dynamic
        ego_transform: Transform from prev_ego to curr_ego frame [4, 4]
            (same as used for EMC: curr_ego^-1 @ prev_ego)
        max_match_distance: Max distance for fallback matching

    Returns:
        dynamic_targets: dict[stream] -> MetaDict(centers, rotations, ...) for matched
        matched_ids: dict[stream] -> [b, n] instance IDs per gaussian (-1 = unmatched)
    """
    # pylint: disable=too-many-locals, too-many-statements

    # Extract boxes and instance IDs from labels
    # labels.boxes is a PackedTensor, labels.instance_ids is a PackedTensor
    prev_boxes = prev_labels.boxes.data  # [M_prev, 10]
    prev_instance_ids = prev_labels.instance_ids.data  # [M_prev]

    curr_boxes = curr_labels.boxes.data  # [M_curr, 10]
    curr_instance_ids = curr_labels.instance_ids.data  # [M_curr]

    # Compute per-object transforms
    object_ids, object_transforms = compute_object_transforms(
        prev_boxes, prev_instance_ids, curr_boxes, curr_instance_ids
    )

    dynamic_targets = {}
    matched_ids_dict = {}

    # Build dynamic class lookup (is_dynamic_class[c] = True if c is dynamic)
    num_classes = (
        dynamic_class_ids.max().item() + 1 if len(dynamic_class_ids) > 0 else 0
    )
    is_dynamic_class = torch.zeros(
        num_classes + 1, dtype=torch.bool, device=dynamic_class_ids.device
    )
    is_dynamic_class[dynamic_class_ids] = True

    # Compute inverse ego transform and ego rotation quaternion (for coordinate corrections)
    ego_transform_inv = torch.inverse(ego_transform.to(torch.float64)).to(
        ego_transform.dtype
    )
    q_ego = rotation_matrix_to_quaternion(ego_transform[:3, :3])  # [4]
    q_ego_inv = _quaternion_conjugate(q_ego)  # [4]

    for stream_name, stream_data in temporal_gaussians.items():
        centers_curr_ego = stream_data.centers  # [b, n, 3] - in curr_ego (after EMC)
        rotations_curr_ego = (
            stream_data.rotations
        )  # [b, n, 4] - in curr_ego (after EMC)
        logits = stream_data.logits  # [b, n, num_classes]

        b, n, _ = centers_curr_ego.shape
        device = centers_curr_ego.device

        # Transform centers back to prev_ego frame for matching with prev_boxes
        centers_prev_ego = _transform_centers(centers_curr_ego, ego_transform_inv)

        # Determine which gaussians are predicted as dynamic classes
        pred_classes = logits.detach().argmax(dim=-1)  # [b, n]
        # Clamp to valid range for lookup
        pred_classes_clamped = pred_classes.clamp(0, len(is_dynamic_class) - 1)
        is_dynamic = is_dynamic_class.to(device)[pred_classes_clamped]  # [b, n]

        # Initialize matched_ids to -1 (unmatched)
        matched_ids = torch.full((b, n), -1, dtype=torch.long, device=device)

        # Only match dynamic gaussians to objects
        # Use prev_ego frame centers for matching (same frame as prev_boxes)
        if is_dynamic.any() and prev_boxes.shape[0] > 0:
            dynamic_matched = match_gaussians_to_objects(
                gaussian_centers=centers_prev_ego,  # In prev_ego frame
                box_centers=prev_boxes[:, :3],  # In prev_ego frame
                box_dims=prev_boxes[:, 3:6],
                box_rots=prev_boxes[:, 6],
                box_instance_ids=prev_instance_ids,
                max_distance=max_match_distance,
            )  # [b, n]

            # Only keep matches for dynamic gaussians
            matched_ids = torch.where(is_dynamic, dynamic_matched, matched_ids)

        # Initialize targets as copies (will be overwritten for matched dynamics)
        # Targets should be in curr_ego frame (same as EMC targets)
        target_centers = centers_curr_ego.clone()
        target_rotations = rotations_curr_ego.clone()

        # For each matched object, compute correct target position/rotation
        # The correct target for a dynamic gaussian is:
        #   center: curr_obj_center + R_obj @ (p_prev - prev_obj_center)
        #   rotation: q_delta_obj @ q_ego^-1 @ q_emc
        for obj_id, obj_transform in zip(object_ids, object_transforms):
            mask = matched_ids == obj_id  # [b, n]

            if not mask.any():
                continue

            # Get gaussian centers in prev_ego frame
            matched_centers_prev = centers_prev_ego[mask]  # [num_matched, 3]

            # Get previous object center (in prev_ego frame)
            prev_obj_mask = prev_instance_ids == obj_id
            prev_obj_center = prev_boxes[prev_obj_mask, :3].squeeze(0)  # [3]

            # Get current object center (in curr_ego frame)
            curr_obj_mask = curr_instance_ids == obj_id
            curr_obj_center = curr_boxes[curr_obj_mask, :3].squeeze(0)  # [3]

            # Compute offset from object center in prev_ego frame
            offset_prev = matched_centers_prev - prev_obj_center  # [num_matched, 3]

            # Extract object rotation from transform
            r_obj = obj_transform[:3, :3]  # [3, 3]

            # Apply object rotation to offset, then add current center
            # Target = curr_obj_center + R_obj @ offset_prev (in curr_ego frame)
            rotated_offset = torch.einsum("ij,...j->...i", r_obj, offset_prev)
            new_centers = curr_obj_center + rotated_offset  # [num_matched, 3]

            target_centers[mask] = new_centers.to(dtype=target_centers.dtype)

            # Update rotations:
            # EMC gave us: q_emc = q_ego @ q_original
            # Correct rotation: q_correct = q_delta_obj @ q_original
            # So: q_correct = q_delta_obj @ q_ego^-1 @ q_emc

            # Extract delta_yaw from object transform
            delta_yaw = torch.atan2(obj_transform[1, 0], obj_transform[0, 0])

            # Create rotation quaternion for delta_yaw (Z-axis rotation)
            q_delta_obj = torch.zeros(4, device=device, dtype=target_rotations.dtype)
            q_delta_obj[0] = torch.cos(delta_yaw / 2)
            q_delta_obj[3] = torch.sin(delta_yaw / 2)

            # Get matched rotations (q_emc)
            matched_rots = rotations_curr_ego[mask]  # [num_matched, 4]

            # Compute: q_correct = q_delta_obj @ q_ego^-1 @ q_emc
            q_ego_inv_exp = q_ego_inv.unsqueeze(0).expand(matched_rots.shape[0], -1)
            q_delta_exp = q_delta_obj.unsqueeze(0).expand(matched_rots.shape[0], -1)

            # First: q_ego^-1 @ q_emc (undo ego rotation)
            intermediate = quaternion_multiply(q_ego_inv_exp, matched_rots)
            # Then: q_delta_obj @ intermediate (apply object rotation)
            new_rots = quaternion_multiply(q_delta_exp, intermediate)
            new_rots = torch.nn.functional.normalize(new_rots, dim=-1)

            target_rotations[mask] = new_rots.to(dtype=target_rotations.dtype)

        # Create target MetaDict with updated centers and rotations
        dynamic_targets[stream_name] = MetaDict(
            {
                "centers": target_centers.detach(),
                "rotations": target_rotations.detach(),
                "scales": stream_data.scales.clone().detach(),
                "opacities": stream_data.opacities.clone().detach(),
                "logits": stream_data.logits.clone().detach(),
            }
        )

        matched_ids_dict[stream_name] = matched_ids

    return dynamic_targets, matched_ids_dict
