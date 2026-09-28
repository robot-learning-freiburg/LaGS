# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import abc
from typing import Any

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from ......config.registry import Registry
from ..utils import quaternion_multiply, rotation_matrix_to_quaternion
from .storage import GaussianInstances, GaussianStreamInstances

# Registry for ego-motion compensation modules
registry = Registry("lags.occupancy.gaussian.ego_motion")


class EgoMotionCompensation(nn.Module):
    """
    Base class for ego-motion compensation of temporal gaussians.

    Transforms gaussian parameters from previous frame's ego coordinate system
    to current frame's ego coordinate system.
    """

    @abc.abstractmethod
    def forward(
        self,
        gaussians: GaussianInstances,
        transform: torch.Tensor,
    ) -> GaussianInstances:
        """
        Apply ego-motion compensation to temporal gaussians.

        Args:
            gaussians: Gaussians from previous frame (in previous ego frame)
            transform: Relative transformation [4, 4] = ego_to_global_curr^-1 @ ego_to_global_prev

        Returns:
            Transformed gaussians (in current ego frame)
        """
        raise NotImplementedError()


def build(conf: OmegaConf | None = None, **kwargs: Any) -> EgoMotionCompensation:
    """Build an ego-motion compensation module from config."""
    return registry.from_config(conf, **kwargs)


@registry.register(key="identity")
class IdentityEgoMotionCompensation(EgoMotionCompensation):
    """
    No-op ego-motion compensation — gaussians stay in the previous frame's
    ego coordinate system.

    Used as ablation baseline to measure the contribution of EMC.
    """

    def __init__(self, voxel_range: list[float]):
        """
        Args:
            voxel_range: Voxel coordinate range [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        # pylint: disable=unused-argument
        super().__init__()

    def forward(
        self,
        gaussians: GaussianInstances,
        transform: torch.Tensor,
    ) -> GaussianInstances:
        return gaussians


@registry.register(key="geometric")
class GeometricEgoMotionCompensation(EgoMotionCompensation):
    """
    Geometric ego-motion compensation using rigid transformations.

    This is the baseline approach that applies pure geometric transformations:
    - Query coords: Transform with full 4x4 matrix (rotation + translation)
    - Centers: Transform with full 4x4 matrix (rotation + translation)
    - Rotations: Compose with ego rotation
    """

    def __init__(self, voxel_range: list[float]):
        """
        Args:
            voxel_range: Voxel coordinate range [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        super().__init__()

        # Register as non-persistent buffer for automatic device handling
        voxel_range_tensor = torch.as_tensor(voxel_range, dtype=torch.float32)
        self.register_buffer("voxel_range", voxel_range_tensor, persistent=False)

    def transform_centers(
        self,
        centers: torch.Tensor,
        transform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Transform gaussian centers from previous ego frame to current ego frame.

        Args:
            centers: Centers [b, n, 3] in world coordinates (meters)
            transform: Transformation matrix [4, 4] = ego_to_global_curr^-1 @ ego_to_global_prev

        Returns:
            Transformed centers [b, n, 3] in current ego frame
        """
        b, n, _ = centers.shape

        # Convert to homogeneous coordinates
        ones = torch.ones(b, n, 1, device=centers.device, dtype=centers.dtype)
        centers = torch.cat([centers, ones], dim=-1)  # [b, n, 4]

        # Apply transformation: (T @ centers.T).T
        # transform is [4, 4], centers is [b, n, 4]
        # Need to broadcast: [4, 4] @ [b, n, 4].T -> [4, b, n].T -> [b, n, 4]
        centers = torch.einsum("ij,bnj->bni", transform, centers)

        # Return xyz coordinates (drop homogeneous coordinate)
        return centers[..., :3]

    def transform_rotations(
        self,
        rotations: torch.Tensor,
        transform: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compose gaussian rotations with ego rotation.

        Args:
            rotations: Quaternions [b, n, 4] in [w, x, y, z] format
            transform: Transformation matrix [4, 4] = ego_to_global_curr^-1 @ ego_to_global_prev

        Returns:
            Composed rotations [b, n, 4]
        """
        # Extract rotation matrix from transformation (top-left 3x3)
        r_ego = transform[:3, :3]  # [3, 3]

        # Convert rotation matrix to quaternion
        q_ego = rotation_matrix_to_quaternion(r_ego)  # [4]

        # Compose quaternions: q_result = q_ego * q_gauss
        # Broadcasting: [4] * [b, n, 4] -> [b, n, 4]
        q_ego = q_ego.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
        rotations = quaternion_multiply(q_ego, rotations)

        return F.normalize(rotations, dim=-1)

    def normalize(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Normalize coords from world coordinates to [0, 1] normalized coordinates.

        Args:
            coords: Coordinates [b, n, 3] in world coordinates

        Returns:
            Normalized coordinates [b, n, 3] in [0, 1]
        """
        vx_min, vx_max = self.voxel_range[:3], self.voxel_range[3:]

        # Normalize: (coords - min) / (max - min)
        normalized = (coords - vx_min) / (vx_max - vx_min)

        return normalized

    def denormalize(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Convert coords from [0, 1] normalized coordinates to world coordinates.

        Args:
            coords: Normalized coordinates [b, n, 3] in [0, 1]

        Returns:
            World coordinates [b, n, 3]
        """
        vx_min, vx_max = self.voxel_range[:3], self.voxel_range[3:]

        # Denormalize: coords * (max - min) + min
        world_coords = coords * (vx_max - vx_min) + vx_min

        return world_coords

    def forward(
        self,
        gaussians: GaussianInstances,
        transform: torch.Tensor,
    ) -> GaussianInstances:
        """
        Apply geometric ego-motion compensation.

        Args:
            gaussians: Gaussians from previous frame
            transform: Transformation matrix [4, 4] = ego_to_global_curr^-1 @ ego_to_global_prev

        Returns:
            Transformed gaussians in current ego frame
        """
        transformed_streams = {}

        for stream_name, stream_data in gaussians.streams.items():
            # Transform reference positions (query_coords)
            query_coords = self.denormalize(stream_data.query_coords)
            query_coords = self.transform_centers(query_coords, transform)
            query_coords = self.normalize(query_coords)

            # Transform decoded centers and rotations
            centers = self.transform_centers(stream_data.centers, transform)
            rotations = self.transform_rotations(stream_data.rotations, transform)

            # Create transformed stream instance
            transformed_stream = GaussianStreamInstances(
                query=stream_data.query,
                query_coords=query_coords,
                logits=stream_data.logits,
                centers=centers,
                scales=stream_data.scales,
                rotations=rotations,
                opacities=stream_data.opacities,
                age=stream_data.age,
                confidence=stream_data.confidence,
                instance_ids=stream_data.instance_ids,
            )

            transformed_streams[stream_name] = transformed_stream

        return GaussianInstances(streams=transformed_streams)


@registry.register(key="feature_aware")
class FeatureAwareEgoMotionCompensation(EgoMotionCompensation):
    """
    Feature-aware ego-motion compensation using FiLM modulation.

    In addition to geometric transformations, this module also transforms
    query features to account for the spatial context shift caused by ego-motion.

    Key components:
    - Per-query motion encoding: Encodes displacement vector for each query
    - Global motion context: Translation magnitude + rotation angle
    - FiLM modulation: Global context modulates per-query updates (gamma * delta + beta)
    - Gated update: Learnable gate allows gradual feature adaptation
    """

    def __init__(
        self,
        voxel_range: list[float],
        embed_dim: int,
        motion_hidden_dim: int = 64,
        init_scale: float = 0.1,
    ):
        """
        Args:
            voxel_range: Voxel coordinate range [x_min, y_min, z_min, x_max, y_max, z_max]
            embed_dim: Dimension of query features
            motion_hidden_dim: Hidden dimension for motion encoders
            init_scale: Scale for weight initialization (smaller = more conservative updates)
        """
        super().__init__()

        self.geometric_emc = GeometricEgoMotionCompensation(voxel_range)
        self.embed_dim = embed_dim

        # Per-query motion encoder: displacement [3] -> [motion_hidden_dim]
        self.motion_encoder = nn.Sequential(
            nn.Linear(3, motion_hidden_dim),
            nn.ReLU(),
            nn.Linear(motion_hidden_dim, motion_hidden_dim),
        )

        # Global motion encoder: [trans_magnitude, rot_angle] -> [motion_hidden_dim]
        self.global_motion_encoder = nn.Sequential(
            nn.Linear(2, motion_hidden_dim),
            nn.ReLU(),
            nn.Linear(motion_hidden_dim, motion_hidden_dim),
        )

        # FiLM generator: global_context -> (gamma, beta)
        self.film_generator = nn.Linear(motion_hidden_dim, embed_dim * 2)

        # Feature update: (query + motion_embed) -> delta
        self.feature_update = nn.Sequential(
            nn.Linear(embed_dim + motion_hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learnable gate initialized near zero
        self.update_gate = nn.Parameter(torch.tensor(0.0))

        # Initialize weights with small scale for conservative initial updates
        self._init_weights(init_scale)

    def _init_weights(self, scale: float):
        """Initialize weights with small scale to preserve pretrained features initially."""
        for module in [
            self.motion_encoder,
            self.global_motion_encoder,
            self.feature_update,
        ]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=scale)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Initialize FiLM generator: gamma near 1, beta near 0
        nn.init.normal_(self.film_generator.weight, mean=0.0, std=scale)
        nn.init.zeros_(self.film_generator.bias)
        # Set gamma bias to 1.0 (identity scaling initially)
        self.film_generator.bias.data[: self.embed_dim] = 1.0

    def _extract_motion_context(
        self, transform: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract translation magnitude and rotation angle from 4x4 matrix.

        Args:
            transform: Transformation matrix [4, 4]

        Returns:
            trans_magnitude: Scalar translation magnitude
            rot_angle: Scalar rotation angle in radians
        """
        translation = transform[:3, 3]
        trans_magnitude = torch.norm(translation)

        rotation_matrix = transform[:3, :3]
        trace = rotation_matrix.diagonal().sum()
        # Clamp to valid range for acos
        rot_angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))

        return trans_magnitude, rot_angle

    def forward(
        self,
        gaussians: GaussianInstances,
        transform: torch.Tensor,
    ) -> GaussianInstances:
        """
        Apply feature-aware ego-motion compensation.

        Args:
            gaussians: Gaussians from previous frame
            transform: Transformation matrix [4, 4] = ego_to_global_curr^-1 @ ego_to_global_prev

        Returns:
            Transformed gaussians with updated features
        """
        # pylint: disable=too-many-locals

        # 1. Apply geometric EMC first
        transformed = self.geometric_emc(gaussians, transform)

        # 2. Extract global motion context
        trans_mag, rot_angle = self._extract_motion_context(transform)
        global_context = torch.stack([trans_mag, rot_angle]).view(1, 1, 2)
        global_embed = self.global_motion_encoder(global_context)  # [1, 1, hidden]

        # 3. Generate FiLM parameters
        film_params = self.film_generator(global_embed)
        gamma = film_params[..., : self.embed_dim]
        beta = film_params[..., self.embed_dim :]

        # 4. Transform features for each stream
        for stream_name, stream_data in transformed.streams.items():
            original = gaussians.streams[stream_name]

            # Compute per-query displacement in world coordinates
            world_coords = self.geometric_emc.denormalize(original.query_coords)
            world_coords_new = self.geometric_emc.transform_centers(
                world_coords, transform
            )
            displacement = world_coords_new - world_coords  # [b, n, 3]

            # Encode motion
            motion_embed = self.motion_encoder(displacement)  # [b, n, hidden]

            # Compute feature update
            query_motion = torch.cat([original.query, motion_embed], dim=-1)
            delta = self.feature_update(query_motion)  # [b, n, embed_dim]

            # Apply FiLM modulation
            delta = gamma * delta + beta

            # Gated update
            gate = torch.sigmoid(self.update_gate)
            stream_data.query = original.query + gate * delta

        return transformed
