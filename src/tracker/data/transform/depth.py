# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, List, Mapping, Tuple

import torch

from ... import t3d
from ...utils.types import MetaDict, Sample
from .registry import registry as transform
from .transform import Transform


def _depth_from_points(
    points: torch.Tensor,
    width: int,
    height: int,
    depth_range: tuple[float, float],
    downsample: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create depth map from LiDAR point cloud.

    Args:
        points: Point cloud in the image plane (N, 3).
        width: Image width.
        height: Image height.
        depth_range: Depth range (min, max).

    Returns:
        depth: Depth map (height, width).
        mask: Depth mask (height, width). True for valid pixels, False for invalid pixels.
    """
    height, width = height // downsample, width // downsample

    coords, z = points[:, :2], points[:, 2]
    coords = coords / downsample
    coords = coords.round()

    # filter points based on depth and image bounds
    mask = (z >= depth_range[0]) & (z < depth_range[1])
    mask = mask & (coords[:, 0] >= 0) & (coords[:, 0] < width)
    mask = mask & (coords[:, 1] >= 0) & (coords[:, 1] < height)

    coords, z = coords[mask], z[mask]

    # order points by coordinate and depth
    # Note: add +2 to retain major pixel order, i.e., ensure (z / max_depth) < 1
    ranks = coords[:, 1] * width + coords[:, 0]
    order = ranks + z / (depth_range[1] + 2)
    order = order.argsort()

    ranks, coords, z = ranks[order], coords[order], z[order]

    # discard all but closest point for each pixel
    mask = torch.ones_like(z, dtype=torch.bool)
    mask[1:] = ranks[1:] != ranks[:-1]

    coords, z = coords[mask], z[mask]
    coords = coords.long()

    # create actual depth map
    depth = torch.zeros((height, width), dtype=z.dtype, device=z.device)
    depth[coords[:, 1], coords[:, 0]] = z

    mask = torch.zeros((height, width), dtype=torch.bool, device=z.device)
    mask[coords[:, 1], coords[:, 0]] = True

    return depth, mask


@transform.register
class DepthFromLidar(Transform):
    """
    Create depth map from LiDAR point cloud.
    """

    def __init__(self, range: tuple[float, float], downsample: int = 1) -> None:
        # pylint: disable=redefined-builtin
        super().__init__()

        self.depth_range = range
        self.downsample = downsample

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "depth_range": self.depth_range,
            "downsample": self.downsample,
        }

    def _generate_depth_map(
        self,
        points: torch.Tensor,
        images: torch.Tensor,
        images_tx: MetaDict,
        lidar_to_global_tx: t3d.Transform,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depths, masks = [], []

        for i in range(images.shape[0]):
            *_, h, w = images[i].shape

            # transformation from global frame to camera frame at time of camera capture
            global_to_camera_tx = t3d.Sequential(
                # camera frame to ego vehicle
                images_tx.extrinsic[i],
                # ego vehicle to global frame (at camera timestamp)
                images_tx.pose[i],
            ).inv

            # transformation from lidar to image plane
            lidar_to_image_tx = t3d.Sequential(
                lidar_to_global_tx,
                global_to_camera_tx,
                images_tx.intrinsic[i],
            )
            lidar_to_image_tx = lidar_to_image_tx.fused()
            lidar_to_image_tx = lidar_to_image_tx.to(dtype=points.dtype)
            lidar_to_image_tx = t3d.ViewProjection(lidar_to_image_tx)

            # transform points to image plane
            points_img = lidar_to_image_tx(points)

            # create depth map
            depth, mask = _depth_from_points(
                points=points_img,
                height=h,
                width=w,
                depth_range=self.depth_range,
                downsample=self.downsample,
            )

            depths.append(depth)
            masks.append(mask)

        depths = torch.stack(depths, dim=0)
        masks = torch.stack(masks, dim=0)

        return depths, masks

    def apply(self, sample: Sample) -> Sample:
        assert sample.batch_size == 1

        points = sample.points.data.get(0)
        points = points[:, :3]

        # transformation from lidar to global frame at time of lidar capture
        tx_lidar_to_global = t3d.Sequential(
            # lidar sensor to ego vehicle frame
            sample.points.meta.transforms.extrinsic,
            # ego vehicle to global frame (at lidar timestamp)
            sample.points.meta.transforms.pose,
        )

        # build depth maps for each camera
        depth, mask = self._generate_depth_map(
            points=points,
            images=sample.images.data,
            images_tx=sample.images.meta.transforms,
            lidar_to_global_tx=tx_lidar_to_global,
        )

        # collect and store
        depth = {
            "data": depth,
            "mask": mask,
        }
        sample.depth = MetaDict(depth)

        return sample


@transform.register
class DepthFromLidarMultiFrame(DepthFromLidar):
    """
    Create depth map from LiDAR point cloud.
    """

    def apply(self, sample: Sample) -> Sample:
        assert sample.batch_size == 1

        # this variant only works with multi-frame point-cloud data
        assert isinstance(sample.points, List | Tuple)

        # generate depth maps for each frame
        depths, masks = [], []
        for t, points in enumerate(sample.points):
            coords = points.data.get(0)
            coords = coords[:, :3]

            # transformation from lidar to global frame at time of lidar capture
            tx_lidar_to_global = t3d.Sequential(
                # lidar sensor to ego vehicle frame
                points.meta.transforms.extrinsic,
                # ego vehicle to global frame (at lidar timestamp)
                points.meta.transforms.pose,
            )

            images_tx = MetaDict(
                {
                    "intrinsic": sample.images.meta.transforms.intrinsic[t],
                    "extrinsic": sample.images.meta.transforms.extrinsic[t],
                    "pose": sample.images.meta.transforms.pose[t],
                }
            )

            # build depth maps for each camera
            depth, mask = self._generate_depth_map(
                points=coords,
                images=sample.images.data[t],
                images_tx=images_tx,
                lidar_to_global_tx=tx_lidar_to_global,
            )

            depths.append(depth)
            masks.append(mask)

        # collect and store
        depth = {
            "data": torch.stack(depths, dim=0),
            "mask": torch.stack(masks, dim=0),
        }
        sample.depth = MetaDict(depth)

        return sample
