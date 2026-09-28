# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any, List, Mapping, Tuple

import torch
from pyquaternion import Quaternion

from ... import t3d
from ...utils.math import normalize_angle, rotate_2d, rotate_3d_z
from ...utils.types import PackedTensor, Sample
from .registry import registry as transform
from .transform import Transform


@transform.register(namespace="augment")
# pylint: disable-next=too-few-public-methods
class RandomShufflePoints(Transform):
    """
    Randomly shuffle point cloud points.
    """

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {"type": f"augment.{self.__class__.__name__}"}

    def apply(self, sample: Sample) -> Sample:
        assert sample.batch_size == 1

        points: PackedTensor = sample.points.data
        points.data[:, :] = points.data[torch.randperm(points.data.shape[0]), :]

        return sample


@transform.register(namespace="augment")
# pylint: disable-next=too-few-public-methods
class RandomTranslatePoints(Transform):
    """
    Randomly translate point cloud points and bounding box annotations.
    """

    def __init__(self, stddev: float | List[float] | torch.Tensor):
        """
        Args:
            stddev (float, list[float], or torch.Tensor): Standard deviation of \
                the translation in each axis. If a float is given, the same \
                value is used for all axes
        """
        super().__init__()

        # convert stddev to a torch tensor of shape [3]
        if isinstance(stddev, torch.Tensor):
            self.stddev = stddev.expand(3)
        elif isinstance(stddev, (list, tuple)):
            self.stddev = torch.tensor(stddev)
        else:
            self.stddev = torch.tensor([stddev, stddev, stddev])

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"augment.{self.__class__.__name__}",
            "stddev": self.stddev.tolist(),
        }

    def _translate_points(self, sample: Sample, offsets: torch.Tensor) -> None:
        points: PackedTensor = sample.points.data
        points.data[:, 0:3] += offsets[None, :]

    def _translate_boxes(self, sample: Sample, offsets: torch.Tensor) -> None:
        if "labels" not in sample or "boxes" not in sample.labels:
            return

        boxes: PackedTensor = sample.labels.boxes
        boxes.data[:, 0:3] += offsets[None, :]

        if "trajectories" in sample.labels:
            centers: PackedTensor = sample.labels.trajectories.center
            centers.data[:, :, :] += offsets[None, None, :]

    def _translate_transforms(self, sample: Sample, offsets: torch.Tensor) -> None:
        transforms = sample.points.meta.transforms

        transforms.extrinsic = t3d.Sequential(
            t3d.Translate(-offsets, dtype=transforms.extrinsic.dtype),
            transforms.extrinsic,
        )

    def apply(self, sample: Sample) -> Sample:
        # generate a random translation vector for each batch element
        mean = torch.tensor(0, dtype=self.stddev.dtype).expand(3)
        std = self.stddev.expand(3)

        offsets = torch.normal(mean, std)

        # apply translation
        self._translate_points(sample, offsets)
        self._translate_boxes(sample, offsets)
        self._translate_transforms(sample, offsets)

        return sample


@transform.register(namespace="augment")
# pylint: disable-next=too-few-public-methods
class RandomUniformScalePoints(Transform):
    """
    Randomly scale point cloud points and bounding box annotations uniformly,
    meaning with same scaling factor for each axis.
    """

    # pylint: disable-next=redefined-builtin
    def __init__(self, range: Tuple[float, float]):
        """
        Args:
            range ([float, float]): Range of random scaling factors (min, max).
        """
        super().__init__()

        self.min, self.max = range

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"augment.{self.__class__.__name__}",
            "range": [self.min, self.max],
        }

    def _scale_points(self, sample: Sample, scale: torch.Tensor) -> None:
        points: PackedTensor = sample.points.data

        points.data[:, 0:3] *= scale

    def _scale_boxes(self, sample: Sample, scale: torch.Tensor) -> None:
        if "labels" not in sample or "boxes" not in sample.labels:
            return

        boxes: PackedTensor = sample.labels.boxes

        # scale everything except rotation (0:3: pos, 3:6 dim, 6: rot, 7:9 vel)
        boxes.data[:, :6] *= scale
        boxes.data[:, 7:] *= scale

        if "trajectories" in sample.labels:
            centers: PackedTensor = sample.labels.trajectories.center
            centers.data[:, :, :] *= scale

    def _scale_transforms(self, sample: Sample, scale: torch.Tensor) -> None:
        transforms = sample.points.meta.transforms

        transforms.extrinsic = t3d.Sequential(
            t3d.Scale(1.0 / scale, dtype=transforms.extrinsic.dtype),
            transforms.extrinsic,
        )

    def apply(self, sample: Sample) -> Sample:
        assert sample.batch_size == 1

        # generate a random scaling factor
        scale = torch.rand(1)
        scale = self.min + scale * (self.max - self.min)

        # apply scaling
        self._scale_points(sample, scale)
        self._scale_boxes(sample, scale)
        self._scale_transforms(sample, scale)

        return sample


@transform.register(namespace="augment")
# pylint: disable-next=too-few-public-methods
class RandomRotatePoints(Transform):
    """
    Randomly rotate point cloud points and bounding box annotations around
    the origin and up-axis (z).
    """

    # pylint: disable-next=redefined-builtin
    def __init__(self, range: Tuple[float, float]):
        """
        Args:
            range ([float, float]): Range of random rotation angles in radians (min, max).
        """
        super().__init__()

        self.min, self.max = range

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"augment.{self.__class__.__name__}",
            "range": [self.min, self.max],
        }

    def _rotate_points(self, sample: Sample, angle: torch.Tensor) -> None:
        points: PackedTensor = sample.points.data

        points.data[:, 0:3] = rotate_3d_z(points.data[:, 0:3], angle)

    def _rotate_boxes(self, sample: Sample, angle: torch.Tensor) -> None:
        if "labels" not in sample or "boxes" not in sample.labels:
            return

        boxes: PackedTensor = sample.labels.boxes

        # rotate box centers and velocities
        boxes.data[:, 0:3] = rotate_3d_z(boxes.data[:, 0:3], angle)
        boxes.data[:, 7:9] = rotate_2d(boxes.data[:, 7:9], angle)

        # update box rotation
        boxes.data[:, 6] = normalize_angle(boxes.data[:, 6] + angle)

        # update trajectories, if present
        if "trajectories" in sample.labels:
            centers: PackedTensor = sample.labels.trajectories.center

            n, t, d = centers.data.shape
            points = centers.data.view(n * t, d)
            points = rotate_3d_z(points, angle)
            centers.data[:, :, :] = points.view(n, t, d)

    def _rotate_transforms(self, sample: Sample, angle: torch.Tensor) -> None:
        transforms = sample.points.meta.transforms

        transforms.extrinsic = t3d.Sequential(
            t3d.Rotate(
                Quaternion(axis=[0, 0, 1], angle=-angle.item()),
                dtype=transforms.extrinsic.dtype,
            ),
            transforms.extrinsic,
        )

    def apply(self, sample: Sample) -> Sample:
        assert sample.batch_size == 1

        # generate a random rotation angle
        angle = torch.rand(1)
        angle = self.min + angle * (self.max - self.min)

        # apply rotation
        self._rotate_points(sample, angle)
        self._rotate_boxes(sample, angle)
        self._rotate_transforms(sample, angle)

        return sample


@transform.register(namespace="augment")
# pylint: disable-next=too-few-public-methods
class RandomFlipPoints(Transform):
    """
    Randomly x/y-flip point cloud points and bounding box annotations.
    """

    def __init__(self, probability=0.5):
        """
        Args:
            probability (float, [float, float], or torch.Tensor, optional): \
                Probability of flipping the point cloud and annotations.
        """
        self.probability = probability

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"augment.{self.__class__.__name__}",
            "probability": self.probability,
        }

    def _flip_points(self, sample: Sample, flip: torch.Tensor) -> None:
        points: PackedTensor = sample.points.data

        points.data[:, 0:2] *= flip[None, :]

    def _flip_boxes(self, sample: Sample, flip: torch.Tensor) -> None:
        if "labels" not in sample or "boxes" not in sample.labels:
            return

        boxes: PackedTensor = sample.labels.boxes

        # flip box centers and velocities
        boxes.data[:, 0:2] *= flip[None, :]
        boxes.data[:, 7:9] *= flip[None, :]

        # Note: We expect to be in nuscenes lidar frame, meaning y points
        # forward, x to the right, and z up. The rotation angle is relative
        # to the x-axis. Therefore:
        # - for x-flips, we get: rot <- 2*pi - rot
        # - for y-flips, we get: rot <- pi - rot
        # Further note that flip contains values (in {-1, 1}) in form
        # (y, x) and not (x, y) as this allows us to just multiply (see
        # above) to flip basic coordinates and vectors.

        flip_y, flip_x = flip < 0
        angle = boxes.data[:, 6]
        angle = 2 * torch.pi - angle if flip_x else angle
        angle = 1 * torch.pi - angle if flip_y else angle

        # normalize rotation to [-pi, pi) again
        boxes.data[:, 6] = normalize_angle(angle)

        # update trajectories, if present
        if "trajectories" in sample.labels:
            centers: PackedTensor = sample.labels.trajectories.center
            centers.data[:, :, 0:2] *= flip[None, None, :]

    def _flip_transforms(self, sample: Sample, flip: torch.Tensor) -> None:
        transforms = sample.points.meta.transforms
        transforms.extrinsic = t3d.Sequential(
            t3d.Scale([flip[0], flip[1], 1.0], dtype=transforms.extrinsic.dtype),
            transforms.extrinsic,
        )

    def apply(self, sample: Sample) -> Sample:
        assert sample.batch_size == 1

        # generate a random bools for each axis (x, y)
        flip = torch.as_tensor(self.probability).expand(2)
        flip = torch.bernoulli(flip)

        # convert bools to "-1" for flip (true) and and "1" for keep-as-is (false)
        # note that this is (y, x) and not (x, y) as we are mirroring at the axis
        flip = 1.0 - flip * 2.0

        # apply flips
        self._flip_points(sample, flip)
        self._flip_boxes(sample, flip)
        self._flip_transforms(sample, flip)

        return sample
