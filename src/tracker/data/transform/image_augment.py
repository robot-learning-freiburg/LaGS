# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from typing import Any, Mapping, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as F

from ... import t3d
from ...utils.types import Sample
from .registry import registry as transform
from .transform import Transform


@transform.register(namespace="augment")
class RandomFlipImages(Transform):
    """
    Randomly flip images along x-/y-axes.

    This transforms all multi-frame images in the exact same way.
    """

    def __init__(self, probability: Tuple[float, float] | float = (0.0, 0.5)):
        self.probability = probability

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"augment.{self.__class__.__name__}",
            "probability": self.probability,
        }

    def apply(self, sample: Sample) -> Sample:
        assert sample.batch_size == 1

        images = sample.images

        # draw random bools for flipping
        flip = torch.as_tensor(self.probability).expand(2)
        flip = torch.bernoulli(flip)

        # apply flips to each image in the multiview sequence
        flip_y, flip_x = flip

        # flip the image
        dims = [(-2, -1)[d] for d in range(2) if flip[d]]
        images.data = images.data.flip(dims=dims)

        # flip the intrinsics
        m = t3d.Identity(dtype=images.meta.transforms.intrinsic.dtype).fused()
        m.matrix[0, 0] = 1.0 - 2.0 * flip_x
        m.matrix[1, 1] = 1.0 - 2.0 * flip_y
        m.matrix[0, 2] = flip_x * images.data.shape[-1]
        m.matrix[1, 2] = flip_y * images.data.shape[-2]

        def update_intrinsic(intrinsic):
            return t3d.Sequential(intrinsic, m)

        images.meta.transforms.intrinsic.map_(update_intrinsic)

        return sample


@transform.register(namespace="augment")
class RandomRotateImages(Transform):
    """
    Randomly rotate images.

    Each camera view is rotated independently with its own random angle.
    """

    def __init__(
        self,
        angle: Tuple[float, float],
        interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
        expand: bool = False,
        center: Tuple[float, float] | None = None,
        fill: float | Tuple[float, ...] = 0.0,
        probability: float = 1.0,
    ):
        """
        Args:
            angle: Range of rotation angles in degrees (min, max).
            interpolation: Interpolation mode for rotation.
            expand: If True, expand output to fit the rotated image.
            center: Optional center of rotation (x, y) as fraction of image size.
                   If None, uses image center.
            fill: Fill value for areas outside the rotated image.
            probability: Probability of applying rotation.
        """
        self.angle = angle
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill if isinstance(fill, (list, tuple)) else [fill]
        self.probability = probability

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"augment.{self.__class__.__name__}",
            "angle": self.angle,
            "interpolation": self.interpolation,
            "expand": self.expand,
            "center": self.center,
            "fill": self.fill,
            "probability": self.probability,
        }

    def apply(self, sample: Sample) -> Sample:
        # pylint: disable=too-many-locals
        if np.random.rand() > self.probability:
            return sample

        images = sample.images

        # Get image dimensions
        *img_b, c, img_h, img_w = images.data.shape

        # Flatten batch dimensions to process each view independently
        num_views = torch.tensor(img_b).prod().item()
        images_flat = images.data.view(num_views, c, img_h, img_w)

        # Sample random angles for each view independently
        angles = torch.empty(num_views).uniform_(self.angle[0], self.angle[1])

        # Rotate each view with its own angle
        rotated_images = []
        for i in range(num_views):
            rotated = F.rotate(
                images_flat[i : i + 1],
                angle=angles[i].item(),
                interpolation=self.interpolation,
                expand=self.expand,
                center=self.center,
                fill=self.fill,
            )
            rotated_images.append(rotated)

        images.data = torch.cat(rotated_images, dim=0).view(*img_b, c, img_h, img_w)

        # Update intrinsics for each view
        # For rotation around image center, we need to apply:
        # 1. Translate to origin
        # 2. Rotate
        # 3. Translate back
        cx = img_w / 2.0 if self.center is None else self.center[0] * img_w
        cy = img_h / 2.0 if self.center is None else self.center[1] * img_h

        def update_intrinsic_with_rotation(intrinsic, angle_deg):
            angle_rad = math.radians(angle_deg)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # Build rotation matrix around the specified center
            m = t3d.Identity(dtype=intrinsic.dtype).fused()
            m.matrix[0, 0] = cos_a
            m.matrix[0, 1] = sin_a
            m.matrix[1, 0] = -sin_a
            m.matrix[1, 1] = cos_a
            m.matrix[0, 2] = cx - cos_a * cx - sin_a * cy
            m.matrix[1, 2] = cy + sin_a * cx - cos_a * cy

            return t3d.Sequential(intrinsic, m)

        # Apply rotation to each view's intrinsics
        angle_idx = 0

        def update_intrinsic(intrinsic):
            nonlocal angle_idx
            result = update_intrinsic_with_rotation(intrinsic, angles[angle_idx].item())
            angle_idx += 1
            return result

        images.meta.transforms.intrinsic.map_(update_intrinsic)

        return sample


@transform.register(namespace="augment")
class RandomScaleAndCropImages(Transform):
    def __init__(
        self,
        size: Tuple[int, int],
        scale: Tuple[float, float],
        h_offset: Tuple[float, float],
        interpolation: F.InterpolationMode = F.InterpolationMode.BILINEAR,
        antialias: bool = True,
    ):
        self.size = size
        self.scale = scale
        self.h_offset = h_offset
        self.interpolation = interpolation
        self.antialias = antialias

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"augment.{self.__class__.__name__}",
            "size": self.size,
            "scale": self.scale,
            "h_offset": self.h_offset,
            "interpolation": self.interpolation,
            "antialias": self.antialias,
        }

    def _sample_params(self, img):
        img_h, img_w = img.shape[-2:]
        crop_h, crop_w = self.size

        # compute the crop size from a randomly sampled image scaling factor
        scale = torch.empty(1).uniform_(self.scale[0], self.scale[1]).item()
        h, w = round(crop_h / scale), round(crop_w / scale)

        # compute random crop offsets
        # torch.randint(0, max(img_h - h, 0) + 1, size=(1,)).item()
        h_range = max(img_h - h, 0)
        h_min = round(self.h_offset[0] * h_range)
        h_max = round(self.h_offset[1] * h_range)

        i = torch.randint(h_min, h_max + 1, size=(1,)).item()
        j = torch.randint(0, max(img_w - w, 0) + 1, size=(1,)).item()

        return i, j, h, w

    def apply(self, sample):
        # pylint: disable=too-many-locals
        images = sample.images

        *img_b, c, img_h, img_w = images.data.shape
        out_h, out_w = self.size

        # choose random crop parameters
        crop_t, crop_l, crop_h, crop_w = self._sample_params(images.data)

        # apply crop to images
        images.data = F.resized_crop(
            images.data.view(torch.tensor(img_b).prod(), c, img_h, img_w),
            top=crop_t,
            left=crop_l,
            height=crop_h,
            width=crop_w,
            size=self.size,
            interpolation=self.interpolation,
            antialias=self.antialias,
        ).view(*img_b, c, out_h, out_w)

        # update the intrinsics
        sy, sx = out_h / crop_h, out_w / crop_w

        m = t3d.Identity(dtype=images.meta.transforms.intrinsic.dtype).fused()
        m.matrix[0, 0] = sx
        m.matrix[1, 1] = sy
        m.matrix[0, 2] = -crop_l * sx
        m.matrix[1, 2] = -crop_t * sy

        def update_intrinsic(intrinsic):
            return t3d.Sequential(intrinsic, m)

        images.meta.transforms.intrinsic.map_(update_intrinsic)

        return sample


def random_grid_mask(
    x: torch.Tensor,
    mask_h: bool = True,
    mask_w: bool = True,
    distance: Tuple[int, int] | None = None,
    ratio: float = 0.5,
    rotate: Tuple[float, float] | None = None,
    invert: bool = False,
    value: torch.Tensor = torch.tensor(0.0),
) -> tuple[torch.Tensor, torch.Tensor]:
    # pylint: disable=too-many-locals

    *_b, c, h, w = x.shape

    # random spacing between mask patches
    patch_dist = np.random.randint(*(distance or (2, h)))

    # compute size of a mask patch via ratio
    patch_size = min(max(int(patch_dist * ratio + 0.5), 1), patch_dist - 1)

    # random offsets/shifts
    offs_h = np.random.randint(-patch_dist // 2, patch_dist // 2)
    offs_w = np.random.randint(-patch_dist // 2, patch_dist // 2)

    # build mask
    if rotate is None:
        h_ext, w_ext = h, w
    else:
        # Note: To avoid unmasked side-bands caused by rotation, extend the
        # mask region.
        h_ext = w_ext = math.ceil(1.42 * max(h, w))

    mask = torch.ones((h_ext, w_ext), device=x.device, dtype=torch.bool)

    if mask_h:
        for i in range(h_ext // patch_dist + 2):
            start = offs_h + i * patch_dist
            end = start + patch_size

            mask[max(start, 0) : min(end, h_ext), :] = False

    if mask_w:
        for i in range(w_ext // patch_dist + 2):
            start = offs_w + i * patch_dist
            end = start + patch_size

            mask[:, max(start, 0) : min(end, w_ext)] = False

    # rotate mask
    if rotate is not None:
        angle = np.random.uniform(rotate[0], rotate[1])

        mask = mask.view(1, 1, h_ext, w_ext)
        mask = F.rotate(mask, angle, fill=(0,))
        mask = mask.squeeze()

        start_h, start_w = (h_ext - h) // 2, (w_ext - w) // 2
        mask = mask[start_h : start_h + h, start_w : start_w + w]

    # invert mask
    if invert:
        mask = mask.logical_not_()

    # apply mask
    value = value.expand(c)[:, None, None]
    x = (~mask).expand_as(x) * x + mask.expand_as(x) * value

    return x, ~mask


@transform.register(namespace="augment")
# pylint: disable-next=too-many-instance-attributes
class RandomGridMask(Transform):
    def __init__(
        self,
        mask_h: bool = True,
        mask_w: bool = True,
        distance: Tuple[int, int] | None = None,
        ratio: float = 0.5,
        rotate: Tuple[float, float] | None = None,
        invert: bool = False,
        value: float | Tuple[float, ...] | torch.Tensor = 0.0,
        probability: float = 1.0,
        mask_depth: bool = False,
    ):
        """
        GridMask Data Augmentation (https://arxiv.org/abs/2001.04086v3)

        Args:
            mask_h (bool):
                Apply mask along the y-axis (horizontal stripes).
            mask_w (bool):
                Apply mask along the x-axis (vertical stripes).
            distance ([int, int] or None):
                Range of patch distances. If None, choose dynamically based on
                image size (between 2 and image height).
            ratio (float):
                Ratio of unmasked width/height to total width/height.
            rotate ([float, float] or None):
                If not None, apply random rotation in the specified range.
            invert (bool):
                Whether to invert the mask before applying it.
            value (float or [float, ...] or Tensor[C]):
                Value to apply for masked areas.
            probability (float):
                Probability of applying grid masking.
            mask_depth (bool):
                Whether to apply grid masking to depth maps.
        """
        assert 0.0 <= ratio <= 1.0
        assert distance is None or (len(distance) == 2 and distance[0] >= 2)

        self.mask_h = mask_h
        self.mask_w = mask_w
        self.distance = distance
        self.ratio = ratio
        self.rotate = rotate
        self.invert = invert
        self.value = torch.as_tensor(value)
        self.probability = probability
        self.mask_depth = mask_depth

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"augment.{self.__class__.__name__}",
            "mask_h": self.mask_h,
            "mask_w": self.mask_w,
            "distance": self.distance,
            "ratio": self.ratio,
            "rotate": self.rotate,
            "invert": self.invert,
            "value": self.value.tolist(),
            "probability": self.probability,
        }

    def apply(self, sample):
        if np.random.rand() > self.probability:
            return sample

        # mask images
        sample.images.data, mask = random_grid_mask(
            sample.images.data,
            mask_h=self.mask_h,
            mask_w=self.mask_w,
            distance=self.distance,
            ratio=self.ratio,
            rotate=self.rotate,
            invert=self.invert,
            value=self.value,
        )

        # apply mask to depth maps
        if self.mask_depth and sample.depth is not None:
            mask = mask.expand_as(sample.depth.data)
            sample.depth.data = torch.where(mask, sample.depth.data, 0.0)
            sample.depth.mask = sample.depth.mask & mask

        return sample
