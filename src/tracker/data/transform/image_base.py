# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import math
from typing import Any, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Fv

from ... import t3d
from ...utils.types import MetaDict, Sample, TensorArray
from .registry import registry as transform
from .transform import Transform


@transform.register
class NormalizeImages(Transform):
    """
    Normalize (multi-view) images.
    """

    def __init__(
        self,
        mean: List[float] | float = 0.5,
        std: List[float] | float = 2,
        dtype: torch.dtype | str = torch.float32,
    ):
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)

        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def apply(self, sample: Sample) -> Sample:
        images = sample.images

        # normalize image
        images.data = images.data.to(dtype=self.dtype)
        images.data = Fv.normalize(images.data, self.mean, self.std, inplace=False)

        # add values used for normalization to sample
        images.meta.norm.mean[:] = self.mean
        images.meta.norm.std[:] = self.std

        return sample


@transform.register
class PadImages(Transform):
    """
    Pad (multi-view) images.
    """

    def __init__(
        self,
        size: Tuple[int, int] | None = None,
        divisor: int | None = None,
        mode: str = "constant",
        value: float | None = None,
    ):
        self.size = size
        self.divisor = divisor
        self.mode = mode
        self.value = value

        assert (self.size is None) != (self.divisor is None)

        assert self.mode in ["constant", "reflect", "replicate", "circular"]
        assert self.size is None or len(self.size) == 2

        if self.size is not None:
            self._compute_pad = self._pad_by_size
        else:
            self._compute_pad = self._pad_by_divisor

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "size": self.size,
            "divisor": self.divisor,
            "mode": self.mode,
            "value": self.value,
        }

    def _pad_by_size(self, shape: torch.Size) -> Tuple[int, int, int, int]:
        source_h, source_w = shape[-2:]
        target_h, target_w = self.size

        # Note: the order is reversed
        return (0, target_w - source_w, 0, target_h - source_h)

    def _pad_by_divisor(self, shape: torch.Size) -> Tuple[int, int, int, int]:
        source_h, source_w = shape[-2:]
        target_h = math.ceil(shape[-2] / self.divisor) * self.divisor
        target_w = math.ceil(shape[-1] / self.divisor) * self.divisor

        # Note: the order is reversed
        return (0, target_w - source_w, 0, target_h - source_h)

    def apply(self, sample: Sample) -> Sample:
        images = sample.images

        # pad images
        pad = self._compute_pad(images.data.shape)
        images.data = F.pad(images.data, pad, mode=self.mode, value=self.value)

        # add padding values to metadata
        images.meta.padding = torch.tensor(pad)
        images.meta.shape = torch.tensor(images.data.shape)

        # update the intrinsic matrices
        def update_intrinsic(intrinsic):
            intrinsic = intrinsic.fused()

            intrinsic.matrix[0, 2] += pad[0]
            intrinsic.matrix[1, 2] += pad[2]

            return intrinsic

        images.meta.transforms.intrinsic.map_(update_intrinsic)

        return sample


@transform.register
class ScaleAndCropImages(Transform):
    def __init__(
        self,
        size: Tuple[int, int],
        h_offset: float = 0.5,
        interpolation: Fv.InterpolationMode = Fv.InterpolationMode.BILINEAR,
        antialias: bool = True,
    ):
        self.size = size
        self.h_offset = h_offset
        self.interpolation = interpolation
        self.antialias = antialias

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
            "size": self.size,
            "h_offset": self.h_offset,
            "interpolation": self.interpolation,
            "antialias": self.antialias,
        }

    def apply(self, sample):
        # pylint: disable=too-many-locals
        images = sample.images

        out_h, out_w = self.size
        *img_b, c, img_h, img_w = images.data.shape

        scale = min(img_h / out_h, img_w / out_w)
        crop_h, crop_w = round(out_h * scale), round(out_w * scale)

        offs_h = round(max(img_h - crop_h, 0) * self.h_offset)
        offs_w = max(img_w - crop_w, 0) // 2

        # apply crop to images
        images.data = Fv.resized_crop(
            images.data.view(torch.tensor(img_b).prod(), c, img_h, img_w),
            top=offs_h,
            left=offs_w,
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
        m.matrix[0, 2] = -offs_w * sx
        m.matrix[1, 2] = -offs_h * sy

        def update_intrinsic(intrinsic):
            return t3d.Sequential(intrinsic, m)

        images.meta.transforms.intrinsic.map_(update_intrinsic)
        images.meta.shape = torch.tensor(images.data.shape)

        return sample


@np.vectorize(otypes="O", excluded={"tx_lidar_to_global", "dtype"})
def _compute_lidar_to_image_tx(pose, extrinsic, intrinsic, tx_lidar_to_global, dtype):
    # transformation from global to camera frome at time of camera capture
    tx_global_to_camera = t3d.Sequential(
        # global frame to ego vehicle (at image timestamp)
        pose.inv,
        # ego vehicle to camera frame
        extrinsic.inv,
    )

    # transformation from camera frame to image plane
    tx_camera_to_image = intrinsic

    # full transformation from lidar to image plane (without normalization)
    tx_lidar_to_image = t3d.Sequential(
        tx_lidar_to_global,
        tx_global_to_camera,
        tx_camera_to_image,
    )

    # fuse and store transformation
    lidar_to_image = tx_lidar_to_image.fused().to(dtype=dtype)
    lidar_to_image = t3d.ViewProjection(lidar_to_image)

    return lidar_to_image


@np.vectorize(otypes="O", excluded={"tx_lidar_to_global"})
def _compute_lidar_to_image_tx_fused(pose, extrinsic, intrinsic, tx_lidar_to_global):
    # transformation from global to camera frome at time of camera capture
    tx_global_to_camera = t3d.Sequential(
        # global frame to ego vehicle (at image timestamp)
        pose.inv,
        # ego vehicle to camera frame
        extrinsic.inv,
    )

    # transformation from camera frame to image plane
    tx_camera_to_image = intrinsic

    # full transformation from lidar to image plane (without normalization)
    tx_lidar_to_image = t3d.Sequential(
        tx_lidar_to_global,
        tx_global_to_camera,
        tx_camera_to_image,
    )

    # fuse and store transformation
    return tx_lidar_to_image.fused().matrix.numpy()


@transform.register
class PrepareLidarToCameraTx(Transform):
    def __init__(self, dtype: torch.dtype | str = torch.float32):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)

        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def _compute_tx(self, points_tx, images_tx):
        # Note: Camera and lidar data each have their respective ego vehicle
        # pose (one for each camera sensor and one for the lidar) because they
        # may not be perfectly in sync. To transform between sensor frames, we
        # therefore first have to transform the data from the source sensor
        # frame to the global frame via extrinsics and ego vehicle pose at time
        # of record, and then transform it to the target sensor frame using the
        # ego pose and extrinsics associated with the sample from that.

        tx_lidar_to_global = t3d.Sequential(
            # lidar sensor to ego vehicle frame (at lidar timestamp)
            points_tx.extrinsic,
            # ego vehicle to global frame
            points_tx.pose,
        )

        lidar_to_image = _compute_lidar_to_image_tx(
            pose=images_tx.pose.data,
            extrinsic=images_tx.extrinsic.data,
            intrinsic=images_tx.intrinsic.data,
            tx_lidar_to_global=tx_lidar_to_global,
            dtype=self.dtype,
        )
        images_tx.lidar_to_image = TensorArray(lidar_to_image)

    def apply(self, sample):
        self._compute_tx(
            points_tx=sample.points.meta.transforms,
            images_tx=sample.images.meta.transforms,
        )

        return sample


@transform.register
class PrepareLidarToCameraTxMultiFrame(PrepareLidarToCameraTx):
    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def apply(self, sample):
        # this variant only works with multi-frame point-cloud data
        assert isinstance(sample.points, List | Tuple)

        # case: image data stored as list
        if isinstance(sample.images, List | Tuple):
            # dimension must match
            assert len(sample.points) == len(sample.images)

            # compute transforms
            for p, i in zip(sample.points, sample.images):
                self._compute_tx(p.meta.transforms, i.meta.transforms)

        # case: packed image data
        else:
            num_frames = len(sample.points)
            images_tx = sample.images.meta.transforms

            # dimension must match
            assert num_frames == images_tx.pose.shape[0]
            assert num_frames == images_tx.extrinsic.shape[0]
            assert num_frames == images_tx.intrinsic.shape[0]

            # compute transforms
            lidar_to_image = []
            for i, p in enumerate(sample.points):
                # get per-frame transforms
                img_tx = {
                    "pose": images_tx.pose[i],
                    "extrinsic": images_tx.extrinsic[i],
                    "intrinsic": images_tx.intrinsic[i],
                }
                img_tx = MetaDict(img_tx)

                # compute transform
                self._compute_tx(p.meta.transforms, img_tx)
                lidar_to_image.append(img_tx.lidar_to_image.data)

            # stack/combine
            images_tx.lidar_to_image = TensorArray(np.stack(lidar_to_image, axis=0))

        return sample


@np.vectorize(otypes="O", excluded={"tx_lidar_to_global"})
def _compute_image_to_lidar_tx_fused(pose, extrinsic, intrinsic, tx_lidar_to_global):
    # transformation from global to camera frome at time of camera capture
    tx_global_to_camera = t3d.Sequential(
        # global frame to ego vehicle (at image timestamp)
        pose.inv,
        # ego vehicle to camera frame
        extrinsic.inv,
    )

    # transformation from camera frame to image plane
    tx_camera_to_image = intrinsic

    # full transformation from lidar to image plane (without normalization)
    tx_lidar_to_image = t3d.Sequential(
        tx_lidar_to_global,
        tx_global_to_camera,
        tx_camera_to_image,
    )

    return tx_lidar_to_image.fused().inv.matrix.numpy()


@transform.register
class PrepareCameraToLidarTxFused(Transform):
    def __init__(self, dtype: torch.dtype | str = torch.float32):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)

        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def _compute_tx(self, points_tx, images_tx):
        tx_lidar_to_global = t3d.Sequential(
            # lidar sensor to ego vehicle frame (at lidar timestamp)
            points_tx.extrinsic,
            # ego vehicle to global frame
            points_tx.pose,
        )

        image_to_lidar = _compute_image_to_lidar_tx_fused(
            pose=images_tx.pose.data,
            extrinsic=images_tx.extrinsic.data,
            intrinsic=images_tx.intrinsic.data,
            tx_lidar_to_global=tx_lidar_to_global,
        )

        image_to_lidar = np.array(image_to_lidar.tolist())
        image_to_lidar = torch.from_numpy(image_to_lidar)
        image_to_lidar = image_to_lidar.to(dtype=self.dtype)

        images_tx.image_to_lidar = image_to_lidar

    def apply(self, sample):
        self._compute_tx(
            points_tx=sample.points.meta.transforms,
            images_tx=sample.images.meta.transforms,
        )

        return sample


@transform.register
class PrepareCameraToLidarTxMultiFrameFused(PrepareCameraToLidarTxFused):
    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def apply(self, sample):
        # this variant only works with multi-frame point-cloud data
        assert isinstance(sample.points, List | Tuple)

        # case: image data stored as list
        if isinstance(sample.images, List | Tuple):
            # dimension must match
            assert len(sample.points) == len(sample.images)

            # compute transforms
            for p, i in zip(sample.points, sample.images):
                self._compute_tx(p.meta.transforms, i.meta.transforms)

        # case: packed image data
        else:
            num_frames = len(sample.points)
            images_tx = sample.images.meta.transforms

            # dimension must match
            assert num_frames == images_tx.pose.shape[0]
            assert num_frames == images_tx.extrinsic.shape[0]
            assert num_frames == images_tx.intrinsic.shape[0]

            # compute transforms
            image_to_lidar = []
            for i, p in enumerate(sample.points):
                # get per-frame transforms
                img_tx = {
                    "pose": images_tx.pose[i],
                    "extrinsic": images_tx.extrinsic[i],
                    "intrinsic": images_tx.intrinsic[i],
                }
                img_tx = MetaDict(img_tx)

                # compute transform
                self._compute_tx(p.meta.transforms, img_tx)
                image_to_lidar.append(img_tx.image_to_lidar)

            # stack/combine
            images_tx.image_to_lidar = torch.stack(image_to_lidar, dim=0)

        return sample


@transform.register
class PrepareLidarEgoToCameraTx(Transform):
    def __init__(self, dtype: torch.dtype | str = torch.float32):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)

        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def _compute_tx(self, points_tx, images_tx):
        # Note: Camera and lidar data each have their respective ego vehicle
        # pose (one for each camera sensor and one for the lidar) because they
        # may not be perfectly in sync. To transform between sensor frames, we
        # therefore first have to transform the data from the source sensor
        # frame to the global frame via extrinsics and ego vehicle pose at time
        # of record, and then transform it to the target sensor frame using the
        # ego pose and extrinsics associated with the sample from that.

        # ego vehicle to global frame at lidar timestamp
        tx_ego_to_global = points_tx.pose

        ego_to_image = _compute_lidar_to_image_tx(
            pose=images_tx.pose.data,
            extrinsic=images_tx.extrinsic.data,
            intrinsic=images_tx.intrinsic.data,
            tx_lidar_to_global=tx_ego_to_global,
            dtype=self.dtype,
        )
        images_tx.ego_to_image = TensorArray(ego_to_image)

    def apply(self, sample):
        self._compute_tx(
            points_tx=sample.points.meta.transforms,
            images_tx=sample.images.meta.transforms,
        )

        return sample


@transform.register
class PrepareLidarEgoToCameraTxMultiFrame(PrepareLidarEgoToCameraTx):
    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def apply(self, sample):
        # this variant only works with multi-frame point-cloud data
        assert isinstance(sample.points, List | Tuple)

        # case: image data stored as list
        if isinstance(sample.images, List | Tuple):
            # dimension must match
            assert len(sample.points) == len(sample.images)

            # compute transforms
            for p, i in zip(sample.points, sample.images):
                self._compute_tx(p.meta.transforms, i.meta.transforms)

        # case: packed image data
        else:
            num_frames = len(sample.points)
            images_tx = sample.images.meta.transforms

            # dimension must match
            assert num_frames == images_tx.pose.shape[0]
            assert num_frames == images_tx.extrinsic.shape[0]
            assert num_frames == images_tx.intrinsic.shape[0]

            # compute transforms
            ego_to_image = []
            for i, p in enumerate(sample.points):
                # get per-frame transforms
                img_tx = {
                    "pose": images_tx.pose[i],
                    "extrinsic": images_tx.extrinsic[i],
                    "intrinsic": images_tx.intrinsic[i],
                }
                img_tx = MetaDict(img_tx)

                # compute transform
                self._compute_tx(p.meta.transforms, img_tx)
                ego_to_image.append(img_tx.ego_to_image.data)

            # stack/combine
            images_tx.ego_to_image = TensorArray(np.stack(ego_to_image, axis=0))

        return sample


@transform.register
class PrepareLidarEgoToCameraTxFused(Transform):
    def __init__(self, dtype: torch.dtype | str = torch.float32):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)

        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def _compute_tx(self, points_tx, images_tx):
        # Note: Camera and lidar data each have their respective ego vehicle
        # pose (one for each camera sensor and one for the lidar) because they
        # may not be perfectly in sync. To transform between sensor frames, we
        # therefore first have to transform the data from the source sensor
        # frame to the global frame via extrinsics and ego vehicle pose at time
        # of record, and then transform it to the target sensor frame using the
        # ego pose and extrinsics associated with the sample from that.

        # ego vehicle to global frame at lidar timestamp
        tx_ego_to_global = points_tx.pose

        ego_to_image = _compute_lidar_to_image_tx_fused(
            pose=images_tx.pose.data,
            extrinsic=images_tx.extrinsic.data,
            intrinsic=images_tx.intrinsic.data,
            tx_lidar_to_global=tx_ego_to_global,
        )

        ego_to_image = np.array(ego_to_image.tolist())
        ego_to_image = torch.from_numpy(ego_to_image)
        ego_to_image = ego_to_image.to(dtype=self.dtype)

        images_tx.ego_to_image = ego_to_image

    def apply(self, sample):
        self._compute_tx(
            points_tx=sample.points.meta.transforms,
            images_tx=sample.images.meta.transforms,
        )

        return sample


@transform.register
class PrepareLidarEgoToCameraTxMultiFrameFused(PrepareLidarEgoToCameraTxFused):
    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def apply(self, sample):
        # this variant only works with multi-frame point-cloud data
        assert isinstance(sample.points, List | Tuple)

        # case: image data stored as list
        if isinstance(sample.images, List | Tuple):
            # dimension must match
            assert len(sample.points) == len(sample.images)

            # compute transforms
            for p, i in zip(sample.points, sample.images):
                self._compute_tx(p.meta.transforms, i.meta.transforms)

        # case: packed image data
        else:
            num_frames = len(sample.points)
            images_tx = sample.images.meta.transforms

            # dimension must match
            assert num_frames == images_tx.pose.shape[0]
            assert num_frames == images_tx.extrinsic.shape[0]
            assert num_frames == images_tx.intrinsic.shape[0]

            # compute transforms
            ego_to_image = []
            for i, p in enumerate(sample.points):
                # get per-frame transforms
                img_tx = {
                    "pose": images_tx.pose[i],
                    "extrinsic": images_tx.extrinsic[i],
                    "intrinsic": images_tx.intrinsic[i],
                }
                img_tx = MetaDict(img_tx)

                # compute transform
                self._compute_tx(p.meta.transforms, img_tx)
                ego_to_image.append(img_tx.ego_to_image.data)

            # stack/combine
            images_tx.ego_to_image = torch.stack(ego_to_image, dim=0)

        return sample


@transform.register
class PrepareCameraToLidarEgoTxFused(Transform):
    def __init__(self, dtype: torch.dtype | str = torch.float32):
        super().__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
            assert isinstance(dtype, torch.dtype)

        self.dtype = dtype

    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def _compute_tx(self, points_tx, images_tx):
        # ego vehicle (at lidar timestamp) to global frame
        tx_ego_to_global = points_tx.pose

        image_to_ego = _compute_image_to_lidar_tx_fused(
            pose=images_tx.pose.data,
            extrinsic=images_tx.extrinsic.data,
            intrinsic=images_tx.intrinsic.data,
            tx_lidar_to_global=tx_ego_to_global,
        )

        image_to_ego = np.array(image_to_ego.tolist())
        image_to_ego = torch.from_numpy(image_to_ego)
        image_to_ego = image_to_ego.to(dtype=self.dtype)

        images_tx.image_to_ego = image_to_ego

    def apply(self, sample):
        self._compute_tx(
            points_tx=sample.points.meta.transforms,
            images_tx=sample.images.meta.transforms,
        )

        return sample


@transform.register
class PrepareCameraToLidarEgoTxMultiFrameFused(PrepareCameraToLidarEgoTxFused):
    @property
    def hparams(self) -> Mapping[str, Any]:
        return {
            "type": f"{self.__class__.__name__}",
        }

    def apply(self, sample):
        # this variant only works with multi-frame point-cloud data
        assert isinstance(sample.points, List | Tuple)

        # case: image data stored as list
        if isinstance(sample.images, List | Tuple):
            # dimension must match
            assert len(sample.points) == len(sample.images)

            # compute transforms
            for p, i in zip(sample.points, sample.images):
                self._compute_tx(p.meta.transforms, i.meta.transforms)

        # case: packed image data
        else:
            num_frames = len(sample.points)
            images_tx = sample.images.meta.transforms

            # dimension must match
            assert num_frames == images_tx.pose.shape[0]
            assert num_frames == images_tx.extrinsic.shape[0]
            assert num_frames == images_tx.intrinsic.shape[0]

            # compute transforms
            image_to_ego = []
            for i, p in enumerate(sample.points):
                # get per-frame transforms
                img_tx = {
                    "pose": images_tx.pose[i],
                    "extrinsic": images_tx.extrinsic[i],
                    "intrinsic": images_tx.intrinsic[i],
                }
                img_tx = MetaDict(img_tx)

                # compute transform
                self._compute_tx(p.meta.transforms, img_tx)
                image_to_ego.append(img_tx.image_to_ego)

            # stack/combine
            images_tx.image_to_ego = torch.stack(image_to_ego, dim=0)

        return sample
