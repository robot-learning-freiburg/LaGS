# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from pyquaternion import Quaternion
from torch.utils.data._utils.collate import default_collate_fn_map

from . import utils


class Transform(ABC):
    def __init__(
        self, dtype: torch.dtype = torch.float32, device: torch.device = "cpu"
    ):
        self.dtype = dtype
        self.device = device

    @property
    @abstractmethod
    def matrix(self):
        pass

    def transform_points(self, points: torch.Tensor):
        n, _3 = points.shape

        # get points as homogeneous coordinates (N, 4)
        points_h = torch.empty(n, 4, dtype=points.dtype, device=points.device)
        points_h[:, :3] = points
        points_h[:, 3] = 1.0

        # build homogeneous transformation matrix (4, 4)
        matrix = self.matrix

        # transform points
        points_h = torch.mm(points_h, matrix.T)

        return points_h[:, :3]

    def __matmul__(self, points: torch.Tensor):
        return self.transform_points(points)

    def __call__(self, points: torch.Tensor):
        return self.transform_points(points)

    @property
    @abstractmethod
    def inv(self):
        pass

    def fused(self):
        return Fused(self.matrix, dtype=self.dtype, device=self.device)

    @abstractmethod
    def to(self, *args, **kwargs):
        pass

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    def cpu(self):
        return self.to(device=torch.device("cpu"))


class Identity(Transform):
    def __init__(
        self, dtype: torch.dtype = torch.float32, device: torch.device = "cpu"
    ):
        super().__init__(dtype=dtype, device=device)

    @property
    def matrix(self):
        return torch.eye(4, dtype=self.dtype, device=self.device)

    @property
    def inv(self):
        return Identity(dtype=self.dtype, device=self.device)

    def transform_points(self, points: torch.Tensor):
        return points.clone()

    def to(self, *args, **kwargs):
        to = utils.torch.parse_to(*args, **kwargs)

        return Identity(dtype=to.dtype, device=to.device)


class Translate(Transform):
    def __init__(
        self, vec, dtype: torch.dtype | None = None, device: torch.device = "cpu"
    ):
        self.vector = torch.as_tensor(vec, dtype=dtype, device=device)
        super().__init__(dtype=self.vector.dtype, device=device)

    @property
    def matrix(self):
        m = torch.eye(4, dtype=self.dtype, device=self.device)
        m[:3, 3] = self.vector

        return m

    @property
    def inv(self):
        return Translate(-self.vector, dtype=self.dtype, device=self.device)

    def to(self, *args, **kwargs):
        to = utils.torch.parse_to(*args, **kwargs)

        return Translate(
            self.vector.to(*args, **kwargs), dtype=to.dtype, device=to.device
        )


class Rotate(Transform):
    def __init__(
        self,
        quaternion,
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cpu",
    ):
        super().__init__(dtype=dtype, device=device)

        self.quaternion = Quaternion(quaternion)

    @property
    def matrix(self):
        m = self.quaternion.transformation_matrix
        m = torch.as_tensor(m, dtype=self.dtype, device=self.device)

        return m

    @property
    def inv(self):
        return Rotate(self.quaternion.inverse, dtype=self.dtype, device=self.device)

    def to(self, *args, **kwargs):
        to = utils.torch.parse_to(*args, **kwargs)

        q = self.quaternion if not to.copy else copy.deepcopy(self.quaternion)
        return Rotate(q, dtype=to.dtype, device=to.device)


class Scale(Transform):
    def __init__(
        self, scale, dtype: torch.dtype | None = None, device: torch.device = "cpu"
    ):
        scale = torch.as_tensor(scale, dtype=dtype, device=device)
        self.scale = scale.expand(3)

        super().__init__(dtype=self.scale.dtype, device=device)

    @property
    def matrix(self):
        m = torch.eye(4, dtype=self.dtype, device=self.device)
        m[0, 0] = self.scale[0]
        m[1, 1] = self.scale[1]
        m[2, 2] = self.scale[2]

        return m

    @property
    def inv(self):
        return Scale(-self.scale, dtype=self.dtype, device=self.device)

    def to(self, *args, **kwargs):
        to = utils.torch.parse_to(*args, **kwargs)

        return Scale(self.scale.to(*args, **kwargs), dtype=to.dtype, device=to.device)


class Sequential(Transform):
    def __init__(
        self,
        *transforms,
        dtype: torch.dtype | None = None,
        device: torch.device = "cpu",
    ):
        if dtype is None:
            dtype = transforms[0].dtype

        super().__init__(dtype=dtype, device=device)

        self.transforms = transforms

    @property
    def matrix(self):
        m = torch.eye(4, dtype=self.dtype, device=self.device)

        for transform in self.transforms:
            m = torch.mm(transform.matrix, m)

        return m

    @property
    def inv(self):
        transforms = list(reversed(self.transforms))
        transforms = [tx.inv for tx in transforms]

        return Sequential(*transforms, dtype=self.dtype, device=self.device)

    def to(self, *args, **kwargs):
        to = utils.torch.parse_to(*args, **kwargs)

        transforms = [tx.to(*args, **kwargs) for tx in self.transforms]
        return Sequential(*transforms, dtype=to.dtype, device=to.device)


class Fused(Transform):
    def __init__(self, matrix, dtype: torch.dtype = None, device: torch.device = "cpu"):
        self._matrix = torch.as_tensor(matrix, dtype=dtype, device=device)
        super().__init__(dtype=self._matrix.dtype, device=device)

    @property
    def matrix(self):
        return self._matrix

    @property
    def inv(self):
        return Fused(torch.inverse(self._matrix), dtype=self.dtype, device=self.device)

    def to(self, *args, **kwargs):
        to = utils.torch.parse_to(*args, **kwargs)

        return Fused(self._matrix.to(*args, **kwargs), dtype=to.dtype, device=to.device)


class RotateTranslate(Transform):
    def __init__(
        self,
        rotate: Rotate,
        translate: Translate,
        dtype: torch.dtype | None = None,
        device: torch.device = "cpu",
    ):
        if dtype is None:
            dtype = rotate.dtype

        super().__init__(dtype=dtype, device=device)

        self.rotate = rotate
        self.translate = translate

    @property
    def matrix(self):
        return torch.mm(self.translate.matrix, self.rotate.matrix)

    @property
    def inv(self):
        return TranslateRotate(
            translate=self.translate.inv,
            rotate=self.rotate.inv,
            dtype=self.dtype,
            device=self.device,
        )

    def to(self, *args, **kwargs):
        to = utils.torch.parse_to(*args, **kwargs)

        translate = self.translate.to(*args, **kwargs)
        rotate = self.rotate.to(*args, **kwargs)

        return RotateTranslate(
            rotate=rotate, translate=translate, device=to.device, dtype=to.dtype
        )


class TranslateRotate(Transform):
    def __init__(
        self,
        translate: Translate,
        rotate: Rotate,
        dtype: torch.dtype | None = None,
        device: torch.device = "cpu",
    ):
        if dtype is None:
            dtype = translate.dtype

        super().__init__(dtype=dtype, device=device)

        self.translate = translate
        self.rotate = rotate

    @property
    def matrix(self):
        return torch.mm(self.rotate.matrix, self.translate.matrix)

    @property
    def inv(self):
        return RotateTranslate(
            rotate=self.rotate.inv,
            translate=self.translate.inv,
            dtype=self.dtype,
            device=self.device,
        )

    def to(self, *args, **kwargs):
        to = utils.torch.parse_to(*args, **kwargs)

        translate = self.translate.to(*args, **kwargs)
        rotate = self.rotate.to(*args, **kwargs)

        return TranslateRotate(
            translate=translate, rotate=rotate, device=to.device, dtype=to.dtype
        )


class Intrinsic(Transform):
    def __init__(
        self, matrix, dtype: torch.dtype | None = None, device: torch.device = "cpu"
    ):
        matrix = torch.as_tensor(matrix, dtype=dtype, device=device)
        assert matrix.shape == (3, 3)

        super().__init__(dtype=matrix.dtype, device=device)

        self._matrix = torch.eye(4, dtype=dtype, device=device)
        self._matrix[:3, :3] = matrix

    @property
    def matrix(self):
        return self._matrix

    @property
    def inv(self):
        return Intrinsic(
            torch.inverse(self._matrix), dtype=self.dtype, device=self.device
        )

    def to(self, *args, **kwargs):
        to = utils.torch.parse_to(*args, **kwargs)

        return Intrinsic(self._matrix[:3, :3], dtype=to.dtype, device=to.device)


class ViewProjection:
    def __init__(self, transform: Transform):
        self.transform = transform

    def project(self, points: torch.Tensor):
        points = self.transform(points)

        points[:, :2] = points[:, :2] / points[:, 2][:, None]
        points[:, 2] = points[:, 2]

        return points

    def __matmul__(self, points: torch.Tensor):
        return self.project(points)

    def __call__(self, points: torch.Tensor):
        return self.project(points)

    @property
    def device(self):
        return self.transform.device

    @property
    def dtype(self):
        return self.transform.dtype

    def to(self, *args, **kwargs):
        return self.transform.to(*args, **kwargs)

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    @staticmethod
    def mask(
        points: torch.Tensor,
        shape: Tuple[int, ...],
        margin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        x, y, depth = points[:, 0], points[:, 1], points[:, 2]
        h, w = shape[-2:]
        mx, my, md = margin

        mask = depth > md
        mask = mask & (x >= mx) & (x < (w - mx))
        mask = mask & (y >= my) & (y < (h - my))

        return mask


# enable collate support for transforms and projections
def _collate_fn(batch, *, collate_fn_map=None):
    # pylint: disable=unused-argument
    return batch


default_collate_fn_map[Transform] = _collate_fn
default_collate_fn_map[ViewProjection] = _collate_fn
