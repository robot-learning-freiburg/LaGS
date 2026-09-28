# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This source code is derived from:
# * BEVDepth (https://github.com/Megvii-BaseDetection/BEVDepth), Copyright (c) Megvii, licensed under MIT,
# * BEVDet (https://github.com/HuangJunJie2017/BEVDet), Copyright (c) Phigent Robotics, licensed under Apache-2.0.
# See the LICENSES/ directory for full license texts.

from typing import Callable

import torch
from torch import nn
from torch.utils import checkpoint as ckpt

from ....utils.types import MetaDict
from .aspp import Aspp


class Mlp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        activation: Callable[[], nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act1 = activation()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.drop2(x)

        return x


class SELayer(nn.Module):
    def __init__(
        self,
        channels: int,
        activation: Callable[[], nn.Module] = nn.ReLU,
        gate_layer: Callable[[], nn.Module] = nn.Sigmoid,
    ):
        super().__init__()

        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = activation()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)

        return x * self.gate(x_se)


class BasicBlock(nn.Module):
    # pylint: disable=too-many-instance-attributes
    """
    Basic ResNet block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.act2(x + identity)


class DepthNet(nn.Module):
    # pylint: disable=too-many-instance-attributes
    """
    DepthNet adapted from BevDepth/BevDet.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        context_channels: int,
        depth_bins: int,
        use_aspp: bool = True,
        aspp_hidden_channels: int | None = None,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.context_channels = context_channels
        self.use_checkpointing = use_checkpointing

        # input conv
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # camera parameter gating networks
        self.n_camera_params = 17  # intrinsics (fx,fy,cx,cy,skew) + extrinsics (3x4)
        self.bn = nn.BatchNorm1d(self.n_camera_params)

        self.depth_mlp = Mlp(self.n_camera_params, hidden_channels, hidden_channels)
        self.depth_se = SELayer(hidden_channels)

        self.context_mlp = Mlp(self.n_camera_params, hidden_channels, hidden_channels)
        self.context_se = SELayer(hidden_channels)

        # main feature network
        self.context_conv = nn.Conv2d(
            hidden_channels,
            context_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # main depth network
        depth_conv_list = [
            BasicBlock(hidden_channels, hidden_channels),
            BasicBlock(hidden_channels, hidden_channels),
            BasicBlock(hidden_channels, hidden_channels),
        ]

        if use_aspp:
            aspp = Aspp(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                hidden_channels=aspp_hidden_channels or hidden_channels,
            )
            depth_conv_list.append(aspp)

        self.depth_conv = nn.Sequential(
            *depth_conv_list,
            nn.Conv2d(hidden_channels, depth_bins, kernel_size=1, stride=1, padding=0),
        )

    def _collect_camera_params(
        self, transforms: MetaDict, dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        # for extrinsics: extract rotation and translation from base 4x4 matrix
        extrinsic = transforms.extrinsic
        extrinsic = extrinsic.map(lambda x: x.matrix).collect()  # [..., 4, 4]
        extrinsic = extrinsic[..., :3, :4]  # [..., 3, 4]
        extrinsic = extrinsic.flatten(start_dim=-2)  # [..., 12]

        # for intrinsics: extract [fx, fy, cx, cy, skew] from base 4x4 matrix
        intrinsic = transforms.intrinsic
        intrinsic = intrinsic.map(lambda x: x.matrix).collect()  # [..., 4, 4]
        intrinsic = intrinsic[..., (0, 1, 0, 1, 0), (0, 1, 2, 2, 1)]  # [..., 5]

        params = torch.cat((intrinsic, extrinsic), dim=-1)  # [..., 17]
        params = params.to(dtype=dtype)

        return params

    def forward(
        self, x: torch.Tensor, img_meta: MetaDict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        *b, c, h, w = x.shape

        # prepare camera parameters
        cam_params = self._collect_camera_params(img_meta.transforms, x.dtype)
        cam_params = cam_params.view(-1, cam_params.shape[-1])  # [prod(b), n_params]
        cam_params = self.bn(cam_params)

        # prepare image features
        x = x.view(-1, c, h, w)  # [prod(b), c, h, w]
        x = self.reduce_conv(x)

        # compute context features
        context_se = self.context_mlp(cam_params)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)

        # compute depth
        depth_se = self.depth_mlp(cam_params)[..., None, None]
        depth = self.depth_se(x, depth_se)

        if self.use_checkpointing:
            depth = ckpt.checkpoint(self.depth_conv, depth, use_reentrant=False)
        else:
            depth = self.depth_conv(depth)

        depth = depth.softmax(dim=-3)

        # reshape to original batch size
        depth = depth.view(*b, -1, h, w)
        context = context.view(*b, -1, h, w)

        return depth, context
