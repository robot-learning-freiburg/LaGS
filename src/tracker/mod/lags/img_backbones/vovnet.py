# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Implements the VoVNetV2 backbone (Lee et al., https://arxiv.org/abs/1911.06667);
# adapted from: https://github.com/WangYueFt/detr3d/blob/main/projects/mmdet3d_plugin/models/backbones/vovnet.py
# (licensed under MIT).

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.utils.checkpoint as ckpt
from torch import nn


@dataclass
class VoVNetSpec:
    stem_chs: Tuple[int, ...]
    stage_conv_chs: Tuple[int, ...]
    stage_out_chs: Tuple[int, ...]
    layers_per_block: int
    blocks_per_stage: Tuple[int, ...]
    use_se: bool
    depthwise: bool


SPECS = {
    "19-slim-dw-eSE": VoVNetSpec(
        stem_chs=(64, 64, 64),
        stage_conv_chs=(64, 80, 96, 112),
        stage_out_chs=(112, 256, 384, 512),
        layers_per_block=3,
        blocks_per_stage=(1, 1, 1, 1),
        use_se=True,
        depthwise=True,
    ),
    "19-dw-eSE": VoVNetSpec(
        stem_chs=(64, 64, 64),
        stage_conv_chs=(128, 160, 192, 224),
        stage_out_chs=(256, 512, 768, 1024),
        layers_per_block=3,
        blocks_per_stage=(1, 1, 1, 1),
        use_se=True,
        depthwise=True,
    ),
    "19-slim-eSE": VoVNetSpec(
        stem_chs=(64, 64, 128),
        stage_conv_chs=(64, 80, 96, 112),
        stage_out_chs=(112, 256, 384, 512),
        layers_per_block=3,
        blocks_per_stage=(1, 1, 1, 1),
        use_se=True,
        depthwise=False,
    ),
    "19-eSE": VoVNetSpec(
        stem_chs=(64, 64, 128),
        stage_conv_chs=(128, 160, 192, 224),
        stage_out_chs=(256, 512, 768, 1024),
        layers_per_block=3,
        blocks_per_stage=(1, 1, 1, 1),
        use_se=True,
        depthwise=False,
    ),
    "39-eSE": VoVNetSpec(
        stem_chs=(64, 64, 128),
        stage_conv_chs=(128, 160, 192, 224),
        stage_out_chs=(256, 512, 768, 1024),
        layers_per_block=5,
        blocks_per_stage=(1, 1, 2, 2),
        use_se=True,
        depthwise=False,
    ),
    "57-eSE": VoVNetSpec(
        stem_chs=(64, 64, 128),
        stage_conv_chs=(128, 160, 192, 224),
        stage_out_chs=(256, 512, 768, 1024),
        layers_per_block=5,
        blocks_per_stage=(1, 1, 4, 3),
        use_se=True,
        depthwise=False,
    ),
    "99-eSE": VoVNetSpec(
        stem_chs=(64, 64, 128),
        stage_conv_chs=(128, 160, 192, 224),
        stage_out_chs=(256, 512, 768, 1024),
        layers_per_block=5,
        blocks_per_stage=(1, 3, 9, 3),
        use_se=True,
        depthwise=False,
    ),
}


class DepthwiseConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()

        self.conv_dw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        self.conv_pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )

        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x


class EffectiveSEBlock(nn.Module):
    """
    Effective Squeeze-Excitation block.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x

        x_se = self.pool(x_se)
        x_se = self.fc(x_se)
        x_se = self.gate(x_se)

        return x * x_se


class OsaBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stage_channels: int,
        concat_channels: int,
        layers_per_block: int,
        use_se: bool,
        residual: bool,
        depthwise: bool,
    ) -> None:
        super().__init__()

        self.residual = residual
        self.depthwise = depthwise

        # input reduction
        if depthwise and in_channels != stage_channels:
            next_in_channels = stage_channels
            self.conv_reduce = ConvBlock(in_channels, stage_channels, kernel_size=1)
        else:
            next_in_channels = in_channels
            self.conv_reduce = None

        # middle layers
        layers = []
        for _ in range(layers_per_block):
            layer = DepthwiseConvBlock if depthwise else ConvBlock
            layer = layer(next_in_channels, stage_channels, kernel_size=3, padding=1)
            layers.append(layer)

            next_in_channels = stage_channels

        self.layers = nn.ModuleList(layers)

        # feature aggregation
        next_in_channels = in_channels + layers_per_block * stage_channels
        self.conv_concat = ConvBlock(next_in_channels, concat_channels, kernel_size=1)

        # squeeze-excitation
        if use_se:
            self.se_block = EffectiveSEBlock(concat_channels)
        else:
            self.se_block = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, x_id = [x], x

        # input reduction
        if self.conv_reduce is not None:
            x = self.conv_reduce(x)

        # middle layers
        for layer in self.layers:
            x = layer(x)
            out.append(x)

        # feature aggregation
        x = torch.cat(out, dim=1)
        x = self.conv_concat(x)

        # squeeze-excitation
        if self.se_block is not None:
            x = self.se_block(x)

        # skip connection
        if self.residual:
            x = x + x_id

        return x


class OsaStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stage_channels: int,
        out_channels: int,
        blocks_per_stage: int,
        layers_per_block: int,
        use_se: bool,
        depthwise: bool,
        downsample: bool,
        use_grad_checkpointing: bool,
    ) -> None:
        super().__init__()

        self.use_grad_checkpointing = use_grad_checkpointing

        if downsample:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        else:
            self.pool = None

        blocks = []
        for i in range(blocks_per_stage):
            block = OsaBlock(
                in_channels=in_channels,
                stage_channels=stage_channels,
                concat_channels=out_channels,
                layers_per_block=layers_per_block,
                use_se=use_se,
                residual=i > 0,
                depthwise=depthwise,
            )
            blocks.append(block)

            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = self.pool(x)

        if self.use_grad_checkpointing and not torch.jit.is_scripting():
            x = ckpt.checkpoint_sequential(
                functions=self.blocks,
                segments=len(self.blocks),
                input=x,
                use_reentrant=False,
            )
        else:
            x = self.blocks(x)

        return x


class VoVNet(nn.Module):
    def __init__(
        self,
        variant: str | VoVNetSpec,
        in_channels: int = 3,
        out_features: List[str] = None,
        freeze_stages: int | None = None,
        freeze_norms: bool = True,
        use_grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()

        self.freeze_stages = freeze_stages
        self.freeze_norms = freeze_norms
        self.out_features = out_features

        if isinstance(variant, str):
            s = SPECS[variant]
        else:
            s = variant

        # stem module
        conv_t = DepthwiseConvBlock if s.depthwise else ConvBlock
        self.stem = nn.Sequential(
            ConvBlock(in_channels, s.stem_chs[0], kernel_size=3, padding=1, stride=2),
            conv_t(s.stem_chs[0], s.stem_chs[1], kernel_size=3, padding=1, stride=1),
            conv_t(s.stem_chs[1], s.stem_chs[2], kernel_size=3, padding=1, stride=2),
        )

        # OSA stages
        stage_in_chs = [s.stem_chs[2]] + list(s.stage_out_chs)

        stages = []
        stage_names = []
        for i in range(4):
            stage = OsaStage(
                in_channels=stage_in_chs[i],
                stage_channels=s.stage_conv_chs[i],
                out_channels=s.stage_out_chs[i],
                blocks_per_stage=s.blocks_per_stage[i],
                layers_per_block=s.layers_per_block,
                use_se=s.use_se,
                depthwise=s.depthwise,
                downsample=i != 0,
                use_grad_checkpointing=use_grad_checkpointing,
            )

            stages.append(stage)
            stage_names.append(f"stage{i+2}")

        self.stages = nn.ModuleList(stages)
        self.stage_names = stage_names

        # initialize weights
        self._init_weights()
        self._freeze_stages()
        self._freeze_norms()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _freeze_stages(self) -> None:
        if self.freeze_stages is None:
            return

        # freeze stem
        self.stem.eval()
        for param in self.stem.parameters():
            param.requires_grad = False

        # freeze stages
        for i in range(self.freeze_stages):
            self.stages[i].eval()
            for param in self.stages[i].parameters():
                param.requires_grad = False

    def _freeze_norms(self) -> None:
        if not self.freeze_norms:
            return

        for m in self.modules():
            # pylint: disable-next=protected-access
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def train(self, mode: bool = True) -> None:
        super().train(mode=mode)

        self._freeze_stages()
        self._freeze_norms()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []

        # stem
        x = self.stem(x)
        if "stem" in self.out_features:
            out.append(x)

        # stages
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self.out_features:
                out.append(x)

        return out
