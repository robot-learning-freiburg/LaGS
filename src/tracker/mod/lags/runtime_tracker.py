# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Any

import torch

from .occupancy.gaussian.temporal import TemporalState


class RunTimeTracker:
    # pylint: disable=too-many-instance-attributes

    current_id: int
    current_seq: Any | None
    gaussian_instance_id: int
    timestamp: float | None
    time_delta: float | None
    track_instances: torch.Tensor | None
    ego_to_global: torch.Tensor | None
    temporal_state: TemporalState | None
    stereo_state: object | None  # depth.stereo.StereoState

    def __init__(
        self,
        output_threshold: float = 0.2,
        score_threshold: float = 0.4,
        record_threshold: float = 0.4,
        max_age_since_update: int = 1,
    ):
        self.threshold = score_threshold
        self.output_threshold = output_threshold
        self.record_threshold = record_threshold
        self.max_age_since_update = max_age_since_update

        self.reset()

    def reset(self) -> None:
        self.current_id = 0
        self.current_seq = None
        self.gaussian_instance_id = 0
        self.timestamp = None
        self.time_delta = None
        self.track_instances = None
        self.ego_to_global = None
        self.temporal_state = None
        self.stereo_state = None

    def update_active_tracks(self, track_instances, active_mask) -> None:
        active = active_mask.to(dtype=torch.bool)
        aging = track_instances.track_query_mask.to(dtype=torch.bool) & ~active

        disappear_time = track_instances.disappear_time
        disappear_time[active] = 0
        disappear_time[aging] += 1

        keep = active | (aging & (disappear_time < self.max_age_since_update))

        self.track_instances = track_instances[keep]
