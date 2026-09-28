# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import os


def is_slurm_job() -> bool:
    return "SLURM_JOB_ID" in os.environ


def is_requeue() -> bool:
    return int(os.environ.get("SLURM_RESTART_COUNT", "0")) > 0


def get_job_id() -> int:
    return os.environ["SLURM_JOB_ID"]


def get_restart_count() -> int:
    return int(os.environ.get("SLURM_RESTART_COUNT", "0"))
