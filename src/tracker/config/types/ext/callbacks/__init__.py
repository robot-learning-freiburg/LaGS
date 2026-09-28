# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from importlib.metadata import version

from packaging.version import Version

lightning_version = Version(version("lightning"))
if lightning_version < Version("2.6.0"):
    from .weight_averaging import EMAWeightAveraging, WeightAveraging
