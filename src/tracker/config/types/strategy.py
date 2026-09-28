# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import timedelta

from lightning.pytorch.strategies import DDPStrategy, StrategyRegistry

StrategyRegistry.register(
    name="ddp_timeout_1h",
    strategy=DDPStrategy,
    description="DDP Strategy with timeout set to 1h",
    timeout=timedelta(hours=1),
)

StrategyRegistry.register(
    name="ddp_timeout_2h",
    strategy=DDPStrategy,
    description="DDP Strategy with timeout set to 1h",
    timeout=timedelta(hours=2),
)

StrategyRegistry.register(
    name="ddp_timeout_1h_find_unused_parameters_true",
    strategy=DDPStrategy,
    description="DDP Strategy with timeout set to 1h and find_unused_parameters set to True",
    timeout=timedelta(hours=1),
    find_unused_parameters=True,
)

StrategyRegistry.register(
    name="ddp_timeout_2h_find_unused_parameters_true",
    strategy=DDPStrategy,
    description="DDP Strategy with timeout set to 1h and find_unused_parameters set to True",
    timeout=timedelta(hours=2),
    find_unused_parameters=True,
)
