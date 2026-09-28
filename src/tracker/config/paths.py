# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path

# top-level directories (defaults)
root = Path(__file__).parent.parent.parent.parent
config = root / "config"
slurm = root / ".slurm"
