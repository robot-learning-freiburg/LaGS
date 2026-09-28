# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import lightning as L

# Note: Generally torch.distributed should be preferred to get the rank and
# world size. However, in the early stages (e.g. setting up datasets), those
# are not initialized yet. In those cases, we can use the global variable
# below, which is derived from environment variables that PyTorch Lightning
# sets.
#
# pylint: disable-next=protected-access
rank = L.fabric.utilities.rank_zero._get_rank() or 0
