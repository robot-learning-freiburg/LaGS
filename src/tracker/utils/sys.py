# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from typing import Optional, Union


def read_file_field(path: Union[Path, str]) -> Optional[str]:
    path = Path(path)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as fd:
        return fd.read().strip()


def get_container_hostname() -> Optional[str]:
    return read_file_field("/etc/container-host")
