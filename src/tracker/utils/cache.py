# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import datetime
import hashlib
import json
import shutil
import struct
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Type

from .. import config
from .log import get_logger

log = get_logger(__name__)


def _hash_update_recursive(m, obj: Any):
    if obj is None:
        m.update(struct.pack("q", 0))

    elif isinstance(obj, int):
        m.update(struct.pack("q", obj))

    elif isinstance(obj, float):
        m.update(struct.pack("f", obj))

    elif isinstance(obj, str):
        m.update(obj.encode(encoding="utf-8", errors="ignore"))

    elif isinstance(obj, Mapping):
        for key in sorted(obj.keys()):
            _hash_update_recursive(m, key)
            _hash_update_recursive(m, obj[key])

    elif isinstance(obj, Sequence):
        for x in obj:
            _hash_update_recursive(m, x)
    else:
        raise RuntimeError(f"unhashable type: {type(obj)}")


def sha256_recursive(obj: Any):
    m = hashlib.sha256()

    _hash_update_recursive(m, obj)

    return m


class Artifact:
    def __init__(self, root: Path, qname: str, hparams: Mapping[str, Any]):
        self.qname = qname
        self.hparams = hparams
        self.hash = sha256_recursive(self.hparams).hexdigest()
        self.path = root / qname / self.hash
        self.timestamp = None

    @property
    def metadata_path(self):
        return self.path / "metadata.json"

    @property
    def storage_path(self):
        return self.path / "storage"

    def _load_metadata(self) -> Dict[str, Any]:
        with open(self.metadata_path, "r", encoding="utf-8") as fd:
            metadata = json.load(fd)

        return metadata

    def remove(self):
        if self.path.exists():
            shutil.rmtree(self.path)

    def prepare(self):
        """
        Prepare the cache structure on disk but do not initialize it yet.

        This function cleans up any remnants of the cache if it does not exist.

        The intended use-case for this function is to set up the cache
        structure without marking the cache as initialized. For example, this
        can be called before the cache is being filled, whereas calling
        initialize() may already advertise the cache as being present when in
        fact it has not been fully constructed yet.
        """
        if self.exists():
            raise RuntimeError("trying to initialize cache, but it already exists")

        # try to remove any remnants
        self.remove()

        self.timestamp = datetime.datetime.now()

        self.metadata_path.parent.mkdir(parents=True, exist_ok=False)
        self.storage_path.mkdir(parents=True, exist_ok=False)

    def initialize(self):
        if self.exists():
            raise RuntimeError("trying to initialize cache, but it already exists")

        timestamp = self.timestamp
        if timestamp is None:
            timestamp = datetime.datetime.now()

        metadata = {
            "name": self.qname,
            "hash": self.hash,
            "timestamp": timestamp.astimezone().isoformat(),
            "hparams": self.hparams,
        }

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        with open(self.metadata_path, "w", encoding="utf-8") as fd:
            json.dump(metadata, fd, indent=2)

    def exists(self) -> bool:
        return self.metadata_path.exists()

    def available(self, raise_on_mismatch=True) -> bool:
        if not self.exists():
            return False

        meta = self._load_metadata()
        matches = meta["hparams"] == self.hparams

        if not matches and raise_on_mismatch:
            log.error("cache mismatch at '%s'", self.path)
            log.error("  name: %s", self.qname)
            log.error("  hash: %s", self.hash)
            log.error("  hparams: %s", self.hparams)
            raise RuntimeError(f"Cache mismatch at '{self.path}'")

        return matches


def get(qname: str | Type, hparams: Mapping[str, Any]) -> Artifact:
    cache_root = Path(config.get().paths.cache)

    if not isinstance(qname, str):
        qname = f"{qname.__module__}.{qname.__name__}"

    return Artifact(cache_root, qname, hparams)
