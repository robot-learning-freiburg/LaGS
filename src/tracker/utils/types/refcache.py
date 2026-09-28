# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass
from typing import Any, Dict


class RefCache:
    def __init__(self):
        self.cache: Dict[str, RefEntry] = {}

    def __contains__(self, key):
        return key in self.cache

    def store(self, key, value):
        if key in self.cache:
            raise ValueError(f"key {key} already present in cache")

        self.cache[key] = RefEntry(value)

    def acquire_or_store(self, key, constructor, *args, **kwargs):
        if key in self.cache:
            return self.acquire(key)

        obj = constructor(*args, **kwargs)
        self.store(key, obj)

        return obj

    def acquire(self, key):
        entry = self.cache[key]
        entry.refcount += 1

        return entry.value

    def try_acquire(self, key):
        entry = self.cache.get(key, None)
        if entry is None:
            return None

        entry.refcount += 1

        return entry.value

    def release(self, key):
        entry = self.cache[key]
        entry.refcount -= 1

        if entry.refcount <= 0:
            del self.cache[key]

    def get(self, key, default=None):
        entry = self.cache.get(key)
        if entry is None:
            return default

        return entry.value

    def __getitem__(self, key):
        return self.cache[key].value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.cache})"


@dataclass
class RefEntry:
    value: Any
    refcount: int = 1
