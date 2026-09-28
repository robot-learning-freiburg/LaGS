# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Mapping, Tuple

import omegaconf
from omegaconf import OmegaConf


def as_config(conf: Mapping[str, Any] | OmegaConf) -> OmegaConf:
    if OmegaConf.is_config(conf):
        return conf

    return OmegaConf.create(conf)


def get_type_and_kwargs(conf: OmegaConf | Mapping[str, Any]) -> Tuple[str, OmegaConf]:
    conf = copy(conf, readonly=False)
    ty = conf.pop("type")

    return ty, conf


def get_kwargs(conf: OmegaConf | Mapping[str, Any]) -> Tuple[str, OmegaConf]:
    conf = copy(conf, readonly=False)

    if "type" in conf:
        del conf["type"]

    return conf


def copy(conf: OmegaConf, readonly: bool = True) -> OmegaConf:
    if conf is None:
        return None

    conf = deepcopy(as_config(conf))

    OmegaConf.set_struct(conf, readonly)
    OmegaConf.set_readonly(conf, readonly)

    return conf


def to_primitive(conf: OmegaConf, resolve: bool = True) -> Any:
    return OmegaConf.to_container(conf, resolve=resolve)


@contextmanager
def context_resolver(
    name: str,
    resolver: omegaconf.Resolver,
    *,
    use_cache: bool = False,
):
    OmegaConf.register_new_resolver(name, resolver, replace=False, use_cache=use_cache)

    try:
        yield
    finally:
        OmegaConf.clear_resolver(name)
