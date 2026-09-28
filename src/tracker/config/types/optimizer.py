# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import re
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from torch.optim import Optimizer

from ...utils.log import get_logger
from .. import utils
from ..registry import Registry

log = get_logger(__name__)

registry = Registry("optimizer")
registry.register_from_module(torch.optim, Optimizer)


def get(key: str) -> Any:
    return registry.get(key)


def build(conf: OmegaConf, params: Any) -> Optimizer:
    return registry.from_config_resolve(conf, params=params)


DEFAULT_GROUP_MATCH = "@default"
GROUP_MATCH_KEYWORDS = [DEFAULT_GROUP_MATCH]


def _normalize_groups_cfg(groups_cfg: OmegaConf) -> OmegaConf:
    groups_cfg = utils.copy(groups_cfg, readonly=False)

    for group in groups_cfg:
        if isinstance(group.match, str):
            group.match = [group.match]

    return groups_cfg


def _get_default_group_index(groups_cfg: OmegaConf) -> int | None:
    index = [i for i, g in enumerate(groups_cfg) if DEFAULT_GROUP_MATCH in g.match]

    if len(index) < 1:
        return None

    if len(index) > 1:
        raise ValueError("multiple default parameter groups specified")

    return index[0]


def _compile_group_matches(groups_cfg: OmegaConf) -> List[List[re.Pattern]]:
    return [
        [re.compile(m) for m in g.match if m not in GROUP_MATCH_KEYWORDS]
        for g in groups_cfg
    ]


def _find_group_index(
    param_name: str, group_matches: List[List[re.Pattern]], default: int | None
) -> int:
    indices = [
        i
        for i, matches in enumerate(group_matches)
        if any(m.match(param_name) for m in matches)
    ]

    if len(indices) == 0:
        if default is not None:
            return default

        raise ValueError(
            f"parameter '{param_name}' has not been assigned to any group, "
            + "consider adding a group with match: '@default'"
        )

    if len(indices) > 1:
        raise ValueError(f"parameter '{param_name}' matches multiple groups: {indices}")

    return indices[0]


def get_grouped_params(
    module: torch.nn.Module, groups_cfg: OmegaConf
) -> List[Dict[str, Any]]:
    groups_cfg = _normalize_groups_cfg(groups_cfg)

    # get index for the default group
    default_group_index = _get_default_group_index(groups_cfg)

    # compile list of matches for the groups
    matches = _compile_group_matches(groups_cfg)

    # build groups
    def args(group):
        group = utils.copy(group, readonly=False)
        del group["match"]

        return group

    params_map = {}
    params = [{"params": [], **args(group)} for group in groups_cfg]

    for name, param in module.named_parameters(recurse=True, remove_duplicate=False):
        # get the group index via matches
        group_index = _find_group_index(name, matches, default_group_index)

        # if we have already delt with this parameter...
        if param in params_map:
            # make sure that any duplicated parameters are in the same group
            if params_map[param] != group_index:
                raise ValueError(
                    f"parameter {name} matches multiple groups: {params_map[param]}, {group_index}"
                )

        # if this is a new parameter...
        else:
            # add the parameter
            params_map[param] = group_index
            params[group_index]["params"].append(param)

    log.info("using %s parameter groups for optimization", len(params))

    # check for empty groups
    for i, group in enumerate(params):
        if len(group["params"]) == 0 and i != default_group_index:
            log.warning("empty parameter group at index %s", i)

    return params
