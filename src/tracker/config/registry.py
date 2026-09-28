# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

import functools
import inspect
from abc import ABC
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Type, TypeVar

from omegaconf import OmegaConf

from ..utils import types
from . import utils

_T = TypeVar("_T")


class FromConfigMixin(ABC):
    # pylint: disable=too-few-public-methods

    @classmethod
    def from_config(
        cls: Type[_T], conf: OmegaConf | Mapping[str, Any], *args, **kwargs
    ) -> _T:
        conf_kwargs = utils.get_kwargs(conf)

        return cls(*args, **kwargs, **conf_kwargs)


class TypeMixin(ABC):
    # pylint: disable=too-few-public-methods

    @classmethod
    def type(cls) -> str:
        return cls.__name__


class RegistryBaseType(TypeMixin, FromConfigMixin, ABC):
    # pylint: disable=too-few-public-methods
    pass


class Registry:
    def __init__(self, name):
        self.name = name
        self.modules: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.modules[key]

    def _register(
        self,
        module: Any,
        key: Optional[str] = None,
        namespace: Optional[str] = None,
        force: bool = False,
    ) -> Any:
        if module is None:
            raise ValueError(f"attempting to register module '{key}' as None")

        if key is None:
            key = module.type() if issubclass(module, TypeMixin) else module.__name__

        if namespace is not None:
            key = f"{namespace}.{key}"

        if not force and key in self.modules:
            raise RuntimeError(f"attempting to register module '{key}' twice")

        self.modules[key] = module
        return module

    def register(
        self,
        module: Optional[Any] = None,
        key: Optional[str] = None,
        namespace: Optional[str] = None,
        force: bool = False,
    ) -> Callable | Any:
        if module is None:
            # if module has not been passed in, return a decorator
            return functools.partial(
                self._register, key=key, namespace=namespace, force=force
            )

        # otherwise just directly register the module
        return self._register(module, key=key, namespace=namespace, force=force)

    def register_from_module(self, module, base_cls, namespace=None):
        def pred(obj):
            return types.is_true_subclass(obj, base_cls)

        for _, v in inspect.getmembers(module, pred):
            self.register(v, namespace=namespace)

    def _from_config(
        self,
        conf: OmegaConf | Mapping[str, Any],
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        resolve: bool,
    ) -> Any:
        conf = utils.as_config(conf)

        cls = self.modules[conf.type]

        if inspect.isclass(cls) and issubclass(cls, FromConfigMixin):
            return cls.from_config(conf, *args, **kwargs)

        ty, conf_kwargs = utils.get_type_and_kwargs(conf)
        assert ty == conf.type

        if resolve:
            conf_kwargs = OmegaConf.to_container(conf_kwargs, resolve=True)

        return cls(*args, **kwargs, **conf_kwargs)

    def from_config(self, conf: OmegaConf | Mapping[str, Any], *args, **kwargs) -> Any:
        return self._from_config(conf, args, kwargs, resolve=False)

    def from_config_resolve(
        self, conf: OmegaConf | Mapping[str, Any], *args, **kwargs
    ) -> Any:
        return self._from_config(conf, args, kwargs, resolve=True)
