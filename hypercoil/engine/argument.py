# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Model argument
~~~~~~~~~~~~~~
Convenience mappings representing arguments to a model or loss function.
Also aliased to loss arguments.
"""
from __future__ import annotations
from collections import namedtuple
from collections.abc import Mapping
from typing import Any

import equinox as eqx


class ModelArgument(Mapping, eqx.Module):
    """
    Representation of a set of arguments to a model or loss function.

    Effectively this is currently little more than a prettified, immutable
    dict. Or a namedtuple that's a mapping rather than a sequence, with a
    dict-like interface.
    """

    # We'd like to use the read-only MappingProxyType, but it's not possible
    # for ``eqx.filter_value_and_grad`` to work with it. So, for now, we'll
    # just use a regular dict.
    _arg_dict: dict  # MappingProxyType

    def __init__(self, **params):
        _arg_dict = params
        # self._arg_dict = MappingProxyType(_arg_dict)
        self._arg_dict = _arg_dict

    def __getattr__(self, name: str) -> Any:
        return self._arg_dict[name]

    def __getitem__(self, k: str) -> Any:
        return self._arg_dict[k]

    def __len__(self) -> int:
        return len(self._arg_dict)

    def __iter__(self) -> iter:
        return iter(self._arg_dict)

    def __add__(self, other) -> "ModelArgument":
        """
        If there's a clash, the other argument wins.
        Also, if either argument subclasses but doesn't override __add__,
        the output is reduced to a base ModelArgument.
        """
        return ModelArgument(**{**self._arg_dict, **other._arg_dict})

    def __eq__(self, other) -> bool:
        left = all(v == other.get(k, None) for k, v in self.items())
        right = all(v == self.get(k, None) for k, v in other.items())
        return left and right

    @classmethod
    def add(cls, arg: "ModelArgument", **params) -> "ModelArgument":
        return arg + cls(**params)

    @classmethod
    def all_except(cls, arg: Mapping, *remove) -> "ModelArgument":
        arg = {k: v for k, v in arg.items() if k not in remove}
        return cls(**arg)

    @classmethod
    def replaced(cls, arg: Mapping, **replace) -> "ModelArgument":
        arg = {
            k: (replace[k] if replace.get(k) is not None else v)
            for k, v in arg.items()
        }
        return cls(**arg)

    @classmethod
    def swap(cls, arg: Mapping, *rm, **new) -> "ModelArgument":
        arg = {k: v for k, v in arg.items() if k not in rm}
        arg.update(**new)
        return cls(**arg)

    def __repr__(self) -> str:
        agmt = namedtuple(type(self).__name__, self._arg_dict.keys())
        return eqx.pretty_print.tree_pformat(agmt(**self._arg_dict))


class UnpackingModelArgument(ModelArgument):
    """
    ``ModelArgument`` variant that is automatically unpacked when it is the
    output of an ``apply`` call. This is only distinguished from
    ``ModelArgument`` within the loss scheme module.
    """

    pass
