# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parameters
~~~~~~~~~~
Transformations and grammar for addressing neural network parameters.
"""
from __future__ import annotations
from dataclasses import field
from functools import reduce
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Union,
)

import equinox as eqx

from ..engine.paramutil import PyTree
from .grammar import (
    Grammar,
    Grouping,
    GroupingPool,
    LeafInterpreter,
    Literalisation,
    TransformPool,
    TransformPrimitive,
)


def retrieve_address(
    model: PyTree,
    where: Union[str, Callable],
):
    if where is None:
        return (model,)
    elif callable(where):
        return where(model)
    return ParameterAddressGrammar().compile(where)(model)


def transform_address(
    model: PyTree,
    where: Union[str, Callable],
    replace_fn: Callable,
) -> PyTree:
    if where is None:
        return model
    elif callable(where):
        f = where
    else:
        f = ParameterAddressGrammar().compile(where)
    return eqx.tree_at(f, model, replace_fn=replace_fn)


def filter_address(
    model: PyTree,
    where: Union[str, Callable],
) -> PyTree:
    def _f(matches):
        def __f(x):
            for m in matches:
                if x is m:
                    return True
            return False

        return __f

    if where is None:
        matches = ()
    elif callable(where):
        matches = where(model)
    else:
        f = ParameterAddressGrammar().compile(where)
        matches = f(model)
    return eqx.filter(model, _f(matches=matches))


class ParameterAddressGrammar(Grammar):
    groupings: GroupingPool = GroupingPool(
        Grouping(open="(", close=")"),
    )
    transforms: TransformPool = field(
        default_factory=lambda: TransformPool(
            ConcatenateNode(),
            KeyNode(),
            IntegerKeyNode(),
        )
    )
    whitespace: bool = False
    shorthand: dict = field(default_factory=lambda: {})
    default_interpreter: Optional[LeafInterpreter] = field(
        default_factory=lambda: ParameterSelectInterpreter()
    )
    default_root_transform: Optional[TransformPrimitive] = field(
        default_factory=lambda: ParameterAddressRootNode()
    )


class ParameterSelectInterpreter(LeafInterpreter):
    def __call__(self, leaf: Any) -> Callable:
        def retrieve_parameter(model: PyTree) -> PyTree:
            if leaf is None:
                return (model,)
            try:
                return (model.__getattribute__(leaf),)
            except AttributeError:
                try:
                    return (model.__getitem__(leaf),)
                except (AttributeError, KeyError, TypeError) as e:
                    raise AttributeError(
                        f"Could not retrieve parameter {leaf} from model {model}."
                    )

        def retrieve_parameters(arg: Any) -> PyTree:
            if isinstance(arg, tuple):
                return reduce(
                    lambda x, y: x + y,
                    tuple(retrieve_parameter(v) for v in arg),
                )
            return retrieve_parameter(arg)

        return retrieve_parameters


class ParameterAddressRootNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: float = float("inf")
    associative: bool = False
    commutative: bool = False
    literals: Sequence[Literalisation] = ()

    def __call__(self, *pparams, **params) -> Callable:
        f = pparams[0]

        def compiled(arg: Any) -> PyTree:
            return f((arg,))

        return compiled


# ----------------------------- Concatenation ----------------------------- #


class ConcatenateInfixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "infix"
    regex: str = r"\;"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class ConcatenateNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = float("inf")
    priority: int = 4
    associative: bool = True
    commutative: bool = True
    literals: Sequence[Literalisation] = (ConcatenateInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:
        def concatenate(arg: Any) -> PyTree:
            out = tuple(f(arg) for f in pparams)
            return reduce(lambda x, y: x + y, out)

        return concatenate


# --------------------------------- Keys ---------------------------------- #


class StringKeyInfixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "infix"
    regex: str = r"\$"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class AttributeInfixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "infix"
    regex: str = r"\."

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class KeyNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 2
    priority: int = 3
    associative: bool = False
    commutative: bool = False
    literals: Sequence[Literalisation] = (
        AttributeInfixLiteralisation(),
        StringKeyInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        if len(pparams) == 2:
            f_acc, f_attr = pparams
        else:
            f_acc, f_attr = lambda x: x, pparams[0]

        def getitem(arg: Any) -> PyTree:
            acc = f_acc(arg)
            out = tuple(f_attr(a) for a in acc)
            return reduce(lambda x, y: x + y, out)

        return getitem


# ------------------------------ Integer Key ------------------------------ #


class IntegerKeySuffixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "suffix"
    regex: str = r"\#(?P<index>[0-9]+)"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["index"] = (int(params["index"]),)
        return params


class IntegerKeyMultiSuffixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "suffix"
    regex: str = r"\#(?P<index>[0-9]+[\,[0-9]+]+)"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["index"] = tuple(int(z) for z in params["index"].split(","))
        return params


class IntegerKeyMultiRangeSuffixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "suffix"
    regex: str = r"\#(?P<index>[0-9]+[\:0-9]*[\,\:0-9]+)"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        keys = params["index"].split(",")
        index = []
        for key in keys:
            if ":" in key:
                lim = [int(z) if z != "" else None for z in key.split(":")]
                if isinstance(lim[1], int):
                    lim[1] += 1
                index += [
                    slice(lim[0], lim[1]),
                ]
            else:
                index += [int(key)]
        params["index"] = tuple(index)
        return params


class IntegerKeyNode(TransformPrimitive):
    min_arity: int = 0  # The index itself is parsed as a parameter
    # rather than an argument.
    max_arity: int = 1
    priority: int = 3
    associative: bool = False
    commutative: bool = False
    literals: Sequence[Literalisation] = (
        IntegerKeyMultiRangeSuffixLiteralisation(),
        IntegerKeyMultiSuffixLiteralisation(),
        IntegerKeySuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        if pparams:
            f = pparams[0]
        else:
            f = lambda x: x

        def getitem_impl(acc):
            out = ()
            for index in params["index"]:
                if isinstance(index, slice):
                    start, stop, step = index.start, index.stop, index.step
                    if start is None:
                        start = 0
                    if stop is None:
                        stop = len(acc)
                    if step is None:
                        step = 1
                    index = list(range(start, stop, step))
                    out = out + tuple(acc[i] for i in index)
                else:
                    out = out + (acc[index],)
            return out

        def getitem(arg: Any) -> PyTree:
            acc = f(arg)
            out = tuple(getitem_impl(a) for a in acc)
            return reduce(lambda x, y: x + y, out)

        return getitem
