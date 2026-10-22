# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
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
    """
    Retrieve the parameters of a model at an address.

    Parameters
    ----------
    model
        Model.
    where
        Address string of :class:`ParameterAddressGrammar`, or a callable that
        takes ``model`` and returns the parameters. None addresses ``model``.

    Returns
    -------
    tuple
        Parameters at the address.
    """
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
    """
    Replace the parameters of a model at an address by a function of them.

    Parameters
    ----------
    model
        Model.
    where
        Address string of :class:`ParameterAddressGrammar`, or a callable that
        takes ``model`` and returns the parameters. None returns ``model``.
    replace_fn
        Function applied to each parameter at the address.

    Returns
    -------
    PyTree
        ``model`` with the parameters replaced, by :func:`equinox.tree_at`.
    """
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
    """
    Filter a model to the parameters at an address.

    Parameters
    ----------
    model
        Model.
    where
        Address string of :class:`ParameterAddressGrammar`, or a callable that
        takes ``model`` and returns the parameters. None keeps no parameter.

    Returns
    -------
    PyTree
        ``model`` with every leaf that is not at the address set to None, by
        :func:`equinox.filter`.
    """
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
    """
    Grammar of parameter address strings.

    An address names parameters of a model:

    - ``name`` selects an attribute, or else an item;
    - ``a.b`` and ``a$b`` select ``b`` of each parameter that ``a`` selects;
    - ``a#i``, ``a#i,j`` and ``a#i:j`` select the items at the indices, or in
      the inclusive range, of each parameter that ``a`` selects;
    - ``a;b`` concatenates the selections of ``a`` and ``b``;
    - parentheses group.
    """
    groupings: GroupingPool = GroupingPool(
        Grouping(open='(', close=')'),
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
    """
    Leaf interpreter that selects a named attribute or item of a model.

    The compiled function raises ``AttributeError`` if a model has neither an
    attribute nor an item of the name.
    """
    def __call__(self, leaf: Any) -> Callable:
        def retrieve_parameter(model: PyTree) -> PyTree:
            if leaf is None:
                return (model,)
            try:
                return (model.__getattribute__(leaf),)
            except AttributeError:
                try:
                    return (model.__getitem__(leaf),)
                except (AttributeError, KeyError, TypeError):
                    raise AttributeError(
                        f'Could not retrieve parameter {leaf} from model {model}.'
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
    """Root transform that applies a compiled address to a model."""
    min_arity: int = 1
    max_arity: int = 1
    priority: float = float('inf')
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
    """Literal ``;`` of :class:`ConcatenateNode`."""
    affix: Literal['prefix', 'suffix', 'infix', 'leaf'] = 'infix'
    regex: str = r'\;'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` unchanged."""
        return params


class ConcatenateNode(TransformPrimitive):
    """Transform that concatenates the selections of its arguments."""
    min_arity: int = 2
    max_arity: int = float('inf')
    priority: int = 4
    associative: bool = True
    commutative: bool = True
    literals: Sequence[Literalisation] = (ConcatenateInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:
        """Return a function concatenating the selections of ``pparams``."""
        def concatenate(arg: Any) -> PyTree:
            out = tuple(f(arg) for f in pparams)
            return reduce(lambda x, y: x + y, out)

        return concatenate


# --------------------------------- Keys ---------------------------------- #


class StringKeyInfixLiteralisation(Literalisation):
    """Literal ``$`` of :class:`KeyNode`."""
    affix: Literal['prefix', 'suffix', 'infix', 'leaf'] = 'infix'
    regex: str = r'\$'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` unchanged."""
        return params


class AttributeInfixLiteralisation(Literalisation):
    """Literal ``.`` of :class:`KeyNode`."""
    affix: Literal['prefix', 'suffix', 'infix', 'leaf'] = 'infix'
    regex: str = r'\.'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` unchanged."""
        return params


class KeyNode(TransformPrimitive):
    """Transform that selects a named key of each selected parameter."""
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

        """Return a function that selects the key of each selection."""
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
    """Literal ``#i`` of :class:`IntegerKeyNode`."""
    affix: Literal['prefix', 'suffix', 'infix', 'leaf'] = 'suffix'
    regex: str = r'\#(?P<index>[0-9]+)'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the index ``i`` into a one-element tuple."""
        params['index'] = (int(params['index']),)
        return params


class IntegerKeyMultiSuffixLiteralisation(Literalisation):
    """Literal ``#i,j,...`` of :class:`IntegerKeyNode`."""
    affix: Literal['prefix', 'suffix', 'infix', 'leaf'] = 'suffix'
    regex: str = r'\#(?P<index>[0-9]+[\,[0-9]+]+)'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the indices into a tuple of integers."""
        params['index'] = tuple(int(z) for z in params['index'].split(','))
        return params


class IntegerKeyMultiRangeSuffixLiteralisation(Literalisation):
    """Literal ``#i:j,...`` (inclusive ranges) of :class:`IntegerKeyNode`."""
    affix: Literal['prefix', 'suffix', 'infix', 'leaf'] = 'suffix'
    regex: str = r'\#(?P<index>[0-9]+[\:0-9]*[\,\:0-9]+)'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the indices and inclusive ranges into integers and slices."""
        keys = params['index'].split(',')
        index = []
        for key in keys:
            if ':' in key:
                lim = [int(z) if z != '' else None for z in key.split(':')]
                if isinstance(lim[1], int):
                    lim[1] += 1
                index += [
                    slice(lim[0], lim[1]),
                ]
            else:
                index += [int(key)]
        params['index'] = tuple(index)
        return params


class IntegerKeyNode(TransformPrimitive):
    """Transform that selects items by integer index or range."""
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

        """Return a function selecting the indexed items of each selection."""
        if pparams:
            f = pparams[0]
        else:

            def f(x):
                return x

        def getitem_impl(acc):
            out = ()
            for index in params['index']:
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
