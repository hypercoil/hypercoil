# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
DataFrames
~~~~~~~~~~
Transformations and grammar for DataFrame operations.
"""
import pandas as pd
from dataclasses import field
from functools import reduce
from typing import (
    Any, Callable, Dict, Literal, Optional, Sequence, Tuple
)
from .grammar import (
    Grammar,
    Literalisation, TransformPrimitive,
    LeafInterpreter
)


class ConfoundFormulaGrammar(Grammar):
    pass


class ColumnSelectInterpreter(LeafInterpreter):
    def __call__(self, leaf: Any) -> Callable:
        def select_column(
            df: pd.DataFrame,
            meta: Optional[Any] = None,
        ) -> pd.DataFrame:
            return df[[leaf]], meta
        return select_column


#------------------------------ Concatenation ------------------------------#


class ConcatenateInfixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'infix'
    regex : str = r'\+'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class ConcatenatePrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'\+\{[^\{^\}]+\}'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class ConcatenateNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = float('inf')
    priority: int = 3
    associative: bool = True
    commutative: bool = True
    literals: Sequence[Literalisation] = field(default_factory = lambda: [
        ConcatenatePrefixLiteralisation(),
        ConcatenateInfixLiteralisation(),
    ])

    def ascend(self, *pparams, **params) -> Callable:

        def collate_metadata(meta: Sequence[Optional[Dict]]) -> Any:
            meta = [m for m in meta if m is not None]
            if len(meta) == 0:
                return None
            else:
                return reduce(lambda x, y: {**x, **y}, meta)

        def concatenate(
            arg: Any,
            meta: Optional[Dict] = None
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = zip(*(f(arg, meta) for f in pparams))
            return pd.concat(
                tuple(arg),
                axis=1
            ), collate_metadata(meta)

        return concatenate


#--------------------------- Backward Difference ---------------------------#


class BackwardDifferencePrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'd(?P<order>[0-9]+)'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['order'] = (int(params['order']),)
        return params


class BackwardDifferenceInclPrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'dd(?P<order>[0-9]+)'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['order'] = tuple(range(int(params['order']) + 1))
        return params


class BackwardDifferenceRangePrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'd(?P<order>[0-9]+\-[0-9]+)'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        lim = [int(z) for z in params['order'].split('-')]
        params['order'] = tuple(range(lim[0], lim[1] + 1))
        return params


class BackwardDifferenceEnumPrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'd\{(?P<order>[0-9\,]+)\}'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['order'] = tuple(int(z) for z in params['order'].split(','))
        return params


class BackwardDifferenceNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 2
    literals: Sequence[Literalisation] = field(default_factory = lambda: [
        BackwardDifferenceEnumPrefixLiteralisation(),
        BackwardDifferenceInclPrefixLiteralisation(),
        BackwardDifferenceRangePrefixLiteralisation(),
        BackwardDifferencePrefixLiteralisation(),
    ])

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]

        def backward_difference(
            arg: Any,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = f(arg, meta)
            acc = [arg]
            for o in range(max(params['order'])):
                arg = arg.diff()
                if o in params['order']:
                    arg.columns = [f'{c}_derivative{o}' if o != 0 else c
                                   for c in arg.columns]
                    acc.append(arg)
            return pd.concat(acc, axis=1), meta

        return backward_difference


#---------------------------------- Power ----------------------------------#


class PowerSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'suffix'
    regex : str = r'\^(?P<order>[0-9]+)'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['order'] = (int(params['order']),)
        return params


class PowerInclSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'suffix'
    regex : str = r'\^\^(?P<order>[0-9]+)'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['order'] = tuple(range(1, int(params['order']) + 1))
        return params


class PowerRangeSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'suffix'
    regex : str = r'\^(?P<order>[0-9]+\-[0-9]+)'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        lim = [int(z) for z in params['order'].split('-')]
        params['order'] = tuple(range(lim[0], lim[1] + 1))
        return params


class PowerEnumSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'suffix'
    regex : str = r'\^\{(?P<order>[0-9\,]+)\}'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['order'] = tuple(int(z) for z in params['order'].split(','))
        return params


class PowerNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 1
    literals: Sequence[Literalisation] = field(default_factory = lambda: [
        PowerEnumSuffixLiteralisation(),
        PowerInclSuffixLiteralisation(),
        PowerRangeSuffixLiteralisation(),
        PowerSuffixLiteralisation(),
    ])

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]

        def power(
            arg: Any,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = f(arg, meta)
            acc = [arg ** o for o in params['order']]
            for df, o in zip(acc, params['order']):
                df.columns = [f'{c}_power{o}' if o != 1 else c
                              for c in df.columns]
            return pd.concat(acc, axis=1), meta

        return power


#-------------------------------- Threshold --------------------------------#


#---------------------------------- Union ----------------------------------#


#------------------------------ Intersection -------------------------------#


#-------------------------------- Negation ---------------------------------#


#-------------------------------- LHS ~ RHS --------------------------------#


#------------------------------- Interaction -------------------------------#


#------------------------------- Cross Terms -------------------------------#


#--------------------------------- First N ---------------------------------#


#--------------------------- Cumulative Variance ---------------------------#


#---------------------------- Noise Components -----------------------------#


#------------------------------ Dummy Coding -------------------------------#


#------------------------------- Deduplicate -------------------------------#
