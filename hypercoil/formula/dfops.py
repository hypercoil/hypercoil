# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
DataFrames
~~~~~~~~~~
Transformations and grammar for DataFrame operations.
"""
import pandas as pd
import equinox as eqx
from dataclasses import field
from functools import reduce
from typing import (
    Any, Callable, Dict, Literal, Optional, Sequence, Tuple
)
from .grammar import (
    Grammar,
    Literalisation, TransformPrimitive,
    LeafInterpreter, Grouping, GroupingPool, TransformPool,
)


def confound_formula_shorthand():
    return {
        'wm': 'white_matter',
        'gsr': 'global_signal',
        'rps': 'trans_x + trans_y + trans_z + rot_x + rot_y + rot_z',
        'fd': 'framewise_displacement',
        'dv': 'std_dvars',
    }


def collate_metadata(meta: Sequence[Optional[Dict]]) -> Any:
    meta = [m for m in meta if m is not None]
    if len(meta) == 0:
        return None
    else:
        return reduce(lambda x, y: {**x, **y}, meta)


class ConfoundFormulaGrammar(Grammar):
    groupings: GroupingPool = GroupingPool(
        Grouping(open='(', close=')'),
    )
    transforms: TransformPool = field(
        default_factory = lambda: TransformPool(
            ConcatenateNode(),
            PowerNode(),
            BackwardDifferenceNode(),
            IndicatorNode(),
            UnionNode(),
            IntersectionNode(),
            NegationNode(),
        )
    )
    whitespace: bool = False
    shorthand: Dict[str, str] = field(
        default_factory = confound_formula_shorthand
    )


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
    literals: Sequence[Literalisation] = (
        ConcatenatePrefixLiteralisation(),
        ConcatenateInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

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
    literals: Sequence[Literalisation] = (
        BackwardDifferenceEnumPrefixLiteralisation(),
        BackwardDifferenceInclPrefixLiteralisation(),
        BackwardDifferenceRangePrefixLiteralisation(),
        BackwardDifferencePrefixLiteralisation(),
    )

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
    literals: Sequence[Literalisation] = (
        PowerEnumSuffixLiteralisation(),
        PowerInclSuffixLiteralisation(),
        PowerRangeSuffixLiteralisation(),
        PowerSuffixLiteralisation(),
    )

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


#-------------------------------- Indicator --------------------------------#


class IndicatorPrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = (
        r'1_\[(?P<compare>[\>\<\=\!]+)'
        r'(?P<threshold>[0-9]+[\.]?[0-9]*)\]'
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        threshold = float(params['order'])
        compare = params['compare']
        if compare == '>':
            params['indicator'] = lambda x: x > threshold
            params['name'] = f'gt_{threshold}'
        elif compare == '>=':
            params['indicator'] = lambda x: x >= threshold
            params['name'] = f'ge_{threshold}'
        elif compare == '<':
            params['indicator'] = lambda x: x < threshold
            params['name'] = f'lt_{threshold}'
        elif compare == '<=':
            params['indicator'] = lambda x: x <= threshold
            params['name'] = f'le_{threshold}'
        elif compare == '==':
            params['indicator'] = lambda x: x == threshold
            params['name'] = f'eq_{threshold}'
        elif compare == '!=':
            params['indicator'] = lambda x: x != threshold
            params['name'] = f'ne_{threshold}'
        return params


class IndicatorNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 1
    literals: Sequence[Literalisation] = (
        IndicatorPrefixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]

        def indicator(
            arg: Any,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = f(arg, meta)
            df = params['indicator'](arg)
            df.columns = [f'{c}_{params["name"]}' for c in df.columns]
            return df, meta

        return indicator


#---------------------------------- Union ----------------------------------#


class UnionPrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'\[OR\]'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class UnionNode(TransformPrimitive):
    min_arity: int = 1 # requires a concatenation but the concatenation
                       # counts only as one argument
    max_arity: int = 1
    priority: int = 1
    literals: Sequence[Literalisation] = (
        UnionPrefixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]

        def union(
            arg: Any,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = f(arg, meta)
            values = reduce((lambda x, y: x | y), arg.values.T)
            column = '_or_'.join(arg.columns)
            df = pd.DataFrame({f'union_{column}': values})
            return df, meta

        return union


#------------------------------ Intersection -------------------------------#


class IntersectionPrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'\[AND\]'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class IntersectionNode(TransformPrimitive):
    min_arity: int = 1 # requires a concatenation but the concatenation
                       # counts only as one argument
    max_arity: int = 1
    priority: int = 1
    literals: Sequence[Literalisation] = (
        IntersectionPrefixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]

        def intersection(
            arg: Any,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = f(arg, meta)
            values = reduce((lambda x, y: x & y), arg.values.T)
            column = '_and_'.join(arg.columns)
            df = pd.DataFrame({f'intersection_{column}': values})
            return df, meta

        return intersection


#-------------------------------- Negation ---------------------------------#


class NegationPrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'\[NOT\]'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class NegationNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 1
    literals: Sequence[Literalisation] = (
        NegationPrefixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]

        def negation(
            arg: Any,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = f(arg, meta)
            values = ~values
            column = [f'negation_{c}' for c in arg.columns]
            df = pd.DataFrame({column: values})
            return df, meta

        return negation


#-------------------------------- LHS ~ RHS --------------------------------#


class LhsRhs(eqx.Module):
    lhs: pd.DataFrame
    rhs: pd.DataFrame


class LhsRhsInfixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'infix'
    regex : str = r'\~'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class LhsRhsNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 100
    literals: Sequence[Literalisation] = (
        LhsRhsInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f_lhs, f_rhs = pparams

        def lhs_rhs(
            arg: Any,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            lhs, meta_lhs = f_lhs(arg, meta)
            rhs, meta_rhs = f_rhs(arg, meta)
            meta = {**meta_lhs, **meta_rhs}
            return LhsRhs(lhs=lhs, rhs=rhs), meta

        return lhs_rhs


#------------------------------- Interaction -------------------------------#


class InteractionInfixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'infix'
    regex : str = r'\:'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class InteractionNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = float('inf')
    priority: int = 2
    literals: Sequence[Literalisation] = (
        InteractionInfixLiteralisation(),
    )
    associative: bool = True
    commutative: bool = True

    def ascend(self, *pparams, **params) -> Callable:

        def interaction(
            arg: Any,
            meta: Optional[Dict] = None
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = zip(*(f(arg, meta) for f in pparams))
            values = reduce((lambda x, y: x.values * y.values), arg)
            column = '_by_'.join(arg.columns)
            df = pd.DataFrame({f'interaction_{column}': values})
            return df, collate_metadata(meta)

        return interaction


#------------------------------- Cross Terms -------------------------------#


#--------------------------------- First N ---------------------------------#


#--------------------------- Cumulative Variance ---------------------------#


#---------------------------- Noise Components -----------------------------#


#------------------------------ Dummy Coding -------------------------------#


#------------------------------- Deduplicate -------------------------------#


#------------------------------- FC-Specific -------------------------------#
