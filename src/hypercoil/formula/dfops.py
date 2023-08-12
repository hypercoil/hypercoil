# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
DataFrames
~~~~~~~~~~
Transformations and grammar for DataFrame operations.
"""
import re
from dataclasses import field
from functools import reduce
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import equinox as eqx
import numpy as np
import pandas as pd

from .grammar import (
    Grammar,
    Grouping,
    GroupingPool,
    LeafInterpreter,
    Literalisation,
    TransformPool,
    TransformPrimitive,
)


def confound_formula_shorthand():
    return {
        "wm": "white_matter",
        "gsr": "global_signal",
        "rps": "trans_x + trans_y + trans_z + rot_x + rot_y + rot_z",
        "fd": "framewise_displacement",
        "dv": "std_dvars",
        "acc": "a_comp_cor",
        "wcc": "w_comp_cor",
        "ccc": "c_comp_cor",
    }


def collate_metadata(meta: Sequence[Optional[Dict]]) -> Any:
    meta = [m for m in meta if m is not None]
    if len(meta) == 0:
        return None
    else:
        return reduce(lambda x, y: {**x, **y}, meta)


class ConfoundFormulaGrammar(Grammar):
    groupings: GroupingPool = GroupingPool(
        Grouping(open="(", close=")"),
    )
    transforms: TransformPool = field(
        default_factory=lambda: TransformPool(
            ConcatenateNode(),
            PowerNode(),
            BackwardDifferenceNode(),
            IndicatorNode(),
            UnionNode(),
            IntersectionNode(),
            NegationNode(),
            ScatterNode(),
            FirstNNode(),
            CumulativeVarianceNode(),
            MetadataFilterNode(),  # This has got to be last because other
            # nodes might have a metadata filter as a
            # substring of their literalisation.
        )
    )
    whitespace: bool = False
    shorthand: Dict[str, str] = field(
        default_factory=confound_formula_shorthand
    )
    default_interpreter: Optional[LeafInterpreter] = field(
        default_factory=lambda: ColumnSelectInterpreter()
    )
    default_root_transform: Optional[TransformPrimitive] = field(
        default_factory=lambda: DeduplicateRootNode()
    )


class ColumnSelectInterpreter(LeafInterpreter):
    def __call__(self, leaf: Any) -> Callable:
        def select_column(
            df: pd.DataFrame,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            return df[[leaf]], meta

        return select_column


class DeduplicateRootNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: float = float("inf")
    associative: bool = False
    commutative: bool = False
    literals: Sequence[Literalisation] = ()

    def __call__(self, *pparams, **params) -> Callable:
        f = pparams[0]

        def compiled(
            arg: pd.DataFrame,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = f(arg, meta)
            arg = arg.loc[:, ~arg.columns.duplicated()].copy()
            return arg, meta

        return compiled


# ----------------------------- Concatenation ----------------------------- #


class ConcatenateInfixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "infix"
    regex: str = r"\+"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class ConcatenatePrefixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "prefix"
    regex: str = r"\+\{[^\{^\}]+\}"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class ConcatenateNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = float("inf")
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
            meta: Optional[Dict] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = zip(*(f(arg, meta) for f in pparams))
            return pd.concat(tuple(arg), axis=1), collate_metadata(meta)

        return concatenate


# -------------------------- Backward Difference -------------------------- #


class BackwardDifferencePrefixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "prefix"
    regex: str = r"d(?P<order>[0-9]+)"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["order"] = (int(params["order"]),)
        return params


class BackwardDifferenceInclPrefixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "prefix"
    regex: str = r"dd(?P<order>[0-9]+)"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["order"] = tuple(range(int(params["order"]) + 1))
        return params


class BackwardDifferenceRangePrefixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "prefix"
    regex: str = r"d(?P<order>[0-9]+\-[0-9]+)"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        lim = [int(z) for z in params["order"].split("-")]
        params["order"] = tuple(range(lim[0], lim[1] + 1))
        return params


class BackwardDifferenceEnumPrefixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "prefix"
    regex: str = r"d\{(?P<order>[0-9\,]+)\}"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["order"] = tuple(int(z) for z in params["order"].split(","))
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
            orig = arg
            acc = []
            for o in range(1, max(params["order"]) + 1):
                arg = arg.diff()
                if o in params["order"]:
                    arg.columns = [
                        f"{c}_derivative{o}" if o != 0 else c
                        for c in arg.columns
                    ]
                    acc.append(arg)
            if 0 in params["order"]:
                acc = [orig] + acc
            return pd.concat(acc, axis=1), meta

        return backward_difference


# --------------------------------- Power --------------------------------- #


class PowerSuffixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "suffix"
    regex: str = r"\^(?P<order>[0-9]+)"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["order"] = (int(params["order"]),)
        return params


class PowerInclSuffixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "suffix"
    regex: str = r"\^\^(?P<order>[0-9]+)"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["order"] = tuple(range(1, int(params["order"]) + 1))
        return params


class PowerRangeSuffixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "suffix"
    regex: str = r"\^(?P<order>[0-9]+\-[0-9]+)"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        lim = [int(z) for z in params["order"].split("-")]
        params["order"] = tuple(range(lim[0], lim[1] + 1))
        return params


class PowerEnumSuffixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "suffix"
    regex: str = r"\^\{(?P<order>[0-9\,]+)\}"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["order"] = tuple(int(z) for z in params["order"].split(","))
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
            acc = [arg**o for o in params["order"]]
            for df, o in zip(acc, params["order"]):
                df.columns = [
                    f"{c}_power{o}" if o != 1 else c for c in df.columns
                ]
            return pd.concat(acc, axis=1), meta

        return power


# ------------------------------- Indicator ------------------------------- #


class IndicatorPrefixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "prefix"
    regex: str = (
        r"1_\[(?P<compare>[\>\<\=\!]+)" r"(?P<threshold>[0-9]+[\.]?[0-9]*)\]"
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        threshold = float(params["threshold"])
        compare = params["compare"]
        if compare == ">":
            params["indicator"] = lambda x: x > threshold
            params["name"] = f"gt_{threshold}"
        elif compare == ">=":
            params["indicator"] = lambda x: x >= threshold
            params["name"] = f"ge_{threshold}"
        elif compare == "<":
            params["indicator"] = lambda x: x < threshold
            params["name"] = f"lt_{threshold}"
        elif compare == "<=":
            params["indicator"] = lambda x: x <= threshold
            params["name"] = f"le_{threshold}"
        elif compare == "==":
            params["indicator"] = lambda x: x == threshold
            params["name"] = f"eq_{threshold}"
        elif compare == "!=":
            params["indicator"] = lambda x: x != threshold
            params["name"] = f"ne_{threshold}"
        return params


class IndicatorNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 1
    literals: Sequence[Literalisation] = (IndicatorPrefixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]

        def indicator(
            arg: Any,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = f(arg, meta)
            df = params["indicator"](arg)
            df.columns = [f'{c}_{params["name"]}' for c in df.columns]
            return df, meta

        return indicator


# --------------------------------- Union --------------------------------- #


class UnionPrefixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "prefix"
    regex: str = r"\[OR\]"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class UnionNode(TransformPrimitive):
    min_arity: int = 1  # requires a concatenation but the concatenation
    # counts only as one argument
    max_arity: int = 1
    priority: int = 1
    literals: Sequence[Literalisation] = (UnionPrefixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]

        def union(
            arg: Any,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = f(arg, meta)
            values = reduce((lambda x, y: x | y), arg.values.T)
            column = "_or_".join(arg.columns)
            df = pd.DataFrame({f"union_{column}": values})
            return df, meta

        return union


# ----------------------------- Intersection ------------------------------ #


class IntersectionPrefixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "prefix"
    regex: str = r"\[AND\]"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class IntersectionNode(TransformPrimitive):
    min_arity: int = 1  # requires a concatenation but the concatenation
    # counts only as one argument
    max_arity: int = 1
    priority: int = 1
    literals: Sequence[Literalisation] = (IntersectionPrefixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]

        def intersection(
            arg: Any,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = f(arg, meta)
            values = reduce((lambda x, y: x & y), arg.values.T)
            column = "_and_".join(arg.columns)
            df = pd.DataFrame({f"intersection_{column}": values})
            return df, meta

        return intersection


# ------------------------------- Negation -------------------------------- #


class NegationPrefixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "prefix"
    regex: str = r"\[NOT\]"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class NegationNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 1
    literals: Sequence[Literalisation] = (NegationPrefixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]

        def negation(
            arg: Any,
            meta: Optional[Any] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = f(arg, meta)
            values = ~arg.values
            column = [f"negation_{c}" for c in arg.columns]
            df = pd.DataFrame({c: v for c, v in zip(column, values.T)})
            return df, meta

        return negation


# ------------------------------- LHS ~ RHS ------------------------------- #


class LhsRhs(eqx.Module):
    lhs: pd.DataFrame
    rhs: pd.DataFrame


class LhsRhsInfixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "infix"
    regex: str = r"\~"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class LhsRhsNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 100
    literals: Sequence[Literalisation] = (LhsRhsInfixLiteralisation(),)

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


# ----------------------------- Dummy Coding ------------------------------ #


# ----------------------------- Conditioning ------------------------------ #


# ------------------------------ Expression ------------------------------- #


# ------------------------------ Interaction ------------------------------ #


class InteractionInfixLiteralisation(Literalisation):
    affix: Literal["prefix", "suffix", "infix", "leaf"] = "infix"
    regex: str = r"\:"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class InteractionNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = float("inf")
    priority: int = 2
    literals: Sequence[Literalisation] = (InteractionInfixLiteralisation(),)
    associative: bool = True
    commutative: bool = True

    def ascend(self, *pparams, **params) -> Callable:
        def interaction(
            arg: Any,
            meta: Optional[Dict] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            arg, meta = zip(*(f(arg, meta) for f in pparams))
            values = reduce((lambda x, y: x.values * y.values), arg)
            column = "_by_".join(arg.columns)
            df = pd.DataFrame({f"interaction_{column}": values})
            return df, collate_metadata(meta)

        return interaction


# ------------------------------ Cross Terms ------------------------------ #


# -------------------------------- First N -------------------------------- #


# TODO: after adding support for logical composition in metadata filters,
#       this should use a metadata filter instead of hard-coding to match
#       the ``Mask`` field. This might actually have some application outside
#       of confound regression, unlike CumulVar, so changing this is probably
#       a good idea here.
class FirstNLeafLiteralisation(Literalisation):
    affix: str = "leaf"
    regex: str = (
        r"n_\{\{(?P<count>[0-9]+);(?P<variable>[a-zA-Z0-9_]+)"
        r"(;)?(Mask=(?P<mask>[A-Za-z,]*))?\}\}"
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        shorthand = confound_formula_shorthand()
        params["count"] = int(params["count"])
        for k, v in shorthand.items():
            params["variable"] = re.sub(k, v, params["variable"])
        return params


class FirstNNode(TransformPrimitive):
    min_arity: int = 0
    max_arity: int = 0
    priority: int = 0
    literals: Sequence[Literalisation] = (FirstNLeafLiteralisation(),)
    associative: bool = False
    commutative: bool = False

    def ascend(self, *pparams, **params) -> Callable:
        def first_n(
            df: "pd.DataFrame",
            meta: Optional[Dict] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            n = params["count"]
            pattern = re.compile(r"^{}_[0-9]+".format(params["variable"]))
            matches = match_metadata(pattern, meta)
            matches.sort(key=numbered_string)

            if params.get("mask", None) is None:
                matches = matches[:n]
            else:
                matches_mask = []
                masks = params["mask"].split(",")
                for mask in masks:
                    filt = [m for m in matches if meta[m].get("Mask") == mask]
                    matches_mask += filt[:n]
                matches_mask.sort(key=numbered_string)
                matches = matches_mask

            columns = tuple(successive_pad_search(df, m) for m in matches)
            return pd.concat(columns, axis=1), meta

        return first_n


# -------------------------- Cumulative Variance -------------------------- #


class CumulativeVarianceLeafLiteralisation(Literalisation):
    affix: str = "leaf"
    regex: str = (
        r"v_\{\{(?P<variance>[0-9\.]+);(?P<variable>[a-zA-Z0-9_]+)"
        r"(;)?(Mask=(?P<mask>[A-Za-z,]*))?\}\}"
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        shorthand = confound_formula_shorthand()
        params["variance"] = float(params["variance"]) / 100
        for k, v in shorthand.items():
            params["variable"] = re.sub(k, v, params["variable"])
        return params


class CumulativeVarianceNode(TransformPrimitive):
    min_arity: int = 0
    max_arity: int = 0
    priority: int = 0
    literals: Sequence[Literalisation] = (
        CumulativeVarianceLeafLiteralisation(),
    )
    associative: bool = False
    commutative: bool = False

    def ascend(self, *pparams, **params) -> Callable:
        def cumulative_variance(
            df: "pd.DataFrame",
            meta: Optional[Dict] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            v = params["variance"]
            pattern = re.compile(r"^{}_[0-9]+".format(params["variable"]))
            matches = match_metadata(pattern, meta)

            if params.get("mask", None) is not None:
                matches_mask = []
                masks = params["mask"].split(",")
                for mask in masks:
                    filt = [m for m in matches if meta[m].get("Mask") == mask]
                    matches_mask += filt
                matches = matches_mask

            matches.sort(key=numbered_string)
            out = []
            done = False
            for m in matches:
                item = meta[m]
                if done:
                    done = item["CumulativeVarianceExplained"] > v
                    if done:
                        continue
                done = item["CumulativeVarianceExplained"] > v
                out += [m]

            columns = tuple(successive_pad_search(df, o) for o in out)
            return pd.concat(columns, axis=1), meta

        return cumulative_variance


# ---------------------------- Metadata Filter ---------------------------- #


# TODO: add simple logical composition (AND, OR, NOT) to filters.
#       Currently everything is AND, which isn't always what we want.
class MetadataFilterLeafLiteralisation(Literalisation):
    affix: str = "leaf"
    regex: str = (
        r"\{\{(?P<variable>[a-zA-Z0-9_]+)"
        r"(;)?(?P<filters>[a-zA-Z0-9_=;]+)\}\}"
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params["filters"] = [
            s.split("=") for s in params["filters"].split(";")
        ]
        for filter in params["filters"]:
            try:
                filter[1] = float(filter[1])
            except ValueError:
                try:
                    filter[1] = bool(filter[1])
                except ValueError:
                    pass
        params["filters"] = tuple(tuple(f) for f in params["filters"])
        return params


class MetadataFilterNode(TransformPrimitive):
    min_arity: int = 0
    max_arity: int = 0
    priority: int = 0
    literals: Sequence[Literalisation] = (MetadataFilterLeafLiteralisation(),)
    associative: bool = False
    commutative: bool = False

    def ascend(self, *pparams, **params) -> Callable:
        def filter_by_metadata(
            df: "pd.DataFrame",
            meta: Optional[Dict] = None,
        ) -> Tuple[pd.DataFrame, Any]:
            pattern = re.compile(r"^{}.*".format(params["variable"]))
            matches = match_metadata(pattern, meta)

            out = []
            for m in matches:
                matched = True
                item = meta[m]
                for k, v in params["filters"]:
                    if item.get(k) != v:
                        matched = False
                        break
                if matched:
                    out += [m]

            columns = tuple(successive_pad_search(df, o) for o in out)
            return pd.concat(columns, axis=1), meta

        return filter_by_metadata


# ---------------------------- Scatter Spikes ----------------------------- #


class ScatterPrefixLiteralisation(Literalisation):
    affix: str = "prefix"
    regex: str = r"\[SCATTER\]"

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class ScatterNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 1
    literals: Sequence[Literalisation] = (ScatterPrefixLiteralisation(),)
    associative: bool = False
    commutative: bool = False

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]

        def scatter_columns(
            arg: Any,
            meta: Optional[Dict] = None,
        ) -> Tuple[pd.DataFrame, Any]:

            arg, meta = f(arg, meta)
            scattered = []
            for col in arg.columns:
                data = arg[[col]].values
                nz = data.nonzero()
                names = [f"{col}_spike{i}" for i in nz[0]]
                cols = np.zeros((data.shape[0], len(nz)))
                cols[nz[0], range(len(nz))] = data[nz]
                scattered += [pd.DataFrame(cols, columns=names)]
            return pd.concat(scattered, axis=1), meta

        return scatter_columns


# --------------------------- Utility functions --------------------------- #


def numbered_string(s):
    """
    Convert a string whose final characters are numeric into a tuple
    containing all characters except the final numeric ones in the first field
    and the numeric characters, cast to an int, in the second field. This
    permits comparison of strings of this form without regard to the number of
    zeros in the final numeric part. It also permits strings of this form to
    be sorted in numeric order.

    Parameters
    ----------
    s : string
        A string whose first characters are non-numeric and whose final
        characters are numeric.

    Returns
    -------
    tuple(str, int)
        Input string split into its initial non-numeric substring and its
        final numeric substring, cast to an int.
    """
    num = int(re.search("(?P<num>[0-9]+$)", s).groupdict()["num"])
    string = re.sub("[0-9]+$", "", s)
    return (string, num)


def match_metadata(pattern, metadata):
    """
    Find all dictionary keys that match a pattern.

    Parameters
    ----------
    pattern : compiled regular expression
        A compiled regular expression (obtained via re.compile) that
        represents the pattern to be matched in the metadata keys.
    metadata : dict
        Dictionary whose keys are to be searched for the specified pattern.
    """
    return list(filter(pattern.match, metadata.keys()))


def successive_pad_search(df, key, pad=0, k=5):
    """
    Find the column of a DataFrame that matches a query, ignoring potential
    zero-padding.

    This function is an unfortunate artefact of an inconsistency in the way
    that an earlier version of fMRIPrep named metadata entries and data
    columns. The metadata entries were named without zero-padding, while the
    data columns were named with zero-padding. This function allows the
    metadata entries to be matched to the data columns without regard to
    zero-padding.

    Parameters
    ----------
    df : DataFrame
        DataFrame whose columns are to be searched for the specified key.
    key : str
        Key to search for in the DataFrame.
    pad : int (default 0)
        Number whose padding to ignore in the search
    k : int
        Maximum number of pads to try before a KeyError is raised.
    """
    for i in range(k):
        try:
            return df[[key]]
        except KeyError:
            p = "{:" + f"{pad}{i + 1}" + "}"
            st, nu = numbered_string(key)
            key = f"{st}" + p.format(nu)
    raise KeyError
