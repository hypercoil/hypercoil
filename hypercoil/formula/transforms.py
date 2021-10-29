# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Transforms
~~~~~~~~~~
Specific column transforms. For base classes, see `coltransforms`.
"""
import pandas as pd
from functools import reduce
from .coltransforms import (
    ColumnTransform,
    OrderedTransform
)
from .match import (
    MatchOnly,
    MatchAndCompare,
)
from .utils import diff_nanpad


class PowerTransform(OrderedTransform):
    """
    Column transform that computes exponential expansions. The nth-order
    transform returns the input data raised to the nth power.

    Formula specifications
    ----------------------
    * (variable)^6 for the 6th power
    * (variable)^^6 for all powers up to the 6th
    * (variable)^4-6 for the 4th through 6th powers
    * 1 must be included in the powers range for the original term to be
      returned when exponential terms are computed.
    """
    def __init__(self):
        super(PowerTransform, self).__init__(
            transform=lambda data, order: data.values ** order,
            all=r'^\((?P<child0>.*)\)\^\^(?P<order>[0-9]+)$',
            select=r'^\((?P<child0>.*)\)\^(?P<order>[0-9]+[\-]?[0-9]*)$',
            first=1,
            name='power',
            identity=1
        )


class DerivativeTransform(OrderedTransform):
    """
    Column transform that computes temporal derivatives as backward
    differences. The nth-order transform returns the nth-order backward
    difference of the input data, padding the output with NaNs so that its
    shape matches that of the input.

    Formula specifications
    ----------------------
    * d6(variable) for the 6th temporal derivative
    * dd6(variable) for all temporal derivatives up to the 6th
    * d4-6(variable) for the 4th through 6th temporal derivatives
    * 0 must be included in the temporal derivative range for the original
      term to be returned when temporal derivatives are computed.
    """
    def __init__(self):
        super(DerivativeTransform, self).__init__(
            transform=lambda data, order: diff_nanpad(data, n=order, axis=0),
            all=r'^dd(?P<order>[0-9]+)\((?P<child0>.*)\)$',
            select=r'^d(?P<order>[0-9]+[\-]?[0-9]*)\((?P<child0>.*)\)$',
            first=0,
            name='derivative',
            identity=0
        )


class ThreshIndicatorTransform(ColumnTransform):
    """
    Column transform that maps each variable to a Boolean indicator that
    specifies whether a comparative relation between each observation and a
    numeric threshold is satisfied. Supported relations include equality and
    standard inequalities (`>`, `>=`, `<`, `<=`, `!=`). Note that any missing
    or NaN observation is always mapped to False.

    Formula specifications
    ----------------------
    * 1_[variable < 0.5] for an indicator of whether each observation is less
      than 0.5
    * 1_[variable <= 0.5] for an indicator of whether each observation is less
      than or equal to 0.5
    * 1_[variable > 0.5] for an indicator of whether each observation is
      greater than 0.5
    * 1_[variable >= 0.5] for an indicator of whether each observation is
      greater than or equal to 0.5
    * 1_[variable == 0.5] for an indicator of whether each observation exactly
      equals 0.5
    * 1_[variable != 0.5] for an indicator of whether each observation is not
      exactly equal to 0.5
    """
    def __init__(self):
        regex = (r'^1_\[(?P<child0>[^\>\<\=\!]*)'
                 r'(?P<compare>[\>\<\=\!]+) *'
                 r'(?P<thresh>[0-9]+[\.]?[0-9]*)\]')
        transform = lambda data, compare, thresh: compare(data.values, thresh)
        matches = [MatchAndCompare(regex=regex)]
        super(ThreshIndicatorTransform, self).__init__(
            transform=transform,
            matches=matches,
            name='thresh'
        )


class UnionTransform(ColumnTransform):
    """
    Column transform that replaces all input variables with a single Boolean-
    valued variable representing their logical union. Input variables should be
    Boolean-valued.

    Formula specifications
    ----------------------
    * or(variable0 + variable1) for the elementwise union of Boolean-valued
      observations in variable0 and variable1
    * or(1_[variable > 1] + 1_[variable < 0]) for an indicator specifying if
      the value of `variable` is in the complement of (0, 1).

    See also
    --------
    IntersectionTransform : logical intersection.
    NegationTransform : logical negation
    """
    def __init__(self):
        regex = r'^or\((?P<child0>.*)\)'
        transform = lambda values: reduce((lambda x, y: x | y), values.T)
        matches = [MatchOnly(regex=regex)]
        super(UnionTransform, self).__init__(
            transform=transform,
            matches=matches,
            name='union'
        )

    def __call__(self, children, **args):
        selected = children[0]
        vars = '_OR_'.join(selected.columns)
        return pd.DataFrame(
            data=self.transform(selected.values),
            columns=[f'union_{vars}']
        )


class IntersectionTransform(ColumnTransform):
    """
    Column transform that replaces all input variables with a single Boolean-
    valued variable representing their logical intersection. Input variables
    should be Boolean-valued.

    Formula specifications
    ----------------------
    * and(variable0 + variable1) for the elementwise intersection of Boolean-
      valued observations in variable0 and variable1
    * and(1_[variable < 1] + 1_[variable > 1]) for an indicator specifying if the
      value of `variable` is in (0, 1).

    See also
    --------
    UnionTransform : logical union.
    NegationTransform : logical negation
    """
    def __init__(self):
        regex = r'^and\((?P<child0>.*)\)'
        transform = lambda values: reduce((lambda x, y: x & y), values.T)
        matches = [MatchOnly(regex=regex)]
        super(IntersectionTransform, self).__init__(
            transform=transform,
            matches=matches,
            name='intersection'
        )

    def __call__(self, children, **args):
        selected = children[0]
        vars = '_AND_'.join(selected.columns)
        return pd.DataFrame(
            data=self.transform(selected.values),
            columns=[f'intersection_{vars}']
        )


class NegationTransform(ColumnTransform):
    """
    Column transform that replaces each input variable with its logical
    negation, computed elementwise. Input variables should be Boolean-valued.
    Note that in many cases a different indicator can be used to binarise
    a variable and thereby avoid the use of negation; this is generally
    preferred.

    Formula specifications
    ----------------------
    * not(variable) for the elementwise negation of Boolean-valued observations
      in variable
    * not(or(variable0 + variable1)) for values that are false in both
      variable0 or variable1; equivalent to
      and(not(variable0) + not(variable1))

    See also
    --------
    UnionTransform : logical union.
    IntersectionTransform : logical intersection.
    """
    def __init__(self):
        regex = r'not\((?P<child0>.*)\)'
        transform = lambda values: ~ values
        matches = [MatchOnly(regex=regex)]
        super(NegationTransform, self).__init__(
            transform=transform,
            matches=matches,
            name='negation'
        )

    def __call__(self, children, **args):
        selected = children[0]
        vars = [f'NOT_{col}' for col in selected.columns]
        return pd.DataFrame(
            data=self.transform(selected.values),
            columns=vars
        )
