# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Model specifications
~~~~~~~~~~~~~~~~~~~~
Specifications for selecting particular columns from a data frame and adding
expansion terms if necessary. Largely adapted and updated from niworkflows.
"""
from .expression import Expression


class ModelSpec(object):
    """
    Parse a confound manipulation formula.

    Recursively parse a model formula by breaking it into additive atoms
    and tracking grouping symbol depth.

    Parameters
    ----------
    model_formula: str
        Expression for the model formula, e.g.
        '(a + b)^^2 + dd1(c + (d + e)^3) + f'
        Note that any expressions to be expanded *must* be in parentheses,
        even if they include only a single variable (e.g., (x)^2, not x^2).
    parent_data: pandas DataFrame
        A tabulation of all values usable in the model formula. Each additive
        term in `model_formula` should correspond either to a variable in this
        data frame or to instructions for operating on a variable (for
        instance, computing temporal derivatives or exponential terms).

    Returns
    -------
    variables: list(str)
        A list of variables included in the model parsed from the provided
        formula.
    data: pandas DataFrame
        All values in the complete model.

    Options
    -------
    Temporal derivative options:
    * d6(variable) for the 6th temporal derivative
    * dd6(variable) for all temporal derivatives up to the 6th
    * d4-6(variable) for the 4th through 6th temporal derivatives
    * 0 must be included in the temporal derivative range for the original
      term to be returned when temporal derivatives are computed.

    Exponential options:
    * (variable)^6 for the 6th power
    * (variable)^^6 for all powers up to the 6th
    * (variable)^4-6 for the 4th through 6th powers
    * 1 must be included in the powers range for the original term to be
      returned when exponential terms are computed.

    Temporal derivatives and exponential terms are computed for all terms
    in the grouping symbols that they adjoin.
    """
    def __init__(self, spec, name=None, shorthand=None, transforms=None):
        self.spec = spec
        self.name = name or spec
        self.shorthand = shorthand
        self.transforms = transforms

    def __call__(self, df, metadata=None):
        if self.shorthand:
            formula = self.shorthand(self.spec, df.columns, metadata)
        else:
            formula = self.spec
        expr = Expression(formula, self.transforms)
        return expr(df)
