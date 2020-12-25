# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Model specifications
~~~~~~~~~~~~~~~~~~~~
Specifications for selecting particular columns from a data frame and adding
expansion terms if necessary. Largely lifted and adapted from niworkflows.
"""
import re
import numpy as np
import pandas as pd
from collections import OrderedDict
from functools import reduce


class ModelSpec(object):
    def __init__(self, spec, name=None, unscramble=True):
        self.spec = spec
        self.unscramble = unscramble
        self.name = name or spec

    def __call__(self, df):
        return parse_formula(self.spec, df, self.unscramble)


class ColumnTransform(object):
    def __init__(self, transform, all, select,
                 first=0, name='transform', identity=None):
        self.transform = transform
        self.all = re.compile(all)
        self.select = re.compile(select)
        self.first = first
        self.name = name
        self.identity = identity


    def check_and_expand(self, expr, variables, data):
        if re.search(self.all, expr):
            order = self.all.findall(expr)
            order = set(range(self.first, int(*order) + 1))
        elif re.search(self.select, expr):
            order = self.select.findall(expr)
            order = self._order_as_range(*order)
        variables, data = self(
            order=order,
            variables=variables,
            data=data
        )
        return variables, data

    def _order_as_range(order):
        """Convert a hyphenated string representing order for derivative or
        exponential terms into a range object that can be passed as input to
        the appropriate expansion function."""
        order = order.split('-')
        order = [int(o) for o in order]
        if len(order) > 1:
            order = range(order[0], (order[-1] + 1))
        return order

    def __call__(self, order, variables, data):
        variables_xfm, data_xfm = OrderedDict(), OrderedDict()
        selected = data[variables]
        if self.identity in order:
            data_xfm[self.identity] = selected
            variables_xfm[self.identity] = variables
            order -= {self.identity}
        for o in order:
            variables_xfm[o] = [f'{v}_{self.name}{o}' for v in variables]
            data_xfm[o] = self.transform(selected, o)
        variables_xfm = reduce((lambda x, y: x + y), variables_xfm.values())
        data_xfm = pd.DataFrame(
            columns=variables_xfm,
            data=np.concatenate([*data_xfm.values()], axis=1)
        )
        return variables_xfm, data_xfm


class PowerTransform(ColumnTransform):
    """
    Compute exponential expansions.

    Parameters
    ----------
    order: set(int)
        A set of exponential terms to include. For instance, {1, 2}
        indicates that the first and second powers should be added.
        To retain the original terms, 1 *must* be included in the list.
    variables: list(str)
        List of variables for which exponential terms should be computed.
    data: pandas DataFrame object
        Table of values of all observations of all variables.

    Returns
    -------
    variables_exp: list
        A list of variables to include in the final data frame after adding
        the specified exponential terms.
    data_exp: pandas DataFrame object
        Table of values of all observations of all variables, including any
        specified exponential terms.

    """
    def __init__(self):
        super(PowerTransform, self).__init__(
            transform=lambda data, order: data ** order,
            all=r'\^\^([0-9]+)$',
            select=r'\^([0-9]+[\-]?[0-9]*)$',
            first=1,
            name='power',
            identity=1
        )


class DerivativeTransform(ColumnTransform):
    """
    Compute temporal derivative terms by the method of backwards differences.

    Parameters
    ----------
    order: set(int)
        A set of temporal derivative terms to include. For instance, {1, 2}
        indicates that the first and second derivative terms should be added.
        To retain the original terms, 0 *must* be included in the set.
    variables: list(str)
        List of variables for which temporal derivative terms should be
        computed.
    data: pandas DataFrame object
        Table of values of all observations of all variables.

    Returns
    -------
    variables_deriv: list
        A list of variables to include in the final data frame after adding
        the specified derivative terms.
    data_deriv: pandas DataFrame object
        Table of values of all observations of all variables, including any
        specified derivative terms.

    """
    def __init__(self):
        super(DerivativeTransform, self).__init__(
            transform=lambda data, order: diff_nanpad(data, n=order, axis=0),
            all=r'^dd([0-9]+)',
            select=r'^d([0-9]+[\-]?[0-9]*)',
            first=0,
            name='derivative',
            identity=0
        )


def diff_nanpad(a, n=1, axis=-1):
    diff = np.empty_like(a) * float('nan')
    diff[n:] = np.diff(a, n=n, axis=axis)
    return diff


def _check_and_expand_subformula(expression, parent_data, variables, data):
    """Check if the current operation contains a suboperation, and parse it
    where appropriate."""
    grouping_depth = 0
    for i, char in enumerate(expression):
        if char == '(':
            if grouping_depth == 0:
                formula_delimiter = i + 1
            grouping_depth += 1
        elif char == ')':
            grouping_depth -= 1
            if grouping_depth == 0:
                expr = expression[formula_delimiter:i].strip()
                return parse_formula(expr, parent_data)
    return variables, data


def parse_expression(expression, parent_data):
    """
    Parse an expression in a model formula.

    Parameters
    ----------
    expression: str
        Formula expression: either a single variable or a variable group
        paired with an operation (exponentiation or differentiation).
    parent_data: pandas DataFrame
        The source data for the model expansion.

    Returns
    -------
    variables: list
        A list of variables in the provided formula expression.
    data: pandas DataFrame
        A tabulation of all terms in the provided formula expression.

    """
    variables = None
    data = None
    variables, data = _check_and_expand_subformula(expression,
                                                   parent_data,
                                                   variables,
                                                   data)
    variables, data = _check_and_expand_exponential(expression,
                                                    variables,
                                                    data)
    variables, data = _check_and_expand_derivative(expression,
                                                   variables,
                                                   data)
    if variables is None:
        expr = expression.strip()
        variables = [expr]
        data = parent_data[expr]
    return variables, data


def _get_matches_from_data(regex, variables):
    matches = re.compile(regex)
    matches = ' + '.join([v for v in variables if matches.match(v)])
    return matches


def _get_variables_from_formula(model_formula):
    symbols_to_clear = [' ', r'\(', r'\)', 'dd[0-9]+', r'd[0-9]+[\-]?[0-9]*',
                        r'\^\^[0-9]+', r'\^[0-9]+[\-]?[0-9]*']
    for symbol in symbols_to_clear:
        model_formula = re.sub(symbol, '', model_formula)
    variables = model_formula.split('+')
    return variables


shorthand = {
    'wm': 'white_matter',
    'gsr': 'global_signal',
    'rps': 'trans_x + trans_y + trans_z + rot_x + rot_y + rot_z',
    'fd': 'framewise_displacement'
}
shorthand_re = {
    'acc': '^a_comp_cor_[0-9]+',
    'tcc': '^t_comp_cor_[0-9]+',
    'dv': '^std_dvars$',
    'dvall': '.*dvars$',
    'nss': '^non_steady_state_outlier[0-9]+',
    'spikes': '^motion_outlier[0-9]+'
}
shorthand_filters = {
    'acc\(n=(?P<n>[0-9]+)\)': FirstN('^a_comp_cor_[0-9]+'),
    'acc\(v=(?P<v>[0-9\.]+)\)': CumulVar('^a_comp_cor_[0-9]+'),
    'tcc\(n=(?P<n>[0-9]+)\)': FirstN('^t_comp_cor_[0-9]+'),
    'tcc\(v=(?P<v>[0-9\.]+)\)': CumulVar('^t_comp_cor_[0-9]+'),
}


def load_metadata(path):
    with open(path) as file:
        metadata = json.load(file)
    return metadata


def numbered_string(s):
    num = int(re.search('(?P<num>[0-9]+$)', s).groupdict()['num'])
    string = re.sub('[0-9]+$', '', s)
    return (string, num)


class FirstN(object):
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)

    def __call__(self, metadata, n):
        n = int(n)
        matches = list(filter(self.pattern.match, metadata.keys()))
        matches.sort(key=numbered_string)
        return ' + '.join(matches[:n])


class CumulVar(object):
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)

    def __call__(self, metadata, v):
        v = float(v)
        if v > 1: v /= 100
        out = []
        matches = list(filter(self.pattern.match, metadata.keys()))
        for m in matches:
            item = metadata.get(m)
            if not item: break
            out += [m]
            if item['CumulativeVarianceExplained'] > v: break
        return ' + '.join(out)



def expand_shorthand(model_formula, variables, metadata=None):
    for k, filt in shorthand_filters.items():
        params = re.search(k, model_formula)
        if params is None: continue
        v = filt(metadata, **params.groupdict())
        model_formula = re.sub(k, v, model_formula)
    for k, v in shorthand_re.items():
        v = find_matching_variables(v, variables)
        model_formula = re.sub(k, v, model_formula)
    for k, v in shorthand.items():
        model_formula = re.sub(k, v, model_formula)

    formula_variables = get_formula_variables(model_formula)
    others = ' + '.join(set(variables) - set(formula_variables))
    model_formula = re.sub('others', others, model_formula)
    return model_formula


def find_matching_variables(regex, variables):
    matches = re.compile(regex)
    matches = ' + '.join([v for v in variables if matches.match(v)])
    return matches


def _unscramble_regressor_columns(parent_data, data):
    """Reorder the columns of a confound matrix such that the columns are in
    the same order as the input data with any expansion columns inserted
    immediately after the originals.
    """
    matches = ['_power[0-9]+', '_derivative[0-9]+']
    var = OrderedDict((c, deque()) for c in parent_data.columns)
    for c in data.columns:
        col = c
        for m in matches:
            col = re.sub(m, '', col)
        if col == c:
            var[col].appendleft(c)
        else:
            var[col].append(c)
    unscrambled = reduce((lambda x, y: x + y), var.values())
    return data[[*unscrambled]]


def parse_formula(model_formula, parent_data, unscramble=False):
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
    variables = {}
    data = {}
    expr_delimiter = 0
    grouping_depth = 0
    model_formula = _expand_shorthand(model_formula, parent_data.columns)
    for i, char in enumerate(model_formula):
        if char == '(':
            grouping_depth += 1
        elif char == ')':
            grouping_depth -= 1
        elif grouping_depth == 0 and char == '+':
            expression = model_formula[expr_delimiter:i].strip()
            variables[expression] = None
            data[expression] = None
            expr_delimiter = i + 1
    expression = model_formula[expr_delimiter:].strip()
    variables[expression] = None
    data[expression] = None
    for expression in list(variables):
        if expression[0] == '(' and expression[-1] == ')':
            (variables[expression],
             data[expression]) = parse_formula(expression[1:-1],
                                               parent_data)
        else:
            (variables[expression],
             data[expression]) = parse_expression(expression,
                                                  parent_data)
    variables = list(set(reduce((lambda x, y: x + y), variables.values())))
    data = pd.concat((data.values()), axis=1)

    if unscramble:
        data = _unscramble_regressor_columns(parent_data, data)

    return variables, data
