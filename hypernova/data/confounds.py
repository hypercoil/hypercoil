# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Select terms for a confound model, and compute any requisite expansions."""

import re
import numpy as np
import pandas as pd
from functools import reduce
from collections import deque, OrderedDict
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import (
    traits, TraitedSpec, BaseInterfaceInputSpec, File, isdefined,
    SimpleInterface
)


def temporal_derivatives(order, variables, data):
    """
    Compute temporal derivative terms by the method of backwards differences.

    Parameters
    ----------
    order: range or list(int)
        A list of temporal derivative terms to include. For instance, [1, 2]
        indicates that the first and second derivative terms should be added.
        To retain the original terms, 0 *must* be included in the list.
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
    variables_deriv = OrderedDict()
    data_deriv = OrderedDict()
    if 0 in order:
        data_deriv[0] = data[variables]
        variables_deriv[0] = variables
        order = set(order) - set([0])
    for o in order:
        variables_deriv[o] = ['{}_derivative{}'.format(v, o)
                              for v in variables]
        data_deriv[o] = np.tile(np.nan, data[variables].shape)
        data_deriv[o][o:, :] = np.diff(data[variables], n=o, axis=0)
    variables_deriv = reduce((lambda x, y: x + y), variables_deriv.values())
    data_deriv = pd.DataFrame(columns=variables_deriv,
                              data=np.concatenate([*data_deriv.values()],
                                                  axis=1))

    return (variables_deriv, data_deriv)


def exponential_terms(order, variables, data):
    """
    Compute exponential expansions.

    Parameters
    ----------
    order: range or list(int)
        A list of exponential terms to include. For instance, [1, 2]
        indicates that the first and second exponential terms should be added.
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
    variables_exp = OrderedDict()
    data_exp = OrderedDict()
    if 1 in order:
        data_exp[1] = data[variables]
        variables_exp[1] = variables
        order = set(order) - set([1])
    for o in order:
        variables_exp[o] = ['{}_power{}'.format(v, o) for v in variables]
        data_exp[o] = data[variables]**o
    variables_exp = reduce((lambda x, y: x + y), variables_exp.values())
    data_exp = pd.DataFrame(columns=variables_exp,
                            data=np.concatenate([*data_exp.values()], axis=1))
    return (variables_exp, data_exp)


def _order_as_range(order):
    """Convert a hyphenated string representing order for derivative or
    exponential terms into a range object that can be passed as input to the
    appropriate expansion function."""
    order = order.split('-')
    order = [int(o) for o in order]
    if len(order) > 1:
        order = range(order[0], (order[-1] + 1))
    return order


def _check_and_expand_exponential(expr, variables, data):
    """Check if the current operation specifies exponential expansion. ^^6
    specifies all powers up to the 6th, ^5-6 the 5th and 6th powers, ^6 the
    6th only."""
    if re.search(r'\^\^[0-9]+$', expr):
        order = re.compile(r'\^\^([0-9]+)$').findall(expr)
        order = range(1, int(*order) + 1)
        variables, data = exponential_terms(order, variables, data)
    elif re.search(r'\^[0-9]+[\-]?[0-9]*$', expr):
        order = re.compile(r'\^([0-9]+[\-]?[0-9]*)').findall(expr)
        order = _order_as_range(*order)
        variables, data = exponential_terms(order, variables, data)
    return variables, data


def _check_and_expand_derivative(expr, variables, data):
    """Check if the current operation specifies a temporal derivative. dd6x
    specifies all derivatives up to the 6th, d5-6x the 5th and 6th, d6x the
    6th only."""
    if re.search(r'^dd[0-9]+', expr):
        order = re.compile(r'^dd([0-9]+)').findall(expr)
        order = range(0, int(*order) + 1)
        (variables, data) = temporal_derivatives(order, variables, data)
    elif re.search(r'^d[0-9]+[\-]?[0-9]*', expr):
        order = re.compile(r'^d([0-9]+[\-]?[0-9]*)').findall(expr)
        order = _order_as_range(*order)
        (variables, data) = temporal_derivatives(order, variables, data)
    return variables, data


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


def _expand_shorthand(model_formula, variables):
    """Expand shorthand terms in the model formula."""
    wm = 'white_matter'
    gsr = 'global_signal'
    rps = 'trans_x + trans_y + trans_z + rot_x + rot_y + rot_z'
    fd = 'framewise_displacement'
    acc = _get_matches_from_data('a_comp_cor_[0-9]+', variables)
    tcc = _get_matches_from_data('t_comp_cor_[0-9]+', variables)
    dv = _get_matches_from_data('^std_dvars$', variables)
    dvall = _get_matches_from_data('.*dvars', variables)
    nss = _get_matches_from_data('non_steady_state_outlier[0-9]+',
                                 variables)
    spikes = _get_matches_from_data('motion_outlier[0-9]+', variables)

    model_formula = re.sub('wm', wm, model_formula)
    model_formula = re.sub('gsr', gsr, model_formula)
    model_formula = re.sub('rps', rps, model_formula)
    model_formula = re.sub('fd', fd, model_formula)
    model_formula = re.sub('acc', acc, model_formula)
    model_formula = re.sub('tcc', tcc, model_formula)
    model_formula = re.sub('dv', dv, model_formula)
    model_formula = re.sub('dvall', dvall, model_formula)
    model_formula = re.sub('nss', nss, model_formula)
    model_formula = re.sub('spikes', spikes, model_formula)

    formula_variables = _get_variables_from_formula(model_formula)
    others = ' + '.join(set(variables) - set(formula_variables))
    model_formula = re.sub('others', others, model_formula)
    return model_formula


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
