# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Model specifications
~~~~~~~~~~~~~~~~~~~~
Specifications for selecting particular columns from a data frame and adding
expansion terms if necessary. Largely lifted and adapted from niworkflows.
"""
import re, json
import numpy as np
import pandas as pd
from collections import OrderedDict, deque
from functools import reduce


class ModelSpec(object):
    def __init__(self, spec, name=None, unscramble=True,
                 shorthand=None, transforms=None):
        self.spec = spec
        self.unscramble = unscramble
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


class Expression(object):
    def __init__(self, expr, transforms=None):
        self.transform = False
        self.transforms = transforms
        self.expr = expr.strip()
        if self.is_parenthetical(self.expr):
            self.expr = self.expr[1:-1]
        self.children = []

        expr_delimiter = 0
        grouping_depth = 0
        for i, char in enumerate(self.expr):
            if char == '(':
                grouping_depth += 1
            elif char == ')':
                grouping_depth -= 1
            elif grouping_depth == 0 and char == '+':
                self.children += [
                    Expression(self.expr[expr_delimiter:i], transforms)]
                expr_delimiter = i + 1
        if expr_delimiter > 0:
            self.children += [
                Expression(self.expr[expr_delimiter:], transforms)]
        else:
            self.set_transform_node()
        self.purge()
        self.n_children = len(self.children)

    def parse(self, df, unscramble=False):
        self.purge()
        if self.n_children == 0:
            self.data = df[self.expr]
            return self.data
        for i, expr in enumerate(self.children):
            self.data[i] = expr.parse(df)
        self.data = pd.concat(self.data, axis=1)
        if self.transform:
            self.data = self.transform.check_and_expand(
                self.expr, self.data.columns, self.data)
        if unscramble:
            self._unscramble_regressor_columns(df)
        return self.data

    def set_transform_node(self):
        for t in self.transforms:
            if re.search(t.all, self.expr) or re.search(t.select, self.expr):
                self.transform = t
                self._transform_arg_as_child()
                return
        self._transform_arg_as_child()

    def purge(self):
        self.data = [None for _ in self.children]

    def _transform_arg_as_child(self):
        """Make the argument of the transform into a child node."""
        grouping_depth = 0
        for i, char in enumerate(self.expr):
            if char == '(':
                if grouping_depth == 0:
                    expr_delimiter = i + 1
                grouping_depth += 1
            elif char == ')':
                grouping_depth -= 1
                if grouping_depth == 0:
                    self.children = [
                        Expression(self.expr[expr_delimiter:i],
                                   self.transforms)]
                    return

    def _unscramble_regressor_columns(self, df):
        """Reorder the columns of a confound matrix such that the columns are in
        the same order as the input data with any expansion columns inserted
        immediately after the originals.
        """
        matches = ['_power[0-9]+', '_derivative[0-9]+']
        var = OrderedDict((c, deque()) for c in df.columns)
        for c in self.data.columns:
            col = c
            for m in matches:
                col = re.sub(m, '', col)
            if col == c:
                var[col].appendleft(c)
            else:
                var[col].append(c)
        unscrambled = reduce((lambda x, y: x + y), var.values())
        return self.data[[*unscrambled]]

    def is_parenthetical(self, expr):
        return (expr[0] == '(' and expr[-1] == ')')

    def __repr__(self):
        if not self.transform:
            return f'Expression({self.expr}, children={self.n_children})'
        else:
            return f'Expression({self.transform}, children=1)'

    def __call__(self, df):
        return self.parse(df, unscramble=True)


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
        data = self(
            order=order,
            variables=variables,
            data=data
        )
        return data

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
        variables_xfm = reduce(
            lambda x, y: x + y,
            [list(v) for v in variables_xfm.values()])
        data_xfm = pd.DataFrame(
            columns=variables_xfm,
            data=np.concatenate([*data_xfm.values()], axis=1)
        )
        return data_xfm

    def __repr__(self):
        return f'{type(self).__name__}()'


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


def numbered_string(s):
    num = int(re.search('(?P<num>[0-9]+$)', s).groupdict()['num'])
    string = re.sub('[0-9]+$', '', s)
    return (string, num)


class ShorthandFilter(object):
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)


class FirstN(ShorthandFilter):
    def __call__(self, metadata, n):
        n = int(n)
        matches = list(filter(self.pattern.match, metadata.keys()))
        matches.sort(key=numbered_string)
        return ' + '.join(matches[:n])


class CumulVar(ShorthandFilter):
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
    'acc\<n=(?P<n>[0-9]+)\>': FirstN('^a_comp_cor_[0-9]+'),
    'acc\<v=(?P<v>[0-9\.]+)\>': CumulVar('^a_comp_cor_[0-9]+'),
    'tcc\<n=(?P<n>[0-9]+)\>': FirstN('^t_comp_cor_[0-9]+'),
    'tcc\<v=(?P<v>[0-9\.]+)\>': CumulVar('^t_comp_cor_[0-9]+'),
}


class Shorthand(object):
    def __init__(self, rules=None, regex=None, filters=None, transforms=None):
        self.rules = rules
        self.regex = regex
        self.filters = filters
        self.transforms = transforms

    def expand(self, model_formula, variables, metadata=None):
        for k, filt in self.filters.items():
            params = re.search(k, model_formula)
            if params is None: continue
            v = filt(metadata, **params.groupdict())
            model_formula = re.sub(k, v, model_formula)
        for k, v in self.regex.items():
            v = self.find_matching_variables(v, variables)
            model_formula = re.sub(k, v, model_formula)
        for k, v in self.rules.items():
            model_formula = re.sub(k, v, model_formula)

        formula_variables = self.get_formula_variables(model_formula)
        others = ' + '.join(set(variables) - set(formula_variables))
        model_formula = re.sub('others', others, model_formula)
        return model_formula

    def find_matching_variables(self, regex, variables):
        matches = re.compile(regex)
        matches = ' + '.join([v for v in variables if matches.match(v)])
        return matches

    def get_formula_variables(self, model_formula):
        symbols_to_clear = [t.all for t in self.transforms] + [
            t.select for t in self.transforms]
        for symbol in symbols_to_clear:
            model_formula = re.sub(symbol, '', model_formula)
        variables = model_formula.split('+')
        return variables

    def __call__(self, model_formula, variables, metadata=None):
        return self.expand(model_formula, variables, metadata)


class FCShorthand(Shorthand):
    def __init__(self):
        transforms = [
            DerivativeTransform(),
            PowerTransform()
        ]
        super(FCShorthand, self).__init__(
            rules=shorthand,
            regex=shorthand_re,
            filters=shorthand_filters,
            transforms=transforms
        )


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


def load_metadata(path):
    with open(path) as file:
        metadata = json.load(file)
    return metadata


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
