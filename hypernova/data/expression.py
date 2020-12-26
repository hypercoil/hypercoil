# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Expression
~~~~~~~~~~
Model expressions and parse trees for generating models from DataFrames.
"""
import re
import pandas as pd
from functools import reduce
from collections import OrderedDict, deque


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

    def is_parenthetical(self, expr):
        return (expr[0] == '(' and expr[-1] == ')')

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

    def __repr__(self):
        if not self.transform:
            return f'Expression({self.expr}, children={self.n_children})'
        else:
            return f'Expression({self.transform}, children=1)'

    def __call__(self, df):
        return self.parse(df, unscramble=True)
