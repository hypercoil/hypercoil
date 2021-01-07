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
from .utils import successive_pad_search


class Expression(object):
    """
    Parsable additive expression object.

    An additive expression recursively decomposed into a parse tree comprising
    sub-expressions and transform nodes. The expression can be applied to a
    DataFrame to select, transform, and compose columns as needed to build the
    model specified by the parent expression.

    Parameters/Attributes
    ---------------------
    expr : str
        Expression to be parsed.
    transforms : list(ColumnTransform objects) or None (default None)
        List containing the column-wise transforms to be parsed in the
        formula. Consult the ColumnTransform documentation for further
        information and the source code of the DerivativeTransform and
        PowerTransform classes for example implementations.

    Additional Attributes
    ---------------------
    transform : ColumnTransform object or None
        Indicates whether the expression node is a transform node. If this is
        None, then the node is an ordinary sub-expression; otherwise, it is a
        transform node that applies the specified transformation to all child
        expressions.
    children : list(Expression objects)
        List of Expression nodes that represent children or sub-expressions of
        the node.
    n_children : int
        Number of immediate child nodes or subexpressions. A value of 0
        indicates that the node is a leaf node.
    data : DataFrame or list
        Stores a working list of DataFrames computed by child nodes (initially
        None); when the parse operation is called, each child node computes
        a DataFrame that is stored in this attribute, and then the parent node
        concatenates all DataFrames produced by child nodes and stores it
        here.
    """
    def __init__(self, expr, transforms=None):
        self.transform = None
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
            self._set_transform_node()
        self.purge()
        self.n_children = len(self.children)

    def parse(self, df, unscramble=False):
        """
        Parse the expression for a provided DataFrame.

        For leaf nodes, the single column represented by the node is extracted
        from the DataFrame. For parent nodes, the parse function is called
        recursively for all children and the outputs received are composed
        into a new DataFrame. For transform nodes, the outputs thus received
        are then transformed.

        Parameters
        ----------
        df : DataFrame
            DataFrame containing the variables to be selected, transformed,
            and composed to reflect the specified expression.
        unscramble : bool (default False)
            Indicates that the columns of the DataFrame assembled from all
            children should be reordered to match the order found in the
            input DataFrame. This is false by default, unless the expression's
            built-in __call__ method is used instead, in which case it is true
            for the parent expression and false for all children.

        Returns
        -------
        self.data : DataFrame
            New DataFrame comprising the columns specified in the expression,
            selected, transformed, and composed from columns in the input
            DataFrame.
        """
        self.purge()
        if self.n_children == 0:
            self.data = successive_pad_search(df, self.expr, pad=0, k=5)
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

    def _set_transform_node(self):
        """
        Set the expression node to be a transform node and specify the
        transform being performed.
        """
        for t in self.transforms:
            if re.search(t.all, self.expr) or re.search(t.select, self.expr):
                self.transform = t
                self._transform_arg_as_child()
                return
        self._transform_arg_as_child()

    def purge(self):
        """
        Reset internal data references. Clear the data attribute of any
        previously computed models in preparation for re-parsing for a new
        input.
        """
        self.data = [None for _ in self.children]

    def is_parenthetical(self, expr):
        """
        Return true if an expression is bounded by parentheses.
        """
        return (expr[0] == '(' and expr[-1] == ')')

    def _transform_arg_as_child(self):
        """
        If the current node is a transform node, create a new child node
        containing the argument of the transform. This function is also
        called for parenthetical sub-expressions.
        """
        grouping_depth = 0
        for i, char in enumerate(self.expr):
            if char == '(':
                if grouping_depth == 0:
                    expr_delimiter = i + 1
                grouping_depth += 1
            elif char == ')':
                grouping_depth -= 1
                if grouping_depth == 0:
                    self.children = [Expression(self.expr[expr_delimiter:i],
                                                self.transforms)]
                    return

    def _unscramble_regressor_columns(self, df):
        """
        Reorder the columns of the output DataFrame such that they are in
        the same order as the input data with any transformed columns inserted
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
