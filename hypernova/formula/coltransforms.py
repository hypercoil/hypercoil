# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Transforms
~~~~~~~~~~
Transforms applied to columns of a pandas DataFrame.
"""
import pandas as pd
from collections import OrderedDict
from .match import AllOrders, SelectOrder


class ColumnTransform(object):
    """
    Generic transformation applied column-wise to a DataFrame. Used to enhance
    hypernova.data.Expression and enable its parse tree to support additional
    transformations.

    Parameters
    ----------
    transform : callable(DataFrame, args)
        A callable that implements the transform to be applied column-wise.
        Its first argument is a DataFrame that contains the columns to be
        transformed; additional arguments correspond to additional groups
        matched and parsed from an expression. It returns a DataFrame whose
        columns have been transformed by the specified function.
    matches : list(MatchRule)
        List of `MatchRule` objects containing regular expressions paired with
        rules for extracting arguments to the `transform` from any matches to
        those regular expressions.
    name : str
        Name of the transform being applied. This determines the names of the
        transformed columns/series, which take the form
        `<variable name>_<transform name><`argform` string>`
    """
    def __init__(self, transform, matches, name='transform'):
        self.transform = transform
        self.matches = matches
        self.name = name

    def parse_expr(self, expr):
        """
        Search a formula string for the regular expressions indicating that
        the transform is to be applied (stored in the `all` and `select`
        attributes). If any matches are identified, extract any additional
        arguments to the transform.
        """
        for match in self.matches:
            parsed = match(expr)
            if parsed is not None:
                return parsed

    def argform(self, **args):
        """
        Additional string to append to names of output columns, to disambiguate
        when the same transform is applied with different arguments to a single
        DataFrame.
        """
        return ''

    def __call__(self, children, **args):
        """
        Apply the transform to a DataFrame/DataFrames.

        Parameters
        ----------
        order : set(int)
            Set of transform orders to be applied.
        children : list(DataFrame)
            DataFrames containing the variables to be transformed. The default
            implementation assumes and transforms a single child.
        """
        selected = children[0]
        data_xfm = pd.DataFrame(
            data=self.transform(selected, **args),
            columns=[f'{v}_{self.name}{self.argform(**args)}'
                     for v in selected.columns])
        return data_xfm

    def __repr__(self):
        return f'{type(self).__name__}()'


class OrderedTransform(ColumnTransform):
    """
    Ordered extension of a transformation applied column-wise to a DataFrame.
    Used to enhance hypernova.data.Expression and enable its parse tree to
    support additional transformations.

    The transformation is a function, which can also have an order. The
    transformation order is an integer that can represent some property of
    the transformation, most naturally the number of times the transformation
    is applied. An nth order transformation of this type is a composition of
    n functions all of that same type. For instance, if the transformation is
    elementwise multiplication by the column vector, then the kth order could
    correspond to the kth power.

    Note, however, that the OrderedTransform object does not automatically
    assume this meaning of order (and therefore does not recursively call the
    same transformation until the requested order is attained). Instead, it is
    expected that the transform function takes two arguments corresponding to
    the columns to be transformed and the order of the transformation. The
    exact implementation of a transformation of a particular order is left to
    the transform function itself. Thus, the exact meaning of order depends on
    the implementing function, and it is best to consult the function's
    documentation.

    Parameters/Attributes
    ---------------------
    transform : callable(DataFrame, int)
        A callable that implements the transform to be applied column-wise.
        Its first argument is a DataFrame that contains the columns to be
        transformed, and its second argument is an integer that indicates the
        order of the transform. It returns a DataFrame of the same shape as
        the input whose elements have been transformed by the specified order
        of function.
    all : str
        String representing the regular expression that indicates that all
        orders of the transform beginning from `first` until a specified order
        are to be applied and concatenated. The regular expression must
        contain the symbols used to represent the transform as well as the
        exact parenthetic string `(?P<order>[0-9]+)` in the position where the
        transform order is specified. Alongside the `select` argument, this is
        the regular expression that will be matched in any Expressions using
        this transform. Consult `PowerTransform` and `DerivativeTransform` for
        examples.
    select : str
        String representing the regular expression that indicates that a
        selected range of orders of the transform denoted `<begin>-<end>` are
        to be applied and concatenated. The regular expression must contain
        the symbols used to represent the transform as well as the exact
        parenthetic string `(?P<order>[0-9]+[\-]?[0-9]*)` in the position where
        the transform order is specified. Alongside the `select` argument, this
        is the regular expression that will be matched in any Expressions using
        this transform. Consult `PowerTransform` and `DerivativeTransform` for
        examples.
    name : str
        Name of the transform being applied. This determines the names of the
        transformed columns/series, which take the form
        `<variable name>_<transform name><transform order>`
    first : int (default 0)
        Integer denoting the smallest possible transformation order for an
        ordered transform. For instance, a transformation representing
        derivatives could begin with the zeroth derivative, corresponding to
        identity. (Including an identity transformation supports easier
        notation if the transform is to be used in a model specification.)
    identity : int or None (default None)
        Integer denoting the transformation order that corresponds to an
        identity transform. This could often be the same as `first`.
    """
    def __init__(self, transform, all, select,
                 name='transform', first=0, identity=None):
        self.identity = identity
        matches = [
            AllOrders(regex=all, first=first),
            SelectOrder(regex=select)
        ]
        super(OrderedTransform, self).__init__(
            transform=transform,
            name=name,
            matches=matches
        )

    def argform(self, order, **args):
        return order

    def __call__(self, children, order, **args):
        """
        Apply the transform of a particular set of orders to specified columns
        of a DataFrame.

        Parameters
        ----------
        children : DataFrame
            DataFrames containing the variables to be transformed. The default
            implementation assumes and transforms a single child.
        order : set(int)
            Set of transform orders to be applied.
        """
        order = order.copy()
        data_xfm = OrderedDict()
        selected = children[0]
        if self.identity in order:
            data_xfm[self.identity] = selected
            order -= {self.identity}
        for o in order:
            data_xfm[o] = super(OrderedTransform, self).__call__(
                children=children,
                order=o
            )
        data_xfm = pd.concat(
            data_xfm.values(), axis=1
        )
        return data_xfm


class NoOrderCallable():
    """
    Wrap a one-argument function to create a callable that is compatible with
    ColumnTransform. This will result in any order argument being ignored.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, arg, order):
        return self.transform(arg)


class IteratedOrderCallable():
    """
    Wrap a one-argument function to create a callable that is compatible with
    ColumnTransform. The order will specify how many times the function is
    iteratively applied. This is not as efficient as a function that caches
    previous results.
    """
    # TODO: use dynamic programming
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, arg, order):
        a = arg
        for _ in range(order):
            a = self.transform(a)
        return a
