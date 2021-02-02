# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Transforms
~~~~~~~~~~
Transforms applied to columns of a pandas DataFrame.
"""
import re
import numpy as np
import pandas as pd
from collections import OrderedDict


class MatchRule(object):
    def __init__(self, regex, rule, typedict=None):
        self.regex = re.compile(regex)
        self.rule = rule
        self.typedict = typedict or {}

    def _cast_all(self, dict):
        for k, v in dict.items():
            t = self.typedict.get(k)
            if t:
                dict[k] = t(v)
        return dict

    def __call__(self, dict):
        return self.rule(self._cast_all(dict))


class MatchOnly(MatchRule):
    def __init__(self, regex, typedict=None):
        rule = lambda x: x
        super(MatchOnly, self).__init__(
            regex=regex, rule=rule, typedict=typedict)


class AllOrders(MatchRule):
    def __init__(self, regex, first=0):
        self.first = first
        rule = lambda x: x.update(order=set(range(self.first, x['order'] + 1)))
        typedict = {
            'order': int
        }
        super(AllOrders, self).__init__(
            regex=regex,
            rule=rule,
            typedict=typedict
        )


class SelectOrder(MatchRule):
    def __init__(self, regex):
        rule = lambda x: x.update(order=set(self._order_as_range(x['order'])))
        super(SelectOrder, self).__init__(regex=regex, rule=rule)

    def _order_as_range(self, order):
        """
        Convert a hyphenated string representing order of the column transform
        (if applicable) into a range object that can be passed as input to the
        appropriate expansion function. For instance, the string `3-5` will be
        converted to range(3, 6).
        """
        order = order.split('-')
        order = [int(o) for o in order]
        if len(order) > 1:
            order = range(order[0], (order[-1] + 1))
        return order


class ColumnTransform(object):
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
            m = re.match(match.regex, expr)
            if m:
                parsed = m.groupdict()
                match(parsed)
                return parsed

    def argform(self, **args):
        return None

    def __call__(self, children, **args):
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
    Generic transformation applied column-wise to a DataFrame. Used to enhance
    hypernova.data.Expression and enable its parse tree to support additional
    transformations.

    The transformation is a function, which can also have an order. The
    transformation order is an integer that can represent some property of
    the transformation, most naturally the number of times the transformation
    is applied. An nth order transformation of this type is a composition of
    n functions all of that same type. For instance, if the transformation is
    elementwise multiplication by the column vector, then the kth order could
    correspond to the kth power.

    Note, however, that the ColumnTransform object does not automatically
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
        exact parenthetic string `([0-9]+)` in the position where the
        transform order is specified. Alongside the `select` argument, this is
        the regular expression that will be matched in any Expressions using
        this transform. Consult `PowerTransform` and `DerivativeTransform` for
        examples.
    select : str
        String representing the regular expression that indicates that a
        selected range of orders of the transform denoted `<begin>-<end>` are
        to be applied and concatenated. The regular expression must contain
        the symbols used to represent the transform as well as the exact
        parenthetic string `([0-9]+[\-]?[0-9]*)` in the position where the
        transform order is specified. Alongside the `select` argument, this is
        the regular expression that will be matched in any Expressions using
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
        order : set(int)
            Set of transform orders to be applied.
        variables : list(str)
            List of names of the variables to be transformed.
        data : DataFrame
            DataFrame containing the variables to be transformed.
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
