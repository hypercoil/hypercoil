# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Transforms
~~~~~~~~~~
Transforms applied to columns of a pandas DataFrame.
"""
import re
import pandas as pd
import numpy as np
from collections import OrderedDict
from functools import reduce


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

    def _order_as_range(self, order):
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
            order -= {self.identity}
        for o in order:
            data_xfm[o] = pd.DataFrame(
                data=self.transform(selected, o),
                columns=[f'{v}_{self.name}{o}' for v in variables])
        data_xfm = pd.concat(
            data_xfm.values(), axis=1
        )
        return data_xfm

    def __repr__(self):
        return f'{type(self).__name__}()'
