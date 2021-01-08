# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Variables
~~~~~~~~~
Dataset variable classes.
"""
from abc import ABC, abstractmethod


def get_col(df, label):
    try:
        return df.index.get_level_values(label)
    except KeyError:
        return df[label]


class DatasetVariable(ABC):
    @abstractmethod
    def __call__(self):
        return None


class CategoricalVariable(DatasetVariable):
    def __init__(self, var, df):
        values = get_col(df, var).unique()
        self.name = var
        self.max_label = len(values)
        self.label_dict = dict(zip(values, range(self.max_label)))
        self.reverse_dict = {v: k for k, v in self.label_dict.items()}

    def __call__(self, df):
        values = get_col(df, self.name)
        return [self.label_dict[v] for v in values]


class ContinuousVariable(DatasetVariable):
    def __init__(self, var, df):
        self.name = var

    def __call__(self, df):
        return get_col(df, self.name)
