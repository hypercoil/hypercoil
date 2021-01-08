# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Variables
~~~~~~~~~
Dataset variable classes.
"""
from abc import ABC, abstractmethod
from .transforms import (
    IdentityTransform, EncodeOneHot, ToTensor,
    ReadNiftiTensorBlock, ReadTableTensorBlock
)


def get_col(df, label):
    try:
        return df.index.get_level_values(label)
    except KeyError:
        return df[label]


class DatasetVariable(ABC):
    def __init__(self, name='data'):
        self.name = name
        self.assignment = None
        self.transform = IdentityTransform()

    @abstractmethod
    def assign(self, arg):
        pass

    def __call__(self):
        return {self.name: self.transform(self.assignment)}

    def __repr__(self):
        s = f'{self.name}={type(self).__name__}('
        s += f'assigned={self.assignment is not None}, '
        s += f'transform={type(self.transform).__name__}'
        s += ')'
        return s


class CategoricalVariable(DatasetVariable):
    def __init__(self, name, df):
        super(CategoricalVariable, self).__init__(name)
        values = get_col(df, name).unique()
        self.max_label = len(values)
        self.label_dict = dict(zip(values, range(self.max_label)))
        self.reverse_dict = {v: k for k, v in self.label_dict.items()}
        self.transform = EncodeOneHot(n_levels=self.max_label)

    def assign(self, df):
        values = get_col(df, self.name)
        self.assignment = [self.label_dict[v] for v in values]


class ContinuousVariable(DatasetVariable):
    def __init__(self, name, df):
        super(ContinuousVariable, self).__init__(name)
        self.transform = ToTensor()

    def assign(self, df):
        self.assignment = get_col(df, self.name)


class NiftiBlockVariable(DatasetVariable):
    def __init__(self, name, levels=None):
        super(NiftiBlockVariable, self).__init__(name)
        self.transform = ReadNiftiTensorBlock(names=levels)

    def assign(self, df):
        self.assignment = get_col(df, self.name).values.tolist()


class TableBlockVariable(DatasetVariable):
    def __init__(self, name, spec=None, levels=None):
        super(TableBlockVariable, self).__init__(name)
        self.transform = ReadTableTensorBlock(names=levels, spec=spec)

    def assign(self, df):
        self.assignment = get_col(df, self.name).values.tolist()


class VariableInitialiser(object):
    def __init__(self, var, **params):
        self.var = var
        self.params = params

    def __call__(self, name, levels=None):
        return self.var(name=name, levels=levels, **self.params)
