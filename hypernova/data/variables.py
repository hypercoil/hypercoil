# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Variables
~~~~~~~~~
Dataset variable subclasses.
"""
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from .transforms import (
    Compose, IdentityTransform, EncodeOneHot, ToTensor,
    ApplyModelSpecsX, ApplyTransform, BlockTransform,
    UnzipTransformedBlock, ConsolidateBlock, ToTensorX,
    ReadDataFrameX, ReadNeuroImageX, DumpX
)


def get_col(df, label):
    try:
        return df.index.get_level_values(label)
    except KeyError:
        return df[label]


class VariableFactory(object):
    def __init__(self, var, **params):
        self.var = var
        self.params = params

    def __getattr__(self, key):
        value = self.params.get(key)
        if value is None:
            raise AttributeError(f'Invalid attribute: {key}')
        return value

    def __call__(self, **params):
        return self.var(**params, **self.params)


class VariableFactoryFactory(VariableFactory):
    def __call__(self, **params):
        return VariableFactory(self.var, **self.params, **params)


class DatasetVariable(ABC):
    def __init__(self, name='data'):
        self.name = name
        self.assignment = None
        self.transform = IdentityTransform()

    @abstractmethod
    def assign(self, arg):
        pass

    def copy(self):
        return deepcopy(self)

    def purge(self):
        pass

    def __call__(self):
        value = self.transform(self.assignment)
        self.purge()
        try:
            [a.purge() for a in self.assignment]
        except AttributeError:
            pass
        if isinstance(value, dict):
            return value
        return {self.name: value}

    def __repr__(self):
        s = f'{self.name}={type(self).__name__}('
        s += f'assigned={self.assignment is not None}, '
        if not isinstance(self.transform, IdentityTransform):
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


class NeuroImageBlockVariable(DatasetVariable):
    def __init__(self, name):
        super(NeuroImageBlockVariable, self).__init__(name)
        self.transform = Compose([
            BlockTransform(Compose([
                ReadNeuroImageX(),
                ToTensorX(),
                DumpX()
            ])),
            ConsolidateBlock()
        ])

    def assign(self, df):
        self.assignment = get_col(df, self.name).values.tolist()


class TableBlockVariable(DatasetVariable):
    def __init__(self, name, spec=None):
        super(TableBlockVariable, self).__init__(name)
        self.transform = Compose([
            BlockTransform(Compose([
                ReadDataFrameX(),
                ApplyModelSpecsX(spec),
                DumpX()
            ])),
            UnzipTransformedBlock(),
            ApplyTransform(Compose([
                BlockTransform(ToTensor(dim=2)),
                ConsolidateBlock()
            ]))
        ])

    def assign(self, df):
        self.assignment = get_col(df, self.name).values.tolist()


class DataObjectVariable(DatasetVariable):
    def __init__(self, name, regex=None, metadata=None,
                 metadata_global=None,
                 metadata_local=None):
        super(DataObjectVariable, self).__init__(name)
        metadata_xfm = metadata_global or IdentityTransform()
        self.regex = regex
        self._metadata = metadata_xfm(metadata) or {}
        self.metadata_local = metadata_local

    @property
    def data(self):
        return self.assignment['data']

    @property
    def metadata(self):
        return self.assignment['metadata']

    def assign(self, data):
        self.assignment = {'data': data}
        self.origin = {'data': deepcopy(data)}
        if self.metadata_local:
            metadata = self.metadata_local(data)
            self._metadata.update(metadata)
        self.assignment['metadata'] = self._metadata
        self.origin['metadata'] = deepcopy(self._metadata)

    def update(self, data=None, metadata=None):
        if data is not None:
            self.assignment['data'] = data
        if metadata is not None:
            self.assignment['metadata'].update(metadata)

    def purge(self):
        self.assignment = deepcopy(self.origin)


class DataPathVariable(DataObjectVariable):
    def __init__(self, name, regex=None, metadata=None,
                 metadata_global=None,
                 metadata_local=None):
        super(DataPathVariable, self).__init__(
            name, regex=regex, metadata=metadata,
            metadata_global=metadata_global,
            metadata_local=metadata_local)
        self.attributes = {}

    def __getattr__(self, key):
        value = self.attributes.get(key)
        if value is None:
            raise AttributeError(f'Invalid attribute: {key}')
        return value

    def assign(self, path):
        self.pathobj = path
        self.path = str(path)
        self.parse_path()
        super(DataPathVariable, self).assign(self.path)

    def parse_path(self):
        vals = {k: re.match(v, self.path) for k, v in self.regex.items()}
        vals = {k: v.groupdict()[k] for k, v in vals.items() if v is not None}
        for k, v in vals.items():
            try:
                self.attributes[k] = int(v)
            except ValueError:
                self.attributes[k] = v

    def get(self, key):
        try:
            return self.attributes.get(int(key))
        except ValueError:
            return self.attributes.get(key)
