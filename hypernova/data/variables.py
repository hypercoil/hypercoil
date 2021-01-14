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
from .functional import get_col
from .transforms import (
    Compose, IdentityTransform, EncodeOneHot, ToTensor,
    ApplyModelSpecsX, ApplyTransform, BlockTransform,
    UnzipTransformedBlock, ConsolidateBlock, ToTensorX,
    ReadDataFrameX, ReadNeuroImageX, DumpX
)


class VariableFactory(object):
    """
    Factory object for producing DatasetVariables.

    Parameters
    ----------
    var : DatasetVariable class
        Class of variable to produce with each call to the factory.

    Any additional parameters are used to complete a partial call to the
    variable constructor.
    """
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
    """
    Factory object for producing VariableFactories.

    Equivalent to `VariableFactory(VariableFactory(var, **params))`.

    Parameters
    ----------
    var : DatasetVariable class
        Class of variable to produce with each call to the factory.

    Any additional parameters are used to complete a partial call to the
    variable constructor.
    """
    def __call__(self, **params):
        return VariableFactory(self.var, **self.params, **params)


class DatasetVariable(ABC):
    """
    Dataset variable object.

    Parameters
    ----------
    name : hashable
        Variable name.

    Attributes
    ----------
    assignment
        Raw (pre-transformation) variable assignment. Set by the `assign`
        method.
    transform : callable
        Transform to automatically apply to the variable assignment when the
        variable is called. A suitable transform can be used as a bridge
        between filesystem-path assignments and model-ready tensor clouds.
    """
    def __init__(self, name='data'):
        self.name = name
        self.assignment = None
        self.transform = IdentityTransform()

    @abstractmethod
    def assign(self, arg):
        """
        Abstract method; subclasses must implement. Sets the raw variable
        assignment on the basis of an argument.
        """
        pass

    def copy(self):
        """
        Create a copy of the variable object with its current assignment
        frozen.
        """
        return deepcopy(self)

    def purge(self):
        """
        Subclasses whose transforms modify the variable in place can implement
        this method to reset the variable to its original assignment. Does
        nothing by default.
        """
        pass

    def __call__(self):
        """
        Return a dictionary of key-value pairs corresponding to named, fully
        transformed assignments of the variable.
        """
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
    """
    Simple categorical variable object.

    Parameters
    ----------
    name : hashable
        Variable name.
    df : DataFrame
        DataFrame containing all assignments of the variable across the entire
        dataset.

    Attributes
    ----------
    max_label : int
        Total number of levels (unique labels) of the variable.
    label_dict : dict
        Dictionary mapping unique labels to an internal integer
        representation.
    reverse_dict : dict
        Dictionary mapping an internal integer representation to each unique
        label.
    transform : EncodeOneHot
        One-to-one transformation encoding mapping each internal variable
        representation to a one-hot encoding.
    """
    def __init__(self, name, df):
        super(CategoricalVariable, self).__init__(name)
        values = get_col(df, name).unique()
        self.max_label = len(values)
        self.label_dict = dict(zip(values, range(self.max_label)))
        self.reverse_dict = {v: k for k, v in self.label_dict.items()}
        self.transform = EncodeOneHot(n_levels=self.max_label)

    def assign(self, df):
        """
        The assignment is a vector containing the entries of a DataFrame
        column sharing its name with the variable.
        """
        values = get_col(df, self.name)
        self.assignment = [self.label_dict[v] for v in values]


class ContinuousVariable(DatasetVariable):
    """
    Simple continuous variable object.

    Parameters
    ----------
    name : hashable
        Variable name.
    df : DataFrame
        Does nothing. Allowed for uniformity with `CategoricalVariable`.
    """
    def __init__(self, name, df=None):
        super(ContinuousVariable, self).__init__(name)
        self.transform = ToTensor()

    def assign(self, df):
        """
        The assignment is a vector containing the entries of a DataFrame
        column sharing its name with the variable.
        """
        self.assignment = get_col(df, self.name)


class NeuroImageBlockVariable(DatasetVariable):
    """
    Variable representation of a block of neuro-images. The assignment is
    a list block of DataPathVariables that include paths to files
    containing data and potentially metadata associated with each image.
    """
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
        """
        The assignment is a vector containing the entries of a DataFrame
        column sharing its name with the variable. The column should contain
        DataPathVariable objects.
        """
        self.assignment = get_col(df, self.name).values.tolist()


class TableBlockVariable(DatasetVariable):
    """
    Variable representation of a block of tabular data. The assignment is
    a list block of DataPathVariables that include paths to files
    containing data and potentially metadata associated with each data
    table.
    """
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
        """
        The assignment is a vector containing the entries of a DataFrame
        column sharing its name with the variable. The column should contain
        DataPathVariable objects.
        """
        self.assignment = get_col(df, self.name).values.tolist()


class DataObjectVariable(DatasetVariable):
    """
    Variable representation of a generic data object with support for a bound
    metadata sidecar.

    Parameters
    ----------
    name : hashable
        Variable name.
    metadata
        Global metadata object to include at the time of assignment.
    metadata_global : transform
        Transform to apply to the global metadata object before assignment.
    metadata_local : transform
        Transform to apply to the data provided at assignment time in order to
        read additional metadata local to the provided data object.
    """
    def __init__(self, name, metadata=None,
                 metadata_global=None,
                 metadata_local=None):
        super(DataObjectVariable, self).__init__(name)
        metadata_xfm = metadata_global or IdentityTransform()
        self._metadata = metadata_xfm(metadata) or {}
        self.metadata_local = metadata_local

    @property
    def data(self):
        """
        Assigned data block.
        """
        return self.assignment['data']

    @property
    def metadata(self):
        """
        Assigned metadata block. This should take the form of a dictionary.
        """
        return self.assignment['metadata']

    def assign(self, data):
        """
        Assign the variable the specified data, find and grab any locally
        referenced metadata, and back up the assignment to permit resets in
        the event of in-place operations.
        """
        self.assignment = {'data': data}
        self.origin = {'data': deepcopy(data)}
        if self.metadata_local:
            metadata = self.metadata_local(data)
            self._metadata.update(metadata)
        self.assignment['metadata'] = self._metadata
        self.origin['metadata'] = deepcopy(self._metadata)

    def update(self, data=None, metadata=None):
        """
        Update the variable's assigned data and/or metadata.
        """
        if data is not None:
            self.assignment['data'] = data
        if metadata is not None:
            self.assignment['metadata'].update(metadata)

    def purge(self):
        """
        Purge the effects of in-place transformations; reset the variable's
        assigned value to its original assignment.
        """
        self.assignment = deepcopy(self.origin)


class DataPathVariable(DataObjectVariable):
    """
    Variable representation of a generic data object with support for a bound
    metadata sidecar.

    Parameters
    ----------
    name : hashable
        Variable name.
    regex : dict
        Dictionary defining a mapping from attributes to regular expressions
        each containing a named subgroup capture sharing a name with the
        attribute.
    metadata
        Global metadata object to include at the time of assignment.
    metadata_global : transform
        Transform to apply to the global metadata object before assignment.
    metadata_local : transform
        Transform to apply to the data provided at assignment time in order to
        read additional metadata local to the provided data object.
    """
    def __init__(self, name, regex=None, metadata=None,
                 metadata_global=None,
                 metadata_local=None):
        super(DataPathVariable, self).__init__(
            name, metadata=metadata,
            metadata_global=metadata_global,
            metadata_local=metadata_local)
        self.regex = regex
        self.attributes = {}

    def __getattr__(self, key):
        value = self.get(key)
        if value is None:
            raise AttributeError(f'Invalid attribute: {key}')
        return value

    def assign(self, path):
        """
        Assign the variable the specified data, find and grab any locally
        referenced metadata, and back up the assignment to permit resets in
        the event of in-place operations.
        """
        self.pathobj = path
        self.path = str(path)
        self.parse_path()
        super(DataPathVariable, self).assign(self.path)

    def parse_path(self):
        """
        Parse variable attributes from the assigned path.
        """
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
