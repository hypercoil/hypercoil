# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Neuroimaging
~~~~~~~~~~~~
Neuroimaging dataset-specific datasets and data references.
"""
from .dataref import DataReference


class fMRIDataReference(DataReference):
    """
    Data reference specifically for functional magnetic resonance neuroimaging
    data.

    A reference to a single observation of multivariate data, or to a single
    level of observations of multivariate data.

    fMRIDataReferences can be nested recursively. For instance, a top-level
    reference can contain sub-references to all of a subject's data, and each
    nested sub-reference can contain references to data from a single run.
    Currently a separate sub-reference is automatically created for each
    variable collection passed to the parent reference; alternative behaviours
    might be implemented in the future should sufficient use cases arise.

    A parent ReferencedDataset can sample, traverse, and split across
    DataReferences at any depth, thereby enabling (for instance) splitting at
    the subject level and sampling at the run level.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing all assignments to the variables of the
        `DataReference`. The name of each variable provided as an argument to
        `DataReference` should be present as a column name in this DataFrame.
    idx : tuple or None
        Index names corresponding to the observation(s) to be included in this
        DataReference from the provided DataFrame. If this is None, then it is
        assumed that all observations in the provided DataFrame are to be
        included.
    level_names : iterable
        Names of any observation levels packaged together into this
        `DataReference`. Each entry should also be an index column name in
        `data`; furthermore, each index column whose corresponding `idx` entry
        is a slice should be included here. In the future, this might be
        computed automatically.
    variables : list(VariableFactory)
        List of VariableFactory objects that, when called, produce a variable
        of each type in the `DataReference`. The use of factories rather than
        the variables themselves enables them to be passed recursively to any
        subreferences with ease and without overwriting the parent variable at
        assignment.
    labels : list(VariableFactory)
        List of VariableFactory objects that, when called, produce a
        CategoricalVariable object for each categorical outcome variable in
        the `DataReference.` The use of factories rather than the variables
        themselves enables them to be passed recursively to any subreferences
        with ease and without overwriting the parent variable at assignment.
    outcomes : list(VariableFactory)
        List of VariableFactory objects that, when called, produce a
        ContinuousVariable object for each continuous outcome variable in the
        `DataReference.` The use of factories rather than the variables
        themselves enables them to be passed recursively to any subreferences
        with ease and without overwriting the parent variable at assignment.

    Attributes
    ----------
    subrefs : list(fMRIDataReference)
        List of dataset sub-references.
    vfactory : list(VariableFactory)
        The factories passed to the `variables` argument at initialisation.
    lfactory : list(VariableFactory)
        The factories passed to the `labels` argument at initialisation.
    ofactory : list(VariableFactory)
        The factories passed to the `outcomes` argument at initialisation.
    variables : list(DatasetVariable)
        List of variable objects produced from calls to the factories
        provided at initialisation and assigned the corresponding fields of
        the input DataFrame. If the references were produced from a call to
        `data_references` or a parent method, then each variable thus produced
        will result from a query to the dataset filesystem directory layout.
    labels : list(DatasetVariable)
        List of CategoricalVariable objects produced from calls to factories
        provided at initialisation and assigned the corresponding fields of
        the input DataFrame. Each corresponds to a specified categorical
        outcome variable.
    outcomes : list(DatasetVariable)
        List of ContinuousVariable objects produced from calls to factories
        provided at initialisation and assigned the corresponding fields of
        the input DataFrame. Each corresponds to a specified continuous
        outcome variable.
    ids : dict
        Dictionary of identifiers for the current data reference.

    Any variable name or identifier can additionally be accessed as an
    attribute of DataReference. This will return the assigned value of the
    fully transformed variable.
    """
    def __init__(self, df, idx, level_names=None,
                 variables=None, labels=None, outcomes=None):
        super(fMRIDataReference, self).__init__(
            data=df, idx=idx, level_names=level_names,
            variables=variables, labels=labels, outcomes=outcomes)

        if self.level_names:
            self.subrefs = self._make_subreferences()
        else:
            self.subrefs = []

    def _make_subreferences(self):
        """
        Automatically create sub-references for the current data reference.
        Called at initialisation.
        """
        subrefs = []
        for idx, _ in self.df.iterrows():
            subrefs += [fMRIDataReference(
                df=self.df,
                idx=idx,
                level_names=None,
                variables=self.vfactory,
                labels=self.lfactory,
                outcomes=self.ofactory
            )]
        return subrefs

    def __repr__(self):
        s = f'{type(self).__name__}('
        s += ', '.join(f'{k}={v}' for k, v in self.ids.items())
        if not self.subrefs:
            s += ')'
            return s
        for sr in self.subrefs:
            s += f',\n {sr}'
        s += '\n)'
        return s
