# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces between neuroimaging data and data loaders.
"""
import pandas as pd
from itertools import product
from collections import ChainMap, OrderedDict
from .variables import (
    VariableFactory,
    CategoricalVariable,
    ContinuousVariable
)


class DataReference(object):
    """
    A reference to a single observation of multivariate data, or to a single
    level of observations of multivariate data.

    If this is a reference to a level of observations, those observations (or
    sub-levels of observations) can be further packaged recursively into
    ``DataReference`` objects and assigned to the ``subrefs`` field. This
    enables a parent ``DataLoader`` to traverse the dataset at any depth. See
    :doc:`fMRIDataReference <hypercoil.data.neuro.fMRIDataReference>`
    for an example implementation of this and other functionality.

    Parameters
    ----------
    data : ``DataFrame``
        ``pandas DataFrame`` containing all assignments to the variables of
        the ``DataReference``. The name of each variable provided as an
        argument to ``DataReference`` should be present as a column name in
        this ``DataFrame``.
    idx : tuple or None
        Index names corresponding to the observation(s) to be included in this
        ``DataReference`` from the provided ``DataFrame``. If this is None,
        then it is assumed that all observations in the provided DataFrame are
        to be included.
    level_names : iterable
        Names of any observation levels packaged together into this
        ``DataReference``. Each entry should also be an index column name in
        ``data``; furthermore, each index column whose corresponding ``idx``
        entry is a slice should be included here. In the future, this might be
        computed automatically.
    variables : list(:doc:`VariableFactory <hypercoil.data.variables.VariableFactory>`)
        List of ``VariableFactory`` objects that, when called, produce a
        variable of each type in the ``DataReference``. The use of factories
        rather than the variables themselves enables them to be passed
        recursively to any subreferences with ease and without overwriting the
        parent variable at assignment.
    labels : list(:doc:`VariableFactory <hypercoil.data.variables.VariableFactory>`)
        List of ``VariableFactory`` objects that, when called, produce a
        ``CategoricalVariable`` object for each categorical outcome variable
        in the ``DataReference.`` The use of factories rather than the
        variables themselves enables them to be passed recursively to any
        subreferences with ease and without overwriting the parent variable at
        assignment.
    outcomes : list(:doc:`VariableFactory <hypercoil.data.variables.VariableFactory>`)
        List of ``VariableFactory`` objects that, when called, produce a
        ``ContinuousVariable`` object for each continuous outcome variable in
        the ``DataReference``. The use of factories rather than the variables
        themselves enables them to be passed recursively to any subreferences
        with ease and without overwriting the parent variable at assignment.

    Attributes
    ----------
    vfactory : list(:doc:`VariableFactory <hypercoil.data.variables.VariableFactory>`)
        The factories passed to the ``variables`` argument at initialisation.
    lfactory : list(:doc:`VariableFactory <hypercoil.data.variables.VariableFactory>`)
        The factories passed to the ``labels`` argument at initialisation.
    ofactory : list(:doc:`VariableFactory <hypercoil.data.variables.VariableFactory>`)
        The factories passed to the ``outcomes`` argument at initialisation.
    variables : list(:doc:`DatasetVariable <hypercoil.data.variables.DatasetVariable>`)
        List of variable objects produced from calls to the factories
        provided at initialisation and assigned the corresponding fields of
        the input ``DataFrame``. If the references were produced from a call
        to ``data_references`` or a parent method, then each variable thus
        produced will result from a query to the dataset filesystem directory
        layout.
    labels : list(:doc:`DatasetVariable <hypercoil.data.variables.DatasetVariable>`)
        List of ``CategoricalVariable`` objects produced from calls to
        factories provided at initialisation and assigned the corresponding
        fields of the input ``DataFrame``. Each corresponds to a specified
        categorical outcome variable.
    outcomes : list(:doc:`DatasetVariable <hypercoil.data.variables.DatasetVariable>`)
        List of ``ContinuousVariable`` objects produced from calls to
        factories provided at initialisation and assigned the corresponding
        fields of the input ``DataFrame``. Each corresponds to a specified
        continuous outcome variable.
    ids : dict
        Dictionary of identifiers for the current data reference.

    Notes
    -----
        Any variable name or identifier can additionally be accessed as an
        attribute of ``DataReference``. This will return the assigned value of
        the fully transformed variable.
    """
    def __init__(self, data, idx=None, level_names=None,
                 variables=None, labels=None, outcomes=None):
        if idx is None:
            self.df = data
        else:
            self.df = data.loc(axis=0)[self._cast_loc(idx)]

        self.vfactory = variables
        self.lfactory = labels
        self.ofactory = outcomes

        self.variables = [v() for v in variables]
        self.labels = [l() for l in labels]
        self.outcomes = [o() for o in outcomes]
        for var in (self.variables + self.outcomes + self.labels):
            var.assign(self.df)

        if level_names:
            self.level_names = [tuple(level_names)]
        else:
            self.level_names = []

        self.ids = self.parse_ids(idx)

    def _cast_loc(self, loc):
        """
        Cast a slicing index (argument to `loc`) to a type that will guarantee
        the output of the `loc` call is a DataFrame (rather than e.g., a
        Series). This operation assumes that every entry in the argument is
        either hashable or a slice; if no entries are slices, the argument is
        is nested in a list.
        """
        if any([isinstance(l, slice) for l in loc]):
            return tuple(loc)
        else:
            return [tuple(loc)]

    def parse_ids(self, values):
        """
        Populate the values of any identifier variables of the
        ``DataReference``.

        Parameters
        ----------
        values : list
            List of values of identifier variables, ordered identically to the
            variables in the ``DataReference`` index.

        Returns
        -------
        ids : dict
            Dictionary mapping identifier variable names to the values they
            are assigned for this ``DataReference``.
        """
        ids = OrderedDict()
        for name, value in zip(self.df.index.names, values):
            if not isinstance(value, slice):
                ids[name] = value
        return ids

    def get_var(self, name):
        """
        Get the variable with the specified name.
        """
        for var in (self.variables + self.outcomes + self.labels):
            if var.name == name: return var

    def __getattr__(self, key):
        """
        If a requested attribute is not present in the object, see if there is
        a variable by that name. If so, return the fully transformed
        assignment to that variable. If the variable's assignment is a
        dictionary with multiple values, then return the entire dictionary;
        otherwise, return only the requested value.
        """
        var = self.get_var(key)
        if var is None:
            out = self.ids.get(key)
            if out is None:
                raise AttributeError(f'Invalid variable: {key}')
            return out
        out = var()
        try:
            return out[var.name]
        except KeyError:
            return out

    @property
    def label(self):
        """
        Fully transformed assignments to all categorical outcome variables.

        Assignments are returned as a dictionary of key-value pairs specifying
        the name of each variable and its fully transformed assignment.
        """
        asgt = [var() for var in self.labels]
        return dict(ChainMap(*asgt))

    @property
    def outcome(self):
        """
        Fully transformed assignments to all continuous outcome variables.

        Assignments are returned as a dictionary of key-value pairs specifying
        the name of each variable and its fully transformed assignment.
        """
        asgt = [var() for var in self.outcomes]
        return dict(ChainMap(*asgt))

    @property
    def data(self):
        """
        Fully transformed assignments to all data variables obtained via
        direct query of the parent dataset.

        Assignments are returned as a dictionary of key-value pairs specifying
        the name of each variable and its fully transformed assignment.
        """
        asgt = [var() for var in self.variables]
        return dict(ChainMap(*asgt))

    def __call__(self):
        """
        Obtain the fully transformed assignments to all variables of the
        ``DataReference``.

        Assignments are returned as a dictionary of key-value pairs specifying
        the name of each variable and its fully transformed assignment.
        """
        asgt = [v() for v in (self.variables + self.outcomes + self.labels)]
        return dict(ChainMap(*asgt))


class DataQuery(object):
    """
    A query to a layout or grabber object representation of a filesystem
    directory containing a dataset.

    Parameters
    ----------
    name : str
        Name of the data object returned by the query.
    pattern : str or None
        Pattern within the search directory that all query results should
        conform to. If this is provided for all queries, then all subsequent
        searches will be substantially narrowed and accelerated.
    variables : dict or :doc:`VariableFactoryFactory <hypercoil.data.variables.VariableFactoryFactory>`
        A ``VariableFactoryFactory`` object whose child factory produces a
        ``DatasetVariable`` of a type suitable for storing the paths returned
        by the query, or a dictionary of such objects.
    transform : transform object or None
        Currently *this does nothing*. Eventually it should override the
        variable's default transform. For now, just write a new variable
        class if the existing ones don't have suitable transforms.
    filters : dict
        Additional keyword arguments to the query are used to filter the
        dataset to return only those data objects matching the query.
    """
    ##TODO: figure out what, if anything, the template argument is doing.
    # Remove arguments that do nothing.
    def __init__(self, name='data', pattern=None, variables=None,
                 transform=None, template=None, **filters):
        if transform is not None:
            raise NotImplementedError(
                'Overriding default transforms is not yet implemented. '
                'Create a new variable class instead.')
        self.name = name
        self.pattern = pattern
        self.variables = variables
        self.transform = transform
        self.template = template
        self.filters = filters

    def __call__(self, layout, **params):
        """
        Submit the query to a filesystem layout.

        Parameters
        ----------
        layout : object
            Object representing a filesystem directory containing a dataset.
            It must implement a ``get`` method that returns all directory
            entries that match a set of filters.
        params : dict
            Additional keyword arguments to the query: these are combined with
            the filters provided at initialisation to form a query. If there is
            a name collision, then filters provided at ``__call__`` take
            precedence.

        Returns
        -------
        result : list(:doc:`DatasetVariable <hypercoil.data.variables.DatasetVariable>`)
            List of data objects in the layout matching the query.
        """
        #TODO: Note that this doesn't at all consider the `pattern` field
        # of the query
        filters = {}
        filters.update(self.filters)
        filters.update(params)
        return layout.get(**filters)

    def variable_factory(self):
        """
        Return the variable factory with the specified identifier levels
        assigned.
        """
        if isinstance(self.variables, dict):
            vars = []
            for k, v in self.variables.items():
                vars += [v(name=k, colname=self.name)]
        else:
            vars = [self.variables(name=self.name)]
        #TODO: This doesn't work unless we have variable factories forward
        # transforms.
        #if self.transform is not None:
        #    var.transform = self.transform
        return vars


def data_references(data_dir, layout, reference, labels, outcomes,
                    observations, levels, queries, filters=None,
                    additional_tables=None, ignore=None):
    """
    Obtain data references for a specified directory.

    Parameters
    ----------
    data_dir : str
        Path to the top-level directory containing all data files.
    layout : object
        Object representing a dataset layout. It must implement the following
        methods:

        - ``getall(i)``: returns a list of all values of ``i`` present in the
                         dataset
        - ``get``: queries the dataset for matching entities
    reference : :doc:`DataReference <hypercoil.data.dataref.DataReference>` class
        Class or subclass of ``DataReference`` to produce as output.
    labels : tuple or None (default ``('subject',)``)
        Names of categorical outcome variables to include in data references.
        These variables can be taken either from data identifiers or from
        additional tables. Labels become available as prediction targets for
        classification models.
    outcomes : tuple or None (default None)
        Names of continuous outcome variables to include in data references.
        These variables can be taken either from data identifiers or from
        additional tables. Outcomes become available as prediction targets for
        regression models.
    observations : tuple (default ``('subject',)``)
        List of data identifiers whose levels are packaged into separate data
        references. Each level should generally have the same values of any
        outcome variables.
    levels : tuple or None (default ``('session', 'run, task')``)
        List of data identifiers whose levels are packaged as sublevels of the
        same data reference. This permits easier augmentation of data via
        pooling across sublevels.
    queries : list(:doc:`DataQuery <hypercoil.data.dataref.DataQuery>`)
        Data queries used to locate files in the dataset. Each entry locates a
        different file for each unique identifier combination.
    filters : dict
        Filters to select data objects in the layout.
    additional_tables : list(str) or None (default None)
        List of paths to files containing additional data. Each file should
        include index columns corresponding to all identifiers present in the
        dataset (e.g., subject, run, etc.).
    ignore : dict(str: list) or None (default None)
        Dictionary indicating identifiers to be ignored. Currently this
        doesn't support any logical composition and takes logical OR over all
        ignore specifications. In other words, data will be ignored so long as
        they satisfy any of the ``ignore`` criteria.

    Returns
    -------
    data_refs : list(:doc:`DataReference <hypercoil.data.dataref.DataReference>`)
        List of data reference objects created from files found in the input
        directory.

    See also
    --------
    fmriprep_references:
        :doc:`Implementation on BIDS-formatted data
        <hypercoil.data.bids.fmriprep_references>`.
    hcp_references:
        :doc:`Implementation on HCP-formatted data
        <hypercoil.data.hcp.hcp_references>`.
    """
    #TODO
    # Why is data_dir an input parameter again? No good reason, other than a
    # bunch of calls to this already have it set as the first argument. Drop
    # it and refactor before it becomes even more of a nuisance.
    labels = labels or []
    outcomes = outcomes or []
    filters = filters or {}
    df = layout.dataset(observations, levels,
                        ignore=ignore, queries=queries,
                        additional_tables=additional_tables,
                        **filters)
    observations = [o for o in observations if o in df.index.names]
    levels = [l for l in levels if l in df.index.names]
    obs = all_observations(df.index, observations, levels)
    labels, outcomes = process_labels_and_outcomes(df, labels, outcomes)
    variables = process_variables(queries)
    data_refs = make_references(reference, df, obs, levels,
                                variables, labels, outcomes)
    return data_refs


# TODO: We need to refactor the below functions away.
def all_observations(index, observations, levels):
    """
    Obtain all extant observation identifier combinations in a DataFrame
    MultiIndex.
    """
    return _all_combinations(index, combine=observations, control=levels)


def _all_combinations(index, combine, control):
    """
    Obtain slices into all subsets of a DataFrame that share common MultiIndex
    subset assignments. Abstract function called by `all_levels` and
    `all_observations`. This function only obtains slicing tuples; it does not
    index the DataFrame to extract them.

    Parameters
    ----------
    index : MultiIndex
        Pandas MultiIndex from which all salient combinations are sliced.
    combine : list(str)
        Indexing variables to combine when creating each slice.
    control : list(str)
        Indexing variables to ignore when creating each slice.

    Returns
    -------
    list(tuple)
        List of indices into all unique MultiIndex assignments of `combine`
        indexing variables.
    """
    idx = []
    gen = []
    for name, c in zip(index.names, index.levels):
        if name in combine:
            idx += [None]
            gen += [list(c)]
        elif name in control:
            idx += [slice(None)]
    combinations = product(*gen)
    return [tuple(_fill_idx_pattern(c, idx)) for c in combinations]


def _fill_idx_pattern(level, idx):
    """
    Complete the indexing pattern in `idx` using the items in `level`. Any
    instance of None in `idx` is replaced by the first unused element of
    `level`. Helper function for `_all_combinations` routines.
    """
    level = list(level)
    return [i if i is not None else level.pop(0) for i in idx]


def process_labels_and_outcomes(df, labels, outcomes):
    """
    Package categorical labels and continuous outcomes into variable objects
    that facilitate sampling.

    Parameters
    ----------
    df : DataFrame
        DataFrame storing the values of label and outcome variables.
    labels : list(str)
        List of names of discrete-valued/categorical outcome variables in the
        input DataFrame.
    outcomes : list(str)
        List of names of continuous-valued outcome variables in the input
        DataFrame.

    Returns
    -------
    labels : list(VariableFactory)
        List of CategoricalVariable factories corresponding to each provided
        label.
    outcomes : list(VariableFactory)
        List of ContinuousVariable factories corresponding to each provided
        outcome.
    """
    return (
        [VariableFactory(CategoricalVariable, name=l, df=df) for l in labels],
        [VariableFactory(ContinuousVariable, name=o, df=df) for o in outcomes]
    )


def process_variables(queries):
    """
    Prepare a variable factory for each query.

    Parameters
    ----------
    queries : list(DatasetQuery)
        List of queries.

    Returns
    -------
    factories : list(VariableFactory)
        List of variable factories corresponding to each provided query.
    """
    factories = []
    for q in queries:
        vars = q.variable_factory()
        factories += vars
    return factories


def make_references(reference, df, obs, levels, variables, labels, outcomes):
    """
    Create a data reference object for each observation in the dataset.

    Parameters
    ----------
    reference : DataReference subclass
        Subclass of DataReference to initialise for each observation.
    df : DataFrame
        DataFrame containing paths to data files and values of outcome
        variables indexed for each observation and level of the dataset.
    obs : list
        List of identifiers for each observation in the dataset.
    levels : list
        List of identifiers that distinguish separate measures of each
        observation in the dataset.
    variables : list(VariableFactory)
        List of factories for each query variable.
    labels : list(VariableFactory)
        List of factories for each categorical outcome variable.
    outcomes : list(VariableFactory)
        List of factories for each continuous outcome variable.

    Returns
    -------
    data_refs : list(DataReference)
        List containing a data reference object for each observation in the
        dataset.
    """
    data_refs = []
    for o in obs:
        try:
            data_refs += [reference(
                df, o,
                level_names=levels,
                variables=variables,
                labels=labels,
                outcomes=outcomes)]
        except KeyError:
            continue
    return data_refs
