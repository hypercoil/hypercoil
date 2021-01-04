# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data references
~~~~~~~~~~~~~~~
Interfaces between neuroimaging data and data loaders.
"""
import pandas as pd
from itertools import product
from .grabber import LightBIDSLayout
from .transforms import (
    ReadNiftiTensorBlock,
    ReadTableTensorBlock,
    ToTensor, EncodeOneHot
)


BIDS_SCOPE = 'derivatives'
BIDS_DTYPE = 'func'

BIDS_IMG_DESC = 'preproc'
BIDS_IMG_SUFFIX = 'bold'
BIDS_IMG_EXT = '.nii.gz'

BIDS_CONF_DESC = 'confounds'
BIDS_CONF_SUFFIX = 'timeseries'
BIDS_CONF_EXT = '.tsv'


class DataReference(object):
    """Does nothing. Yet."""
    def __init__(self):
        pass


class fMRIReferenceBase(DataReference):
    @property
    def data(self):
        return self.data_transform(self.data_ref)

    @property
    def confounds(self):
        return self.confounds_transform(self.confounds_ref)

    @property
    def label(self):
        return {k: t(v) for t, (k, v) in
                zip(self.label_transform, self.label_ref.items())}

    @property
    def outcome(self):
        return {k: t(v) for t, (k, v) in
                zip(self.outcome_transform, self.outcome_ref.items())}


class fMRISubReference(fMRIReferenceBase):
    def __init__(
        self,
        data,
        confounds=None,
        labels=None,
        outcomes = None,
        data_transform=None,
        confounds_transform=None,
        label_transform=None,
        **ids):
        self.data_ref = data
        self.data_transform = data_transform or ReadNiftiTensorBlock()
        self.confounds_ref = confounds
        self.confounds_transform = (
            confounds_transform or ReadTableTensorBlock())
        self.label_ref = labels or []
        self.outcome_ref = outcomes or []
        self.label_transform = label_transform or IdentityTransform()
        self.subject = ids.get('subject')
        self.session = ids.get('session')
        self.run = ids.get('run')
        self.task = ids.get('task')

    def __repr__(self):
        s = f'{type(self).__name__}(sub={self.subject}'
        if self.session:
            s += f', ses={self.session}'
        if self.run:
            s += f', run={self.run}'
        if self.task:
            s += f', task={self.task}'
        s += ')'
        return s


class fMRIDataReference(fMRIReferenceBase):
    def __init__(
        self,
        df,
        idx,
        labels=None,
        outcomes = None,
        data_transform=None,
        confounds_transform=None,
        label_transform=None,
        outcome_transform=None,
        level_names=None):
        if level_names is not None:
            level_names = [tuple(level_names)]
        self.df = df.loc(axis=0)[idx]
        self.labels = labels or []
        self.outcomes = outcomes or []
        self.data_transform = (data_transform or
                               ReadNiftiTensorBlock(names=level_names))
        self.confounds_transform = (confounds_transform or
                                    ReadTableTensorBlock(names=level_names))
        self.label_transform = label_transform or [
            EncodeOneHot(n_levels=l.max_label) for l in self.labels]
        self.outcome_transform = outcome_transform or [
            ToTensor() for _ in self.outcomes]

        self.data_ref = self.df.images.values.tolist()
        self.confounds_ref = self.df.confounds.values.tolist()
        self.label_ref = {l.name: l(self.df) for l in self.labels}
        self.outcome_ref = {o.name: o(self.df) for o in self.outcomes}
        self.subrefs = self.make_subreferences()

        ids = self.parse_ids(idx)
        self.subject = ids.get('subject')
        self.session = ids.get('session')
        self.run = ids.get('run')
        self.task = ids.get('task')

    def parse_ids(self, idx):
        ids = {}
        for entity, value in zip(self.df.index.names, idx):
            if not isinstance(value, slice):
                ids[entity] = value
        return ids

    def make_subreferences(self):
        subrefs = []
        for idx, ref in self.df.iterrows():
            ids = dict(zip(self.df.index.names, idx))
            subrefs += [fMRISubReference(
                data=ref.images,
                confounds=ref.confounds,
                labels=self.labels,
                outcomes=self.outcomes,
                data_transform=self.data_transform,
                confounds_transform=self.confounds_transform,
                label_transform=self.label_transform,
                **ids)
            ]
        return subrefs

    def __repr__(self):
        s = f'{type(self).__name__}(sub={self.subject}'
        if self.session:
            s += f', ses={self.session}'
        if self.run:
            s += f', run={self.run}'
        if self.task:
            s += f', task={self.task}'
        for sr in self.subrefs:
            s += f',\n {sr}'
        s += '\n)'
        return s


class CategoricalVariable(object):
    def __init__(self, var, df):
        values = get_col(df, var).unique()
        self.name = var
        self.max_label = len(values)
        self.label_dict = dict(zip(values, range(self.max_label)))
        self.reverse_dict = {v: k for k, v in self.label_dict.items()}

    def __call__(self, df):
        values = get_col(df, self.name)
        return [self.label_dict[v] for v in values]


class ContinuousVariable(object):
    def __init__(self, var, df):
        self.name = var

    def __call__(self, df):
        return get_col(df, self.name)


class DataQuery(object):
    def __init__(self, name='data', pattern=None, **filters):
        self.name = name
        self.pattern = pattern
        self.filters = filters

    def __call__(self, layout, **filters):
        return layout.get(**self.filters, **filters)


def fmriprep_references(fmriprep_dir, space=None, additional_tables=None,
                        ignore=None, labels=('subject',), outcomes=None,
                        observations=('subject',),
                        levels=('session', 'run', 'task')):
    """
    Obtain data references for a directory containing data processed with
    fMRIPrep.

    Parameters
    ----------
    fmriprep_dir : str
        Path to the top-level directory containing all neuroimaging data
        preprocessed with fMRIPrep.
    space : str or None (default None)
        String indicating the stereotaxic coordinate space from which the
        images are referenced.
    additional_tables : list(str) or None (default None)
        List of paths to files containing additional data. Each file should
        include index columns corresponding to all identifiers present in the
        dataset (e.g., subject, run, etc.).
    ignore : dict(str: list) or None (default None)
        Dictionary indicating identifiers to be ignored. Currently this
        doesn't support any logical composition and takes logical OR over all
        ignore specifications. In other words, data will be ignored if they
        satisfy any of the ignore criteria.
    labels : tuple or None (default ('subject',))
        List of categorical outcome variables to include in data references.
        These variables can be taken either from data identifiers or from
        additional tables. Labels become available as prediction targets for
        classification models. By default, the subject identifier is included.
    outcomes : tuple or None (default None)
        List of continuous outcome variables to include in data references.
        These variables can be taken either from data identifiers or from
        additional tables. Labels become available as prediction targets for
        regression models. By default, the subject identifier is included.
    observations : tuple (default ('subject',))
        List of data identifiers whose levels are packaged into separate data
        references. Each level should generally have the same values of any
        outcome variables.
    levels : tuple or None (default ('session', 'run, task'))
        List of data identifiers whose levels are packaged as sublevels of the
        same data reference. This permits easier augmentation of data via
        pooling across sublevels.

    Returns
    -------
    data_refs : list(fMRIPrepDataReference)
        List of data reference objects created from files found in the input
        directory.
    """
    #layout = bids.BIDSLayout(
    #    fmriprep_dir,
    #    derivatives=[fmriprep_dir],
    #    validate=False)
    images = DataQuery(
        name='images',
        pattern='func/**/*preproc*.nii.gz',
        scope=BIDS_SCOPE,
        datatype=BIDS_DTYPE,
        desc=BIDS_IMG_DESC,
        suffix=BIDS_IMG_SUFFIX,
        extension=BIDS_IMG_EXT)
    confounds = DataQuery(
        name='confounds',
        pattern='func/**/*confounds*.tsv',
        scope=BIDS_SCOPE,
        datatype=BIDS_DTYPE,
        desc=BIDS_CONF_DESC,
        suffix=BIDS_CONF_SUFFIX,
        extension=BIDS_CONF_EXT)
    layout = LightBIDSLayout(
        fmriprep_dir,
        queries=[images, confounds])
    return data_references(
        data_dir=fmriprep_dir,
        layout=layout,
        labels=labels,
        outcomes=outcomes,
        observations=observations,
        levels=levels,
        queries=[images, confounds],
        filters={'space': space},
        additional_tables=additional_tables,
        ignore=ignore
    )


def data_references(data_dir, layout, labels, outcomes, observations, levels,
                    queries=None, filters=None, additional_tables=None,
                    ignore=None):
    """
    Obtain data references for a specified directory.

    Parameters
    ----------
    data_dir : str
        Path to the top-level directory containing all data files.
    layout : object
        Object representing a dataset layout. It must implement the following
        methods:
        - `getall(i)`: returns a list of all values of `i` present in the
                       dataset
        - `get`: queries the dataset for matching entities
    labels : tuple or None (default ('subject',))
        List of categorical outcome variables to include in data references.
        These variables can be taken either from data identifiers or from
        additional tables. Labels become available as prediction targets for
        classification models. By default, the subject identifier is included.
    outcomes : tuple or None (default None)
        List of continuous outcome variables to include in data references.
        These variables can be taken either from data identifiers or from
        additional tables. Labels become available as prediction targets for
        regression models. By default, the subject identifier is included.
    observations : tuple (default ('subject',))
        List of data identifiers whose levels are packaged into separate data
        references. Each level should generally have the same values of any
        outcome variables.
    levels : tuple or None (default ('session', 'run, task'))
        List of data identifiers whose levels are packaged as sublevels of the
        same data reference. This permits easier augmentation of data via
        pooling across sublevels.
    queries : list(DataQuery objects)
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
        ignore specifications. In other words, data will be ignored if they
        satisfy any of the ignore criteria.

    Returns
    -------
    data_refs : list(DataReference)
        List of data reference objects created from files found in the input
        directory.
    """
    labels = labels or []
    outcomes = outcomes or []
    ident = list(observations) + list(levels)
    entities = assemble_entities(layout, ident, ignore)
    index, observations, levels, entities = collate_product(
        entities, observations, levels)
    data = query_all(layout, index, queries, **filters)
    df = pd.DataFrame(
        data=data,
        index=index)
    df_aux = read_additional_tables(additional_tables, entities)
    df = _concat_frames(df, df_aux)
    df = delete_null_levels(df, observations, levels)
    df = delete_null_obs(df, observations, levels)
    obs = all_observations(df, observations, levels)
    labels, outcomes = process_labels_and_outcomes(df, labels, outcomes)
    data_refs = make_references(df, obs, levels, labels, outcomes)
    return data_refs


def assemble_entities(layout, ident, ignore=None):
    """
    Assemble all entities corresponding to the specified identifiers from a
    layout.

    Parameters
    ----------
    layout : object
        Object representing a dataset layout with a `getall` method that
        returns a list of all values of its argument entity present in the
        dataset.
    ident : list(str)
        List of identifier variables present in the dataset and its layout.
    ignore : dict(str: list) or None (default None)
        Dictionary indicating identifiers to be ignored. Currently this
        doesn't support any logical composition and takes logical OR over all
        ignore specifications. In other words, data will be ignored if they
        satisfy any of the ignore criteria.

    Returns
    -------
    entities : dict
        Dictionary enumerating the existing levels of each identifier variable
        in the dataset.
    """
    entities = {}
    ignore = ignore or {}
    for i in ident:
        if ignore.get(i) is None:
            ignore[i] = []

    for i in ident:
        entities[i] = layout.getall(i)
        entities[i] = [n for n in entities[i] if n not in ignore[i]]

    return entities


def collate_product(values, observations, levels):
    """
    Collate a product of all possible combinations of within-observation
    variable assignments.

    This returns all possible combinations, including potentially invalid ones
    that are absent in the actual dataset. Additional functions like
    `delete_null_levels` and `delete_null_obs` can help remove the invalid
    combinations returned by `collate_product`.

    Parameters
    ----------
    values : dict
        Dictionary enumerating the existing levels of each identifier variable
        in the dataset.
    observations : list(str)
        List of identifier variables that distinguish observations from one
        another.
    levels : list(str)
        List of identifier variables that denote sub-levels of a single
        observation.

    Returns
    -------
    index : MultiIndex
        Pandas MultiIndex object enumerating all possible combinations of
        variable assignments in the input `values`.
    observations : list
        The input `observations`, filtered to exclude levels absent from the
        data.
    levels : list
        The input `levels`, filtered to exclude levels absent from the data.
    entities : list
        List of all identifier levels present in the data.
    """
    entities = []
    prod_gen = []
    for name, l in values.items():
        if l:
            entities += [name]
            prod_gen += [l]
    observations = [o for o in observations if o in entities]
    levels = [l for l in levels if l in entities]
    index = pd.MultiIndex.from_product(prod_gen, names=entities)
    return index, observations, levels, entities


def query_all(layout, index, queries, **filters):
    """
    Query a layout for all values assigned to entities in an index.

    Parameters
    ----------
    layout : object
        Object representing a dataset layout with a `get` method that
        returns a list of all data files that match a specified query.
    index : MultiIndex
        Pandas MultiIndex object enumerating all possible combinations of
        variable assignments among dataset identifiers.
    queries : list(DataQuery)
        Data queries used to locate files in the dataset. Each entry locates a
        different file for each unique identifier combination.
    filters : dict
        Additional filters to apply for each query.

    Returns
    -------
    results : dict
        Results for each query in `queries` submitted to the `layout` after
        applying any `filters` additionally specified.
    """
    results = {q.name: [] for q in queries}
    entities = index.names
    for ident in index:
        ident_filters = dict(zip(entities, ident))
        for q in queries:
            result = q(layout, **ident_filters, **filters)
            if not result: result = [None]
            results[q.name] += result
    return results


def read_additional_tables(paths, entities):
    """
    Read a set of TSV-formatted data files into DataFrames with a specified
    set of index columns.

    Parameters
    ----------
    paths : list(str)
        List of paths to additional TSV-formatted data files.
    entities : list
        List of identifier entities corresponding to index columns in the
        data files. These must be the first columns in the data files.

    Returns
    -------
    list(DataFrame)
        List of DataFrames read from the files in `paths`.
    """
    if paths is None:
        return []
    idx = list(range(len(entities)))
    return [pd.read_csv(p, sep='\t', index_col=idx) for p in paths]


def _concat_frames(df, df_aux):
    """
    Concatenate a data frame with a list of additional data frames.
    """
    dfs = [df] + df_aux
    return pd.concat(dfs, axis=1)


def delete_null_levels(df, observations, levels):
    return delete_nulls(df, observations, levels, all_levels)


def delete_null_obs(df, observations, levels):
    return delete_nulls(df, observations, levels, all_observations)


def delete_nulls(df, observations, levels, all_combinations):
    for c in all_combinations(df, observations, levels):
        if level_null(df, c):
            df = df.drop(df.loc(axis=0)[c].index)
    return df


def all_levels(df, observations, levels):
    return all_combinations(df, observations, levels, what='levels')


def all_observations(df, observations, levels):
    return all_combinations(df, observations, levels, what='observations')


def all_combinations(df, observations, levels, what):
    idx = []
    gen = []
    for name, c in zip(df.index.names, df.index.levels):
        if what == 'observations':
            if name in observations:
                idx += [None]
                gen += [list(c)]
            elif name in levels:
                idx += [slice(None)]
        elif what == 'levels':
            if name in observations:
                idx += [slice(None)]
            elif name in levels:
                idx += [None]
                gen += [list(c)]
    combinations = product(*gen)
    return [tuple(fill_idx_pattern(c, idx)) for c in combinations]


def level_null(df, level):
    img_null = df.loc(axis=0)[level].images.isnull()
    conf_null = df.loc(axis=0)[level].confounds.isnull()
    return (img_null | conf_null).all()


def fill_idx_pattern(level, idx):
    level = list(level)
    return [i if i is not None else level.pop(0) for i in idx]


def process_labels_and_outcomes(df, labels, outcomes):
    ls, os = [], []
    for l in labels:
        ls += [CategoricalVariable(l, df)]
    for o in outcomes:
        os += [ContinuousVariable(o, df)]
    return ls, os


def make_references(df, obs, levels, labels, outcomes):
    data_refs = []
    for o in obs:
        try:
            data_refs += [fMRIDataReference(
                df, o, level_names=levels, labels=labels, outcomes=outcomes)]
        except KeyError:
            continue
    return data_refs


def get_col(df, label):
    try:
        return df.index.get_level_values(label)
    except KeyError:
        return df[label]
