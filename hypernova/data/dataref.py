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
from .transforms import ReadNiftiTensor, ReadTableTensor, IdentityTransform


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
        return self.label_transform(self.label_ref)


class fMRISubReference(fMRIReferenceBase):
    def __init__(
        self,
        data,
        confounds=None,
        label=None,
        data_transform=None,
        confounds_transform=None,
        label_transform=None,
        **ids):
        self.data_ref = data
        self.data_transform = data_transform or ReadNiftiTensorBlock()
        self.confounds_ref = confounds
        self.confounds_transform = confounds_transform or ReadTableTensorBlock()
        self.label_ref = label or 'sub'
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
        data_transform=None,
        confounds_transform=None,
        label_transform=None,
        level_names=None):
        if level_names is not None:
            level_names = [tuple(level_names)]
        self.df = df.loc(axis=0)[idx]
        self.labels = labels
        self.data_transform = data_transform or ReadNiftiTensorBlock(names=level_names)
        self.confounds_transform = confounds_transform or ReadTableTensorBlock(names=level_names)
        self.label_transform = label_transform or IdentityTransform()

        self.data_ref = self.df.images.values.tolist()
        self.confounds_ref = self.df.confounds.values.tolist()
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
                label=self.labels,
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


def fmriprep_references(fmriprep_dir, space=None, additional_tables=None,
                        ignore=None, label='subject',
                        observations=('subject',),
                        levels=('session', 'run', 'task')):
    #layout = bids.BIDSLayout(
    #    fmriprep_dir,
    #    derivatives=[fmriprep_dir],
    #    validate=False)
    layout = LightGrabber(
        fmriprep_dir,
        patterns=['func/**/*preproc*.nii.gz',
                  'func/**/*confounds*.tsv'])
    sub, ses, run, task = assemble_entities(layout, ignore)
    index, observations, levels, entities = collate_product(
        sub, ses, run, task, observations, levels)
    images, confounds = query_all(layout, index, space)
    df = pd.DataFrame(
        data={'images': images, 'confounds': confounds},
        index=index)
    df_aux = read_additional_tables(additional_tables, entities)
    df = concat_frames(df, df_aux)
    df = delete_null_levels(df, observations, levels)
    df = delete_null_obs(df, observations, levels)
    obs = all_observations(df, observations, levels)
    data_refs = make_references(df, obs, levels)
    return data_refs


def assemble_entities(layout, ignore=None):
    ignore = ignore or {
        'sub': [],
        'ses': [],
        'run': [],
        'task': []
    }
    sub = layout.get_subjects()
    ses = layout.get_sessions()
    run = layout.get_runs()
    task = layout.get_tasks()

    sub = [i for i in sub if i not in ignore['sub']]
    ses = [i for i in ses if i not in ignore['ses']]
    run = [i for i in run if i not in ignore['run']]
    task = [i for i in task if i not in ignore['task']]
    return sub, ses, run, task


def collate_product(sub, ses, run, task, observations, levels):
    entities = []
    prod_gen = []
    levs = [
        (sub, 'subject'),
        (ses, 'session'),
        (run, 'run'),
        (task, 'task')
    ]
    for l, name in levs:
        if l:
            entities += [name]
            prod_gen += [l]
    observations = [o for o in observations if o in entities]
    levels = [l for l in levels if l in entities]
    index = pd.MultiIndex.from_product(prod_gen, names=entities)
    return index, observations, levels, entities


def get_filters(entities, query):
    filters = {}
    for k, v in zip(entities, query):
        filters[k] = v
    return filters


def read_additional_tables(paths, entities):
    idx = list(range(len(entities)))
    return [pd.read_csv(p, sep='\t', index_col=idx) for p in paths]


def concat_frames(df, df_aux):
    dfs = [df] + df_aux
    return pd.concat(dfs, axis=1)



def n_levels(df, label):
    try:
        return len(df.index.get_level_values(label).unique())
    except KeyError:
        return len(df[label].unique())


def fill_idx_pattern(level, idx):
    level = list(level)
    return [i if i is not None else level.pop(0) for i in idx]


def all_levels(df, observations, levels):
    idx = []
    level_gen = []
    for name, levs in zip(df.index.names, df.index.levels):
        if name in observations:
            idx += [slice(None)]
        elif name in levels:
            idx += [None]
            level_gen += [list(levs)]
    levs = product(*level_gen)
    return [tuple(fill_idx_pattern(l, idx)) for l in levs]


def all_observations(df, observations, levels):
    idx = []
    obs_gen = []
    for name, obs in zip(df.index.names, df.index.levels):
        if name in observations:
            idx += [None]
            obs_gen += [list(obs)]
        elif name in levels:
            idx += [slice(None)]
    obs = product(*obs_gen)
    return [tuple(fill_idx_pattern(o, idx)) for o in obs]


def level_null(df, level):
    img_null = df.loc(axis=0)[level].images.isnull()
    conf_null = df.loc(axis=0)[level].confounds.isnull()
    return (img_null | conf_null).all()


def delete_null_levels(df, observations, levels):
    for level in all_levels(df, observations, levels):
        if level_null(df, level):
            df = df.drop(df.loc(axis=0)[level].index)
    return df


def delete_null_obs(df, observations, levels):
    for obs in all_observations(df, observations, levels):
        if level_null(df, obs):
            df = df.drop(df.loc(axis=0)[obs].index)
    return df


def query_all(layout, index, space=None):
    images = []
    confounds = []
    entities = index.names
    for query in index:
        filters = get_filters(entities, query)
        image = layout.get(
            scope=BIDS_SCOPE,
            datatype=BIDS_DTYPE,
            desc=BIDS_IMG_DESC,
            space=space,
            suffix=BIDS_IMG_SUFFIX,
            extension=BIDS_IMG_EXT,
            **filters)
        confound = layout.get(
            scope=BIDS_SCOPE,
            datatype=BIDS_DTYPE,
            desc=BIDS_CONF_DESC,
            suffix=BIDS_CONF_SUFFIX,
            extension=BIDS_CONF_EXT,
            **filters)
        if not image: image = [None]
        if not confound: confound = [None]
        images += image
        confounds += confound
    return images, confounds


def make_references(df, obs, levels):
    data_refs = []
    for o in obs:
        try:
            data_refs += [fMRIDataReference(df, o, level_names=levels)]
        except KeyError:
            continue
    return data_refs
