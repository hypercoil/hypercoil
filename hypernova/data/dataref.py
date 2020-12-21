# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data references
~~~~~~~~~~~~~~~
Interfaces between neuroimaging data and data loaders.
"""
import bids
from .transforms import ReadNiftiTensor, ReadTableTensor, IdentityTransform


bids.config.set_option('extension_initial_dot', True)

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


class FunctionalConnectivityDataReference(DataReference):
    def __init__(
        self,
        data,
        data_transform=None,
        confounds=None,
        confounds_transform=None,
        label=None,
        label_transform=None,
        **ids):
        self.data_ref = data
        self.data_transform = data_transform or ReadNiftiTensor()
        self.confounds_ref = confounds
        self.confounds_transform = confounds_transform or ReadTableTensor()
        self.label_ref = label or 'sub'
        self.label_transform = label_transform or IdentityTransform()
        self.subject = ids.get('subject')
        self.session = ids.get('session')
        self.run = ids.get('run')
        self.task = ids.get('task')

    @property
    def data(self):
        return self.data_transform(self.data_ref)

    @property
    def confounds(self):
        return self.confounds_transform(self.confounds_ref)

    @property
    def label(self):
        return self.label_transform(self.label_ref)

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
    return collate_product(sub, ses, run, task)


def collate_product(sub, ses, run, task):
    entities = []
    prod_gen = []
    if sub:
        entities += ['subject']
        prod_gen += [sub]
    if ses:
        entities += ['session']
        prod_gen += [ses]
    if run:
        entities += ['run']
        prod_gen += [run]
    if task:
        entities += ['task']
        prod_gen += [task]
    return entities, list(product(*prod_gen))


def get_filters(entities, query):
    filters = {}
    for k, v in zip(entities, query):
        filters[k] = v
    return filters


def query_and_reference_all(layout, entities, queries, space=None):
    data_refs = []
    for query in queries:
        filters = get_filters(entities, query)
        image = layout.get(
            scope=BIDS_SCOPE,
            datatype=BIDS_DTYPE,
            desc=BIDS_IMG_DESC,
            space=space,
            suffix=BIDS_IMG_SUFFIX,
            extension=BIDS_IMG_EXT,
            **filters)
        confounds = layout.get(
            scope=BIDS_SCOPE,
            datatype=BIDS_DTYPE,
            desc=BIDS_CONF_DESC,
            suffix=BIDS_CONF_SUFFIX,
            extension=BIDS_CONF_EXT,
            **filters)
        if not image or not confounds:
            continue
        ref = FunctionalConnectivityDataReference(
            data=image[0],
            confounds=confounds[0],
            label=int(filters['subject']),
            **filters
        )
        data_refs += [ref]
    return data_refs


def fmriprep_references(
    fmriprep_dir,
    space=None,
    additional_tables=None,
    ignore=None):
    layout = bids.BIDSLayout(
        fmriprep_dir,
        derivatives=[fmriprep_dir],
        validate=False)
    entities, queries = assemble_entities(layout, ignore)
    return query_and_reference_all(layout, entities, queries, space)
