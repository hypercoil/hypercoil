# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Neuroimaging datasets
~~~~~~~~~~~~~~~~~~~~~
Dataset objects for neuroimaging data.
"""
import bids
from itertools import product
from torch.utils.data import Dataset, DataLoader
from .dataref import FunctionalConnectivityDataReference


bids.config.set_option('extension_initial_dot', True)

BIDS_SCOPE = 'derivatives'
BIDS_DTYPE = 'func'

BIDS_IMG_DESC = 'preproc'
BIDS_IMG_SUFFIX = 'bold'
BIDS_IMG_EXT = '.nii.gz'

BIDS_CONF_DESC = 'confounds'
BIDS_CONF_SUFFIX = 'timeseries'
BIDS_CONF_EXT = '.tsv'


class NeuroimagingDataset(Dataset):
    def __init__(self, data_refs):
        self.data_refs = data_refs

    def __len__(self):
        return len(self.data_refs)

    def __getitem__(self, idx):
        ref = self.data_refs[idx]
        return {
            'data': ref.data,
            'confounds': ref.confounds,
            'label': ref.label
        }

    def __repr__(self):
        s = f'{type(self).__name__}(n={len(self)})'


class BIDSDataset(NeuroimagingDataset):
    def __init__(self, data_refs):
        super(BIDSDataset, self).__init__(data_refs)

    def _assemble_entities(self, layout, ignore=None):
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
        return self._collate_product(sub, ses, run, task)

    def _collate_product(self, sub, ses, run, task):
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

    def _get_filters(self, entities, query):
        filters = {}
        for k, v in zip(entities, query):
            filters[k] = v
        return filters

    def _query_and_reference_all(self, layout, entities, queries, space=None):
        data_refs = []
        for query in queries:
            filters = self._get_filters(entities, query)
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


class fMRIPrepDataset(BIDSDataset):
    def __init__(self, fmriprep_dir, space=None):
        data_refs = self.fmriprep_references(fmriprep_dir, space)
        super(fMRIPrepDataset, self).__init__(data_refs)

    def fmriprep_references(self, fmriprep_dir, space=None,
                            additional_tables=None, ignore=None):
        layout = bids.BIDSLayout(
            fmriprep_dir,
            derivatives=[fmriprep_dir],
            validate=False)
        entities, queries = self._assemble_entities(layout, ignore)
        return self._query_and_reference_all(layout, entities, queries, space)

    def add_data(self, fmriprep_dir, space=None):
        self.data_refs += self.fmriprep_references(fmriprep_dir, space)
