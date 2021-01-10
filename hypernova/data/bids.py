# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
BIDS interfaces
~~~~~~~~~~~~~~~
Interfaces for loading BIDS-conformant neuroimaging data.
"""
# import bids
from ..formula import ModelSpec, FCConfoundModelSpec
from .dataref import data_references, DataQuery
from .grabber import LightBIDSLayout
from .neuro import fMRIDataReference
from .variables import (
    VariableInitialiser,
    NeuroImageBlockVariable,
    TableBlockVariable
)


BIDS_SCOPE = 'derivatives'
BIDS_DTYPE = 'func'

BIDS_IMG_DESC = 'preproc'
BIDS_IMG_SUFFIX = 'bold'
BIDS_IMG_EXT = '.nii.gz'

BIDS_CONF_DESC = 'confounds'
BIDS_CONF_SUFFIX = 'timeseries'
BIDS_CONF_EXT = '.tsv'


def fmriprep_references(fmriprep_dir, space=None, additional_tables=None,
                        ignore=None, labels=('subject',), outcomes=None,
                        model=None, observations=('subject',),
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
    data_refs : list(fMRIDataReference)
        List of data reference objects created from files found in the input
        directory.
    """
    if isinstance(model, str):
        model = [FCConfoundModelSpec(model)]
    elif model is not None and not isinstance(model, ModelSpec):
        model = [FCConfoundModelSpec(m, name=m)
                 if isinstance(m, str) else m
                 for m in model]
    images = DataQuery(
        name='images',
        pattern='func/**/*preproc*.nii.gz',
        variable=NeuroImageBlockVariable,
        scope=BIDS_SCOPE,
        datatype=BIDS_DTYPE,
        desc=BIDS_IMG_DESC,
        suffix=BIDS_IMG_SUFFIX,
        extension=BIDS_IMG_EXT)
    confounds = DataQuery(
        name='confounds',
        pattern='func/**/*confounds*.tsv',
        variable=VariableInitialiser(TableBlockVariable, spec=model),
        scope=BIDS_SCOPE,
        datatype=BIDS_DTYPE,
        desc=BIDS_CONF_DESC,
        suffix=BIDS_CONF_SUFFIX,
        extension=BIDS_CONF_EXT)
    #layout = bids.BIDSLayout(
    #    fmriprep_dir,
    #    derivatives=[fmriprep_dir],
    #    validate=False)
    layout = LightBIDSLayout(
        fmriprep_dir,
        queries=[images, confounds])
    return data_references(
        data_dir=fmriprep_dir,
        layout=layout,
        reference=fMRIDataReference,
        labels=labels,
        outcomes=outcomes,
        observations=observations,
        levels=levels,
        queries=[images, confounds],
        filters={'space': space},
        additional_tables=additional_tables,
        ignore=ignore
    )
