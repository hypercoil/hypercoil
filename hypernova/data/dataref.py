# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data references
~~~~~~~~~~~~~~~
Interfaces between neuroimaging data and data loaders.
"""
import bids


BIDS_SCOPE = 'derivatives'
BIDS_DTYPE = 'func'

BIDS_IMG_DESC = 'preproc'
BIDS_IMG_SUFFIX = 'bold'
BIDS_IMG_EXT = 'nii.gz'

BIDS_CONF_DESC = 'confounds'
BIDS_CONF_SUFFIX = 'timeseries'
BIDS_CONF_EXT = 'tsv'


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
        label_transform=None):
        self.data_ref = data
        self.data_transform = data_transform or ReadNiftiTensor
        self.confounds_ref = confounds
        self.confounds_transform = confounds_transform or ReadTableTensor
        self.label_ref = label
        self.label_transform = label_transform or EncodeOneHot

    @property
    def data():
        return self.data_transform(self.data_ref)

    @property
    def confounds():
        return self.confounds_transform(self.confounds_ref)

    @property
    def label():
        return self.label_transform(self.label_ref)


def bids_references(
    fmriprep_dir, space='MNI152NLin2009cAsym', additional_tables=None):
    layout = bids.BIDSLayout(
        fmriprep_dir,
        derivatives=[fmriprep_dir],
        validate=False)
    images = layout.get(
        scope=BIDS_SCOPE,
        datatype=BIDS_DTYPE,
        desc=BIDS_IMG_DESC,
        space=space,
        suffix=BIDS_IMG_SUFFIX,
        extension=BIDS_IMG_EXT)
    confounds = layout.get(
        scope=BIDS_SCOPE,
        datatype=BIDS_DTYPE,
        desc=BIDS_CONF_DESC,
        suffix=BIDS_CONF_SUFFIX,
        extension=BIDS_CONF_EXT)
