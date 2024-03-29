# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interfaces for loading BIDS-conformant neuroimaging data.

The reference retrieval utility is written with the assumption that the input
path is a BIDS derivatives dataset created by fMRIPrep.

The recommended use is first retrieving data references through the
:doc:`fmriprep_references <hypercoil.data.bids.fmriprep_references>`
function call, and then passing the references to the
:doc:`make_wds <api/hypercoil.data.wds.make_wds>`
function. A
:doc:`fMRIPrepDataset <hypercoil.data.bids.fMRIPrepDataset>`
class also exists, but porting references to a ``webdataset`` will yield far
superior data loading performance in general.

.. warning::
    ``ciftify`` compatibility is likely absent. This should be a development
    priority, given that a great deal of development has operated under the
    assumption of CIfTI inputs.

.. note::
    Currently we acquire dataset paths using an internal class called a
    ``LightGrabber``, but we'd eventually like to use an actual ``BIDSLayout``
    from ``pybids`` for robustness.
"""
##TODO: ciftify compatibility
##TODO: use pybids wherever possible
# import bids
from .dataref import data_references, DataQuery
from .dataset import ReferencedDataset
from .grabber import LightGrabber
from .neuro import fMRIDataReference, fc_reference_prep
from .transforms import (
    Compose,
    ChangeExtension,
    ReadJSON
)
from .variables import (
    VariableFactory,
    DataPathVariable
)


#bids.config.set_option('extension_initial_dot', True)


BIDS_SCOPE = 'derivatives'
BIDS_DTYPE = 'func'

BIDS_IMG_DESC = 'preproc'
BIDS_IMG_SUFFIX = 'bold'
BIDS_IMG_EXT = '.nii.gz'

BIDS_CONF_DESC = 'confounds'
BIDS_CONF_SUFFIX = 'timeseries'
BIDS_CONF_EXT = '.tsv'


bids_regex = {
    'datatype': '.*/(?P<datatype>[^/]*)/[^/]*',
    'subject': '.*/[^/]*sub-(?P<subject>[^_]*)[^/]*',
    'session': '.*/[^/]*ses-(?P<subject>[^_]*)[^/]*',
    'run': '.*/[^/]*run-(?P<run>[^_]*)[^/]*',
    'task': '.*/[^/]*task-(?P<task>[^_]*)[^/]*',
    'space': '.*/[^/]*space-(?P<space>[^_]*)[^/]*',
    'desc': '.*/[^/]*desc-(?P<desc>[^_]*)[^/]*',
    'suffix': '.*/[^/]*_(?P<suffix>[^/_\.]*)\..*',
    'extension': '.*/[^/\.]*(?P<extension>\..*)$'
}


class BIDSObjectFactory(VariableFactory):
    """
    Factory for producing
    :doc:`LightBIDSObjects <hypercoil.data.bids.LightBIDSObject>`.
    Thin wrapper around
    :doc:`VariableFactory <hypercoil.data.variables.VariableFactory>`; consult
    for further documentation.
    """
    def __init__(self):
        super(BIDSObjectFactory, self).__init__(
            var=LightBIDSObject,
            regex=bids_regex,
            metadata_local=Compose([
                ChangeExtension(new_ext='json'),
                ReadJSON()])
        )


class LightBIDSObject(DataPathVariable):
    """
    Thin wrapper around
    :doc:`DataPathVariable <hypercoil.data.variables.DataPathVariable>`;
    consult for further documentation.
    """
    def __repr__(self):
        return (
            f'LightBIDSObject({self.pathobj.name}, '
            f'dtype={self.datatype})'
        )


class LightBIDSLayout(LightGrabber):
    """
    Lightweight and minimal ``DataGrabber``-inspired class for BIDS-like data.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset on the file system.
    patterns : list(str)
        String patterns to constrain the scope of the layout. If patterns are
        provided, then the layout will not include any files that do not match
        at least one pattern.
    queries : list(:doc:`DataQuery <hypercoil.data.dataref.DataQuery>`)
        Query objects defining the variables to extract from the dataset via
        query.
    """
    def __init__(self, root, queries):
        super(LightBIDSLayout, self).__init__(
            root=root,
            patterns=queries,
            template=BIDSObjectFactory()
        )


class fMRIPrepDataset(ReferencedDataset):
    """
    Referenced dataset created from a directory of fMRIPrep output data.

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
    model : str, list, or None (default None)
        Formula expressions representing confound models to create for each
        subject. For example, a 36-parameter expanded model can be specified as
        `(dd1(rps + wm + csf + gsr))^^2`.
    tmask : str or None (default None)
        A formula expression representing the temporal mask to create for each
        subject. For instance `and(uthr0.5(fd) + uthr1.5(dv))` results in a mask
        that includes time points with less than 0.5 framewise displacement and
        less than 1.5 standardised DVARS.
    observations : tuple (default ('subject',))
        List of data identifiers whose levels are packaged into separate data
        references. Each level should generally have the same values of any
        outcome variables.
    levels : tuple or None (default ('session', 'run, task'))
        List of data identifiers whose levels are packaged as sublevels of the
        same data reference. This permits easier augmentation of data via
        pooling across sublevels.
    depth : int (default 0)
        Sampling depth of the dataset: if the passed references are nested
        hierarchically, then each minibatch is sampled from data references
        at the indicated level. By default, the dataset is sampled from the
        highest level (no nesting).
    """
    def __init__(self, fmriprep_dir, space=None, additional_tables=None,
                 ignore=None, labels=('subject',), outcomes=None,
                 model=None, tmask=None, observations=('subject',),
                 levels=('session', 'run', 'task'), depth=0):
        data_refs = fmriprep_references(
            fmriprep_dir=fmriprep_dir, space=space,
            additional_tables=additional_tables, ignore=ignore,
            labels=labels, outcomes=outcomes, model=model, tmask=tmask,
            observations=observations, levels=levels
        )
        super(fMRIPrepDataset, self).__init__(data_refs, depth=depth)

    def add_data(self, fmriprep_dir, space=None, additional_tables=None,
                 ignore=None, labels=('subject',), outcomes=None,
                 model=None, tmask=None, observations=('subject',),
                 levels=('session', 'run', 'task')):
        """
        Add data from another directory. Call follows the same pattern as
        the constructor.
        """
        data_refs = fmriprep_references(
            fmriprep_dir=fmriprep_dir, space=space,
            additional_tables=additional_tables, ignore=ignore,
            labels=labels, outcomes=outcomes, model=model, tmask=tmask,
            observations=observations, levels=levels
        )
        super(fMRIPrepDataset, self).add_data(data_refs)


def fmriprep_references(fmriprep_dir, space=None, additional_tables=None,
                        ignore=None, labels=('subject',), outcomes=None,
                        model=None, tmask=None, observations=('subject',),
                        levels=('session', 'run', 'task'),
                        dtype=None, device=None):
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
    model : str, list, or None (default None)
        Formula expressions representing confound models to create for each
        subject. For example, a 36-parameter expanded model can be specified as
        `(dd1(rps + wm + csf + gsr))^^2`.
    tmask : str or None (default None)
        A formula expression representing the temporal mask to create for each
        subject. For instance `and(uthr0.5(fd) + uthr1.5(dv))` results in a mask
        that includes time points with less than 0.5 framewise displacement and
        less than 1.5 standardised DVARS.
    observations : tuple (default ('subject',))
        List of data identifiers whose levels are packaged into separate data
        references. Each level should generally have the same values of any
        outcome variables.
    levels : tuple or None (default ('session', 'run', 'task'))
        List of data identifiers whose levels are packaged as sublevels of the
        same data reference. This permits easier augmentation of data via
        pooling across sublevels.
    dtype : torch datatype (default None)
        Datatype of sampled DataReferences at creation. Note that, if you are
        using a `WebDataset` for training (strongly recommended), this will
        not constrain the data type used at training.
    device : str or torch device (default None)
        Device on which DataReferences are to be sampled at creation. Note
        that, if you are using a `WebDataset` for training (strongly
        recommended), this will not constrain the device used at training.

    Returns
    -------
    data_refs : list(:doc:`fMRIDataReference <hypercoil.data.neuro.fMRIDataReference>`)
        List of data reference objects created from files found in the input
        directory.
    """
    image_and_trep, model_and_tmask = fc_reference_prep(
        model,
        tmask,
        dtype=dtype,
        device=device
    )
    images = DataQuery(
        name='images',
        pattern='func/**/*preproc*.nii.gz',
        variables=image_and_trep,
        scope=BIDS_SCOPE,
        datatype=BIDS_DTYPE,
        desc=BIDS_IMG_DESC,
        suffix=BIDS_IMG_SUFFIX,
        extension=BIDS_IMG_EXT)
    confounds = DataQuery(
        name='confounds',
        pattern='func/**/*confounds*.tsv',
        variables=model_and_tmask,
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
