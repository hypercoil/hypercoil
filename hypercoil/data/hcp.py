# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
HCP interfaces
~~~~~~~~~~~~~~~
Interfaces for loading HCP-like neuroimaging data.
"""
import subprocess
from ..formula import ModelSpec, FCConfoundModelSpec
from .dataset import ReferencedDataset
from .grabber import LightGrabber
from .neuro import fMRIDataReference, fc_reference_prep
from .dataref import data_references, DataQuery
from .transforms import (
    NIfTIHeader, CWBCIfTIHeader, Compose, ChangeExtension, ReadJSON
)
from .variables import (
    VariableFactory,
    DataPathVariable
)


HCP_IMG_EXT = {
    'surf': '.dtseries.nii', 'msmall': '.dtseries.nii',
    'surf_clean': '.dtseries.nii', 'msmall_clean': '.dtseries.nii',
    'vol': '.nii.gz', 'vol_clean': '.nii.gz'
}

HCP_CONF_EXT = '.tsv'


hcp_regex = {
    'subject': '.*/(?P<subject>[0-9]*)/.*',
    'session': ('.*/[rt]fMRI_[A-Za-z]*(?P<session>[0-9]*).*', 1),
    'run': '.*/[rt]fMRI_[^_]*_(?P<run>[^/_]*).*',
    'task': '.*/[rt]fMRI_(?P<task>[A-Za-z]*).*',
    'desc': '.*_(?P<desc>[^_]*)_[^_]*',
    'suffix': '.*/[^/]*_(?P<suffix>[^/_\.]*)\..*',
    'extension': '.*/[^/\.]*(?P<extension>\..*)$'
}

#TODO: adapt code to support simple 'vol' and 'vol_clean'
hcp_image_patterns = {
    'surf': '*Atlas.dtseries.nii',
    'msmall': '*Atlas_MSMAll.dtseries.nii',
    'surf_clean': '*Atlas_hp2000_clean.dtseries.nii',
    'msmall_clean': '*Atlas_MSMAll_hp2000_clean.dtseries.nii',
}
pattern = 'msmall'


class HCPObjectFactory(VariableFactory):
    """
    Factory for producing HCPObjects. Thin wrapper around
    `VariableFactory`. Consult `hypercoil.data.VariableFactory` for further
    documentation.
    """
    def __init__(self, pattern=pattern):
        if pattern in ('surf', 'msmall', 'surf_clean', 'msmall_clean'):
            wb = subprocess.Popen(
                'which wb_command',
                stdout=subprocess.PIPE,
                shell=True)
            if wb.communicate()[0]:
                meta_func = CWBCIfTIHeader()
            else:
                meta_func = NIfTIHeader()
        elif pattern in ('vol', 'vol_clean'):
            meta_func = NIfTIHeader()
        super(HCPObjectFactory, self).__init__(
            var=HCPObject,
            regex=hcp_regex,
            metadata_local=meta_func
        )


class ConfoundsFactory(VariableFactory):
    def __init__(self):
        super(ConfoundsFactory, self).__init__(
            var=HCPObject,
            regex=hcp_regex,
            metadata_local=Compose([
                ChangeExtension(new_ext='json'),
                ReadJSON()])
        )


class HCPObject(DataPathVariable):
    """
    Thin wrapper around `DataPathVariable`.
    Consult the `hypercoil.data.DataPathVariable` documentation.
    """
    def __repr__(self):
        return f'HCPObject({self.pathobj.name})'


class HCPLayout(LightGrabber):
    """
    DataGrabber for HCP-like data.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset on the file system.
    patterns : list(str)
        String patterns to constrain the scope of the layout. If patterns are
        provided, then the layout will not include any files that do not match
        at least one pattern.
    queries : list(DataQuery)
        Query objects defining the variables to extract from the dataset via
        query.
    """
    def __init__(self, root, queries):
        super(HCPLayout, self).__init__(
            root=root,
            patterns=queries,
            template=HCPObjectFactory()
        )


class HCPDataset(ReferencedDataset):
    def __init__(self, hcp_dir, additional_tables=None,
                 ignore=None, labels=('task',), outcomes=None,
                 model=None, tmask=None, observations=('subject',),
                 levels=('task', 'session', 'run'), depth=0):
        data_refs = hcp_references(
            hcp_dir=hcp_dir,
            additional_tables=additional_tables, ignore=ignore,
            labels=labels, outcomes=outcomes, model=model, tmask=tmask,
            observations=observations, levels=levels
        )
        super(HCPDataset, self).__init__(data_refs, depth=depth)

    def add_data(self, fmriprep_dir, space=None, additional_tables=None,
                 ignore=None, labels=('task',), outcomes=None,
                 model=None, tmask=None, observations=('subject',),
                 levels=('task', 'session', 'run')):
        """
        Add data from another directory. Call follows the same pattern as
        the constructor.
        """
        data_refs = hcp_references(
            hcp_dir=hcp_dir,
            additional_tables=additional_tables, ignore=ignore,
            labels=labels, outcomes=outcomes, model=model, tmask=tmask,
            observations=observations, levels=levels
        )
        super(HCPDataset, self).add_data(data_refs)


def hcp_references(hcp_dir, additional_tables=None,
                   ignore=None, labels=('task',), outcomes=None,
                   model=None, tmask=None, observations=('subject',),
                   levels=('task', 'session', 'run'),
                   dtype=None, device=None):
    """
    Obtain data references for a directory containing data processed with
    fMRIPrep.

    Parameters
    ----------
    hcp_dir : str
        Path to the top-level directory containing all HCP neuroimaging data.
    additional_tables : list(str) or None (default None)
        List of paths to files containing additional data. Each file should
        include index columns corresponding to all identifiers present in the
        dataset (e.g., subject, run, etc.).
    ignore : dict(str: list) or None (default None)
        Dictionary indicating identifiers to be ignored. Currently this
        doesn't support any logical composition and takes logical OR over all
        ignore specifications. In other words, data will be ignored if they
        satisfy any of the ignore criteria.
    labels : tuple or None (default ('task',))
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
    dtype : torch datatype
        Datatype of sampled DataReferences at creation. Note that, if you are
        using a `WebDataset` for training (strongly recommended), this will
        not constrain the data type used at training.
    device : str (default 'cpu')
        Device on which DataReferences are to be sampled at creation. Note
        that, if you are using a `WebDataset` for training (strongly
        recommended), this will not constrain the device used at training.

    Returns
    -------
    data_refs : list(fMRIDataReference)
        List of data reference objects created from files found in the input
        directory.
    """
    image_and_trep, model_and_tmask = fc_reference_prep(
        model, tmask,
        dtype=dtype,
        device=device
    )
    images = DataQuery(
        name='images',
        pattern=hcp_image_patterns[pattern],
        variables=image_and_trep,
        extension=HCP_IMG_EXT[pattern])
    confounds = DataQuery(
        name='confounds',
        pattern='*Confound_Regressors.tsv',
        variables=model_and_tmask,
        template=ConfoundsFactory(),
        extension=HCP_CONF_EXT)
    layout = HCPLayout(
        root=hcp_dir,
        queries=[images, confounds])
    return data_references(
        data_dir=hcp_dir,
        layout=layout,
        reference=fMRIDataReference,
        labels=labels,
        outcomes=outcomes,
        observations=observations,
        levels=levels,
        queries=[images, confounds],
        filters = {},
        additional_tables=additional_tables,
        ignore=ignore
    )
