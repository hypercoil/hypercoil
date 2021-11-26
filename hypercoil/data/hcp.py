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
from .neuro import fMRIDataReference
from .dataref import data_references, DataQuery
from .transforms import (
    NIfTIHeader, CWBCIfTIHeader, Compose, ChangeExtension, ReadJSON
)
from hypercoil.data.variables import (
    VariableFactory,
    VariableFactoryFactory,
    NeuroImageBlockVariable,
    TableBlockVariable,
    MetaValueBlockVariable,
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
                   levels=('task', 'session', 'run')):
    if isinstance(model, str):
        model = [FCConfoundModelSpec(model)]
    elif model is not None and not isinstance(model, ModelSpec):
        model = [FCConfoundModelSpec(m, name=m)
                 if isinstance(m, str) else m
                 for m in model]
    if isinstance(tmask, str):
        tmask = [FCConfoundModelSpec(tmask)]
    image_and_trep = {
        'images': VariableFactoryFactory(NeuroImageBlockVariable),
        't_r': VariableFactoryFactory(MetaValueBlockVariable,
                                      key='RepetitionTime')
    }
    model_and_tmask = {
        'confounds': VariableFactoryFactory(TableBlockVariable, spec=model),
        'tmask': VariableFactoryFactory(TableBlockVariable, spec=tmask)
    }
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
