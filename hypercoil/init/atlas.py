# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas initialisation
~~~~~~~~~~~~~~~~~~~~
Tools for initialising parameters corresponding to brain atlases.
"""
import torch
import templateflow.api as tflow
from functools import partial
from abc import ABC, abstractmethod
from .atlasmixins import (
    _ObjectReferenceMixin,
    _SingleReferenceMixin,
    _MultiReferenceMixin,
    _PhantomReferenceMixin,
    _CIfTIReferenceMixin,
    _MaskLogicMixin,
    _CortexSubcortexMaskCIfTIMixin,
    _MaskFromNullMixin,
    _SingleCompartmentMixin,
    _MultiCompartmentMixin,
    _CortexSubcortexCompartmentCIfTIMixin,
    _DiscreteLabelMixin,
    _ContinuousLabelMixin,
    _DirichletLabelMixin,
    _VolumetricMeshMixin,
    _VertexCoordinatesCIfTIMixin,
    _SpatialConvMixin,
)
from .base import DomainInitialiser
from .dirichlet import DirichletInit
from ..functional import UnstructuredNoiseSource


#TODO: restore doc strings


class BaseAtlas(ABC):
    def __init__(self, ref_pointer, mask_source, dtype=None, device=None,
                 clear_cache=True, **kwargs):
        self.ref_pointer = ref_pointer
        self.ref = self._load_reference(ref_pointer)

        self._create_mask(mask_source, device=device)
        names_dict = self._compartment_names_dict(**kwargs)
        self._create_compartments(names_dict)

        self._configure_decoders()
        self._configure_compartment_maps(dtype=dtype, device=device)
        self._init_coors(source=mask_source, names_dict=names_dict,
                         dtype=dtype, device=device)
        if clear_cache:
            del self.cached_ref_data

    @abstractmethod
    def _load_reference(self, ref_pointer):
        pass

    @abstractmethod
    def _create_mask(self, mask_source, device=None):
        pass

    @abstractmethod
    def _compartment_names_dict(**kwargs):
        pass

    @abstractmethod
    def _create_compartments(self, names_dict, ref=None):
        pass

    @abstractmethod
    def _configure_decoders(self, null_label=None):
        pass

    @abstractmethod
    def _configure_compartment_maps(self, dtype=None, device=None):
        pass

    @abstractmethod
    def _init_coors(self, source=None, names_dict=None,
                    dtype=None, device=None):
        pass

    def __call__(self, compartments=None, normalise=False, sigma=None,
                 noise=None, max_bin=10000, spherical_scale=1, truncate=None):
        ret = {}
        if compartments is None:
            compartments = ['_all']
        elif isinstance(compartments, str):
            compartments = [compartments]
        for c in compartments:
            c_map = self.maps[c]
            if sigma is not None:
                sigma = self._configure_sigma(sigma)
                c_map = self._convolve(
                    map=c_map, compartment=c, sigma=sigma, max_bin=max_bin,
                    spherical_scale=spherical_scale, truncate=truncate
                )
            if noise is not None:
                c_map = noise(c_map)
            if normalise:
                c_map = c_map / c_map.sum(1, keepdim=True)
            ret[c] = c_map
        return ret

    def _configure_compartment_maps(self, dtype=None, device=None):
        self.maps = {}
        for c, mask in self.compartments.items():
            labels = self.decoder[c]
            dim_out = len(labels)
            if dim_out == 0:
                self.maps[c] = torch.tensor([], dtype=dtype, device=device)
                continue
            dim_in = mask.sum()
            map = torch.empty((dim_out, dim_in), dtype=dtype, device=device)
            self.maps[c] = self._populate_map_from_ref(map, labels, mask, c)

        mask = self.mask
        labels = self.decoder['_all']
        dim_out = len(labels)
        dim_in = self.mask.sum()
        map = torch.empty((dim_out, dim_in), dtype=dtype, device=device)
        self.maps['_all'] = self._populate_map_from_ref(
            map, labels, mask, '_all')


class DirichletInitBaseAtlas(
    _PhantomReferenceMixin,
    _DirichletLabelMixin,
    BaseAtlas,
):
    def __init__(self, mask_source, compartment_labels, conc=100.,
                 template_image=None, init=None, dtype=None, device=None,
                 **kwargs):
        if template_image is None:
            template_image = mask_source
        if isinstance(compartment_labels, int):
            compartment_labels = {'all', compartment_labels}
        self.compartment_labels = compartment_labels
        if init is None:
            init = {
                c : DirichletInit(
                    n_classes=i,
                    concentration=torch.tensor([conc for _ in range (i)]),
                    axis=-2
                )
                for c, i in compartment_labels.items()
            }
            self.init = init
        self._global_compartment_init()
        super().__init__(ref_pointer=template_image,
                         mask_source=mask_source,
                         dtype=dtype,
                         device=device,
                         **kwargs)

    def _global_compartment_init(self):
        if self.init.get('_all'):
            return
        if self.init.get('all'):
            self.init['_all'] = self.init['all']
            return
        concentrations = [d.concentration for d in self.init.values()]
        concentrations = torch.cat(concentrations)
        self.init['_all'] = DirichletInit(
            n_classes=len(concentrations),
            concentration=concentrations,
            axis=-2
        )


class DiscreteVolumetricAtlas(
    _SingleReferenceMixin,
    _MaskFromNullMixin,
    _SingleCompartmentMixin,
    _DiscreteLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    BaseAtlas,
):
    def __init__(self, ref_pointer, clear_cache=True,
                 dtype=None, device=None,):
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=0,
                         clear_cache=clear_cache,
                         dtype=dtype,
                         device=device)


class MultiVolumetricAtlas(
    _SingleReferenceMixin,
    _MaskFromNullMixin,
    _SingleCompartmentMixin,
    _ContinuousLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    BaseAtlas,
):
    def __init__(self, ref_pointer, clear_cache=True,
                 dtype=None, device=None,):
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=0,
                         clear_cache=clear_cache,
                         dtype=dtype,
                         device=device)


class MultifileVolumetricAtlas(
    _MultiReferenceMixin,
    _MaskFromNullMixin,
    _SingleCompartmentMixin,
    _ContinuousLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    BaseAtlas,
):
    def __init__(self, ref_pointer, clear_cache=True,
                 dtype=None, device=None,):
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=0,
                         clear_cache=clear_cache,
                         dtype=dtype,
                         device=device)


class CortexSubcortexCIfTIAtlas(
    _CIfTIReferenceMixin,
    _SingleReferenceMixin,
    _CortexSubcortexMaskCIfTIMixin,
    _CortexSubcortexCompartmentCIfTIMixin,
    _DiscreteLabelMixin,
    _VertexCoordinatesCIfTIMixin,
    _SpatialConvMixin,
    BaseAtlas,
):
    def __init__(self, ref_pointer,
                 mask_L=None, mask_R=None, surf_L=None, surf_R=None,
                 cortex_L='CIFTI_STRUCTURE_CORTEX_LEFT',
                 cortex_R='CIFTI_STRUCTURE_CORTEX_RIGHT',
                 clear_cache=True, dtype=None, device=None):
        default_mask_query_args = {
            'template' : 'fsLR',
            'density' : '32k',
            'desc' : 'nomedialwall',
            'suffix' : 'dparc'
        }
        default_surf_query_args = {
            'template' : 'fsLR',
            'density' : '32k',
            'suffix' : 'sphere',
            'space' : None
        }
        if mask_L is None:
            mask_L = tflow.get(hemi='L', **default_mask_query_args)
        if mask_R is None:
            mask_R = tflow.get(hemi='R', **default_mask_query_args)
        if surf_L is None:
            surf_L = tflow.get(hemi='L', **default_surf_query_args)
        if surf_R is None:
            surf_R = tflow.get(hemi='R', **default_surf_query_args)
        self.surf = {
            'cortex_L' : surf_L,
            'cortex_R' : surf_R
        }
        mask_source = {
            'cortex_L' : mask_L,
            'cortex_R' : mask_R,
            'subcortex' : None
        }
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=mask_source,
                         clear_cache=clear_cache,
                         dtype=dtype,
                         device=device,
                         cortex_L=cortex_L,
                         cortex_R=cortex_R)


class DirichletInitVolumetricAtlas(
    _MaskLogicMixin,
    _SingleCompartmentMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    DirichletInitBaseAtlas,
):
    pass


class DirichletInitSurfaceAtlas(
    _CIfTIReferenceMixin,
    _CortexSubcortexMaskCIfTIMixin,
    _CortexSubcortexCompartmentCIfTIMixin,
    _VertexCoordinatesCIfTIMixin,
    _SpatialConvMixin,
    DirichletInitBaseAtlas,
):
    def __init__(self, cifti_template, compartment_labels,
                 conc=100., init=None, dtype=None, device=None,
                 mask_L=None, mask_R=None, surf_L=None, surf_R=None,
                 cortex_L='CIFTI_STRUCTURE_CORTEX_LEFT',
                 cortex_R='CIFTI_STRUCTURE_CORTEX_RIGHT'):
        default_mask_query_args = {
            'template' : 'fsLR',
            'density' : '32k',
            'desc' : 'nomedialwall',
            'suffix' : 'dparc'
        }
        default_surf_query_args = {
            'template' : 'fsLR',
            'density' : '32k',
            'suffix' : 'sphere',
            'space' : None
        }
        if mask_L is None:
            mask_L = tflow.get(hemi='L', **default_mask_query_args)
        if mask_R is None:
            mask_R = tflow.get(hemi='R', **default_mask_query_args)
        if surf_L is None:
            surf_L = tflow.get(hemi='L', **default_surf_query_args)
        if surf_R is None:
            surf_R = tflow.get(hemi='R', **default_surf_query_args)
        self.surf = {
            'cortex_L' : surf_L,
            'cortex_R' : surf_R
        }
        mask_source = {
            'cortex_L' : mask_L,
            'cortex_R' : mask_R,
            'subcortex' : None
        }
        super().__init__(template_image=cifti_template,
                         mask_source=mask_source,
                         compartment_labels=compartment_labels,
                         conc=conc,
                         init=init,
                         dtype=dtype,
                         device=device,
                         cortex_L=cortex_L,
                         cortex_R=cortex_R)


class _MemeAtlas(
    _SingleReferenceMixin,
    _MaskLogicMixin,
    _MultiCompartmentMixin,
    _DiscreteLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    BaseAtlas,
):
    def __init__(self):
        from .atlasmixins import (
            MaskIntersection, MaskUnion, MaskNegation
        )
        ref_pointer = tflow.get(
            template='MNI152NLin2009cAsym',
            resolution=2,
            desc='100Parcels17Networks'
        )
        eye = tflow.get(
            template='MNI152NLin2009cAsym',
            resolution=2, desc='eye', suffix='mask')
        face = tflow.get(
            template='MNI152NLin2009cAsym',
            resolution=2, desc='face', suffix='mask')
        brain = tflow.get(
            template='MNI152NLin2009cAsym',
            resolution=2, desc='brain', suffix='mask')
        mask_source = MaskIntersection(
            MaskUnion(eye, face),
            MaskNegation(brain)
        )
        compartments_dict = {
            'eye': eye,
            'face': face
        }
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=mask_source,
                         **compartments_dict)


#TODO: Fix the below. Also, spatial convolution.
def atlas_init_(tensor, compartment, atlas, kernel_sigma=None,
                noise_sigma=None, normalise=False):
    """
    Voxel-to-label mapping initialisation.

    Initialise a tensor such that its entries characterise a matrix that maps
    a relevant subset of image voxels to a set of labels. The initialisation
    uses an existing atlas with the option of blurring labels or injecting
    noise.

    Dimension
    ---------
    - tensor : :math:`(L, V)`
      L denotes the total number of labels in the atlas, and V denotes the
      number of voxels to be labelled.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in-place.
    atlas : Atlas object
        Atlas object to use for tensor initialisation.
    kernel_sigma : float or None (default None)
        If this is a float, then a Gaussian smoothing kernel with the
        specified width is applied to each label after it is extracted.
    noise_sigma : float or None (default None)
        If this is a float, then Gaussian noise with the specified standard
        deviation is added to the label.
    """
    if noise_sigma is not None:
        distr = torch.distributions.normal.Normal(0, noise_sigma)
        noise = UnstructuredNoiseSource(distr=distr)
    else:
        noise = None
    val = atlas(compartment=compartment,
                sigma=kernel_sigma,
                noise=noise,
                normalise=normalise)
    tensor.copy_(val)


class AtlasInit(DomainInitialiser):
    def __init__(self, atlas, kernel_sigma=None, noise_sigma=None,
                 normalise=False, domain=None):
        init = partial(atlas_init_, atlas=atlas, kernel_sigma=kernel_sigma,
                       noise_sigma=noise_sigma, normalise=normalise)
        super(AtlasInit, self).__init__(init=init, domain=domain)

    def __call__(self, tensor):
        for k, v in tensor.items():
            super(AtlasInit, self).__call__(v, compartment=k)
