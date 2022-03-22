# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas initialisation
~~~~~~~~~~~~~~~~~~~~
Tools for initialising parameters corresponding to brain atlases.
"""
import torch
import numpy as np
import nibabel as nb
import templateflow.api as tflow
from functools import partial
from pathlib import PosixPath
from scipy.ndimage import gaussian_filter
from .base import DomainInitialiser
from .dirichlet import DirichletInit
from ..functional import UnstructuredNoiseSource
from ..functional.domain import Identity
from ..functional.sphere import spherical_conv, euclidean_conv


#TODO: restore doc strings


def _to_mask(path):
    return nb.load(path).get_fdata().round().astype(bool)


def _is_path(obj):
    return isinstance(obj, str) or isinstance(obj, PosixPath)


class _ObjectReferenceMixin:
    """
    For when a NIfTI image object is already provided as the `ref_pointer`
    argument.
    """
    def _load_reference(self, ref_pointer):
        self.cached_ref_data = ref_pointer.get_fdata()
        return ref_pointer


class _SingleReferenceMixin:
    def _load_reference(self, ref_pointer):
        ref = nb.load(ref_pointer)
        self.cached_ref_data = ref.get_fdata()
        return ref


class _MultiReferenceMixin:
    def _load_reference(self, ref_pointer):
        ref = [nb.load(path) for path in ref_pointer]
        self.cached_ref_data = np.stack([r.get_fdata() for r in ref], -1)
        ref = nb.Nifti1Image(
            dataobj=np.copy(self.cached_ref_data),
            affine=ref[0].affine,
            header=ref[0].header
        )
        return ref


class _PhantomReferenceMixin:
    def _load_reference(self, ref_pointer):
        ref = nb.load(ref_pointer)
        affine = ref.affine
        header = ref.header
        dataobj = _PhantomDataobj(ref)
        self.ref = nb.Nifti1Image(
            header=header, affine=affine, dataobj=dataobj
        )
        self.cached_ref_data = None
        return self.ref


class _PhantomDataobj:
    def __init__(self, base):
        self.shape = base.shape
        self.ndim = base.ndim


class _CIfTIReferenceMixin:
    @property
    def axes(self):
        """
        Thanks to Chris Markiewicz for tutorials that shaped this
        implementation.
        """
        return [
            self.ref.header.get_axis(i)
            for i in range(self.ref.ndim)
        ]

    @property
    def model_axis(self):
        return [a for a in self.axes
                if isinstance(a, nb.cifti2.cifti2_axes.BrainModelAxis)][0]


class MaskLeaf:
    """
    Leaf node for mask logic operations.
    """
    def __init__(self, mask):
        self.mask = mask

    def __call__(self):
        return _to_mask(self.mask)


class MaskNegation:
    """
    Negation node for mask logic operations.
    """
    def __init__(self, child):
        if _is_path(child):
            self.child = MaskLeaf(child)
        else:
            self.child = child

    def __call__(self):
        return ~self.child()


class MaskUnion:
    """
    Union node for mask logic operations.
    """
    def __init__(self, *children):
        self.children = [
            child if not _is_path(child) else MaskLeaf(child)
            for child in children
        ]

    def __call__(self):
        child = self.children[0]
        mask = child()
        for child in self.children[1:]:
            mask = mask + child()
        return mask


class MaskIntersection:
    """
    Intersection node for mask logic operations.
    """
    def __init__(self, *children):
        self.children = [
            child if not _is_path(child) else MaskLeaf(child)
            for child in children
        ]

    def __call__(self):
        child = self.children[0]
        mask = child()
        for child in self.children[1:]:
            mask = mask * child()
        return mask


class _MaskFileMixin:
    def _create_mask(self, source, device=None):
        init = _to_mask(source)
        self.mask = torch.tensor(init.ravel(), device=device)


class _MaskLogicMixin:
    def _create_mask(self, source, device=None):
        init = source()
        self.mask = torch.tensor(init.ravel(), device=device)


class _CortexSubcortexMaskCIfTIMixin:
    def _create_mask(self, source, device=None):
        init = []
        for k, v in source.items():
            try:
                init += [
                    nb.load(v).darrays[0].data.round().astype(bool)
                ]
            except Exception:
                init += [
                    np.ones(self.model_axis.volume_mask.sum()).astype(bool)
                ]
        self.mask = torch.tensor(np.concatenate(init), device=device)


class _MaskFromNullMixin:
    def _create_mask(self, source, device=None):
        if self.ref.ndim <= 3:
            init = (self.cached_ref_data.round() != source)
            self.mask = torch.tensor(init.ravel(), device=device)
        else:
            init = (self.cached_ref_data.sum(-1) > source)
            self.mask = torch.tensor(init.ravel(), device=device)


class _SingleCompartmentMixin:
    def _compartment_names_dict(self, **kwargs):
        return {}

    def _create_compartments(self, names_dict=None, ref=None):
        ref = ref or self.ref

        if ref.ndim == 2: # surface, time by vox greyordinates
            self.compartments = {
                'all': self.mask
            }
        else: # volume, x by y by z by t
            self.compartments = {
                'all' : self.mask
            }


class _MultiCompartmentMixin:
    def _compartment_names_dict(self, **kwargs):
        return kwargs

    def _create_compartments(self, names_dict=None, ref=None):
        ref = ref or self.ref
        dtype = self.mask.dtype # should be torch.bool
        device = self.mask.device

        self.compartments = {}
        for name, vol in names_dict.items():
            if isinstance(name, str):
                self.compartments[name] = torch.tensor(
                    _to_mask(vol), dtype=dtype, device=device
                ).ravel()
            else:
                init = _to_mask(vol)
                for name, data in zip(name, init):
                    self.compartments[name] = torch.tensor(
                        data, dtype=dtype, device=device
                    ).ravel()


class _CortexSubcortexCompartmentCIfTIMixin:
    def _compartment_names_dict(self, **kwargs):
        return kwargs

    def _create_compartments(self, names_dict, ref=None):
        ref = ref or self.ref
        self.compartments = {
            'cortex_L' : torch.zeros(
                self.ref.shape[-1],
                dtype=torch.bool,
                device=self.mask.device),
            'cortex_R' : torch.zeros(
                self.ref.shape[-1],
                dtype=torch.bool,
                device=self.mask.device),
            'subcortex': torch.zeros(
                self.ref.shape[-1],
                dtype=torch.bool,
                device=self.mask.device),
        }
        model_axis = self.model_axis
        for struc, slc, _ in (model_axis.iter_structures()):
            if struc == names_dict['cortex_L']:
                self.compartments['cortex_L'][(slc,)] = True
            elif struc == names_dict['cortex_R']:
                self.compartments['cortex_R'][(slc,)] = True
        try:
            vol_mask = np.where(model_axis.volume_mask)[0]
            vol_min, vol_max = vol_mask.min(), vol_mask.max() + 1
            self.compartments['subcortex'][(slice(vol_min, vol_max),)] = True
        except ValueError:
            pass


class _DiscreteLabelMixin:
    def _configure_decoders(self, null_label=0):
        self.decoder = {}
        for c, mask in self.compartments.items():
            mask = mask.reshape(self.ref.shape)
            labels_in_compartment = np.unique(self.cached_ref_data[mask])
            labels_in_compartment = labels_in_compartment[
                labels_in_compartment != null_label]
            self.decoder[c] = torch.tensor(
                labels_in_compartment, dtype=torch.long, device=mask.device)

        try:
            mask = self.mask.reshape(self.ref.shape)
        except RuntimeError:
            # The reference is already masked. In this case, we're using a
            # CIfTI.
            assert self.mask.sum() == self.ref.shape[-1]
            mask = True
        unique_labels = np.unique(self.cached_ref_data[mask])
        unique_labels = unique_labels[unique_labels != null_label]
        self.decoder['_all'] = torch.tensor(
            unique_labels, dtype=torch.long, device=self.mask.device)

    def _populate_map_from_ref(self, map, labels, mask, compartment=None):
        for i, l in enumerate(labels):
            try:
                map[i] = torch.tensor(
                    self.cached_ref_data.ravel()[mask] == l.item())
            except IndexError:
                # Again the reference is already masked. In this case, we
                # assume we're using a CIfTI.
                assert self.mask.sum() == len(self.cached_ref_data.ravel())
                map[i] = torch.tensor(
                    self.cached_ref_data.ravel() == l.item())
        return map


class _ContinuousLabelMixin:
    def _configure_decoders(self, null_label=None):
        self.decoder = {}
        for c, mask in self.compartments.items():
            mask = mask.reshape(self.ref.shape[:-1])
            #TODO
            # This is a relatively slow step and is repeated work.
            # Can we minimise the extent to which we load the ref data
            # from disk? We should temporarily cache it in memory during
            # the atlas init and then lose it when init is complete.
            # Let's take care of this after the atlas init is fully
            # operational.
            labels_in_compartment = np.where(
                self.cached_ref_data[mask].sum(0))[0]
            labels_in_compartment = labels_in_compartment[
                labels_in_compartment != null_label]
            self.decoder[c] = torch.tensor(
                labels_in_compartment, dtype=torch.long, device=mask.device)

        mask = self.mask.reshape(self.ref.shape[:-1])
        unique_labels = np.where(self.cached_ref_data[mask].sum(0))[0]
        self.decoder['_all'] = torch.tensor(
            unique_labels, dtype=torch.long, device=self.mask.device)

    def _populate_map_from_ref(self, map, labels, mask, compartment=None):
        ref_data = np.moveaxis(
            self.cached_ref_data,
            (0, 1, 2, 3),
            (1, 2, 3, 0)
        ).squeeze()
        for i, l in enumerate(labels):
            map[i] = torch.tensor(ref_data[l].ravel()[mask])
        return map


class _DirichletLabelMixin:
    def _configure_decoders(self, null_label=None):
        self.decoder = {}
        n_labels = 0
        for c, i in self.compartment_labels.items():
            self.decoder[c] = torch.arange(
                n_labels, i, dtype=torch.long, device=self.mask.device)
            n_labels += i
        self.decoder['_all'] = torch.arange(
            0, n_labels, dtype=torch.long, device=self.mask.device)

    def _populate_map_from_ref(self, map, labels, mask, compartment=None):
        self.init[compartment](map)
        return map


class _VolumetricMeshMixin:
    def _init_coors(self, source=None, names_dict=None,
                    dtype=None, device=None):
        axes = None
        shape = self.ref.shape[:3]
        scale = self.ref.header.get_zooms()[:3]
        for i, ax in enumerate(shape[::-1]):
            extra_dims = [...] + [None] * i
            ax = np.arange(ax) * scale[i] #[extra_dims]
            if axes is not None:
                out_shape_new = (1, *ax.shape, *axes.shape[1:])
                out_shape_old = (i, *ax.shape, *axes.shape[1:])
                axes = np.concatenate([
                    np.broadcast_to(ax[tuple(extra_dims)], out_shape_new),
                    np.broadcast_to(np.expand_dims(axes, 1), out_shape_old)
                ], axis=0)
            else:
                axes = np.expand_dims(ax, 0)
        self.coors = torch.tensor(
            axes.reshape(i + 1, -1).T,
            dtype=dtype,
            device=device
        )
        self.topology = {c: 'euclidean' for c in self.compartments.keys()}


class _VertexCoordinatesCIfTIMixin:
    def _init_coors(self, source=None, names_dict=None,
                    dtype=None, device=None):
        model_axis = self.model_axis
        coor = np.empty(model_axis.voxel.shape)
        vox = model_axis.volume_mask
        coor[vox] = model_axis.voxel[vox]

        names2surf = {
            v: (self.surf[k], source[k])
            for k, v in names_dict.items()
        }
        for name, (surf, mask) in names2surf.items():
            if surf is None:
                continue
            surf = nb.load(surf).darrays[0].data
            if mask is not None:
                mask = nb.load(mask)
                mask = mask.darrays[0].data.astype(bool)
                surf = surf[mask]
            coor[model_axis.name == name] = surf
        self.coors = torch.tensor(
            coor,
            dtype=dtype,
            device=device
        )
        self.topology = {}
        euc_mask = torch.BoolTensor(self.model_axis.volume_mask)
        for c, mask in self.compartments.items():
            if (mask * euc_mask).sum() == 0:
                self.topology[c] = 'spherical'
            else:
                self.topology[c] = 'euclidean'


class _EvenlySampledConvMixin:
    def _configure_sigma(self, sigma):
        if sigma is not None:
            scale = self.ref.header.get_zooms()[:3]
            sigma = [sigma / s for s in scale]
            sigma = [0] + [sigma]
        return sigma

    def _convolve(self, sigma, map):
        if sigma is not None:
            gaussian_filter(map, sigma=sigma, output=map)
        return map


class _SpatialConvMixin:
    def _configure_sigma(self, sigma):
        return sigma

    def _convolve(self, map, compartment, sigma, max_bin=10000,
                  spherical_scale=1, truncate=None):
        compartment_mask = self.compartments[compartment]
        if self.topology[compartment] == 'euclidean':
            map = euclidean_conv(
                data=map.T, coor=coors[compartment_mask],
                scale=sigma, max_bin=max_bin,
                truncate=truncate
            ).T
        elif self.topology[compartment] == 'spherical':
            map = spherical_conv(
                data=map.T, coor=coors[compartment_mask],
                scale=(spherical_scale * sigma), r=100,
                max_bin=max_bin, truncate=truncate
            ).T
        return map


class BaseAtlas:
    def __init__(self, ref_pointer, mask_source, dtype=None, device=None,
                 **kwargs):
        self.ref_pointer = ref_pointer
        self.ref = self._load_reference(ref_pointer)

        self._create_mask(mask_source, device=device)
        names_dict = self._compartment_names_dict(**kwargs)
        self._create_compartments(names_dict)

        self._configure_decoders()
        self._configure_compartment_maps(dtype=dtype, device=device)
        self._init_coors(source=mask_source, names_dict=names_dict,
                         dtype=dtype, device=device)

    def __call__(self, compartments=None, normalise=False, sigma=None,
                 noise=None, max_bin=10000, spherical_scale=1, truncate=None):
        ret = {}
        if compartments is None:
            compartments = ['_all']
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


class DiscreteVolumetricAtlas(
    BaseAtlas,
    _SingleReferenceMixin,
    _MaskFromNullMixin,
    _SingleCompartmentMixin,
    _DiscreteLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin
):
    def __init__(self, ref_pointer, dtype=None, device=None,):
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=0,
                         dtype=dtype,
                         device=device)


class MultiVolumetricAtlas(
    BaseAtlas,
    _SingleReferenceMixin,
    _MaskFromNullMixin,
    _SingleCompartmentMixin,
    _ContinuousLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin
):
    def __init__(self, ref_pointer, dtype=None, device=None,):
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=0,
                         dtype=dtype,
                         device=device)


class MultifileVolumetricAtlas(
    BaseAtlas,
    _MultiReferenceMixin,
    _MaskFromNullMixin,
    _SingleCompartmentMixin,
    _ContinuousLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin
):
    def __init__(self, ref_pointer, dtype=None, device=None,):
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=0,
                         dtype=dtype,
                         device=device)


class CortexSubcortexCIfTIAtlas(
    BaseAtlas,
    _CIfTIReferenceMixin,
    _SingleReferenceMixin,
    _CortexSubcortexMaskCIfTIMixin,
    _CortexSubcortexCompartmentCIfTIMixin,
    _DiscreteLabelMixin,
    _VertexCoordinatesCIfTIMixin,
    _SpatialConvMixin
):
    def __init__(self, ref_pointer,
                 mask_L=None, mask_R=None, surf_L=None, surf_R=None,
                 cortex_L='CIFTI_STRUCTURE_CORTEX_LEFT',
                 cortex_R='CIFTI_STRUCTURE_CORTEX_RIGHT',
                 dtype=None, device=None):
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
                         dtype=dtype,
                         device=device,
                         cortex_L=cortex_L,
                         cortex_R=cortex_R)


class _MemeAtlas(
    BaseAtlas,
    _SingleReferenceMixin,
    _MaskLogicMixin,
    _MultiCompartmentMixin,
    _DiscreteLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin
):
    def __init__(self):
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


#TODO: atlas without a reference for anything except coors
class DirichletInitBaseAtlas(
    BaseAtlas,
    _PhantomReferenceMixin,
    _MaskFileMixin,
    _SingleCompartmentMixin,
    _DirichletLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin
):
    def __init__(self, mask_source, compartment_labels, conc=100.,
                 init=None, dtype=None, device=None):
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
        super().__init__(ref_pointer=mask_source,
                         mask_source=mask_source,
                         dtype=dtype,
                         device=device)

    def _global_compartment_init(self):
        if self.init.get('_all'):
            return
        if self.init.get('all'):
            self.init['_all'] = self.init['all']
            return
        concentrations = [d.concentration for d in self.init.values()]
        concentrations = torch.cat(concentrations)
        self.init['_all'] = DirichletInit(concentrations)


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
