# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas mixins
~~~~~~~~~~~~
Mixins for designing atlas classes.
"""
import torch
import numpy as np
import nibabel as nb
from collections import OrderedDict
from pathlib import PosixPath
from scipy.ndimage import gaussian_filter
from ..functional.sphere import spherical_conv, euclidean_conv


def _to_mask(path):
    return nb.load(path).get_fdata().round().astype(bool)


def _is_path(obj):
    return isinstance(obj, str) or isinstance(obj, PosixPath)


class FloatLeaf:
    def __init__(self, img):
        self.img = img

    def __call__(self, nifti=False):
        if not nifti:
            return nb.load(self.img).get_fdata()
        else:
            img = nb.load(self.img)
            return {
                'affine': img.affine,
                'header': img.header,
                'dataobj': img.get_fdata()
            }


class MaskLeaf:
    """
    Leaf node for mask logic operations.
    """
    def __init__(self, mask):
        self.mask = mask

    def __call__(self, nifti=False):
        if not nifti:
            mask = _to_mask(self.mask)
            return mask
        else:
            img = nb.load(self.mask)
            mask = img.get_fdata().round().astype(bool)
            return {
                'affine': img.affine,
                'header': img.header,
                'dataobj': mask
            }


class MaskThreshold:
    def __init__(self, child, threshold):
        self.threshold = threshold
        if _is_path(child):
            self.child = FloatLeaf(child)
        else:
            self.child = child

    def __call__(self, nifti=False):
        if not nifti:
            return (self.child() >= self.threshold)
        else:
            img = self.child(nifti=True)
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': (img['dataobj'] >= self.threshold)
            }


class MaskUThreshold:
    def __init__(self, child, threshold):
        self.threshold = threshold
        if _is_path(child):
            self.child = FloatLeaf(child)
        else:
            self.child = child

    def __call__(self, nifti=False):
        if not nifti:
            return (self.child() <= self.threshold)
        else:
            img = self.child(nifti=True)
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': (img['dataobj'] <= self.threshold)
            }


class MaskNegation:
    """
    Negation node for mask logic operations.
    """
    def __init__(self, child):
        if _is_path(child):
            self.child = MaskLeaf(child)
        else:
            self.child = child

    def __call__(self, nifti=False):
        if not nifti:
            return ~self.child()
        else:
            img = self.child(nifti=True)
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': ~img['dataobj']
            }


class MaskUnion:
    """
    Union node for mask logic operations.
    """
    def __init__(self, *children):
        self.children = [
            child if not _is_path(child) else MaskLeaf(child)
            for child in children
        ]

    def __call__(self, nifti=False):
        child = self.children[0]
        if not nifti:
            mask = child()
            for child in self.children[1:]:
                mask = mask + child()
            return mask
        else:
            img = child(nifti=True)
            dataobj = img['dataobj']
            for child in self.children[1:]:
                cur = child(nifti=True)
                mask = cur['dataobj']
                dataobj = dataobj + mask
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': dataobj
            }


class MaskIntersection:
    """
    Intersection node for mask logic operations.
    """
    def __init__(self, *children):
        self.children = [
            child if not _is_path(child) else MaskLeaf(child)
            for child in children
        ]

    def __call__(self, nifti=False):
        child = self.children[0]
        if not nifti:
            mask = child()
            for child in self.children[1:]:
                mask = mask * child()
            return mask
        else:
            img = child(nifti=True)
            dataobj = img['dataobj']
            for child in self.children[1:]:
                cur = child(nifti=True)
                mask = cur['dataobj']
                dataobj = dataobj * mask
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': dataobj
            }


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
        try:
            ref = nb.load(ref_pointer)
        except TypeError:
            ref = nb.Nifti1Image(**ref_pointer(nifti=True))
        try: # Volumetric NIfTI
            affine = ref.affine
            header = ref.header
            dataobj = _PhantomDataobj(ref)
            self.ref = nb.Nifti1Image(
                header=header, affine=affine, dataobj=dataobj
            )
        except AttributeError: # No affine: Surface/CIfTI
            self.ref = nb.load(ref_pointer)
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


class _LogicMaskMixin:
    def _create_mask(self, source, device=None):
        if _is_path(source):
            source = MaskLeaf(source)
        init = source()
        self.mask = torch.tensor(init.ravel(), device=device)


class _CortexSubcortexCIfTIMaskMixin:
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


class _FromNullMaskMixin:
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

        self.compartments = OrderedDict()
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


class _CortexSubcortexCIfTICompartmentMixin:
    def _compartment_names_dict(self, **kwargs):
        return kwargs

    def _create_compartments(self, names_dict, ref=None):
        ref = ref or self.ref
        self.compartments = OrderedDict([
            ('cortex_L', torch.zeros(
                self.mask.shape,
                dtype=torch.bool,
                device=self.mask.device)),
            ('cortex_R', torch.zeros(
                self.mask.shape,
                dtype=torch.bool,
                device=self.mask.device)),
            ('subcortex', torch.zeros(
                self.mask.shape,
                dtype=torch.bool,
                device=self.mask.device)),
        ])
        # This could not be more stupid. All thanks to the amazing design
        # choice that indexing returns a copy in numpy/torch.
        if self.mask.shape == self.ref.shape[-1]:
            mask = torch.ones(
                self.mask.shape,
                dtype=torch.bool,
                device=self.mask.device
            )
        else:
            mask = self.mask
        model_axis = self.model_axis
        for struc, slc, _ in (model_axis.iter_structures()):
            if struc == names_dict['cortex_L']:
                self._mask_hack(mask, 'cortex_L', slc)
            elif struc == names_dict['cortex_R']:
                self._mask_hack(mask, 'cortex_R', slc)
        try:
            vol_mask = np.where(model_axis.volume_mask)[0]
            vol_min, vol_max = vol_mask.min(), vol_mask.max() + 1
            slc = slice(vol_min, vol_max)
            self._mask_hack(mask, 'subcortex', slc)
        except ValueError:
            pass

    def _mask_hack(self, src_mask, struc, slc):
        compartment_mask = src_mask.clone()
        inner_compartment_mask = torch.zeros(
            compartment_mask.sum(),
            dtype=torch.bool,
            device=compartment_mask.device
        )
        inner_compartment_mask[(slc,)] = True
        compartment_mask[compartment_mask.clone()] = (
            inner_compartment_mask)
        self.compartments[struc][compartment_mask] = True


class _DiscreteLabelMixin:
    def _configure_decoders(self, null_label=0):
        self.decoder = OrderedDict()
        for c, mask in self.compartments.items():
            try:
                mask = mask.reshape(self.ref.shape)
            except RuntimeError:
                mask = mask[self.mask].reshape(self.ref.shape)
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
                    self.cached_ref_data.ravel()[mask[self.mask]] == l.item())
        return map


class _ContinuousLabelMixin:
    def _configure_decoders(self, null_label=None):
        self.decoder = OrderedDict()
        for c, mask in self.compartments.items():
            mask = mask.reshape(self.ref.shape[:-1])
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
        self.decoder = OrderedDict()
        n_labels = 0
        for c, i in self.compartment_labels.items():
            if i == 0:
                self.decoder[c] = torch.tensor(
                    [], dtype=torch.long, device=self.mask.device)
                continue
            self.decoder[c] = torch.arange(
                n_labels, n_labels + i,
                dtype=torch.long, device=self.mask.device)
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
        self.topology = OrderedDict(
            (c, 'euclidean') for c in self.compartments.keys())


class _VertexCIfTIMeshMixin:
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
        self.topology = OrderedDict()
        euc_mask = torch.BoolTensor(self.model_axis.volume_mask)
        for c, mask in self.compartments.items():
            if mask.shape != euc_mask.shape:
                mask = mask[self.mask]
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
        if len(self.coors) < len(compartment_mask):
            compartment_mask = compartment_mask[self.mask]
        if self.topology[compartment] == 'euclidean':
            map = euclidean_conv(
                data=map.T, coor=self.coors[compartment_mask],
                scale=sigma, max_bin=max_bin,
                truncate=truncate
            ).T
        elif self.topology[compartment] == 'spherical':
            map = spherical_conv(
                data=map.T, coor=self.coors[compartment_mask],
                scale=(spherical_scale * sigma), r=100,
                max_bin=max_bin, truncate=truncate
            ).T
        return map
