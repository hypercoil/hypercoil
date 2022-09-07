# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Mixins for designing atlas classes.
"""
import jax
import numpy as np
import nibabel as nb
import jax.numpy as jnp
import equinox as eqx
from collections import OrderedDict
from pathlib import Path, PosixPath
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
from scipy.ndimage import (
    gaussian_filter,
    binary_dilation,
    binary_erosion,
    binary_opening,
    binary_closing,
    binary_fill_holes
)
from ..engine import Tensor
from ..functional.sphere import spherical_conv, euclidean_conv


#TODO: Consider caching the ref as a tensor for possible speedup/efficiency


def _to_mask(path):
    return nb.load(path).get_fdata().round().astype(bool)


def _is_path(obj):
    return isinstance(obj, str) or isinstance(obj, PosixPath)


class FloatLeaf:
    """
    Leaf node for floating-point data in mask logic operations. Used with
    thresholding operations.
    """
    def __init__(self, img):
        self.img = img

    def __call__(self, nifti=False):
        """
        Load the specified image as a floating point-valued array. If
        ``nifti`` is set to True (default False), then the call returns values
        for all fields required to initialise a ``nibabel`` NIfTI image
        object.
        """
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
        """
        Load the specified image as a Boolean-valued array. If ``nifti`` is
        set to True (default False), then the call returns values for all
        fields required to initialise a ``nibabel`` NIfTI image object.
        """
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
    """
    Create a mask by thresholding and binarising a continuous-valued image.
    ``MaskThreshold`` zeros all values less than the specified threshold and
    then uses surviving (larger) values as a Boolean mask. To zero larger
    values, use ``MaskUThreshold``.
    """
    def __init__(self, child, threshold):
        self.threshold = threshold
        if _is_path(child):
            self.child = FloatLeaf(child)
        else:
            self.child = child

    def __call__(self, nifti=False):
        """
        Apply the threshold operation. If ``nifti`` is set to True (default
        False), then the call returns values for all fields required to
        initialise a ``nibabel`` NIfTI image object.
        """
        if not nifti:
            return (self.child() > self.threshold)
        else:
            img = self.child(nifti=True)
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': (img['dataobj'] >= self.threshold)
            }


class MaskUThreshold:
    """
    Create a mask by thresholding and binarising a continuous-valued image.
    ``MaskUThreshold`` zeros all values greater than the specified threshold
    and then uses surviving (smaller) values as a Boolean mask. To zero
    smaller values, use ``MaskThreshold``.
    """
    def __init__(self, child, threshold):
        self.threshold = threshold
        if _is_path(child):
            self.child = FloatLeaf(child)
        else:
            self.child = child

    def __call__(self, nifti=False):
        """
        Apply the upper threshold operation. If ``nifti`` is set to True
        (default False), then the call returns values for all fields required
        to initialise a ``nibabel`` NIfTI image object.
        """
        if not nifti:
            return (self.child() < self.threshold)
        else:
            img = self.child(nifti=True)
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': (img['dataobj'] <= self.threshold)
            }


class MaskDilation:
    """
    Morphological dilation node for mask logic operations. Returns the
    binary dilation of the mask output by child operations.
    """
    def __init__(self, child, structure=None, iterations=1):
        if _is_path(child):
            self.child = MaskLeaf(child)
        else:
            self.child = child
        self.structure = structure
        self.iterations = iterations

    def __call__(self, nifti=False):
        """
        Apply the morphological dilation operation. If ``nifti`` is set to
        True (default False), then the call returns values for all fields
        required to initialise a ``nibabel`` NIfTI image object.
        """
        if not nifti:
            return binary_dilation(
                input=self.child(),
                structure=self.structure,
                iterations=self.iterations
            )
        else:
            img = self.child(nifti=True)
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': binary_dilation(
                    input=img['dataobj'],
                    structure=self.structure,
                    iterations=self.iterations
                )
            }


class MaskErosion:
    """
    Morphological erosion node for mask logic operations. Returns the
    binary erosion of the mask output by child operations.
    """
    def __init__(self, child, structure=None, iterations=1):
        if _is_path(child):
            self.child = MaskLeaf(child)
        else:
            self.child = child
        self.structure = structure
        self.iterations = iterations

    def __call__(self, nifti=False):
        """
        Apply the morphological erosion operation. If ``nifti`` is set to True
        (default False), then the call returns values for all fields required
        to initialise a ``nibabel`` NIfTI image object.
        """
        if not nifti:
            return binary_erosion(
                input=self.child(),
                structure=self.structure,
                iterations=self.iterations
            )
        else:
            img = self.child(nifti=True)
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': binary_erosion(
                    input=img['dataobj'],
                    structure=self.structure,
                    iterations=self.iterations
                )
            }


class MaskOpening:
    """
    Morphological opening node for mask logic operations. Returns the
    binary opening of the mask output by child operations.
    """
    def __init__(self, child, structure=None, iterations=1):
        if _is_path(child):
            self.child = MaskLeaf(child)
        else:
            self.child = child
        self.structure = structure
        self.iterations = iterations

    def __call__(self, nifti=False):
        """
        Apply the morphological opening operation. If ``nifti`` is set to True
        (default False), then the call returns values for all fields required
        to initialise a ``nibabel`` NIfTI image object.
        """
        if not nifti:
            return binary_opening(
                input=self.child(),
                structure=self.structure,
                iterations=self.iterations
            )
        else:
            img = self.child(nifti=True)
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': binary_opening(
                    input=img['dataobj'],
                    structure=self.structure,
                    iterations=self.iterations
                )
            }


class MaskClosing:
    """
    Morphological closing node for mask logic operations. Returns the
    binary closing of the mask output by child operations.
    """
    def __init__(self, child, structure=None, iterations=1):
        if _is_path(child):
            self.child = MaskLeaf(child)
        else:
            self.child = child
        self.structure = structure
        self.iterations = iterations

    def __call__(self, nifti=False):
        """
        Apply the morphological closing operation. If ``nifti`` is set to True
        (default False), then the call returns values for all fields required
        to initialise a ``nibabel`` NIfTI image object.
        """
        if not nifti:
            return binary_closing(
                input=self.child(),
                structure=self.structure,
                iterations=self.iterations
            )
        else:
            img = self.child(nifti=True)
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': binary_closing(
                    input=img['dataobj'],
                    structure=self.structure,
                    iterations=self.iterations
                )
            }


class MaskFillHoles:
    """
    Morphological hole-filling node for mask logic operations. Transforms the
    mask output by child operations by filling any holes in the mask.
    """
    def __init__(self, child, structure=None):
        if _is_path(child):
            self.child = MaskLeaf(child)
        else:
            self.child = child
        self.structure = structure

    def __call__(self, nifti=False):
        """
        Apply the morphological hole-filling operation. If ``nifti`` is set to
        True (default False), then the call returns values for all fields
        required to initialise a ``nibabel`` NIfTI image object.
        """
        if not nifti:
            return binary_fill_holes(
                input=self.child(),
                structure=self.structure,
            )
        else:
            img = self.child(nifti=True)
            return {
                'affine': img['affine'],
                'header': img['header'],
                'dataobj': binary_fill_holes(
                    input=img['dataobj'],
                    structure=self.structure,
                )
            }


class MaskNegation:
    """
    Negation node for mask logic operations. Returns the negation of the mask
    output by child operations.
    """
    def __init__(self, child):
        if _is_path(child):
            self.child = MaskLeaf(child)
        else:
            self.child = child

    def __call__(self, nifti=False):
        """
        Apply the negation operation, recursively calling the operations trees
        of all child nodes. If ``nifti`` is set to True (default False), then
        the call returns values for all fields required to initialise a
        ``nibabel`` NIfTI image object.
        """
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
    Union node for mask logic operations. Returns the union of all masks
    output by child operations.
    """
    def __init__(self, *children):
        self.children = [
            child if not _is_path(child) else MaskLeaf(child)
            for child in children
        ]

    def __call__(self, nifti=False):
        """
        Apply the union operation, recursively calling the operations trees
        of all child nodes. If ``nifti`` is set to True (default False), then
        the call returns values for all fields required to initialise a
        ``nibabel`` NIfTI image object.
        """
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
    Intersection node for mask logic operations. Returns the intersection of
    all masks output by child operations.
    """
    def __init__(self, *children):
        self.children = [
            child if not _is_path(child) else MaskLeaf(child)
            for child in children
        ]

    def __call__(self, nifti=False):
        """
        Apply the intersection operation, recursively calling the operations
        trees of all child nodes. If ``nifti`` is set to True (default False),
        then the call returns values for all fields required to initialise a
        ``nibabel`` NIfTI image object.
        """
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
    Use to load a reference into an atlas class.

    For use when a NIfTI image object is already provided as the
    ``ref_pointer`` argument.
    """
    def _load_reference(
        self,
        ref_pointer: 'nb.nifti1.Nifti1Image',
    ) -> 'nb.nifti1.Nifti1Image':
        self.cached_ref_data = ref_pointer.get_fdata()
        return ref_pointer


class _SingleReferenceMixin:
    """
    Use to load a reference into an atlas class.

    For use when the ``ref_pointer`` object references a single path to an
    image on disk.
    """
    def _load_reference(
        self,
        ref_pointer: Union[str, Path],
    ) -> 'nb.nifti1.Nifti1Image':
        ref = nb.load(ref_pointer)
        self.cached_ref_data = ref.get_fdata()
        return ref


class _MultiReferenceMixin:
    """
    Use to load a reference into an atlas class.

    For use when the ``ref_pointer`` object is an iterable of paths to images
    on disk.
    """
    def _load_reference(
        self,
        ref_pointer: Sequence[Union[str, Path]],
    ) -> 'nb.nifti1.Nifti1Image':
        ref = [nb.load(path) for path in ref_pointer]
        self.cached_ref_data = np.stack([r.get_fdata() for r in ref], -1)
        ref = nb.Nifti1Image(
            dataobj=np.copy(self.cached_ref_data),
            affine=ref[0].affine,
            header=ref[0].header
        )
        return ref


class _PhantomReferenceMixin:
    """
    Use to load a reference into an atlas class.

    For use when the data content of the reference is unimportant, for
    instance when instances of the atlas class are to be initialised from a
    distribution rather than from an existing reference. The reference is
    still used for inferring dimensions of the atlas and input images.
    """
    def _load_reference(
        self,
        ref_pointer: Union[str, Path],
    ) -> 'nb.nifti1.Nifti1Image':
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


class _PhantomDataobj(eqx.Module):
    """
    For tricking ``nibabel`` into instantiating ``Nifti1Image`` objects
    without any real data. Used to reduce data overhead when initialising
    atlases from distributions.
    """

    shape : tuple
    ndim : int

    def __init__(self, base):
        self.shape = base.shape
        self.ndim = base.ndim


class _CIfTIReferenceMixin:
    """
    Use if an atlas uses a CIfTI as its reference. This class implements the
    additional methods ``axes`` and ``model_axis``, which facilitate access to
    CIfTI model axes. Note that this is *not* a substitute for a reference
    loader mixin like ``_ObjectReferenceMixin`` or ``_SingleReferenceMixin``.
    """
    @property
    def axes(self) -> Tuple:
        """
        Thanks to Chris Markiewicz for tutorials that shaped this
        implementation.
        """
        return tuple([
            self.ref.header.get_axis(i)
            for i in range(self.ref.ndim)
        ])

    @property
    def model_axis(self) -> int:
        return [a for a in self.axes
                if isinstance(a, nb.cifti2.cifti2_axes.BrainModelAxis)][0]

    def to_image(
        self,
        save: Optional[str] = None,
        maps: Optional[Dict[str, Tensor]] = None
    ) -> Optional['nb.nifti1.Nifti1Image']:
        if maps is None:
            maps = self.maps
        offset = 1
        dataobj = np.zeros_like(self.ref.get_fdata())
        for k, v in maps.items():
            n_labels = v.shape[0]
            mask = self.compartments[k][self.mask][None, ...]
            data = v.argmax(0) + offset
            dataobj[mask] = data
            offset += n_labels
        new_cifti = nb.Cifti2Image(
            dataobj,
            header=self.ref.header,
            nifti_header=self.ref.nifti_header
        )
        if save is None:
            return new_cifti
        new_cifti.to_filename(save)


class _LogicMaskMixin:
    """
    Use to create an overall mask that specifies atlas inclusion status of
    spatial locations.

    For use when the mask source is either a path on the file system
    containing Boolean-valued data or a nested logical expression tree
    (comprising operation nodes such as ``MaskIntersection`` or
    ``MaskThreshold`` with filesystem paths as leaf nodes).
    """
    def _create_mask(
        self,
        source: Union[str, Path, Callable],
    ) -> None:
        if _is_path(source):
            source = MaskLeaf(source)
        init = source()
        self.mask = jnp.asarray(init.ravel())


class _CortexSubcortexCIfTIMaskMixin:
    """
    Use to create an overall mask that specifies atlas inclusion status of
    spatial locations.

    For use when creating a CIfTI atlas with separate cortical and subcortical
    compartments, and when the provided mask source indicates medial wall
    regions of the cortical surface marked for exclusion from the atlas.
    """
    def _create_mask(
        self,
        source: Dict[str, Union[str, Path]],
    ) -> None:
        init = []
        for v in source.values():
            try:
                init += [
                    nb.load(v).darrays[0].data.round().astype(bool)
                ]
            except Exception:
                init += [
                    np.ones(self.model_axis.volume_mask.sum()).astype(bool)
                ]
        self.mask = jnp.asarray(np.concatenate(init))


class _FromNullMaskMixin:
    """
    Use to create an overall mask that specifies atlas inclusion status of
    spatial locations.

    For use when automatically creating a mask by excluding all background-
    or null-valued (typically 0-valued) spatial locations from the reference.
    For single-volume references (typically discrete-valued), this mixin
    creates a mask that includes all locations not labelled as background.
    For multi-volume references (often continuous-valued), this mixin
    creates a mask that includes all locations that are greater than or equal
    to the provided source parameter after taking the sum across volumes.
    """
    def _create_mask(
        self,
        source: float = 0.0,
    ) -> None:
        if self.ref.ndim <= 3:
            init = (self.cached_ref_data.round() != source)
            self.mask = jnp.asarray(init.ravel())
        else:
            init = (self.cached_ref_data.sum(-1) > source)
            self.mask = jnp.asarray(init.ravel())


class _SingleCompartmentMixin:
    """
    Use to isolate spatial subcompartments of the overall atlas such that each
    has separate label sets.

    For use when no isolation is desired, and the entire atlased region is a
    single compartment.
    """
    def _compartment_names_dict(
        self,
        **kwargs
    ) -> Dict[str, str]:
        return {}

    def _create_compartments(
        self,
        names_dict: Dict[str, Tensor] = None,
        ref: Optional['nb.nifti1.Nifti1Image'] = None
    ) -> None:
        ref = ref or self.ref

        if ref.ndim == 2: # surface, time by vox greyordinates
            self.compartments = {
                'all': self.mask
            }
        else: # volume, x by y by z by t
            self.compartments = {
                'all' : self.mask
            }


#TODO: Intersect compartment mask with overall mask before saving.
#TODO: (low-priority) We need to either make sure these work with CIfTI, or we
# need to make CIfTI-compatible compartment masking other than
# cortex/subcortex.
class _MultiCompartmentMixin:
    """
    Use to isolate spatial subcompartments of the overall atlas such that each
    has separate label sets.

    For use when isolation into multiple compartments is desired. With this
    mixin, each extra keyword argument passed to the atlas constructor is
    interpreted as a name-mask path pair defining an atlas compartment.
    """
    def _compartment_names_dict(
        self,
        **kwargs
    ) -> Dict[str, str]:
        return kwargs

    def _create_compartments(
        self,
        names_dict: Dict[Union[str, Tuple[str]], Tensor] = None,
        ref: Optional['nb.nifti1.Nifti1Image'] = None
    ) -> None:
        ref = ref or self.ref

        self.compartments = OrderedDict()
        for name, vol in names_dict.items():
            if isinstance(name, str):
                self.compartments[name] = jnp.asarray(
                    _to_mask(vol)).ravel()
            else:
                init = _to_mask(vol)
                for name, data in zip(name, init):
                    self.compartments[name] = jnp.asarray(
                        data).ravel()


class _CortexSubcortexCIfTICompartmentMixin:
    """
    Use to isolate spatial subcompartments of the overall atlas such that each
    has separate label sets.

    For use when creating a CIfTI-based atlas with separate subcompartments
    for the left and right cortical hemispheres and for subcortical locations.
    """
    def _compartment_names_dict(self, **kwargs) -> Dict[str, str]:
        return kwargs

    def _create_compartments(self, names_dict, ref=None):
        ref = ref or self.ref
        self.compartments = OrderedDict([
            ('cortex_L', jnp.zeros(
                self.mask.shape, dtype=bool,)),
            ('cortex_R', jnp.zeros(
                self.mask.shape, dtype=bool,)),
            ('subcortex', jnp.zeros(
                self.mask.shape, dtype=bool,)),
        ])
        #TODO
        # This could not be more stupid. Gotta be a better way to do this.
        if self.mask.shape == self.ref.shape[-1]:
            mask = jnp.ones(self.mask.shape, dtype=bool,)
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

    def _mask_hack(
        self,
        src_mask: Tensor,
        struc: str,
        slc: slice,
    ) -> None:
        compartment_mask = src_mask.copy()
        inner_compartment_mask = jnp.zeros(
            compartment_mask.sum(), dtype=bool,)
        inner_compartment_mask = inner_compartment_mask.at[(slc,)].set(True)
        compartment_mask = compartment_mask.at[compartment_mask].set(inner_compartment_mask)
        self.compartments[struc] = jnp.where(compartment_mask, True, False)
        #self.compartments[struc] = self.compartments[struc].at[compartment_mask].set(True)


class _DiscreteLabelMixin:
    """
    Use to decode label sets present in an atlas and to create a linear map
    representation of the atlas.

    For use when the label sets are encoded as discrete values in a single
    reference volume or surface.
    """
    def _configure_decoders(
        self,
        null_label: int = 0
    ) -> None:
        self.decoder = OrderedDict()
        for c, mask in self.compartments.items():
            try:
                mask = mask.reshape(self.ref.shape)
            except jax.core.InconclusiveDimensionOperation:
                mask = mask[self.mask].reshape(self.ref.shape)
            labels_in_compartment = np.unique(
                jnp.asarray(self.cached_ref_data)[mask])
            labels_in_compartment = labels_in_compartment[
                labels_in_compartment != null_label]
            self.decoder[c] = jnp.asarray(
                labels_in_compartment, dtype=int)

        try:
            mask = self.mask.reshape(self.ref.shape)
        except jax.core.InconclusiveDimensionOperation:
            # The reference is already masked. In this case, we're using a
            # CIfTI.
            assert self.mask.sum() == self.ref.shape[-1]
            mask = True
        unique_labels = np.unique(self.cached_ref_data[mask])
        unique_labels = unique_labels[unique_labels != null_label]
        self.decoder['_all'] = jnp.asarray(
            unique_labels, dtype=int)

    def _populate_map_from_ref(
        self,
        map: Dict[str, Tensor],
        labels: Dict[int, int],
        mask: Tensor,
        compartment: Optional[str] = None
    ) -> Dict[str, Tensor]:
        for i, l in enumerate(labels):
            try:
                map = map.at[i].set(jnp.asarray(
                    self.cached_ref_data.ravel()[mask] == l))
            except IndexError:
                # Again the reference is already masked. In this case, we
                # assume we're using a CIfTI.
                assert self.mask.sum() == len(self.cached_ref_data.ravel())
                map = map.at[i].set(jnp.asarray(
                    self.cached_ref_data.ravel()[mask[self.mask]] == l))
        return map


class _ContinuousLabelMixin:
    """
    Use to decode label sets present in an atlas and to create a linear map
    representation of the atlas.

    For use when the label sets are encoded across multiple volumes. This is
    necessary for continuous-valued atlases or atlases with overlapping
    labels, but is also a valid encoding scheme for discrete-valued atlases.
    If the reference uses a single volume or surface, use
    ``DiscreteLabelMixin`` instead.
    """
    def _configure_decoders(
        self,
        null_label: Optional[int] = None
    ) -> None:
        self.decoder = OrderedDict()
        for c, mask in self.compartments.items():
            mask = mask.reshape(self.ref.shape[:-1])
            labels_in_compartment = np.where(
                self.cached_ref_data[mask].sum(0))[0]
            labels_in_compartment = labels_in_compartment[
                labels_in_compartment != null_label]
            self.decoder[c] = jnp.asarray(
                labels_in_compartment + 1,
                dtype=int,
            )

        mask = self.mask.reshape(self.ref.shape[:-1])
        unique_labels = np.where(self.cached_ref_data[mask].sum(0))[0]
        unique_labels = labels_in_compartment[unique_labels != null_label]
        self.decoder['_all'] = jnp.asarray(unique_labels + 1, dtype=int)

    def _populate_map_from_ref(
        self,
        map: Dict[str, Tensor],
        labels: Dict[int, int],
        mask: Tensor,
        compartment: Optional[str] = None
    ) -> Dict[str, Tensor]:
        ref_data = np.moveaxis(
            self.cached_ref_data,
            (0, 1, 2, 3),
            (1, 2, 3, 0)
        ).squeeze()
        for i, l in enumerate(labels):
            map = map.at[i].set(jnp.asarray(ref_data[(l - 1)].ravel()[mask]))
        return map


class _DirichletLabelMixin:
    """
    Use to decode label sets present in an atlas and to create a linear map
    representation of the atlas.

    For use when the linear map is to be initialised from a Dirichlet
    distribution rather than a reference. This requires the prior existence of
    a dictionary attribute called ``compartment_labels`` for the Atlas object,
    whose key-value pairs associate to each atlas compartment an integer
    number of labels. It additionally requires a second key-value mapping
    ``init``, whose entries associate to each atlas compartment the Dirichlet
    distribution from which that compartment's parcel assignment probability
    distributions are to be sampled. These mappings can be instantiated in the
    atlas class's constructor method, potentially from user arguments.
    """
    def _configure_decoders(
        self,
        null_label: Optional[int] = None
    ) -> None:
        self.decoder = OrderedDict()
        n_labels = 0
        for c, i in self.compartment_labels.items():
            if i == 0:
                self.decoder[c] = jnp.array([], dtype=int,)
                continue
            self.decoder[c] = jnp.arange(
                n_labels, n_labels + i, dtype=int,) + 1
            n_labels += i
        self.decoder['_all'] = jnp.arange(0, n_labels, dtype=int,) + 1

    def _populate_map_from_ref(
        self,
        map: Dict[str, Tensor],
        labels: Dict[int, int],
        mask: Tensor,
        compartment=None
    ) -> Dict[str, Tensor]:
        map = self.init[compartment](
            model=map,
            param_name=None
        )
        return map


class _VolumetricMeshMixin:
    """
    Used to establish a coordinate system over the linear map representations
    of the atlas.

    For use when the atlas reference comprises evenly spaced volumetric
    samples (i.e., voxels).
    """
    def _init_coors(
        self,
        source: Optional[Any] = None,
        names_dict: Optional[Any] = None,
    ) -> None:
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
        self.coors = jnp.asarray(
            axes.reshape(i + 1, -1).T,
        )
        self.topology = OrderedDict(
            (c, 'euclidean') for c in self.compartments.keys())


class _VertexCIfTIMeshMixin:
    """
    Used to establish a coordinate system over the linear map representations
    of the atlas.

    For use when the atlas reference is a CIfTI that includes some samples
    associated to cortical surface meshes. This mixin establishes a spherical
    topology for cortical samples and a Euclidean topology for subcortical
    samples.
    """
    def _init_coors(
        self,
        source: Optional[Any] = None,
        names_dict: Optional[Any] = None,
    ) -> None:
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
        self.coors = jnp.asarray(coor)
        self.topology = OrderedDict()
        euc_mask = jnp.asarray(
            self.model_axis.volume_mask, dtype=bool)
        for c, mask in self.compartments.items():
            if mask.shape != euc_mask.shape:
                mask = mask[self.mask]
            if (mask * euc_mask).sum() == 0:
                self.topology[c] = 'spherical'
            else:
                self.topology[c] = 'euclidean'


class _EvenlySampledConvMixin:
    """
    Used to spatially convolve atlas parcels for smoothing.

    This mixin is currently unsupported and likely will not function without
    substantial extra code, although it is likely to perform better under
    many conditions. Its use is not currently advised.
    """
    def _configure_sigma(
        self,
        sigma: Union[float, None],
    ) -> Union[float, None]:
        if sigma is not None:
            scale = self.ref.header.get_zooms()[:3]
            sigma = [sigma / s for s in scale]
            sigma = [0] + [sigma]
        return sigma

    def _convolve(
        self,
        sigma: Union[float, None],
        map: Tensor
    ) -> Tensor:
        if sigma is not None:
            gaussian_filter(map, sigma=sigma, output=map)
        return map


class _SpatialConvMixin:
    """
    Used to spatially convolve atlas parcels for smoothing.

    For use when a coordinate mesh is available for the atlas and either
    Euclidean or spherical topologies are assigned to each of its
    compartments.
    """
    def _configure_sigma(
        self,
        sigma: Union[float, None],
    ) -> Union[float, None]:
        return sigma

    def _convolve(
        self,
        map: Tensor,
        compartment: str,
        sigma: Union[float, None],
        max_bin: int = 10000,
        spherical_scale: float = 1,
        truncate: Optional[float] = None
    ) -> Tensor:
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
