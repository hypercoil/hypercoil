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
from functools import reduce
from pathlib import Path, PosixPath
from typing import (
    Any, Callable, Dict, Generator, Literal,
    Mapping, Optional, Sequence, Tuple, Union
)
from scipy.ndimage import (
    gaussian_filter,
)

from hypercoil.formula.imops import (
    ImageMathsGrammar as ImageMaths,
    NiftiFileInterpreter,
    NiftiObjectInterpreter
)
from ..engine import (
    Tensor, standard_axis_number, promote_axis, demote_axis, axis_complement
)
from ..engine.paramutil import _to_jax_array
from ..functional.linear import form_dynamic_slice
from ..functional.sphere import spherical_conv, euclidean_conv
from ..functional.utils import conform_mask


#TODO: Mixins for building an atlas class from a parameterised atlas module
#      object instead of a NIfTI image.
#TODO: Work out what should be in `numpy` and what should be in `jax.numpy`.


class CiftiError(Exception):
    pass


def modelobj_from_dataobj_and_maskobj(
    dataobj: Tensor,
    model_axes: Sequence[int],
    mask: Optional['Mask'] = None,
    model_axis_out: Optional[int] = None,
) -> Tensor:
    dataobj_model_shape = reduce(
        lambda x, y: x * y,
        (dataobj.shape[ax] for ax in model_axes)
    )
    if isinstance(dataobj, _PhantomDataobj):
        return _PhantomDataobj(
            shape=(dataobj_model_shape, Ellipsis)
        ) #TODO: not sure this is always the right thing to do
    if mask.shape[0] == dataobj_model_shape:
        return modelobj_from_dataobj(
            dataobj,
            model_axes,
            mask=mask.mask_array,
            model_axis_out=model_axis_out,
        )
    else:
        return modelobj_from_dataobj(
            dataobj,
            model_axes,
            mask=jnp.ones(dataobj_model_shape, dtype=bool),
            model_axis_out=model_axis_out,
        )


#TODO: we should ensure this is JIT-able when mask is None. Obviously won't
#      work when mask is not None.
def modelobj_from_dataobj(
    dataobj: Tensor,
    model_axes: Sequence[int],
    mask: Optional[Tensor] = None,
    model_axis_out: Optional[int] = None,
) -> Tensor:
    model_ndim = len(model_axes)
    axes = promote_axis(dataobj.ndim, model_axes)
    modelobj = dataobj.transpose(axes)
    modelobj = modelobj.reshape((-1, *modelobj.shape[model_ndim:]))
    if mask is not None:
        modelobj = modelobj[mask]
    if model_axis_out is not None:
        axes = demote_axis(modelobj.ndim, model_axis_out)
        modelobj = modelobj.transpose(axes)
    return modelobj


#TODO: (low priority): revisit this later. Probably begin downstream with
#      decoders, topologies, etc. Reference types (e.g., surface,
#      multi-volume) might just be too different to have any hope of
#      harmonisation.
#
#      Update: I think we mostly worked this out in the end using model axes.
#      We'll see how it goes in practice.
class Reference(eqx.Module):
    pointer: Any
    model_axes: Union[Sequence[int], Literal['cifti']]
    imobj: Union['nb.Nifti1Image', 'nb.Cifti2Image']
    _imobj_from_pointer: Callable = lambda x: x
    _dataobj_from_imobj: Callable = lambda x: x.get_fdata()
    cached: bool = False
    dataobj: Optional[Tensor] = None
    modelobj: Optional[Tensor] = None

    def __init__(
        self,
        pointer: Any,
        model_axes: Sequence[int],
        dataobj: Optional[Tensor] = None,
        imobj_from_pointer: Optional[Callable] = None,
        dataobj_from_imobj: Optional[Callable] = None,
    ) -> None:
        self.pointer = pointer

        if imobj_from_pointer is not None:
            self._imobj_from_pointer = imobj_from_pointer
        else:
            self._imobj_from_pointer = lambda x: x
        if dataobj_from_imobj is not None:
            self._dataobj_from_imobj = dataobj_from_imobj
        else:
            self._dataobj_from_imobj = lambda x: x.get_fdata()

        self.imobj = self.load_imobj()

        if model_axes == 'cifti':
            model_axes = self.cifti_model_axes()
        self.model_axes = tuple(standard_axis_number(ax, self.ndim)
                                for ax in model_axes)

        if dataobj is None:
            dataobj = self.dataobj_from_imobj()
        self.dataobj = dataobj

        self.cached = True
        self.modelobj = None

    @staticmethod
    def cache_dataobj(ref, dataobj_from_imobj: Callable) -> 'Reference':
        dataobj = dataobj_from_imobj(ref.imobj)
        return eqx.tree_at(
            where=lambda r: (r.dataobj, r._dataobj_from_imobj, r.cached),
            pytree=ref,
            replace=(dataobj, dataobj_from_imobj, True),
        )

    @staticmethod
    def cache_modelobj(
        ref,
        *,
        modelobj_from_dataobj: Callable = modelobj_from_dataobj_and_maskobj,
        mask: Optional[Any] = None,
        model_axis_out: Optional[int] = None,
    ) -> 'Reference':
        modelobj = modelobj_from_dataobj(
            dataobj=ref.dataobj,
            model_axes=ref.model_axes,
            mask=mask,
            model_axis_out=model_axis_out,
        )
        return eqx.tree_at(
            where=lambda r: (r.modelobj, r.cached),
            pytree=ref,
            replace=(modelobj, True),
        )

    @staticmethod
    def purge_cache(ref) -> 'Reference':
        return eqx.tree_at(
            where=lambda r: (r.dataobj, r.modelobj, r.cached),
            pytree=ref,
            replace=(None, None, False),
        )

    @property
    def header(self) -> Any:
        return self.imobj.header

    @property
    def nifti_header(self) -> Any:
        try:
            return self.imobj.nifti_header
        except AttributeError:
            return self.imobj.header

    @property
    def axobj(self) -> Any:
        """
        Thanks to Chris Markiewicz for tutorials that shaped this
        implementation.
        """
        hdr = self.header
        try:
            return tuple(hdr.get_axis(i) for i in range(self.ndim))
        except AttributeError:
            raise CiftiError(
                'The reference image might not have a CIFTI header.'
            )

    @property
    def model_axobj(self) -> Any:
        try:
            return tuple(
                a for a in self.axobj
                if isinstance(a, nb.cifti2.cifti2_axes.BrainModelAxis)
            )[0]
        except IndexError:
            raise ValueError(
                "No BrainModelAxis found in axes of reference image."
            )

    @property
    def ndim(self) -> Any:
        return self.imobj.ndim

    @property
    def shape(self) -> Any:
        return self.imobj.shape

    @property
    def data(self) -> Any:
        if self.cached:
            return self.dataobj
        else:
            return self.dataobj_from_imobj()

    def _zooms(self, axes: Optional[Sequence[int]] = None) -> Sequence[float]:
        zooms = self.imobj.header.get_zooms()
        if axes is None:
            return zooms
        axes = tuple(standard_axis_number(ax, self.ndim) for ax in axes)
        return tuple(zooms[ax] for ax in axes)

    @property
    def zooms(self) -> Sequence[float]:
        return self._zooms()

    @property
    def other_axes(self):
        return axis_complement(self.ndim, self.model_axes)

    @property
    def model_shape(self) -> Tuple[int, ...]:
        return tuple(self.shape[ax] for ax in self.model_axes)

    @property
    def model_zooms(self) -> Sequence[float]:
        return self._zooms(axes=self.model_axes)

    def imobj_from_pointer(self) -> Any:
        return self._imobj_from_pointer(self.pointer)

    def dataobj_from_imobj(self) -> Tensor:
        return self._dataobj_from_imobj(self.imobj)

    def load_imobj(
        self,
        imobj_from_pointer: Optional[Callable] = None
    ) -> Any:
        if imobj_from_pointer is None:
            imobj_from_pointer = self.imobj_from_pointer
        return imobj_from_pointer()

    def cifti_model_axes(
        self,
    ) -> Sequence[int]:
        return tuple(
            i for i, a in enumerate(self.axobj)
            if isinstance(a, nb.cifti2.cifti2_axes.BrainModelAxis)
        )


class Mask(eqx.Module):
    mask_array : Tensor

    @property
    def data(self):
        return self.mask_array

    @property
    def shape(self):
        return self.mask_array.shape

    @property
    def size(self):
        return self.mask_array.sum()

    @staticmethod
    def _map_to_masked_impl(
        mask_array: Tensor,
        model_axes: Sequence[int] = (-2,),
        model_axis_out: Optional[int] = None,
        out_mode: Literal['index', 'zero'] = 'index'
    ) -> Callable:

        def default_out_axis():
            return model_axis_out if model_axis_out is not None else 0

        def standardise_model_axes(
            model_axes: Sequence[int],
            data: Tensor,
        ) -> Sequence[int]:
            return tuple(standard_axis_number(ax, data.ndim)
                         for ax in model_axes)

        def prepare_mask(data: Tensor) -> Tuple[Tensor, Tensor]:
            data = modelobj_from_dataobj(
                dataobj=data,
                model_axes=standardise_model_axes(model_axes, data),
                mask=None,
                model_axis_out=model_axis_out,
            )
            mask = conform_mask(
                tensor=data,
                mask=mask_array,
                axis=default_out_axis(),
            )
            return data, mask

        def apply_mask_index(data: Tensor, mask: Tensor) -> Tensor:
            out = data[mask]
            return out.reshape(
                *data.shape[:default_out_axis()],
                -1,
                *data.shape[(default_out_axis() + 1):]
            )

        def apply_mask_zero(data: Tensor, mask: Tensor) -> Tensor:
            index = (
                (Ellipsis,) +
                (slice(None),) * (data.ndim - (default_out_axis() + 1))
            )
            return jnp.where(mask[index], data, 0.0)

        out_mode_fn = {
            'index': apply_mask_index,
            'zero': apply_mask_zero
        }
        return lambda data: out_mode_fn[out_mode](*prepare_mask(data))

    def map_to_masked(
        self,
        model_axes: Sequence[int] = (-2,),
        model_axis_out: Optional[int] = None,
        out_mode: Literal['index', 'zero'] = 'index'
    ) -> Callable:
        mask_array = _to_jax_array(self.mask_array)
        return self._map_to_masked_impl(
            mask_array,
            model_axes=model_axes,
            model_axis_out=model_axis_out,
            out_mode=out_mode
        )


class Compartment(eqx.Module):
    name : str
    slice_index : int
    slice_size : int
    mask_array : Optional[Tensor] = None

    @property
    def data(self):
        return self.mask_array

    @property
    def shape(self):
        return (self.slice_size,)

    @property
    def size(self):
        return self.slice_size

    def dynamic_slice_map(self) -> Callable:
        def dynamic_slice(data: Tensor) -> Tensor:
            indices, sizes = form_dynamic_slice(
                shape=data.shape,
                slice_axis=-2,
                slice_index=self.slice_index,
                slice_size=self.slice_size,
            )
            return jax.lax.dynamic_slice(input, indices, sizes)
        return dynamic_slice

    def map_to_masked(
        self,
        model_axes: Sequence[int] = (-2,),
        model_axis_out: Optional[int] = None,
        out_mode: Literal['index', 'zero'] = 'index'
    ) -> Callable:
        mask_array = _to_jax_array(self.mask_array)
        return Mask._map_to_masked_impl(
            mask_array,
            model_axes=model_axes,
            model_axis_out=model_axis_out,
            out_mode=out_mode
        )


class CompartmentSet(eqx.Module):
    compartments : Mapping[str, Compartment]

    def __init__(
        self,
        compartment_dict: Dict[str, Tensor]
    ):
        index = 0
        compartments = ()
        for k, v in compartment_dict.items():
            size = v.sum()
            # We absolutely must make sure that the indices and sizes are not
            # JAX arrays -- otherwise we will get a JAX error when we try to
            # compile the function.
            compartments += ((k, Compartment(
                name=k,
                slice_index=int(index),
                slice_size=int(size),
                mask_array=v,
            )),)
            index += size
        self.compartments = OrderedDict(compartments)

    def keys(self):
        return self.compartments.keys()

    def values(self):
        return self.compartments.values()

    def items(self):
        return self.compartments.items()

    def __getitem__(self, key):
        return self.compartments[key]

    def __iter__(self):
        return iter(self.compartments)

    def __len__(self):
        return len(self.compartments)

    def __contains__(self, key):
        return key in self.compartments

    def get(self, key, default=None):
        return self.compartments.get(key, default)

    def map_to_contiguous(self) -> Callable:
        def make_contiguous(data: Tensor) -> Tensor:
            compartment_data = ()
            for compartment in self.compartments.values():
                compartment_data += (compartment.map_to_masked(
                    in_mode='timeseries',
                    out_mode='index'
                ),)
            return jnp.concatenate(compartment_data, axis=-2)
        return make_contiguous

    def dynamic_slice_map(self, compartment: str) -> Callable:
        return self[compartment].dynamic_slice_map()


def _to_mask(path):
    return nb.load(path).get_fdata().round().astype(bool)


def _is_path(obj):
    return isinstance(obj, str) or isinstance(obj, PosixPath)


class _VolumeObjectReferenceMixin:
    """
    Use to load a reference into an atlas class.

    For use when a NIfTI image object is already provided as the
    ``ref_pointer`` argument.
    """
    def _load_reference(
        self,
        ref_pointer: 'nb.nifti1.Nifti1Image',
    ) -> 'Reference':
        return Reference(
            pointer=ref_pointer,
            model_axes=(0, 1, 2),
        )


class _SurfaceObjectReferenceMixin:
    """
    Use to load a reference into an atlas class.

    For use when a CIfTI image object is already provided as the
    ``ref_pointer`` argument.
    """
    def _load_reference(
        self,
        ref_pointer: 'nb.cifti2.Cifti2Image',
    ) -> 'Reference':
        return Reference(
            pointer=ref_pointer,
            model_axes='cifti',
        )


class _VolumeSingleReferenceMixin:
    """
    Use to load a reference into an atlas class.

    For use when the ``ref_pointer`` object references a single path to a
    volumetric image on disk.
    """
    def _load_reference(
        self,
        ref_pointer: Union[str, Path],
    ) -> 'Reference':
        return Reference(
            pointer=ref_pointer,
            model_axes=(0, 1, 2),
            imobj_from_pointer=lambda x: nb.load(x),
        )


class _SurfaceSingleReferenceMixin:
    """
    Use to load a reference into an atlas class.

    For use when the ``ref_pointer`` object references a single path to a
    surface image on disk.
    """
    def _load_reference(
        self,
        ref_pointer: Union[str, Path],
    ) -> 'Reference':
        return Reference(
            pointer=ref_pointer,
            model_axes='cifti',
            imobj_from_pointer=lambda x: nb.load(x),
        )


class _VolumeMultiReferenceMixin:
    """
    Use to load a reference into an atlas class.

    For use when the ``ref_pointer`` object is an iterable of paths to
    volumetric images on disk.
    """
    def _load_reference(
        self,
        ref_pointer: Sequence[Union[str, Path]],
    ) -> 'nb.nifti1.Nifti1Image':
        ref = tuple(nb.load(path) for path in ref_pointer)
        dataobj = np.stack((r.get_fdata() for r in ref), -1)
        ref = nb.Nifti1Image(
            dataobj=dataobj,
            affine=ref[0].affine,
            header=ref[0].header
        )
        return Reference(
            pointer=ref,
            model_axes=(0, 1, 2),
            dataobj=dataobj,
        )


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
            ref = nb.Nifti1Image(
                header=ref.header,
                affine=ref.affine,
                dataobj=_PhantomDataobj.from_base(ref),
            )
            model_axes = (0, 1, 2)
        except AttributeError: # No affine: Surface/CIfTI
            ref = nb.load(ref_pointer)
            model_axes = 'cifti'
        return Reference(
            pointer=ref,
            model_axes=model_axes,
            dataobj_from_imobj=lambda x: _PhantomDataobj.from_base(x),
        )


class _PhantomDataobj(eqx.Module):
    """
    For tricking ``nibabel`` into instantiating ``Nifti1Image`` objects
    without any real data. Used to reduce data overhead when initialising
    atlases from distributions.
    """
    shape : tuple
    ndim : int

    def __init__(self, shape: tuple):
        self.shape = shape
        self.ndim = len(shape)

    @classmethod
    def from_base(cls, base: Any):
        shape = base.shape
        return cls(shape=shape)


class _CIfTIReferenceMixin:
    """
    Use if an atlas uses a CIfTI as its reference. This class implements the
    additional methods ``axes`` and ``model_axis``, which facilitate access to
    CIfTI model axes. Note that this is *not* a substitute for a reference
    loader mixin like ``_ObjectReferenceMixin`` or ``_SingleReferenceMixin``.
    """
    def to_image(
        self,
        save: Optional[str] = None,
        maps: Optional[Dict[str, Tensor]] = None
    ) -> Optional['nb.nifti1.Nifti1Image']:
        if maps is None:
            maps = self.maps
        offset = 1
        dataobj = np.zeros_like(self.ref.dataobj)
        for k, v in maps.items():
            if v.shape == (0,):
                continue
            n_labels = v.shape[0]
            mask = self.compartments[k].data[None, ...]
            data = v.argmax(0) + offset
            dataobj[np.array(mask)] = data
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
        source: Union[str, Sequence[str]],
    ) -> 'Mask':
        if isinstance(source, str):
            formula = None
            if _is_path(source):
                def f(*pparams):
                    return nb.load(pparams[0]).get_fdata()
            else:
                def f(*pparams):
                    return pparams[0].get_fdata()
        else:
            formula, source = source
            if isinstance(source, str):
                source = (source,)
            if _is_path(source[0]):
                interpreter = NiftiFileInterpreter()
            else:
                interpreter = NiftiObjectInterpreter()
            h = ImageMaths().compile(formula, interpreter=interpreter)
            def f(*pparams):
                img, _ = h(*pparams)
                return img
        init = f(*source).astype(bool)
        return Mask(jnp.asarray(init.ravel()))


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
    ) -> 'Mask':
        init = ()
        for v in source.values():
            try:
                init += (
                    nb.load(v).darrays[0].data.round().astype(bool),
                )
            except Exception:
                init += (
                    np.ones(self.ref.model_axobj.volume_mask.sum()).astype(bool),
                )
        return Mask(jnp.asarray(np.concatenate(init)))


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
    ) -> 'Mask':
        if self.ref.ndim <= 3:
            init = (self.ref.dataobj.round() != source)
        else:
            init = (self.ref.dataobj.sum(self.ref.other_axes) > source)
        return Mask(jnp.asarray(init.ravel()))


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
    ) -> 'CompartmentSet':
        compartments = OrderedDict((
            ('all', jnp.ones((self.mask.size,), dtype=bool)),
        ))
        return CompartmentSet(compartment_dict=compartments)


#TODO: Intersect compartment mask with overall mask before saving.
#TODO: (low-priority) We need to either make sure these work with CIfTI, or we
#      need to make CIfTI-compatible compartment masking other than
#      cortex/subcortex.
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
        names_dict: Dict[Union[str, Tuple[str]], Tensor],
    ) -> 'CompartmentSet':

        apply_mask = self.mask.map_to_masked(
            model_axes=self.ref.model_axes,
        )

        def _get_masks(
            names_dict: Dict[Union[str, Tuple[str]], Tensor]
        ) -> Generator:
            for name, vol in names_dict.items():
                mask = _to_mask(vol)
                if isinstance(name, str):
                    mask = apply_mask(mask).ravel()
                    yield name, mask
                else:
                    for n, m in zip(name, mask):
                        m = apply_mask(mask).ravel()
                        yield n, m

        compartments = OrderedDict(_get_masks(names_dict))
        return CompartmentSet(compartment_dict=compartments)


class _CortexSubcortexCIfTICompartmentMixin:
    """
    Use to isolate spatial subcompartments of the overall atlas such that each
    has separate label sets.

    For use when creating a CIfTI-based atlas with separate subcompartments
    for the left and right cortical hemispheres and for subcortical locations.
    """
    def _compartment_names_dict(self, **kwargs) -> Dict[str, str]:
        return kwargs

    def _create_compartments(
        self,
        names_dict: Dict[Union[str, Tuple[str]], Tensor],
    ) -> 'CompartmentSet':
        compartments = OrderedDict([
            ('cortex_L', jnp.zeros(self.mask.size, dtype=bool,)),
            ('cortex_R', jnp.zeros(self.mask.size, dtype=bool,)),
            ('subcortex', jnp.zeros(self.mask.size, dtype=bool,)),
        ])

        apply_mask = self.mask.map_to_masked(
            model_axes=(0,),
        )

        if self.mask.size == self.ref.shape[-1]:
            src = jnp.arange(self.mask.size)
        else:
            src = apply_mask(jnp.arange(self.mask.shape[0]))
        model_axis = self.ref.model_axobj
        names_dict = {v: k for k, v in names_dict.items()}

        for struc, slc, _ in (model_axis.iter_structures()):
            name = names_dict.get(struc, None)
            if name is not None:
                start, stop = slc.start, slc.stop
                start = start if start is not None else 0
                stop = stop if stop is not None else self.mask.size
                compartments[name] = jnp.logical_and(
                    src >= start, src < stop
                )

        try:
            #TODO: If it's not contiguous, this will fail.
            vol_mask = np.where(model_axis.volume_mask)[0]
            start, stop = vol_mask.min(), vol_mask.max() + 1
            compartments['subcortex'] = jnp.logical_and(
                src >= start, src < stop
            )
        except ValueError:
            pass

        return CompartmentSet(compartment_dict=compartments)


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
    ) -> Dict[str, Tensor]:

        modelobj = self.ref.modelobj

        decoder = OrderedDict()
        for c, compartment in self.compartments.items():
            select_compartment = compartment.map_to_masked(model_axes=(0,))
            compartment_data = select_compartment(modelobj)
            labels_in_compartment = jnp.unique(compartment_data)
            label_mask = (labels_in_compartment != null_label)
            labels_in_compartment = labels_in_compartment[label_mask]
            decoder[c] = jnp.asarray(labels_in_compartment, dtype=int)
        return decoder

    def _populate_map_from_ref(
        self,
        map: Dict[str, Tensor],
        labels: Dict[int, int],
        compartment: Optional[str] = None
    ) -> Dict[str, Tensor]:

        select_compartment = self.compartments[compartment].map_to_masked(
            model_axes=(0,),
        )
        modelobj = select_compartment(self.ref.modelobj).squeeze()

        for i, l in enumerate(labels):
            map = map.at[i].set(jnp.asarray(modelobj == l).T)
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

        modelobj = self.ref.modelobj

        decoder = OrderedDict()
        for c, compartment in self.compartments.items():
            select_compartment = compartment.map_to_masked(model_axes=(0,))
            compartment_data = select_compartment(modelobj)
            labels_in_compartment = jnp.where(compartment_data.sum(0))[0]
            if null_label is not None:
                label_mask = (labels_in_compartment != null_label)
                labels_in_compartment = labels_in_compartment[label_mask]
            decoder[c] = jnp.asarray(labels_in_compartment + 1, dtype=int)
        return decoder

    def _populate_map_from_ref(
        self,
        map: Dict[str, Tensor],
        labels: Dict[int, int],
        compartment: Optional[str] = None
    ) -> Dict[str, Tensor]:

        modelobj = self.ref.modelobj

        #TODO: This will fail for multiple compartments because we need an
        #      offset for each compartment.
        for i, l in enumerate(labels):
            map = map.at[i].set(modelobj[:, (l - 1)])
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
        decoder = OrderedDict()
        n_labels = 0
        for c, i in self.compartment_labels.items():
            if i == 0:
                decoder[c] = jnp.array([], dtype=int,)
                continue
            decoder[c] = jnp.arange(n_labels, n_labels + i, dtype=int,) + 1
            n_labels += i
        return decoder

    def _populate_map_from_ref(
        self,
        map: Dict[str, Tensor],
        labels: Dict[int, int],
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
        shape = self.ref.model_shape
        scale = self.ref.model_zooms
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
        model_axis = self.ref.model_axobj
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
            model_axis.volume_mask, dtype=bool)
        #TODO: Use functions instead of attributes below to interact with
        #      compartments and masks.
        for c, compartment in self.compartments.items():
            mask = compartment.data
            if mask.shape != euc_mask.shape:
                mask = mask[self.mask.data]
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
            scale = self.ref.model_zooms
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
        #TODO: Use functions instead of attributes below to interact with
        #      compartments and masks.
        compartment_mask = self.compartments[compartment].data
        if len(self.coors) < len(compartment_mask):
            compartment_mask = compartment_mask[self.mask.data]
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
