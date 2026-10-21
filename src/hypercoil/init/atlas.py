# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Tools for initialising parameters corresponding to brain atlases.

Neuroimaging atlases are principally defined and distributed with reference to
either volumetric or surface-based coordinate spaces. Additionally, atlases
might be either discrete-valued, hard-label parcellations or continuous-valued
maps that can overlap with one another. This module handles these cases,
together with random or unstructured initialisations that use no prior
knowledge.

The initialisation class itself,
:class:`~hypercoil.init.atlas.AtlasInitialiser`,
takes as its required argument an atlas object.
All atlas objects subclass the abstract base class
:class:`~hypercoil.init.atlas.BaseAtlas`
and incorporate a combination of
:mod:`atlas mixins <hypercoil.init.atlasmixins>`
to implement diverse functionalities. These subclasses generally take a
pointer to a reference as an input argument; this reference either forms the
base for the atlas (for classes that use previously existing knowledge
annotations) or defines its dimensions (for classes based on random
initialisations). Existing subclasses include

- :class:`~hypercoil.init.atlas.DiscreteVolumetricAtlas`
  implements a volumetric atlas with discrete parcels, loaded from a single
  reference image and a single volume in which each unique integer identifies
  a single parcel.
- :class:`~hypercoil.init.atlas.MultiVolumetricAtlas`
  implements a volumetric atlas whose parcels are each encoded in a separate
  volume of the reference image.
- :class:`~hypercoil.init.atlas.MultifileVolumetricAtlas`
  implements a volumetric atlas whose parcels (either discrete or continuous)
  are each encoded in a separate reference image file.
- :class:`~hypercoil.init.atlas.CortexSubcortexCIfTIAtlas`
  implements a surface-based atlas with discrete parcels, loaded from a CIfTI
  image. Parcels are compartmentalised into left and right cerebral cortex and
  subcortex.
- :class:`~hypercoil.init.atlas.DirichletInitVolumetricAtlas`
  implements a volumetric atlas whose voxel-label annotations are initialised
  as random, i.i.d. samples from a Dirichlet distribution.
- :class:`~hypercoil.init.atlas.DirichletInitSurfaceAtlas`
  implements a surface-based atlas compartmentalised into left and right
  cerebral cortex and subcortex. Voxel- and vertex-label annotations are
  initialised as random, i.i.d. samples from a Dirichlet distribution.

.. note::
    Composing the :mod:`atlas mixins <hypercoil.init.atlasmixins>` into a new
    atlas subclass is not currently supported in general.

.. image:: ../_images/atlas_linearmap.svg
  :width: 400
  :align: center

Each subclass instance of :class:`BaseAtlas` has a ``maps`` attribute that
stores voxel-label annotations as a dictionary. The keys of this dictionary
correspond to isolated compartments of the atlas. (The single key ``'all'`` is
used for atlases not compartmentalised.) The dictionary values are
:math:`L_{compartment} \times V_{compartment}` matrices, where
:math:`V_{compartment}` is the number of spatial locations (voxels or vertices)
in the compartment, and :math:`L_{compartment}` is the number of distinct
parcel labels in the compartment. These matrices can be used in a linear
mapping to reduce the dimension of an input time series from the number of
voxels to the number of parcels. See
:mod:`the linear atlas module <hypercoil.nn.atlas>` for more details.
"""
from __future__ import annotations
from abc import abstractmethod
from collections import OrderedDict
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import jax
import jax.numpy as jnp
import equinox as eqx
import templateflow.api as tflow
from numpyro.distributions import Normal

from ..engine import PyTree, Tensor
from ..engine.noise import ScalarIIDAddStochasticTransform
from ..formula.nnops import retrieve_address
from ..neuro.const import CIfTIStructures
from .atlasmixins import (
    Reference,
    _CIfTIOutputMixin,
    _ContinuousLabelMixin,
    _CortexSubcortexCIfTICompartmentMixin,
    _CortexSubcortexCIfTIMaskMixin,
    _CortexSubcortexGIfTICompartmentMixin,
    _DirichletLabelMixin,
    _DiscreteLabelMixin,
    _FromNullMaskMixin,
    _GIfTIOutputMixin,
    _GIfTIReferenceMixin,
    _IcosphereReferenceMixin,
    _is_path,
    _LogicMaskMixin,
    _MultiCompartmentMixin,
    _NIfTIOutputMixin,
    _PhantomReferenceMixin,
    _SingleCompartmentMixin,
    _SpatialConvMixin,
    _SurfaceSingleReferenceMixin,
    _VertexCIfTIMeshMixin,
    _VertexGIfTIMeshMixin,
    _VolumeMultiReferenceMixin,
    _VolumeSingleReferenceMixin,
    _VolumetricMeshMixin,
)
from .base import MappedInitialiser
from .dirichlet import DirichletInitialiser
from .mapparam import MappedParameter, ProbabilitySimplexParameter


class BaseAtlas(eqx.Module):
    """
    Base class for atlases: linear maps from spatial locations to labels.

    Discrete and continuous atlas containers inherit from this class.

    Several atlas classes are included to cover frequent scenarios, but users
    can also create their own atlas class compositionally using the available
    :mod:`mixins <hypercoil.init.atlasmixins>`.
    Each atlas class must implement the following methods:

    ``_load_reference``
        Implemented by a ``~ReferenceMixin`` class.
    ``_create_mask``
        Implemented by a ``~MaskMixin`` class.
    ``_compartment_names_dict``
        Implemented by a ``~CompartmentMixin`` class.
    ``_create_compartments``
        Implemented by a ``~CompartmentMixin`` class.
    ``_configure_decoders``
        Implemented by a ``~LabelMixin`` class.
    ``_populate_map_from_ref``
        Implemented by a ``~LabelMixin`` class.
    ``_init_coors``
        Implemented by a ``~MeshMixin`` class.
    ``_configure_sigma``
        Implemented by a ``~ConvMixin`` class.
    ``_convolve``
        Implemented by a ``~ConvMixin`` class.

    Abstractly, atlas creation proceeds through the following steps:

    - Loading a reference that contains the atlas data;
    - Creating an overall mask that indicates regions of space that are
      candidates for inclusion in the atlas
    - Defining isolated subcompartments of the atlas (for instance, left and
      right cortical hemispheres)
    - Decoding the sets of labels that are present in each subcompartment
    - Creating a linear map representation of each subcompartment's atlas
    - Establishing a coordinate system over each linear map

    Parameters
    ----------
    ref_pointer
        Pointer to a reference path, image, or object used to instantiate the
        atlas.
    mask_source
        Source of data used to create an overall mask for the atlas.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.
    clear_cache
        Indicates that data loaded in from the reference should be cleared
        away when atlas creation is complete.

    Attributes
    ----------
    ref
        The reference whose pointer was provided at construction time.
    mask
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors
        Spatial position of each location in the atlas.
    topology
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """

    name: str
    ref_pointer: Any
    ref: Tensor
    mask: Tensor
    compartments: Dict[str, Tensor]
    decoder: Dict[str, Tensor]
    maps: Dict[str, Tensor]
    coors: Dict[str, Tensor]
    topology: Dict[str, str]

    def __init__(
        self,
        ref_pointer: Any,
        mask_source: Any,
        clear_cache: bool = True,
        name: Optional[str] = None,
        **params,
    ):
        if name is None:
            name = ''
        self.name = name
        self.ref_pointer = ref_pointer
        self.ref = self._load_reference(ref_pointer)

        self.mask = self._create_mask(mask_source)
        self.ref = Reference.cache_modelobj(ref=self.ref, mask=self.mask)
        names_dict = self._compartment_names_dict(**params)
        self.compartments = self._create_compartments(names_dict)

        self.decoder = self._configure_decoders()
        self._configure_compartment_maps()
        self._init_coors(source=mask_source, names_dict=names_dict)
        if clear_cache:
            self.ref = Reference.purge_cache(self.ref)

    @abstractmethod
    def _load_reference(
        self,
        ref_pointer: Any,
    ) -> Tensor:
        ...

    @abstractmethod
    def _create_mask(
        self,
        mask_source: Any,
    ) -> None:
        ...

    @abstractmethod
    def _compartment_names_dict(**params) -> Dict[str, str]:
        ...

    @abstractmethod
    def _create_compartments(
        self,
        names_dict: Dict[str, str],
        ref: Optional[Tensor] = None,
    ) -> None:
        ...

    @abstractmethod
    def _configure_decoders(
        self,
        null_label: Optional[Any] = None,
    ) -> None:
        ...

    @abstractmethod
    def _populate_map_from_ref(
        self,
        map: Tensor,
        labels: Tensor,
        mask: Tensor,
        compartment: Optional[str] = None,
    ) -> Tensor:
        ...

    @abstractmethod
    def _init_coors(
        self,
        source: Optional[Any] = None,
        names_dict: Optional[Dict[str, str]] = None,
    ) -> None:
        ...

    def __call__(
        self,
        compartments: Union[bool, Sequence[str]] = True,
        normalise: bool = False,
        sigma: Optional[float] = None,
        noise: Optional[Callable] = None,
        max_bin: int = 10000,
        spherical_scale: float = 1,
        truncate: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute transformed maps for selected atlas subcompartments.

        Parameters
        ----------
        compartments
            Compartments for which transformed maps should be computed. If
            this is set to False, then a single transformed map is returned
            for the combined compartment containing all spatial locations. If
            this is set to True, then a transformed map is returned for every
            defined compartment.
        normalise
            Indicates that maps should be spatially normalised such that the
            sum over all assignments to a parcel is equal to 1. When the map
            is used as a linear transformation, this option results in
            computation of a weighted average over each parcel.
        sigma
            If this is not None, then spatial smoothing using a Gaussian
            kernel is applied over each parcel's assignments. Distances are
            established by the atlas's coordinate system and the topology of
            each compartment. The value of sigma establishes the width of the
            Gaussian kernel.
        noise
            If this is a callable, then it should be a mapping from a tensor
            to another tensor of the same dimensions. After spatial smoothing
            but before normalisation, this callable is applied to each map
            tensor to inject noise.
        max_bin
            Because spatial smoothing has substantial memory overhead, this
            flag sets a ceiling on the number of spatial locations to be
            smoothed at a time.
        spherical_scale
            During spatial smoothing, the Gaussian kernel is scaled by this
            value for spherical coordinate systems only. This enables the use
            of different kernel widths for spherical and Euclidean
            coordinates.
        truncate
            Maximum distance at which data points are convolved together
            during smoothing.

        Returns
        -------
        dict(tensor)
            Dictionary of transformed maps for each specified atlas
            compartment.
        """
        ret = OrderedDict()
        if compartments is False:
            compartments = ['_all']
        elif isinstance(compartments, str):
            compartments = [compartments]
        elif compartments is True:
            compartments = list(self.compartments.keys())
        for c in compartments:
            c_map = self.maps[c]
            if sigma is not None:
                sigma = self._configure_sigma(sigma)
                c_map = self._convolve(
                    map=c_map,
                    compartment=c,
                    sigma=sigma,
                    max_bin=max_bin,
                    spherical_scale=spherical_scale,
                    truncate=truncate,
                )
            if noise is not None:
                c_map = noise(c_map)
            if normalise:
                c_map = c_map / c_map.sum(1, keepdims=True)
            ret[c] = c_map
        return ret

    def _configure_compartment_maps(self) -> None:
        self.maps = OrderedDict()
        for c, compartment in self.compartments.items():
            labels = self.decoder[c]
            dim_out = len(labels)
            if dim_out == 0:
                self.maps[c] = jnp.array([])
                continue
            dim_in = compartment.size
            map = jnp.empty(
                (dim_out, dim_in)
            )  # TODO: this smells like torch...
            self.maps[c] = jnp.array(
                self._populate_map_from_ref(map, labels, c)
            )


class DirichletInitBaseAtlas(
    _PhantomReferenceMixin,
    _DirichletLabelMixin,
    BaseAtlas,
):
    """
    Abstract base class for atlases initialised from a Dirichlet distribution.

    See :class:`BaseAtlas` for general details.

    Parameters
    ----------
    mask_source
        Source of data used to create an overall mask for the atlas.
    compartment_labels
        Number of labels to initialise for each compartment.
    conc
        If this is provided and ``init`` is not, then the Dirichlet
        distributions used to sample assignments of spatial locations to
        parcels are defined with the same concentration parameter, ``conc``,
        for each parcel.
    template_image
        Template used to define the spatial dimensions and coordinates of the
        atlas. If this is not provided explicitly, then the mask source image
        is used by default.
    init
        Dict mapping from compartment names to the Dirichlet distributions
        used for parcel assignment initialisation in each compartment.
    """

    compartment_labels: Dict[str, int]
    init: Dict[str, Callable]

    def __init__(
        self,
        mask_source: Any,
        compartment_labels: Dict[str, int],
        conc: Optional[float] = 100.0,
        template_image: Optional[str] = None,
        init: Optional[Dict[str, Callable]] = None,
        name: Optional[str] = None,
        *,
        key: 'jax.random.PRNGKey',
        **params,
    ):
        if template_image is None:
            template_image = mask_source
            if not _is_path(template_image):
                template_image = template_image[1]
                if not _is_path(template_image):
                    template_image = template_image[0]
        if isinstance(compartment_labels, int):
            compartment_labels = {'all', compartment_labels}
        self.compartment_labels = compartment_labels
        keys = jax.random.split(key, len(compartment_labels))
        if init is None:
            default_init = True
            init = OrderedDict(
                (
                    c,
                    partial(
                        DirichletInitialiser.init,
                        concentration=(conc,),
                        num_classes=i,
                        axis=-2,
                        mapper=None,
                        where=None,
                        key=k,
                    ),
                )
                for k, (c, i) in zip(keys, compartment_labels.items())
            )
        global_key = jax.random.split(key, 1)[0]
        self.init = init
        self._global_compartment_init(key=global_key)
        super().__init__(
            ref_pointer=template_image,
            mask_source=mask_source,
            name=name,
            **params,
        )
        if default_init:
            for k, v in self.init.items():
                v.domain = partial(ProbabilitySimplexParameter, axis=-2)

    def _global_compartment_init(self, *, key: 'jax.random.PRNGKey') -> None:
        if self.init.get('_all'):
            return
        if self.init.get('all'):
            self.init['_all'] = self.init['all']
            return
        concentrations = ()
        for initialiser in self.init.values():
            conc = initialiser.keywords['concentration']
            num_classes = initialiser.keywords['num_classes']
            concentrations = concentrations + conc * num_classes
        self.init['_all'] = partial(
            DirichletInitialiser.init,
            concentration=concentrations,
            num_classes=len(concentrations),
            axis=-2,
            mapper=None,
            where=None,
            key=key,
        )


class DiscreteVolumetricAtlas(
    _VolumeSingleReferenceMixin,
    _FromNullMaskMixin,
    _SingleCompartmentMixin,
    _DiscreteLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    _NIfTIOutputMixin,
    BaseAtlas,
):
    """
    Atlas with discrete, non-overlapping parcels in a single volume.

    A value of 0 in the reference image is taken to mean no label.

    Parameters
    ----------
    ref_pointer
        Path to a NIfTI file containing the atlas.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.
    clear_cache
        Indicates that data loaded in from the reference should be cleared
        away when atlas creation is complete.

    Attributes
    ----------
    ref
        The reference whose pointer was provided at construction time.
    mask
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors
        Spatial position of each location in the atlas.
    topology
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """

    def __init__(
        self,
        ref_pointer: str,
        clear_cache: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(
            ref_pointer=ref_pointer,
            mask_source=0,
            clear_cache=clear_cache,
            name=name,
        )


class MultiVolumetricAtlas(
    _VolumeSingleReferenceMixin,
    _FromNullMaskMixin,
    _SingleCompartmentMixin,
    _ContinuousLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    _NIfTIOutputMixin,
    BaseAtlas,
):
    """
    Continuous atlas with each label in a separate volume of one image.

    Overlapping labels, such as probabilistic segmentations or ICA results, are
    stored across volumes. For labels stored across multiple files, use
    :class:`MultifileVolumetricAtlas`.

    Parameters
    ----------
    ref_pointer
        Path to a NIfTI file containing the atlas.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.
    clear_cache
        Indicates that data loaded in from the reference should be cleared
        away when atlas creation is complete.

    Attributes
    ----------
    ref
        The reference whose pointer was provided at construction time.
    mask
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors
        Spatial position of each location in the atlas.
    topology
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """

    def __init__(
        self,
        ref_pointer: str,
        clear_cache: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(
            ref_pointer=ref_pointer,
            mask_source=0,
            clear_cache=clear_cache,
            name=name,
        )


class MultifileVolumetricAtlas(
    _VolumeMultiReferenceMixin,
    _FromNullMaskMixin,
    _SingleCompartmentMixin,
    _ContinuousLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    _NIfTIOutputMixin,
    BaseAtlas,
):
    """
    Continuous atlas with each label in a separate image file.

    Overlapping labels, such as probabilistic segmentations or ICA results, are
    stored across files. For labels stored across the volumes of one file, use
    :class:`MultiVolumetricAtlas`.

    Parameters
    ----------
    ref_pointer
        Paths to NIfTI files containing the atlas.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.
    clear_cache
        Indicates that data loaded in from the reference should be cleared
        away when atlas creation is complete.

    Attributes
    ----------
    ref
        The reference whose pointer was provided at construction time.
    mask
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors
        Spatial position of each location in the atlas.
    topology
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """

    def __init__(
        self,
        ref_pointer: Sequence[str],
        clear_cache: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__(
            ref_pointer=ref_pointer,
            mask_source=0,
            clear_cache=clear_cache,
            name=name,
        )


class CortexSubcortexGIfTIAtlas(
    _GIfTIReferenceMixin,
    _CortexSubcortexCIfTIMaskMixin,
    _CortexSubcortexGIfTICompartmentMixin,
    _DiscreteLabelMixin,
    _VertexGIfTIMeshMixin,
    _SpatialConvMixin,
    _GIfTIOutputMixin,
    BaseAtlas,
):
    surf: Dict[str, str]

    def __init__(
        self,
        data_L: str,
        data_R: str,
        data_subcortex: Optional[str] = None,
        mask_L: Optional[str] = None,
        mask_R: Optional[str] = None,
        surf_L: Optional[str] = None,
        surf_R: Optional[str] = None,
        subcortex: Optional[str] = None,
        clear_cache: bool = True,
        name: Optional[str] = None,
    ):
        self.surf, mask_source = _surface_atlas_common_args(
            mask_L=mask_L,
            mask_R=mask_R,
            surf_L=surf_L,
            surf_R=surf_R,
        )
        super().__init__(
            ref_pointer=(data_L, data_R, data_subcortex),
            mask_source=mask_source,
            clear_cache=clear_cache,
            name=name,
        )


class CortexSubcortexCIfTIAtlas(
    _SurfaceSingleReferenceMixin,
    _CortexSubcortexCIfTIMaskMixin,
    _CortexSubcortexCIfTICompartmentMixin,
    _DiscreteLabelMixin,
    _VertexCIfTIMeshMixin,
    _SpatialConvMixin,
    _CIfTIOutputMixin,
    _GIfTIOutputMixin,
    BaseAtlas,
):
    """
    Surface-based atlas loaded from a CIfTI image.

    The atlas has three compartments: the left and right cortical hemispheres,
    which are surfaces with a spherical coordinate system, and the subcortical
    grey matter, which has a volumetric Euclidean coordinate system.

    Parameters
    ----------
    ref_pointer
        Path to a CIfTI file containing the atlas.
    mask_L
        Path to a GIfTI file containing an atlas mask for the left cortical
        hemisphere. If none is provided, the 32K fsLR mask excluding medial
        wall vertices is used by default.
    mask_R
        Path to a GIfTI file containing an atlas mask for the right cortical
        hemisphere. If none is provided, the 32K-vertex fsLR mask excluding
        medial wall vertices is used by default.
    surf_L
        Path to a GIfTI file containing the mesh of the surface of the left
        cortical hemisphere. If none is provided, the 32K-vertex spherical
        fsLR surface is used by default.
    surf_L
        Path to a GIfTI file containing the mesh of the surface of the right
        cortical hemisphere. If none is provided, the 32K-vertex spherical
        fsLR surface is used by default.
    cortex_L and cortex_R : str
        Names of brain model axis objects corresponding to cortex in the CIfTI
        reference. Default to ``'CIFTI_STRUCTURE_CORTEX_LEFT'`` and
        ``'CIFTI_STRUCTURE_CORTEX_RIGHT'``.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.
    clear_cache
        Indicates that data loaded in from the reference should be cleared
        away when atlas creation is complete.

    Attributes
    ----------
    ref
        The reference whose pointer was provided at construction time.
    mask
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors
        Spatial position of each location in the atlas.
    topology
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """

    surf: Dict[str, str]

    def __init__(
        self,
        ref_pointer: str,
        mask_L: Optional[str] = None,
        mask_R: Optional[str] = None,
        surf_L: Optional[str] = None,
        surf_R: Optional[str] = None,
        cortex_L: str = CIfTIStructures.LEFT,
        cortex_R: str = CIfTIStructures.RIGHT,
        clear_cache: bool = True,
        name: Optional[str] = None,
    ):
        self.surf, mask_source = _surface_atlas_common_args(
            mask_L=mask_L,
            mask_R=mask_R,
            surf_L=surf_L,
            surf_R=surf_R,
        )
        super().__init__(
            ref_pointer=ref_pointer,
            mask_source=mask_source,
            clear_cache=clear_cache,
            name=name,
            cortex_L=cortex_L,
            cortex_R=cortex_R,
        )


class DirichletInitVolumetricAtlas(
    _LogicMaskMixin,
    _SingleCompartmentMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    _NIfTIOutputMixin,
    DirichletInitBaseAtlas,
):
    """
    Volumetric atlas sampled from a Dirichlet distribution.

    The parcel assignment of each spatial location is a categorical
    distribution sampled from a Dirichlet distribution. For a surface-based
    atlas, use :class:`DirichletInitSurfaceAtlas`.

    Parameters
    ----------
    mask_source
        Source of data used to create an overall mask for the atlas.
    n_labels
        Number of labels to initialise for the atlas.
    conc
        If this is provided and `init` is not, then the Dirichlet
        distributions used to sample assignments of spatial locations to
        parcels are defined with the same concentration parameter, `conc`,
        for each parcel.
    init
        The Dirichlet distribution used for parcel assignment initialisation.
        If this is not provided, then a distribution with equal concentrations
        will be created using the `conc` parameter.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.

    Attributes
    ----------
    ref
        The reference whose pointer was provided at construction time.
    mask
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors
        Spatial position of each location in the atlas.
    topology
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data
        Nothing here. The field exists for consistency with other atlas
        classes.
    """

    def __init__(
        self,
        mask_source: Union[str, Callable],
        n_labels: Dict[str, int],
        conc: Optional[float] = 100.0,
        name: Optional[str] = None,
        init: Optional[Dict[str, DirichletInitialiser]] = None,
        *,
        key: 'jax.random.PRNGKey',
        **params,
    ):
        if init is not None:
            init = {'all': init}
        super().__init__(
            mask_source=mask_source,
            compartment_labels={'all': n_labels},
            conc=conc,
            init=init,
            name=name,
            key=key,
        )


class DirichletInitSurfaceAtlas(
    _CortexSubcortexCIfTIMaskMixin,
    _CortexSubcortexCIfTICompartmentMixin,
    _VertexCIfTIMeshMixin,
    _SpatialConvMixin,
    _CIfTIOutputMixin,
    _GIfTIOutputMixin,
    DirichletInitBaseAtlas,
):
    """
    Surface atlas sampled from a Dirichlet distribution.

    The parcel assignment of each spatial location is a categorical
    distribution sampled from a Dirichlet distribution. The atlas has three
    compartments: the left and right cortical hemispheres, which are surfaces
    with a spherical coordinate system, and the subcortical grey matter, which
    has a volumetric Euclidean coordinate system. For a volumetric atlas, use
    :class:`DirichletInitVolumetricAtlas`.

    Parameters
    ----------
    cifti_template
        Template CIfTI for establishing compartment extents.
    compartment_labels
        Number of labels to initialise for each compartment.
    conc
        If this is provided and `init` is not, then the Dirichlet
        distributions used to sample assignments of spatial locations to
        parcels are defined with the same concentration parameter, `conc`,
        for each parcel.
    init
        Dict mapping from compartment names to the Dirichlet distributions
        used for parcel assignment initialisation in each compartment.
    mask_L
        Path to a GIfTI file containing an atlas mask for the left cortical
        hemisphere. If none is provided, the 32K fsLR mask excluding medial
        wall vertices is used by default.
    mask_R
        Path to a GIfTI file containing an atlas mask for the right cortical
        hemisphere. If none is provided, the 32K-vertex fsLR mask excluding
        medial wall vertices is used by default.
    surf_L
        Path to a GIfTI file containing the mesh of the surface of the left
        cortical hemisphere. If none is provided, the 32K-vertex spherical
        fsLR surface is used by default.
    surf_R
        Path to a GIfTI file containing the mesh of the surface of the right
        cortical hemisphere. If none is provided, the 32K-vertex spherical
        fsLR surface is used by default.
    cortex_L and cortex_R : str
        Names of brain model axis objects corresponding to cortex in the CIfTI
        reference. Default to 'CIFTI_STRUCTURE_CORTEX_LEFT' and
        'CIFTI_STRUCTURE_CORTEX_RIGHT'.

    Attributes
    ----------
    ref
        The reference whose pointer was provided at construction time.
    mask
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors
        Spatial position of each location in the atlas.
    topology
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """

    surf: Dict[str, str]

    def __init__(
        self,
        cifti_template: str,
        compartment_labels: Dict[str, int],
        conc: Optional[float] = 100.0,
        init: Optional[Dict[str, DirichletInitialiser]] = None,
        name: Optional[str] = None,
        mask_L: Optional[str] = None,
        mask_R: Optional[str] = None,
        surf_L: Optional[str] = None,
        surf_R: Optional[str] = None,
        cortex_L: str = CIfTIStructures.LEFT,
        cortex_R: str = CIfTIStructures.RIGHT,
        *,
        key: 'jax.random.PRNGKey',
    ):
        self.surf, mask_source = _surface_atlas_common_args(
            mask_L=mask_L,
            mask_R=mask_R,
            surf_L=surf_L,
            surf_R=surf_R,
        )
        super().__init__(
            template_image=cifti_template,
            mask_source=mask_source,
            compartment_labels=compartment_labels,
            conc=conc,
            init=init,
            name=name,
            cortex_L=cortex_L,
            cortex_R=cortex_R,
            key=key,
        )


class IcosphereAtlas(
    _IcosphereReferenceMixin,
    _CortexSubcortexCIfTIMaskMixin,
    _CortexSubcortexGIfTICompartmentMixin,
    _DiscreteLabelMixin,
    _VertexGIfTIMeshMixin,
    _SpatialConvMixin,
    _GIfTIOutputMixin,
    BaseAtlas,
):
    """
    Atlas of approximately evenly spaced discs on a spherical surface.

    The surface is homeomorphic to the cortical sheet; the disc centres are the
    vertices of a subdivided icosahedron.

    Parameters
    ----------
    n_subdivisions
        Number of subdivisions of the icosahedron used to define disc centres.
        The default number of labels per hemisphere follows the formula
        ``12 + 10 * (n_subdivisions**2 -1)``. Then, the coordinates of each
        label are compared against the mask coordinates using a 1-nearest
        neighbour search to determine the inclusion status of each label.
    rotation_target
        If provided, the coordinates of the first vertex of the icosphere will be
        rotated to align with the provided target coordinate.
    rotation_secondary
        If provided, the coordinates of the second vertex of the icosphere will
        be rotated to align with the provided secondary coordinate.
    mask_L
        Path to a GIfTI file containing an atlas mask for the left cortical
        hemisphere. If none is provided, the 32K fsLR mask excluding medial
        wall vertices is used by default.
    mask_R
        Path to a GIfTI file containing an atlas mask for the right cortical
        hemisphere. If none is provided, the 32K-vertex fsLR mask excluding
        medial wall vertices is used by default.
    surf_L
        Path to a GIfTI file containing the mesh of the surface of the left
        cortical hemisphere. If none is provided, the 32K-vertex spherical
        fsLR surface is used by default.
    surf_R
        Path to a GIfTI file containing the mesh of the surface of the right
        cortical hemisphere. If none is provided, the 32K-vertex spherical
        fsLR surface is used by default.
    """

    surf: Dict[str, str]
    vertices: Dict[str, Tensor]

    def __init__(
        self,
        n_subdivisions: int,
        rotation_target: Optional[Tensor] = None,
        rotation_secondary: Optional[Tensor] = None,
        disc_scale: float = 0.5,
        name: Optional[str] = None,
        mask_L: Optional[str] = None,
        mask_R: Optional[str] = None,
        surf_L: Optional[str] = None,
        surf_R: Optional[str] = None,
        clear_cache: bool = True,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        self.surf, mask_source = _surface_atlas_common_args(
            mask_L=mask_L,
            mask_R=mask_R,
            surf_L=surf_L,
            surf_R=surf_R,
        )
        ref_pointer = {
            'n_subdivisions': n_subdivisions,
            'rotation_target': rotation_target,
            'rotation_secondary': rotation_secondary,
            'disc_scale': disc_scale,
            'mask_L': mask_source['cortex_L'],
            'mask_R': mask_source['cortex_R'],
            'surf_L': self.surf['cortex_L'],
            'surf_R': self.surf['cortex_R'],
        }

        super().__init__(
            ref_pointer=ref_pointer,
            mask_source=mask_source,
            clear_cache=clear_cache,
            name=name,
        )


class _MemeAtlas(
    _VolumeSingleReferenceMixin,
    _LogicMaskMixin,
    _MultiCompartmentMixin,
    _DiscreteLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    _NIfTIOutputMixin,
    BaseAtlas,
):
    """
    For testing purposes only.
    """

    def __init__(self):
        ref_pointer = tflow.get(
            template='MNI152NLin2009cAsym',
            resolution=2,
            desc='100Parcels17Networks',
        )
        eye = tflow.get(
            template='MNI152NLin2009cAsym',
            resolution=2,
            desc='eye',
            suffix='mask',
        )
        face = tflow.get(
            template='MNI152NLin2009cAsym',
            resolution=2,
            desc='face',
            suffix='mask',
        )
        brain = tflow.get(
            template='MNI152NLin2009cAsym',
            resolution=2,
            desc='brain',
            suffix='mask',
        )
        mask_formula = '((IMGa -bin) -or (IMGb -bin)) -and (IMGc -neg)'
        mask_source = (eye, face, brain)
        compartments_dict = {
            'eye': eye,
            'face': face,
        }
        super().__init__(
            ref_pointer=ref_pointer,
            mask_source=(mask_formula, mask_source),
            name='Meme',
            **compartments_dict,
        )


def atlas_init(
    *,
    shape: Optional[Any] = None,
    atlas: BaseAtlas,
    compartments: Union[bool, Tuple[str]] = True,
    normalise: bool = False,
    max_bin: int = 10000,
    spherical_scale: float = 1.0,
    truncate: Optional[float] = None,
    kernel_sigma: Optional[float] = None,
    noise_sigma: Optional[float] = None,
    key: 'jax.random.PRNGKey',
):
    r"""
    Voxel-to-label mapping initialisation.

    Initialise a tensor such that its entries characterise a matrix that maps
    a relevant subset of image voxels to a set of labels. The initialisation
    uses an existing atlas with the option of blurring labels or injecting
    noise.

    .. note::
        The ``shape`` argument is unused and is only present for compatibility
        with the initialisation API.

    Parameters
    ----------
    atlas
        Atlas object to use for tensor initialisation.
    kernel_sigma
        If this is a float, then a Gaussian smoothing kernel with the
        specified width is applied to each label after it is extracted.
    noise_sigma
        If this is a float, then Gaussian noise with the specified standard
        deviation is added to the label.
    """
    if noise_sigma is not None:
        distr = Normal(0.0, noise_sigma)
        noise = ScalarIIDAddStochasticTransform(distr, key=key)
    else:
        noise = None
    return atlas(
        compartments=compartments,
        normalise=normalise,
        sigma=kernel_sigma,
        noise=noise,
        max_bin=max_bin,
        spherical_scale=spherical_scale,
        truncate=truncate,
    )


class AtlasInitialiser(MappedInitialiser):
    r"""
    Voxel-to-label mapping initialisation.

    Initialise a tensor dictionary such that its entries characterise a set of
    matrices that each map a relevant subset of image voxels to a set of
    labels. The initialisation can use a previously existing atlas,
    instantiated as a subclass of
    :class:`~hypercoil.init.atlas.BaseAtlas`
    with the option of blurring labels using pointwise spatial convolution or
    injecting noise.

    Parameters
    ----------
    atlas
        Atlas object to use for tensor initialisation.
    normalise
        Indicates that maps should be spatially normalised such that the sum
        over all assignments to a parcel is equal to 1. When the map is used
        as a linear transformation, this option results in computation of a
        weighted average over each parcel.
    kernel_sigma
        If this is not None, then spatial smoothing using a Gaussian kernel is
        applied over each parcel's assignments. Distances are established by
        the atlas's coordinate system and the topology of each compartment.
        The value of sigma establishes the width of the Gaussian kernel.
    noise_sigma
        If this is not None, then Gaussian noise with the specified standard
        deviation is added to each label.
    """

    atlas: BaseAtlas
    normalise: bool = False
    max_bin: int = 10000
    spherical_scale: float = 1.0
    truncate: Optional[float] = None
    kernel_sigma: Optional[float] = None
    noise_sigma: Optional[float] = None

    def __init__(
        self,
        atlas: BaseAtlas,
        normalise: bool = False,
        max_bin: int = 10000,
        spherical_scale: float = 1.0,
        truncate: Optional[float] = None,
        kernel_sigma: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        mapper: Optional[Type[MappedParameter]] = None,
    ):
        self.atlas = atlas
        self.normalise = normalise
        self.max_bin = max_bin
        self.spherical_scale = spherical_scale
        self.truncate = truncate
        self.kernel_sigma = kernel_sigma
        self.noise_sigma = noise_sigma
        if mapper is None:
            try:
                mapper = atlas.init['_all'].mapper
            except (AttributeError, KeyError):
                pass
        super().__init__(mapper=mapper)

    def __call__(
        self,
        model: PyTree,
        *,
        where: Union[str, Callable] = 'weight',
        key: jax.random.PRNGKey,
        **params,
    ):
        parameters = retrieve_address(model, where=where)
        if key is not None:
            keys = jax.random.split(key, len(parameters))
        else:
            keys = (None,) * len(parameters)
        return tuple(
            self._init(
                shape=None,
                key=key,
                **params,
            )
            for key, parameter in zip(keys, parameters)
        )

    def _init(
        self,
        shape: Optional[Any],
        key: jax.random.PRNGKey,
    ) -> Tensor:
        return atlas_init(
            shape=shape,
            atlas=self.atlas,
            normalise=self.normalise,
            max_bin=self.max_bin,
            spherical_scale=self.spherical_scale,
            kernel_sigma=self.kernel_sigma,
            noise_sigma=self.noise_sigma,
            truncate=self.truncate,
            compartments=True,
            key=key,
        )

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        atlas: BaseAtlas,
        normalise: bool = False,
        max_bin: int = 10000,
        spherical_scale: float = 1.0,
        truncate: Optional[float] = None,
        kernel_sigma: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        where: Union[str, Callable] = 'weight',
        key: jax.random.PRNGKey,
        **params,
    ) -> PyTree:
        init = cls(
            mapper=mapper,
            atlas=atlas,
            normalise=normalise,
            max_bin=max_bin,
            spherical_scale=spherical_scale,
            truncate=truncate,
            kernel_sigma=kernel_sigma,
            noise_sigma=noise_sigma,
        )
        # TODO: We're going to need our own version of _init_impl here to
        #      handle the separation of compartments into a tensor dict.
        return super()._init_impl(
            init=init,
            model=model,
            where=where,
            key=key,
            **params,
        )


def _surface_atlas_common_args(
    mask_L=None,
    mask_R=None,
    mask_sub=None,
    surf_L=None,
    surf_R=None,
    coor_sub=None,
):
    default_mask_query_args = {
        'template': 'fsLR',
        'density': '32k',
        'desc': 'nomedialwall',
        'suffix': 'dparc',
    }
    default_surf_query_args = {
        'template': 'fsLR',
        'density': '32k',
        'suffix': 'sphere',
        'space': None,
    }
    if mask_L is None:
        mask_L = tflow.get(hemi='L', **default_mask_query_args)
    if mask_R is None:
        mask_R = tflow.get(hemi='R', **default_mask_query_args)
    if surf_L is None:
        surf_L = tflow.get(hemi='L', **default_surf_query_args)
    if surf_R is None:
        surf_R = tflow.get(hemi='R', **default_surf_query_args)
    surf = {
        'cortex_L': surf_L,
        'cortex_R': surf_R,
        'subcortex': coor_sub,
    }
    mask_source = {
        'cortex_L': mask_L,
        'cortex_R': mask_R,
        'subcortex': mask_sub,
    }
    return surf, mask_source
