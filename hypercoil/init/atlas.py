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
    _LogicMaskMixin,
    _CortexSubcortexCIfTIMaskMixin,
    _FromNullMaskMixin,
    _SingleCompartmentMixin,
    _MultiCompartmentMixin,
    _CortexSubcortexCIfTICompartmentMixin,
    _DiscreteLabelMixin,
    _ContinuousLabelMixin,
    _DirichletLabelMixin,
    _VolumetricMeshMixin,
    _VertexCIfTIMeshMixin,
    _SpatialConvMixin,
)
from .base import DomainInitialiser
from .dirichlet import DirichletInit
from ..functional import UnstructuredNoiseSource


class BaseAtlas(ABC):
    """
    Atlas object encoding linear mappings from voxels to labels.
    Base class inherited by discrete and continuous atlas containers.

    About
    -----
    Several atlas classes are included to cover frequent scenarios, but users
    can also create their own atlas class compositionally using the available
    mixins. Each atlas class must implement the following methods:

    `_load_reference`
        Implemented by a `~ReferenceMixin` class.
    `_create_mask`
        Implemented by a `~MaskMixin` class.
    `_compartment_names_dict`
        Implemented by a `~CompartmentMixin` class.
    `_create_compartments`
        Implemented by a `~CompartmentMixin` class.
    `_configure_decoders`
        Implemented by a `~LabelMixin` class.
    `_populate_map_from_ref`
        Implemented by a `~LabelMixin` class.
    `_init_coors`
        Implemented by a `~MeshMixin` class.

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
    clear_cache : bool (True)
        Indicates that data loaded in from the reference should be cleared
        away when atlas creation is complete.

    Attributes
    ----------
    ref
        The reference whose pointer was provided at construction time.
    mask : bool tensor
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments : dict(bool tensor)
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder : dict(long tensor)
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps : dict(tensor)
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors : dict(tensor)
        Spatial position of each location in the atlas.
    topology : dict(str)
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data : ndarray
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """
    def __init__(self, ref_pointer, mask_source, dtype=None, device=None,
                 clear_cache=True, name=None, **kwargs):
        if name is None: name = ''
        self.name = name
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
    def _populate_map_from_ref(self, map, labels, mask, compartment=None):
        pass

    @abstractmethod
    def _init_coors(self, source=None, names_dict=None,
                    dtype=None, device=None):
        pass

    def __repr__(self):
        return f'{type(self).__name__}({self.name})'

    def __call__(self, compartments=True, normalise=False, sigma=None,
                 noise=None, max_bin=10000, spherical_scale=1, truncate=None):
        """
        Compute transformed maps for selected atlas subcompartments.

        Parameters
        ----------
        compartments : iterable, False, or True (default True)
            Compartments for which transformed maps should be computed. If
            this is set to False, then a single transformed map is returned
            for the combined compartment containing all spatial locations. If
            this is set to True, then a transformed map is returned for every
            defined compartment.
        normalise : bool (default False)
            Indicates that maps should be spatially normalised such that the
            sum over all assignments to a parcel is equal to 1. When the map
            is used as a linear transformation, this option results in
            computation of a weighted average over each parcel.
        sigma : float or None (default None)
            If this is not None, then spatial smoothing using a Gaussian
            kernel is applied over each parcel's assignments. Distances are
            established by the atlas's coordinate system and the topology of
            each compartment. The value of sigma establishes the width of the
            Gaussian kernel.
        noise : callable or None
            If this is a callable, then it should be a mapping from a tensor
            to another tensor of the same dimensions. After spatial smoothing
            but before normalisation, this callable is applied to each map
            tensor to inject noise.
        max_bin : int (default 10000)
            Because spatial smoothing has substantial memory overhead, this
            flag sets a ceiling on the number of spatial locations to be
            smoothed at a time.
        spherical_scale : float (default 1)
            During spatial smoothing, the Gaussian kernel is scaled by this
            value for spherical coordinate systems only. This enables the use
            of different kernel widths for spherical and Euclidean
            coordinates.
        truncate : float
            Maximum distance at which data points are convolved together
            during smoothing.
        """
        ret = {}
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
    """
    Abstract base class for atlases initialised from a Dirichlet distribution.
    See `BaseAtlas` for general details.

    Parameters
    ----------
    mask_source
        Source of data used to create an overall mask for the atlas.
    compartment_labels : dict(int)
        Number of labels to initialise for each compartment.
    conc : float
        If this is provided and `init` is not, then the Dirichlet
        distributions used to sample assignments of spatial locations to
        parcels are defined with the same concentration parameter, `conc`,
        for each parcel.
    template_image
        Template used to define the spatial dimensions and coordinates of the
        atlas. If this is not provided explicitly, then the mask source image
        is used by default.
    init : dict(Dirichlet)
        Dict mapping from compartment names to the Dirichlet distributions
        used for parcel assignment initialisation in each compartment.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.
    """
    def __init__(self, mask_source, compartment_labels, conc=100.,
                 template_image=None, init=None, dtype=None, device=None,
                 name=None, **kwargs):
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
                         name=name,
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
    _FromNullMaskMixin,
    _SingleCompartmentMixin,
    _DiscreteLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    BaseAtlas,
):
    """
    Discrete atlas container object. Use for atlases stored in single-volume
    images with non-overlapping, discrete-valued parcels. It is assumed that
    a value of 0 in the reference image corresponds to no label.

    Parameters
    ----------
    ref_pointer
        Path to a NIfTI file containing the atlas.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.
    clear_cache : bool (True)
        Indicates that data loaded in from the reference should be cleared
        away when atlas creation is complete.

    Attributes
    ----------
    ref : nb.Nifti1Image
        The reference whose pointer was provided at construction time.
    mask : bool tensor
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments : dict(bool tensor)
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder : dict(long tensor)
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps : dict(tensor)
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors : dict(tensor)
        Spatial position of each location in the atlas.
    topology : dict(str)
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data : ndarray
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """
    def __init__(self, ref_pointer, clear_cache=True, name=None,
                 dtype=None, device=None,):
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=0,
                         clear_cache=clear_cache,
                         name=name,
                         dtype=dtype,
                         device=device)


class MultiVolumetricAtlas(
    _SingleReferenceMixin,
    _FromNullMaskMixin,
    _SingleCompartmentMixin,
    _ContinuousLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    BaseAtlas,
):
    """
    Continuous atlas container object. Use for atlases whose labels overlap
    and must therefore be stored across multiple image volumes -- for
    instance, probabilistic segmentations or ICA results. If the labels are
    stored across multiple files, use `MultifileVolumetricAtlas` instead.

    Parameters
    ----------
    ref_pointer
        Path to a NIfTI file containing the atlas.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.
    clear_cache : bool (True)
        Indicates that data loaded in from the reference should be cleared
        away when atlas creation is complete.

    Attributes
    ----------
    ref : nb.Nifti1Image
        The reference whose pointer was provided at construction time.
    mask : bool tensor
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments : dict(bool tensor)
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder : dict(long tensor)
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps : dict(tensor)
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors : dict(tensor)
        Spatial position of each location in the atlas.
    topology : dict(str)
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data : ndarray
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """
    def __init__(self, ref_pointer, clear_cache=True, name=None,
                 dtype=None, device=None,):
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=0,
                         clear_cache=clear_cache,
                         name=name,
                         dtype=dtype,
                         device=device)


class MultifileVolumetricAtlas(
    _MultiReferenceMixin,
    _FromNullMaskMixin,
    _SingleCompartmentMixin,
    _ContinuousLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    BaseAtlas,
):
    """
    Continuous atlas container object. Use for atlases whose labels overlap
    and must therefore be stored across multiple image files -- for
    instance, probabilistic segmentations or ICA results. If the labels are
    stored across multiple volumes of a single file, use `MultiVolumetricAtlas`
    instead.

    Parameters
    ----------
    ref_pointer
        Path to a NIfTI file containing the atlas.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.
    clear_cache : bool (True)
        Indicates that data loaded in from the reference should be cleared
        away when atlas creation is complete.

    Attributes
    ----------
    ref : nb.Nifti1Image
        The reference whose pointer was provided at construction time.
    mask : bool tensor
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments : dict(bool tensor)
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder : dict(long tensor)
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps : dict(tensor)
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors : dict(tensor)
        Spatial position of each location in the atlas.
    topology : dict(str)
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data : ndarray
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """
    def __init__(self, ref_pointer, clear_cache=True, name=None,
                 dtype=None, device=None,):
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=0,
                         clear_cache=clear_cache,
                         name=name,
                         dtype=dtype,
                         device=device)


class CortexSubcortexCIfTIAtlas(
    _CIfTIReferenceMixin,
    _SingleReferenceMixin,
    _CortexSubcortexCIfTIMaskMixin,
    _CortexSubcortexCIfTICompartmentMixin,
    _DiscreteLabelMixin,
    _VertexCIfTIMeshMixin,
    _SpatialConvMixin,
    BaseAtlas,
):
    """
    CIfTI surface-based atlas container object. This class automatically
    defines three subcompartments for the left and right hemispheric cortices
    and for the subcortical grey matter. The cortical hemispheres are surfaces
    endowed with a spherical coordinate system, while the subcortex uses a
    volumetric, Euclidean coordinate system.

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
        reference. Default to 'CIFTI_STRUCTURE_CORTEX_LEFT' and
        'CIFTI_STRUCTURE_CORTEX_RIGHT'.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.
    clear_cache : bool (True)
        Indicates that data loaded in from the reference should be cleared
        away when atlas creation is complete.

    Attributes
    ----------
    ref : nb.Nifti1Image
        The reference whose pointer was provided at construction time.
    mask : bool tensor
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments : dict(bool tensor)
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder : dict(long tensor)
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps : dict(tensor)
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors : dict(tensor)
        Spatial position of each location in the atlas.
    topology : dict(str)
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data : ndarray
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """
    def __init__(self, ref_pointer,
                 mask_L=None, mask_R=None, surf_L=None, surf_R=None,
                 cortex_L='CIFTI_STRUCTURE_CORTEX_LEFT',
                 cortex_R='CIFTI_STRUCTURE_CORTEX_RIGHT',
                 clear_cache=True, name=None, dtype=None, device=None):
        self.surf, mask_source = _cifti_atlas_common_args(
            mask_L=mask_L,
            mask_R=mask_R,
            surf_L=surf_L,
            surf_R=surf_R
        )
        super().__init__(ref_pointer=ref_pointer,
                         mask_source=mask_source,
                         clear_cache=clear_cache,
                         name=name,
                         dtype=dtype,
                         device=device,
                         cortex_L=cortex_L,
                         cortex_R=cortex_R)


class DirichletInitVolumetricAtlas(
    _LogicMaskMixin,
    _SingleCompartmentMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    DirichletInitBaseAtlas,
):
    """
    Volumetric atlas object created from random samples of a Dirichlet
    distribution. Each spatial location's parcel assignment is a categorical
    distribution sampled from a Dirichlet distribution. For a surface-based
    Dirichlet atlas, use `DirichletInitSurfaceAtlas`.

    Parameters
    ----------
    mask_source
        Source of data used to create an overall mask for the atlas.
    n_labels : dict(int)
        Number of labels to initialise for the atlas.
    conc : float
        If this is provided and `init` is not, then the Dirichlet
        distributions used to sample assignments of spatial locations to
        parcels are defined with the same concentration parameter, `conc`,
        for each parcel.
    init : dict(Dirichlet)
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
    ref : nb.Nifti1Image
        The reference whose pointer was provided at construction time.
    mask : bool tensor
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments : dict(bool tensor)
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder : dict(long tensor)
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps : dict(tensor)
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors : dict(tensor)
        Spatial position of each location in the atlas.
    topology : dict(str)
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data : None
        Nothing here. The field exists for consistency with other atlas
        classes.
    """
    def __init__(self, mask_source, n_labels, conc=100., name=None,
                 init=None, dtype=None, device=None, **kwargs):
        if init is not None: init = {'all': init}
        super().__init__(mask_source=mask_source,
                         compartment_labels={'all': n_labels},
                         conc=conc,
                         init=init,
                         name=name,
                         dtype=dtype,
                         device=device)


class DirichletInitSurfaceAtlas(
    _CIfTIReferenceMixin,
    _CortexSubcortexCIfTIMaskMixin,
    _CortexSubcortexCIfTICompartmentMixin,
    _VertexCIfTIMeshMixin,
    _SpatialConvMixin,
    DirichletInitBaseAtlas,
):
    """
    Surface atlas object created from random samples of a Dirichlet
    distribution. Each spatial location's parcel assignment is a categorical
    distribution sampled from a Dirichlet distribution. This class
    defines three subcompartments for the left and right hemispheric cortices
    and for the subcortical grey matter. The cortical hemispheres are surfaces
    endowed with a spherical coordinate system, while the subcortex uses a
    volumetric, Euclidean coordinate system. For a volumetric Dirichlet atlas,
    use `DirichletInitVolumetricAtlas`.

    Parameters
    ----------
    cifti_template
        Template CIfTI for establishing compartment extents.
    compartment_labels : dict(int)
        Number of labels to initialise for each compartment.
    conc : float
        If this is provided and `init` is not, then the Dirichlet
        distributions used to sample assignments of spatial locations to
        parcels are defined with the same concentration parameter, `conc`,
        for each parcel.
    init : dict(Dirichlet)
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
    surf_L
        Path to a GIfTI file containing the mesh of the surface of the right
        cortical hemisphere. If none is provided, the 32K-vertex spherical
        fsLR surface is used by default.
    cortex_L and cortex_R : str
        Names of brain model axis objects corresponding to cortex in the CIfTI
        reference. Default to 'CIFTI_STRUCTURE_CORTEX_LEFT' and
        'CIFTI_STRUCTURE_CORTEX_RIGHT'.
    dtype
        Datatype for non-Boolean (non-mask) and non-Long (non-label) tensors
        created as part of the atlas.
    device
        Device on which all tensors created as part of the atlas reside.

    Attributes
    ----------
    ref : nb.Nifti1Image
        The reference whose pointer was provided at construction time.
    mask : bool tensor
        Boolean tensor indicating the inclusion status of each spatial
        location in the atlas.
    compartments : dict(bool tensor)
        Boolean tensor indicating the spatial extent of each atlas
        subcompartment.
    decoder : dict(long tensor)
        Numerical identity of the parcel in each row of a subcompartment's
        map tensor.
    maps : dict(tensor)
        Assignment of each spatial location in each subcompartment to a set
        of regions.
    coors : dict(tensor)
        Spatial position of each location in the atlas.
    topology : dict(str)
        Type of topology (spherical or Euclidean) over each compartment.
    cached_ref_data : ndarray
        Data loaded from the reference to enable construction. By default,
        this is purged when construction is complete.
    """
    def __init__(self, cifti_template, compartment_labels,
                 conc=100., init=None, name=None, dtype=None, device=None,
                 mask_L=None, mask_R=None, surf_L=None, surf_R=None,
                 cortex_L='CIFTI_STRUCTURE_CORTEX_LEFT',
                 cortex_R='CIFTI_STRUCTURE_CORTEX_RIGHT'):
        self.surf, mask_source = _cifti_atlas_common_args(
            mask_L=mask_L,
            mask_R=mask_R,
            surf_L=surf_L,
            surf_R=surf_R
        )
        super().__init__(template_image=cifti_template,
                         mask_source=mask_source,
                         compartment_labels=compartment_labels,
                         conc=conc,
                         init=init,
                         name=name,
                         dtype=dtype,
                         device=device,
                         cortex_L=cortex_L,
                         cortex_R=cortex_R)


class _MemeAtlas(
    _SingleReferenceMixin,
    _LogicMaskMixin,
    _MultiCompartmentMixin,
    _DiscreteLabelMixin,
    _VolumetricMeshMixin,
    _SpatialConvMixin,
    BaseAtlas,
):
    """
    For testing purposes only.
    """
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
                         name='Meme',
                         **compartments_dict)


def atlas_init_(tensor, compartment, atlas, normalise=False,
                max_bin=10000, spherical_scale=1, truncate=None,
                kernel_sigma=None, noise_sigma=None):
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
    val = atlas(compartments=compartment,
                normalise=normalise,
                sigma=kernel_sigma,
                noise=noise,
                max_bin=max_bin,
                spherical_scale=spherical_scale,
                truncate=truncate)
    tensor.copy_(val[compartment[0]])


class AtlasInit(DomainInitialiser):
    def __init__(self, atlas, normalise=False, max_bin=10000,
                 spherical_scale=1, truncate=None, kernel_sigma=None,
                 noise_sigma=None, domain=None):
        init = partial(atlas_init_, atlas=atlas, normalise=normalise,
                       max_bin=max_bin, spherical_scale=spherical_scale,
                       kernel_sigma=kernel_sigma, noise_sigma=noise_sigma,
                       truncate=truncate)
        if domain is None:
            try:
                domain = atlas.init['all'].domain
            except AttributeError:
                pass
        super(AtlasInit, self).__init__(init=init, domain=domain)

    def __call__(self, tensor):
        for k, v in tensor.items():
            super(AtlasInit, self).__call__(v, compartment=[k])


def _cifti_atlas_common_args(
    mask_L=None, mask_R=None, surf_L=None, surf_R=None
):
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
    surf = {
        'cortex_L' : surf_L,
        'cortex_R' : surf_R
    }
    mask_source = {
        'cortex_L' : mask_L,
        'cortex_R' : mask_R,
        'subcortex' : None
    }
    return surf, mask_source
