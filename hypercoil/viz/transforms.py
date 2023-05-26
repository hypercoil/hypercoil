# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Visualisation transforms
~~~~~~~~~~~~~~~~~~~~~~~~
Functions for transforming the input and output of visualisation functions.
See also ``flows.py`` for functions that transform the control flow of
visualisation functions.
"""
from itertools import chain
from math import ceil
from pkg_resources import resource_filename as pkgrf
from typing import Literal, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import nibabel as nb
import pandas as pd
from neuromaps.transforms import mni152_to_fsaverage, mni152_to_fslr
import matplotlib.pyplot as plt
import pyvista as pv

from hypercoil.engine import Tensor
from hypercoil.init.atlas import BaseAtlas
from .flows import direct_transform
from .netplot import filter_adjacency_data, filter_node_data
from .surf import CortexTriSurface, make_cmap
from .utils import plot_to_image, auto_focus


def surf_from_archive(
    allowed: Sequence[str] = ("templateflow", "neuromaps")
) -> callable:
    """
    Load a surface from a cloud-based data archive.

    Parameters
    ----------
    allowed : sequence of str (default: ("templateflow", "neuromaps")
        The archives to search for the surface.

    Returns
    -------
    callable
        A transformer function. Transformer functions accept a plotting
        function and return a new plotting function that takes the
        following arguments:
        * The transformed plotter will no longer require a ``surf`` argument,
          but will require a ``template`` argument.
        * The ``template`` argument should be a string that identifies the
          template space to load the surface from. The ``template`` argument
          will be passed to the archive loader function.
        * An optional ``load_mask`` argument can be passed to the transformed
          plotter to indicate whether the surface mask should be loaded
        (defaults to ``True``).
        * An optional ``projections`` argument can be passed to the
          transformed plotter to indicate which projections to load
          (defaults to ``("veryinflated",)``).
    """
    archives = {
        "templateflow": CortexTriSurface.from_tflow,
        "neuromaps": CortexTriSurface.from_nmaps
    }
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            template: str,
            load_mask: bool,
            projections: Sequence[str],
        ):
            for a in allowed:
                constructor = archives[a]
                try:
                    surf = constructor(
                        template=template,
                        load_mask=load_mask,
                        projections=projections,
                    )
                    return {"surf": surf}
                except Exception as e:
                    print(f"Failed to load {template} from {a}: {e}")
                    continue
            raise ValueError(
                f"Could not load {template} with projections {projections} "
                f"from any of {allowed}."
            )

        def f_transformed(
            *,
            template: str = "fsLR",
            load_mask: bool = True,
            **params: Mapping,
        ):
            try:
                projections = (params["projection"],)
            except KeyError:
                projections = ("veryinflated",)
            return xfm(f, transformer_f)(**params)(
                template=template,
                load_mask=load_mask,
                projections=projections,
            )

        return f_transformed
    return transform


def scalars_from_cifti(
    scalars: str,
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = 0.,
    plot: bool = False,
) -> callable:
    """
    Load a scalar dataset from a CIFTI file onto a CortexTriSurface.

    Parameters
    ----------
    scalars : str
        The name that the scalar dataset loaded from the CIFTI file is given
        on the surface.
    is_masked : bool (default: True)
        Indicates whether the CIFTI file contains a dataset that is already
        masked.
    apply_mask : bool (default: False)
        Indicates whether the surface mask should be applied to the CIFTI
        dataset.
    null_value : float or None (default: 0.)
        The value to use for masked-out vertices.
    plot : bool (default: False)
        Indicates whether the scalar dataset should be plotted.

    Returns
    -------
    callable
        A transformer function. Transformer functions accept a plotting
        function and return a new plotting function that takes the
        following arguments:
        * The transformed plotter will now require a ``<scalars>_cifti``
          argument, where ``<scalars>`` is the name of the scalar dataset
          provided as an argument to this function. The value of this
          argument should be either a ``Cifti2Image`` object or a path to a
          CIFTI file. This is the CIFTI image whose data will be loaded onto
          the surface.
        * The ``surf`` argument should be a ``CortexTriSurface`` object.
    """
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            surf: CortexTriSurface,
            cifti: nb.Cifti2Image,
            scalars_to_plot: Optional[Sequence[str]],
        ) -> Mapping:
            surf.add_cifti_dataset(
                name=scalars,
                cifti=cifti,
                is_masked=is_masked,
                apply_mask=apply_mask,
                null_value=null_value,
            )
            ret = {"surf": surf}
            if plot:
                scalars_to_plot = tuple(scalars_to_plot + [scalars])
                ret["scalars"] = scalars_to_plot
            return ret

        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            try:
                cifti = params[f"{scalars}_cifti"]
                params = {
                    k: v for k, v in params.items()
                    if k != f"{scalars}_cifti"
                }
            except KeyError:
                raise TypeError(
                    "Transformed plot function missing one required "
                    f"keyword-only argument: {scalars}_cifti"
                )
            scalars_to_plot = None
            if plot:
                scalars_to_plot = params.get("scalars", [])
            return xfm(f, transformer_f)(**params)(
                cifti=cifti,
                surf=surf,
                scalars_to_plot=scalars_to_plot
            )

        return f_transformed
    return transform


def scalars_from_atlas(
    scalars: str,
    compartment: str = ("cortex_L", "cortex_R"),
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
    plot: bool = False,
) -> callable:
    """
    Load a scalar dataset from an atlas object onto a CortexTriSurface.

    Parameters
    ----------
    scalars : str
        The name that the scalar dataset loaded from the atlas is given on
        the surface.
    compartment : str or Sequence[str] (default: ("cortex_L", "cortex_R"))
        The atlas compartment(s) from which to load the scalar dataset.
    select : Sequence[int] or None (default: None)
        If not None, the indices of the scalar maps to load from the atlas.
        If None, all scalar maps are loaded.
    exclude : Sequence[int] or None (default: None)
        If not None, the indices of the scalar maps to exclude from the
        atlas. If None, no scalar maps are excluded.
    plot : bool (default: True)
        Indicates whether the scalar dataset should be plotted.

    Returns
    -------
    callable
        A transformer function. Transformer functions accept a plotting
        function and return a new plotting function that takes the
        following arguments:
        * The transformed plotter will now require a ``<scalars>_atlas``
          argument, where ``<scalars>`` is the name of the scalar dataset
          provided as an argument to this function. The value of this
          argument should bean instance of an object derived from a
          ``hypercoil`` ``BaseAtlas``. This is the atlas whose data will be
          loaded onto the surface.
        * The ``surf`` argument should be a ``CortexTriSurface`` object.
        * The optional ``maps`` argument can contain parcel-wise maps that
          override those in the atlas.
    """
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            surf: CortexTriSurface,
            atlas: BaseAtlas,
            maps: Optional[Mapping] = None,
            scalars_to_plot: Optional[Sequence[str]] = None,
            hemi_to_plot: Optional[Sequence[str]] = None,
        ) -> Mapping:
            if maps is None:
                maps = atlas.maps
            maps_L = maps_R = None
            if "cortex_L" in compartment:
                maps_L = maps["cortex_L"]
            if "cortex_R" in compartment:
                maps_R = maps["cortex_R"]
            excl = exclude or []
            if select is not None:
                excl = [
                    i for i in range(len(maps_L) + len(maps_R))
                    if i not in select
                ]
            i = 0
            hemis = []
            for m in maps_L:
                if i in excl:
                    i += 1
                    continue
                surf.add_vertex_dataset(
                    name=f"{scalars}_{i}",
                    left_data=m,
                    is_masked=True,
                )
                i += 1
                hemis.append("left")
            for m in maps_R:
                if i in excl:
                    i += 1
                    continue
                surf.add_vertex_dataset(
                    name=f"{scalars}_{i}",
                    right_data=m,
                    is_masked=True,
                )
                i += 1
                hemis.append("right")
            ret = {"surf": surf}
            if plot:
                scalars_to_plot = tuple(
                    scalars_to_plot + [
                        f"{scalars}_{j}" for j in range(i) if j not in excl
                    ]
                )
                hemi_to_plot = tuple(hemi_to_plot + hemis)
                ret["scalars"] = scalars_to_plot
                ret["hemi"] = hemi_to_plot
            return ret

        def f_transformed(
            *,
            surf: CortexTriSurface,
            maps: Optional[Mapping] = None,
            **params: Mapping,
        ):
            try:
                atlas = params[f"{scalars}_atlas"]
                params = {
                    k: v for k, v in params.items()
                    if k != f"{scalars}_atlas"
                }
            except KeyError:
                raise TypeError(
                    "Transformed plot function missing one required "
                    f"keyword-only argument: {scalars}_atlas"
                )
            scalars_to_plot = None
            hemi_to_plot = None
            if plot:
                scalars_to_plot = params.get("scalars", [])
                hemi_to_plot = params.get("hemi", [])
            return xfm(f, transformer_f)(**params)(
                surf=surf,
                atlas=atlas,
                maps=maps,
                scalars_to_plot=scalars_to_plot,
                hemi_to_plot=hemi_to_plot,
            )

        return f_transformed
    return transform


def resample_to_surface(
    scalars: str,
    template: str = "fsLR",
    null_value: Optional[float] = 0.,
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
) -> callable:
    templates = {
        "fsLR": mni152_to_fslr,
        "fsaverage": mni152_to_fsaverage,
    }
    f_resample = templates[template]
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            nii: nb.Nifti1Image,
            surf: CortexTriSurface,
        ) -> Mapping:
            left, right = f_resample(nii)
            scalar_names = surf.add_gifti_dataset(
                name=scalars,
                left_gifti=left,
                right_gifti=right,
                is_masked=False,
                apply_mask=True,
                null_value=null_value,
                select=select,
                exclude=exclude,
            )
            if len(scalar_names) == 1:
                return {
                    "surf": surf,
                    "scalars": scalar_names[0],
                }
            return {
                "surf": surf,
                "scalars": tuple(scalar_names),
            }

        def f_transformed(
            *,
            nii: nb.Nifti1Image,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            return xfm(f, transformer_f)(**params)(nii=nii, surf=surf)

        return f_transformed
    return transform


def parcellate_colormap(
    cmap_name: str,
    parcellation_name: str,
) -> callable:
    cmaps = {
        "network": "viz/resources/cmap_network.nii",
        "modal": "viz/resources/cmap_modal.nii",
    }
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(surf: CortexTriSurface):
            cmap = pkgrf(
                'hypercoil',
                cmaps[cmap_name]
            )
            surf.add_cifti_dataset(
                name=f'cmap_{cmap_name}',
                cifti=cmap,
                is_masked=True,
                apply_mask=False,
                null_value=0.
            )
            (
                (cmap_left, clim_left),
                (cmap_right, clim_right),
                (cmap, clim)
            ) = make_cmap(
                surf, f'cmap_{cmap_name}', parcellation_name, return_both=True
            )

            return{
                'surf': surf,
                'cmap': (cmap_left, cmap_right),
                'clim': (clim_left, clim_right),
                'node_cmap': cmap,
                'node_cmap_range': clim,
            }

        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            return xfm(f, transformer_f)(**params)(surf=surf)

        return f_transformed
    return transform


def parcellate_scalars(
    scalars: str,
    parcellation_name: str,
) -> callable:
    sink = f"{scalars}_parcellated"
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(surf: CortexTriSurface) -> Mapping:
            parcellated = surf.parcellate_vertex_dataset(
                name=scalars,
                parcellation=parcellation_name
            )
            surf.scatter_into_parcels(
                data=parcellated,
                parcellation=parcellation_name,
                sink=sink
            )
            return {
                'surf': surf,
                'scalars': sink,
            }

        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            return xfm(f, transformer_f)(**params)(surf=surf)

        return f_transformed
    return transform


def scatter_into_parcels(
    scalars: str,
    parcellation_name: str,
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(surf: CortexTriSurface, parcellated: Tensor) -> Mapping:
            surf.scatter_into_parcels(
                data=parcellated,
                parcellation=parcellation_name,
                sink=scalars,
            )
            return {
                'surf': surf,
                'scalars': scalars,
            }

        def f_transformed(
            *,
            surf: CortexTriSurface,
            parcellated: Tensor,
            **params: Mapping,
        ):
            return xfm(f, transformer_f)(**params)(
                surf=surf, parcellated=parcellated)

        return f_transformed
    return transform


def parcel_centres(
    parcellation_name: str,
    projection: str = "pial",
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(surf: CortexTriSurface) -> Mapping:
            return {
                "surf": surf,
                "coor": surf.parcel_centres_of_mass(
                    parcellation_name,
                    projection
                ),
            }

        def f_transformed(*, surf, **params: Mapping):
            return xfm(f, transformer_f)(**params)(surf=surf)

        return f_transformed
    return transform


#TODO: replace threshold arg with the option to provide one of our hypermaths
#      expressions.
def vol_from_nifti(
    threshold: float = 0.0,
    return_val: bool = True,
    return_coor: bool = True,
    return_voxdim: bool = True,
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(nii: nb.Nifti1Image) -> Mapping:
            vol = nii.get_fdata()
            loc = np.where(vol > threshold)
            ret = {}
            if return_val:
                val = vol[loc]
                ret["val"] = val
            if return_coor:
                coor = np.stack(loc)
                coor = (nii.affine @ np.concatenate(
                    (coor, np.ones((1, coor.shape[-1])))
                ))[:3]
                ret["coor"] = coor
            if return_voxdim:
                ret["voxdim"] = nii.header.get_zooms()
            return ret

        def f_transformed(*, nii: nb.Nifti1Image, **params: Mapping):
            return xfm(f, transformer_f)(**params)(nii=nii)

        return f_transformed
    return transform


def vol_from_atlas(
    compartment: str = "all",
    return_val: bool = True,
    return_coor: bool = True,
    return_voxdim: bool = True,
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            atlas: BaseAtlas,
            maps: Optional[Mapping] = None,
        ) -> Mapping:
            if maps is None:
                maps = atlas.maps
            ret = {}
            if return_val:
                ret["val"] = maps[compartment]
            if return_coor:
                coor = atlas.coors
                coor = atlas.mask.map_to_masked(model_axes=(-2,))(coor)
                coor = atlas.compartments[compartment].map_to_masked(
                    model_axes=(-2,)
                )(coor).T
                ret["coor"] = np.asarray(coor)
            if return_voxdim:
                ret["voxdim"] = atlas.ref.zooms
            return ret

        def f_transformed(
            *,
            atlas: BaseAtlas,
            maps: Optional[Mapping] = None,
            **params: Mapping,
        ):
            return xfm(f, transformer_f)(**params)(atlas=atlas, maps=maps)

        return f_transformed
    return transform


def add_edge_variable(
    name: str,
    threshold: float = 0.0,
    percent_threshold: bool = False,
    topk_threshold_nodewise: bool = False,
    absolute: bool = True,
    incident_node_selection: Optional[np.ndarray] = None,
    connected_node_selection: Optional[np.ndarray] = None,
    edge_selection: Optional[np.ndarray] = None,
    removed_val: Optional[float] = None,
    surviving_val: Optional[float] = 1.0,
    emit_degree: Union[bool, Literal["abs", "+", "-"]] = False,
    emit_incident_nodes: Union[bool, tuple] = False,
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            adj: Union[np.ndarray, str] = None,
        ) -> Mapping:
            if isinstance(adj, str):
                adj = pd.read_csv(adj, sep='\t', header=None).values

            ret = filter_adjacency_data(
                adj=adj,
                name=name,
                threshold=threshold,
                percent_threshold=percent_threshold,
                topk_threshold_nodewise=topk_threshold_nodewise,
                absolute=absolute,
                incident_node_selection=incident_node_selection,
                connected_node_selection=connected_node_selection,
                edge_selection=edge_selection,
                removed_val=removed_val,
                surviving_val=surviving_val,
                emit_degree=emit_degree,
                emit_incident_nodes=emit_incident_nodes,
            )

            if emit_degree is not False or emit_incident_nodes is not False:
                edge_df, node_df = ret
                return {
                    "edge_values": edge_df,
                    "node_values": node_df,
                }
            return {"edge_values": ret}

        def f_transformed(**params: Mapping):
            adj = params[f"{name}_adjacency"]
            params = {
                k: v for k, v in params.items()
                if k != f"{name}_adjacency"
            }
            return xfm(f, transformer_f)(**params)(adj=adj)

        return f_transformed
    return transform


def add_node_variable(
    name: str = "node",
    threshold: Union[float, int] = 0.0,
    percent_threshold: bool = False,
    topk_threshold: bool = False,
    absolute: bool = True,
    node_selection: Optional[np.ndarray] = None,
    incident_edge_selection: Optional[np.ndarray] = None,
    removed_val: Optional[float] = None,
    surviving_val: Optional[float] = 1.0,
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            val: Union[np.ndarray, str] = None,
        ) -> Mapping:
            if isinstance(val, str):
                val = pd.read_csv(val, sep='\t', header=None).values

            node_df = filter_node_data(
                val=val,
                name=name,
                threshold=threshold,
                percent_threshold=percent_threshold,
                topk_threshold=topk_threshold,
                absolute=absolute,
                node_selection=node_selection,
                incident_edge_selection=incident_edge_selection,
                removed_val=removed_val,
                surviving_val=surviving_val,
            )

            return {
                "node_values": node_df,
            }

        def f_transformed(**params: Mapping):
            val = params[f"{name}_nodal"]
            params = {
                k: v for k, v in params.items()
                if k != f"{name}_nodal"
            }
            return xfm(f, transformer_f)(**params)(val=val)

        return f_transformed
    return transform


def _planar_cam_transformer(
    surf: CortexTriSurface,
    hemi: Optional[str],
    initial: Sequence,
    normal: Optional[Sequence[float]],
    n_steps: int,
) -> Mapping:
    if normal is None:
        _x, _y, _z = initial
        if (_x, _y, _z) == (0, 0, 1) or (_x, _y, _z) == (0, 0, -1):
            normal = (1, 0, 0)
        elif (_x, _y, _z) == (0, 0, 0):
            raise ValueError("initial view cannot be (0, 0, 0)")
        else:
            ref = np.asarray((0, 0, 1))
            initial = np.asarray(initial) / np.linalg.norm(initial)
            rejection = ref - np.dot(ref, initial) * initial
            normal = rejection / np.linalg.norm(rejection)

    assert np.isclose(np.dot(initial, normal), 0)
    angles = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    cos = np.cos(angles)
    sin = np.sin(angles)
    ax_x = initial / np.linalg.norm(initial)
    ax_z = np.asarray(normal) / np.linalg.norm(normal)
    ax_y = np.cross(ax_z, ax_x)
    ax = np.stack((ax_x, ax_y, ax_z), axis=-1)

    lin = np.zeros((n_steps, 3, 3))
    lin[:, 0, 0] = cos
    lin[:, 0, 1] = -sin
    lin[:, 1, 0] = sin
    lin[:, 1, 1] = cos
    lin[:, -1, -1] = 1

    lin = ax @ lin
    vectors = lin @ np.asarray(initial)

    if hemi is None:
        _hemi = ("left", "right")
    else:
        _hemi = (hemi,)
    cpos = []
    for h in _hemi:
        if h == "left":
            vecs_hemi = vectors.copy()
            vecs_hemi[:, 0] = -vecs_hemi[:, 0]
        else:
            vecs_hemi = vectors
        for vector in vecs_hemi:
            v, focus = auto_focus(
                vector=vector,
                plotter=surf.__getattribute__(h),
                slack=2,
            )
            cpos.append(
                (v, focus, (0, 0, 1))
            )
    return cpos


def planar_sweep_cameras(
    initial: Sequence[float] = (1, 0, 0),
    normal: Optional[Sequence[float]] = None,
    n_steps: int = 10,
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            surf: CortexTriSurface,
            hemi: Optional[str],
            hemi_params: Sequence,
        ) -> Mapping:
            views = _planar_cam_transformer(
                surf=surf,
                hemi=hemi,
                initial=initial,
                normal=normal,
                n_steps=n_steps,
            )
            hemi_params = set(hemi_params).union({"views"})
            if hemi is None:
                n_per_hemi = len(views) // 2
                views = (views[:n_per_hemi], views[n_per_hemi:])
            else:
                views = (views,)
            return {
                "views": views,
                "surf": surf,
                "hemi": hemi,
                "hemi_params": hemi_params,
            }

        def f_transformed(
            *,
            surf: CortexTriSurface,
            hemi: Optional[str] = None,
            hemi_params: Sequence = (),
            **params: Mapping,
        ):
            return xfm(f, transformer_f)(**params)(
                surf=surf, hemi=hemi, hemi_params=hemi_params,
            )

        return f_transformed
    return transform


def _focused_cam_transformer(
    surf: CortexTriSurface,
    hemi: Optional[str],
    scalars: str,
    projection: str,
    kind: str,
) -> Mapping:
    if hemi is None:
        _hemi = ("left", "right")
    else:
        _hemi = (hemi,)
    cpos = []
    for h in _hemi:
        if kind == "centroid":
            coor = surf.scalars_centre_of_mass(
                hemisphere=h,
                scalars=scalars,
                projection=projection,
            )
        elif kind == "peak":
            coor = surf.scalars_peak(
                hemisphere=h,
                scalars=scalars,
                projection=projection,
            )
        vector, focus = auto_focus(
            vector=coor,
            plotter=surf.__getattribute__(h),
            slack=1.1,
        )
        cpos.append(
            (vector, focus, (0, 0, 1))
        )
    return cpos


def scalar_focus_camera(
    projection: str,
    kind: Literal["centroid", "peak"] = "centroid",
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            surf: CortexTriSurface,
            hemi: Optional[str],
            hemi_params: Sequence,
            scalars: str,
        ) -> Mapping:
            views = _focused_cam_transformer(
                surf=surf,
                hemi=hemi,
                scalars=scalars,
                projection=projection,
                kind=kind,
            )
            hemi_params = set(hemi_params).union({"views"})
            if hemi is None:
                n_per_hemi = len(views) // 2
                views = (views[:n_per_hemi], views[n_per_hemi:])
            else:
                views = (views,)
            return {
                "views": views,
                "surf": surf,
                "scalars": scalars,
                "hemi": hemi,
                "hemi_params": hemi_params,
            }

        def f_transformed(
            *,
            surf: CortexTriSurface,
            scalars: str,
            hemi: Optional[str] = None,
            hemi_params: Sequence = (),
            **params: Mapping,
        ):
            return xfm(f, transformer_f)(**params)(
                surf=surf,
                hemi=hemi,
                hemi_params=hemi_params,
                scalars=scalars,
            )
        return f_transformed
    return transform


def _ortho_cam_transformer(
    surf: CortexTriSurface,
    hemi: Optional[str],
    scalars: str,
    projection: str,
    n_ortho: int,
) -> Mapping:
    if projection == "sphere":
        metric = "spherical"
        cur_proj = surf.projection
        surf.left.project("sphere")
        surf.right.project("sphere")
    else:
        metric = "euclidean"
    if hemi is None:
        _hemi = ("left", "right")
    else:
        _hemi = (hemi,)
    closest_poles = []
    for h in _hemi:
        coor = surf.scalars_centre_of_mass(
            hemisphere=h,
            scalars=scalars,
            projection=projection,
        )
        closest_poles.append(surf.closest_poles(
            hemisphere=h,
            coors=coor,
            metric=metric,
            n_poles=n_ortho,
        ).tolist())
    if projection == "sphere":
        surf.left.project(cur_proj)
        surf.right.project(cur_proj)
    return closest_poles


def closest_ortho_cameras(
    projection: str,
    n_ortho: int = 1,
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            surf: CortexTriSurface,
            hemi: Optional[str],
            hemi_params: Sequence,
            scalars: str,
        ) -> Mapping:
            views = _ortho_cam_transformer(
                surf=surf,
                hemi=hemi,
                scalars=scalars,
                projection=projection,
                n_ortho=n_ortho,
            )
            hemi_params = set(hemi_params).union({"views"})
            return {
                "views": views,
                "surf": surf,
                "scalars": scalars,
                "hemi": hemi,
                "hemi_params": hemi_params,
            }

        def f_transformed(
            *,
            surf: CortexTriSurface,
            scalars: str,
            hemi: Optional[str] = None,
            hemi_params: Sequence = (),
            **params: Mapping,
        ):
            return xfm(f, transformer_f)(**params)(
                surf=surf,
                hemi=hemi,
                hemi_params=hemi_params,
                scalars=scalars,
            )

        return f_transformed
    return transform


def auto_cameras(
    projection: str,
    n_ortho: int = 0,
    focus: Optional[Literal["centroid", "peak"]] = None,
    n_angles: int = 0,
    initial_angle: Tuple[float, float, float] = (1, 0, 0),
    normal_vector: Optional[Tuple[float, float, float]] = None,
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            surf: CortexTriSurface,
            hemi: Optional[str],
            hemi_params: Sequence,
            scalars: Optional[str],
        ) -> Mapping:
            views_ortho = views_focused = views_planar = []
            if n_ortho > 0:
                views_ortho = _ortho_cam_transformer(
                    surf=surf,
                    hemi=hemi,
                    scalars=scalars,
                    projection=projection,
                    n_ortho=n_ortho,
                )
            if focus is not None:
                views_focused = _focused_cam_transformer(
                    surf=surf,
                    hemi=hemi,
                    scalars=scalars,
                    projection=projection,
                    kind=focus,
                )
            if n_angles > 0:
                views_planar = _planar_cam_transformer(
                    surf=surf,
                    hemi=hemi,
                    initial=initial_angle,
                    normal=normal_vector,
                    n_steps=n_angles,
                )
            hemi_params = set(hemi_params).union({"views"})
            if hemi != "left" and hemi != "right":
                views_left = []
                views_right = []
                if views_ortho:
                    views_left.extend(views_ortho[0])
                    views_right.extend(views_ortho[1])
                if views_focused:
                    views_left.append(views_focused[0])
                    views_right.append(views_focused[1])
                if views_planar:
                    n = len(views_planar) // 2
                    views_left.extend(views_planar[:n])
                    views_right.extend(views_planar[n:])
                views = (views_left, views_right)
            else:
                views = (tuple(views_ortho[0] + views_focused + views_planar),)
            return {
                "views": views,
                "surf": surf,
                "scalars": scalars,
                "hemi": hemi,
                "hemi_params": hemi_params,
            }

        def f_transformed(
            *,
            surf: CortexTriSurface,
            scalars: Optional[str] = None,
            hemi: Optional[str] = None,
            hemi_params: Sequence = (),
            **params: Mapping,
        ):
            return xfm(f, transformer_f)(**params)(
                surf=surf,
                hemi=hemi,
                hemi_params=hemi_params,
                scalars=scalars,
            )

        return f_transformed
    return transform


def ax_grid(
    ncol: Optional[int] = None,
    nrow: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    hide_axes: bool = True,
    tight_layout: bool = True,
    num_panels: Optional[int] = None,
    order: Literal["row-major", "col-major"] = "row-major",
) -> callable:
    if num_panels is None:
        num_panels = float("inf")
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            plotter: Sequence[pv.Plotter],
            views: Sequence,
            window_size: Tuple[int, int],
            hemi: Optional[Literal["left", "right", "both"]],
            **params,
        ) -> Union[Tuple[plt.Figure, ...], plt.Figure]:
            if isinstance(views[0], str):
                views = [views] * len(plotter)
            if isinstance(window_size[0], int):
                window_size = [window_size] * len(plotter)
            if isinstance(hemi, str):
                hemi = [hemi] * len(plotter)
            out = tuple(
                plot_to_image(
                    _p,
                    views=_v,
                    window_size=_w,
                    hemi=_h,
                ) for _p, _v, _w, _h in zip(plotter, views, window_size, hemi)
            )
            out = list(chain(*out))
            try:
                nout = len(out)
                if nout > num_panels:
                    n_panels = num_panels
                    nfigs = ceil(nout / n_panels)
                    out = tuple(
                        out[i * n_panels:(i + 1) * n_panels]
                        for i in range(nfigs)
                    )
                else:
                    nfigs = 1
                    n_panels = nout
                    out = (out,)
                if ncol is None:
                    ncols = ceil(n_panels / nrow)
                    nrows = nrow
                elif nrow is None:
                    ncols = ncol
                    nrows = ceil(n_panels / ncol)
            except TypeError:
                out = ((out,),)
                nout = 1
                ncols = 1
                nrows = 1

            figs = []
            for _out in out:
                fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
                if order == "col-major":
                    axes = axes.T
                for i, ax in enumerate(axes.flat):
                    if i < len(_out):
                        ax.imshow(_out[i])
                    if hide_axes:
                        ax.axis("off")
                if tight_layout:
                    fig.tight_layout()
                figs.append(fig)
            return {"fig": tuple(figs)}

        def f_transformed(
            views: Sequence = (
                "dorsal", "ventral", "anterior", "posterior",
            ),
            window_size: Tuple[int, int] = (1300, 1000),
            hemi: Optional[Literal["left", "right", "both"]] = "both",
            **params,
        ) -> Union[Tuple[plt.Figure, ...], plt.Figure]:
            return xfm(transformer_f, f)(
                views=views, window_size=window_size, hemi=hemi)(**params)

        return f_transformed
    return transform


def row_major_grid(
    ncol: Optional[int] = None,
    nrow: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    hide_axes: bool = True,
    tight_layout: bool = True,
    num_panels: Optional[int] = None,
) -> callable:
    return ax_grid(
        ncol=ncol,
        nrow=nrow,
        figsize=figsize,
        hide_axes=hide_axes,
        tight_layout=tight_layout,
        num_panels=num_panels,
        order="row-major",
    )


def col_major_grid(
    ncol: Optional[int] = None,
    nrow: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    hide_axes: bool = True,
    tight_layout: bool = True,
    num_panels: Optional[int] = None,
) -> callable:
    return ax_grid(
        ncol=ncol,
        nrow=nrow,
        figsize=figsize,
        hide_axes=hide_axes,
        tight_layout=tight_layout,
        num_panels=num_panels,
        order="col-major",
    )


def plot_and_save():
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            plotter: pv.Plotter,
            basename: str,
            views: Sequence,
            window_size: Tuple[int, int],
            hemi: Optional[Literal["left", "right", "both"]],
            **params,
        ):
            imgs = plot_to_image(
                plotter,
                basename=basename,
                views=views,
                window_size=window_size,
                hemi=hemi,
            )
            return {**params, **{"screenshots": imgs}}

        def f_transformed(
            *,
            basename: str = None,
            views: Sequence = (
                "medial", "lateral", "dorsal",
                "ventral", "anterior", "posterior",
            ),
            window_size: Tuple[int, int] = (1300, 1000),
            hemi: Optional[Literal["left", "right", "both"]] = None,
            **params,
        ):
            return xfm(transformer_f, f)(
                basename=basename,
                views=views,
                window_size=window_size,
                hemi=hemi,
            )(**params)

        return f_transformed
    return transform


def save_fig():
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(fig: plt.Figure, filename: str, **params):
            if not isinstance(fig, plt.Figure):
                for i, _fig in enumerate(fig):
                    _fig.savefig(filename.format(index=i))
            else:
                fig.savefig(filename)
            plt.close("all")
            return {"fig": fig}

        def f_transformed(*, filename: str = None, **params):
            return xfm(transformer_f, f)(filename=filename)(**params)

        return f_transformed
    return transform


def save_html(backend: Literal["panel", "pythreejs"] = "panel"):
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(plotter: pv.Plotter, filename: str, **params):
            plotter.export_html(filename, backend=backend)
            return {"plotter": plotter}

        def f_transformed(*, filename: str = None, **params):
            return xfm(transformer_f, f)(filename=filename)(**params)

        return f_transformed
    return transform
