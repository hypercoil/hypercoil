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
from neuromaps.transforms import mni152_to_fsaverage, mni152_to_fslr
import matplotlib.pyplot as plt
import pyvista as pv

from hypercoil.init.atlas import BaseAtlas
from .flows import direct_transform
from .surf import CortexTriSurface, make_cmap
from .utils import plot_to_image, auto_focus


def surf_from_archive(
    allowed: Sequence[str] = ("templateflow", "neuromaps")
) -> callable:
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
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(surf: CortexTriSurface, cifti: nb.Cifti2Image):
            surf.add_cifti_dataset(
                name=scalars,
                cifti=cifti,
                is_masked=is_masked,
                apply_mask=apply_mask,
                null_value=null_value,
            )
            return {"surf": surf}

        def f_transformed(
            *,
            surf: CortexTriSurface,
            cifti: nb.Cifti2Image,
            **params: Mapping,
        ):
            return xfm(f, transformer_f)(**params)(cifti=cifti, surf=surf)

        return f_transformed
    return transform


def scalars_from_atlas(
    scalars: str,
    compartment: str = ("cortex_L", "cortex_R"),
    select: Optional[Sequence[int]] = None,
    exclude: Optional[Sequence[int]] = None,
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            surf: CortexTriSurface,
            atlas: BaseAtlas,
            maps: Optional[Mapping] = None,
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
            return {
                "surf": surf,
                # "scalars": tuple(
                #     f"{scalars}_{i}" for i in range(i)
                # ),
                # "hemi": tuple(
                #     "left" if i < len(maps_L) else "right" for i in range(i)
                # ),
            }

        def f_transformed(
            *,
            surf: CortexTriSurface,
            atlas: BaseAtlas,
            maps: Optional[Mapping] = None,
            **params: Mapping,
        ):
            return xfm(f, transformer_f)(**params)(
                surf=surf, atlas=atlas, maps=maps)

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
            surf.add_gifti_dataset(
                name=scalars,
                left_gifti=left,
                right_gifti=right,
                is_masked=False,
                apply_mask=True,
                null_value=null_value,
                select=select,
                exclude=exclude,
            )
            return {"surf": surf}

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

            (cmap_left, clim_left), (cmap_right, clim_right) = make_cmap(
                surf, f'cmap_{cmap_name}', parcellation_name)

            return{
                'surf': surf,
                'cmap': (cmap_left, cmap_right),
                'clim': (clim_left, clim_right),
            }

        def f_transformed(
            *,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
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
        ):
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


def _planar_cam_transformer(
    surf: CortexTriSurface,
    hemi: Optional[str],
    initial: Sequence,
    normal: Sequence,
    n_steps: int,
) -> Mapping:
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
                slack=1.3,
            )
            cpos.append(
                (v, focus, (0, 0, 1))
            )
    return cpos


def planar_sweep_cameras(
    initial: Sequence[float] = (1, 0, 0),
    normal: Sequence[float] = (0, 0, 1),
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
            return {
                "views": (views,),
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
            return {
                "views": (views,),
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
    normal_vector: Tuple[float, float, float] = (0, 0, 1),
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
            if hemi is not "left" and hemi is not "right":
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
