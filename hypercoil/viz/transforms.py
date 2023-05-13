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
from .utils import plot_to_image


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
                except Exception:
                    continue
            return {"surf": surf}

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
            return xfm(f, transformer_f, unpack_dict=True)(**params)(
                template=template,
                load_mask=load_mask,
                projections=projections,
            )

        return f_transformed
    return transform


def scalars_from_cifti(
    scalar_name: str,
    is_masked: bool = True,
    apply_mask: bool = False,
    null_value: Optional[float] = None,
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(surf: CortexTriSurface, cifti: nb.Cifti2Image):
            surf.add_cifti_dataset(
                name=scalar_name,
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
            return xfm(f, transformer_f, unpack_dict=True)(
                **params)(cifti=cifti, surf=surf)

        return f_transformed
    return transform


def resample_to_surface(
    scalar_name: str,
    template: str = "fsLR",
) -> callable:
    templates = {
        "fsLR": mni152_to_fslr,
        "fsaverage": mni152_to_fsaverage,
    }
    f_resample = templates[template]
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(nii: nb.Nifti1Image, surf: CortexTriSurface):
            data = f_resample(nii)
            surf.add_gifti_dataset(
                name=scalar_name,
                left_gifti=data[0],
                right_gifti=data[1],
                is_masked=False,
                apply_mask=True,
                null_value=None,
            )
            return {"surf": surf}

        def f_transformed(
            *,
            nii: nb.Nifti1Image,
            surf: CortexTriSurface,
            **params: Mapping,
        ):
            return xfm(f, transformer_f, unpack_dict=True)(**params)(
                nii=nii,
                surf=surf,
            )

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
            return xfm(f, transformer_f, unpack_dict=True)(**params)(surf=surf)

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
            return xfm(f, transformer_f, unpack_dict=True)(
                **params)(nii=nii)

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
            return xfm(f, transformer_f, unpack_dict=True)(
                **params)(atlas=atlas, maps=maps)

        return f_transformed
    return transform


def ax_grid(
    ncol: Optional[int] = None,
    nrow: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    hide_axes: bool = True,
    tight_layout: bool = True,
    order: Literal["row-major", "col-major"] = "row-major",
) -> callable:
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            out: Sequence[pv.Plotter],
            views: Sequence,
            window_size: Tuple[int, int],
            hemi: Optional[Literal["left", "right", "both"]],
            **params,
        ) -> Union[Tuple[plt.Figure, ...], plt.Figure]:
            out = tuple(
                plot_to_image(
                    o,
                    views=views,
                    window_size=window_size,
                    hemi=hemi
                ) for o in out
            )
            out = list(chain(*out))
            try:
                nout = len(out)
                if ncol is None:
                    ncols = ceil(nout / nrow)
                    nrows = nrow
                elif nrow is None:
                    ncols = ncol
                    nrows = ceil(nout / ncol)
            except TypeError:
                nout = 1
                out = (out,)
                ncols = 1
                nrows = 1

            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            if order == "col-major":
                axes = axes.T
            for i, ax in enumerate(axes.flat):
                if i < nout:
                    ax.imshow(out[i])
                if hide_axes:
                    ax.axis("off")
            if tight_layout:
                fig.tight_layout()
            return fig

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
) -> callable:
    return ax_grid(
        ncol=ncol,
        nrow=nrow,
        figsize=figsize,
        hide_axes=hide_axes,
        tight_layout=tight_layout,
        order="row-major",
    )


def col_major_grid(
    ncol: Optional[int] = None,
    nrow: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    hide_axes: bool = True,
    tight_layout: bool = True,
) -> callable:
    return ax_grid(
        ncol=ncol,
        nrow=nrow,
        figsize=figsize,
        hide_axes=hide_axes,
        tight_layout=tight_layout,
        order="col-major",
    )


def plot_and_save():
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(
            pl: pv.Plotter,
            basename: str,
            views: Sequence,
            window_size: Tuple[int, int],
            hemi: Optional[Literal["left", "right", "both"]],
            **params,
        ):
            imgs = plot_to_image(
                pl,
                basename=basename,
                views=views,
                window_size=window_size,
                hemi=hemi,
            )
            return imgs

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
            fig.savefig(filename)
            return fig

        def f_transformed(*, filename: str = None, **params):
            return xfm(transformer_f, f)(filename=filename)(**params)

        return f_transformed
    return transform
