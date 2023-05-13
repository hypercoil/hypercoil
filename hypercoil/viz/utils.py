# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Plot and report utilities.
~~~~~~~~~~~~~~~~~~~~~~~~~~
Utilities for plotting and reporting.
"""
from itertools import chain
from math import ceil
from typing import Literal, Mapping, Optional, Sequence, Tuple, Union
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
import pyvista as pv

from hypercoil.engine.axisutil import promote_axis
from hypercoil.init.atlas import BaseAtlas
from .surfplot import plot_to_image


def source_chain(*pparams) -> callable:
    def transform(f: callable) -> callable:
        for p in reversed(pparams):
            f = p(f)
        return f
    return transform


def sink_chain(*pparams) -> callable:
    def transform(f: callable) -> callable:
        for p in pparams:
            f = p(f)
        return f
    return transform


def transform_chain(
    f: callable,
    source_chain: Optional[callable] = None,
    sink_chain: Optional[callable] = None,
) -> callable:
    if source_chain is not None:
        f = source_chain(f)
    if sink_chain is not None:
        f = sink_chain(f)
    return f


def split_chain(
    *chains: Sequence[callable],
) -> Sequence[callable]:
    def transform(f: callable) -> callable:
        fxfm = tuple(c(f) for c in chains)
        try:
            fxfm = tuple(chain(*fxfm))
        except TypeError:
            pass

        def f_transformed(**params: Mapping):
            return tuple(
                fx(**params)
                for fx in fxfm
            )

        return f_transformed
    return transform


def map_over_sequence(
    xfm: callable,
    mapping: Mapping[str, Sequence],
    output_name: str,
) -> callable:
    mapping_transform = close_mapping_transform(mapping, output_name=output_name)
    def transform(f: callable) -> callable:
        return xfm(f, mapping_transform)
    return transform


def replicate_and_map(
    var: str,
    vals: Sequence,
) -> callable:
    def transform(f: callable) -> callable:
        def f_transformed(**params: Mapping):
            return tuple(
                f(**{**params, **{var: val}})
                for val in vals
            )

        return f_transformed
    return transform


def map_over_split_chain(
    *chains: Sequence[callable],
    mapping: Mapping[str, Sequence],
) -> callable:
    for vals in mapping.values():
        assert len(chains) == len(vals), "Number of chains must match number of values."
    def transform(f: callable) -> callable:
        fxfm = tuple(c(f) for c in chains)
        try:
            fxfm = tuple(chain(*fxfm))
        except TypeError:
            pass

        def f_transformed(**params: Mapping):
            return tuple(
                fxfm[i](**params, **{k: mapping[k][i] for k in mapping})
                for i in range(len(fxfm))
            )

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
        def transformer_f(nii: nb.Nifti1Image):
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


def apply_along_axis(var: str, axis: int) -> callable:
    def transform(f: callable) -> callable:
        def f_transformed(**params):
            val = params[var]
            new_ax = promote_axis(val.ndim, axis)
            val = np.transpose(val, new_ax)
            return tuple(
                f(**{**params, var: val[i]})
                for i in range(val.shape[0])
            )

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


def save_fig():
    def transform(f: callable, xfm: callable = direct_transform) -> callable:
        def transformer_f(fig: plt.Figure, filename: str, **params):
            fig.savefig(filename)
            return fig

        def f_transformed(*, filename: str = None, **params):
            return xfm(transformer_f, f)(filename=filename)(**params)

        return f_transformed
    return transform


def direct_transform(
    f_outer: callable,
    f_inner: callable,
    unpack_dict: bool = False,
) -> callable:
    def transformed_f_outer(**f_outer_params):
        def transformed_f_inner(**f_inner_params):
            if unpack_dict:
                return f_outer(**{**f_outer_params, **f_inner(**f_inner_params)})
            return f_outer(f_inner(**f_inner_params), **f_outer_params)
        return transformed_f_inner
    return transformed_f_outer


def close_replicating_transform(
    mapping: Mapping,
    default_params: Literal["inner", "outer"] = "inner",
) -> callable:
    n_replicates = len(next(iter(mapping.values())))
    for v in mapping.values():
        assert len(v) == n_replicates, (
            "All mapped values must have the same length")
    def replicating_transform(
        f_outer: callable,
        f_inner: callable,
        unpack_dict: bool = False,
    ) -> callable:
        def transformed_f_outer(**f_outer_params):
            def transformed_f_inner(**f_inner_params):
                ret = []
                for i in range(n_replicates):
                    if default_params == "inner":
                        mapped_params_inner = {
                            k: v[i] for k, v in mapping.items()
                            if k not in f_outer_params
                        }
                        mapped_params_outer = {
                            k: v[i] for k, v in mapping.items()
                            if k in f_outer_params
                        }
                    elif default_params == "outer":
                        mapped_params_inner = {
                            k: v[i] for k, v in mapping.items()
                            if k in f_inner_params
                        }
                        mapped_params_outer = {
                            k: v[i] for k, v in mapping.items()
                            if k not in f_inner_params
                        }
                    f_inner_params_i = {
                        **f_inner_params,
                        **mapped_params_inner
                    }
                    f_outer_params_i = {
                        **f_outer_params,
                        **mapped_params_outer
                    }
                    if unpack_dict:
                        ret.append(
                            f_outer(**{**f_inner(**f_inner_params_i), **f_outer_params_i})
                        )
                    else:
                        ret.append(
                            f_outer(f_inner(**f_inner_params_i)[i], **f_outer_params_i)
                        )
                return tuple(ret)
            return transformed_f_inner
        return transformed_f_outer
    return replicating_transform


def close_mapping_transform(
    mapping: Mapping,
    output_name: str,
) -> callable:
    n_replicates = len(next(iter(mapping.values())))
    for v in mapping.values():
        assert len(v) == n_replicates, (
            "All mapped values must have the same length")
    def mapping_transform(
        f_outer: callable,
        f_inner: callable,
        unpack_dict: bool = False,
    ) -> callable:
        def transformed_f_outer(**f_outer_params):
            def transformed_f_inner(**f_inner_params):
                ret = []
                out = f_inner(**f_inner_params)
                assert len(out) == n_replicates, (
                    "The length of the output of the inner function must be "
                    "equal to the length of the mapped values")
                for i, o in enumerate(out):
                    mapped_params = {k: v[i] for k, v in mapping.items()}
                    if unpack_dict:
                        f_outer_params_i = {
                            **f_outer_params,
                            **mapped_params,
                            **o
                        }
                    else:
                        f_outer_params_i = {
                            **f_outer_params,
                            **mapped_params,
                            output_name: o
                        }
                    ret.append(f_outer(**f_outer_params_i))
                return tuple(ret)
            return transformed_f_inner
        return transformed_f_outer
    return mapping_transform
