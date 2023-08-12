# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Differentiably compute a weight's centre of mass.

Functionality is available that operates on the "intrinsic" mesh grid
coordinates of a tensor
(:func:`cmass`, :func:`cmass_reference_displacement_grid`)
or that takes the last axis of a tensor to correspond
to different locations and accepts a second argument that indicates explicitly
the coordinates of each location
(:func:`cmass_coor`, :func:`diffuse`, :func:`cmass_reference_displacement`).
"""
from __future__ import annotations
from typing import Any, Optional, Sequence

import jax.numpy as jnp

from ..engine import Tensor
from .sphere import spherical_geodesic


def cmass(
    X: Tensor,
    axes: Optional[Sequence[int]] = None,
    na_rm: bool = False,
) -> Tensor:
    r"""
    Differentiably compute a weight's centre of mass. This can be used to
    regularise the weight so that its centre of mass is close to a provided
    coordinate.

    :Dimension: **Input :** :math:`(*, k_1, k_2, ..., k_n)`
                    `*` denotes any number of batch and intervening
                    dimensions, and the `k_i`s are dimensions along which the
                    centre of mass is computed
                **Output :** :math:`(*, n)`
                    `n` denotes the number of dimensions of each centre of
                    mass vector

    Parameters
    ----------
    X : Tensor
        Tensor containing the weights whose centres of mass are to be
        computed.
    axes : iterable or None (default None)
        Axes of the input tensor that together define each slice of the tensor
        within which a single centre-of-mass vector is computed. If this is
        set to None, then the centre of mass is computed across all axes. If
        this is [-3, -2, -1], then the centre of mass is computed separately
        for each 3-dimensional slice spanned by the last three axes of the
        tensor.
    na_rm : float or False (default False)
        If any single slice of the input tensor has zero mass, then the centre
        of mass within that slice is undefined and populated with NaN. The
        `na_rm` parameter specified how such undefined values are handled. If
        this is False, then NaN values are left intact; if this is a float,
        then NaN values are replaced by the specified float.

    Returns
    -------
    cmass : Tensor
        Centre of mass vectors for each slice from the input tensor. The
        coordinates are ordered according to the specification in ``axes``.
    """
    dim = X.shape
    ndim = X.ndim
    all_axes = list(range(ndim))
    axes = [all_axes[ax] for ax in axes] if axes is not None else all_axes
    axes = tuple(axes)
    out_dim = [s for ax, s in enumerate(dim) if all_axes[ax] not in axes]
    out_dim += [len(axes)]
    out = jnp.zeros(out_dim)
    for i, ax in enumerate(axes):
        coor = jnp.arange(1, X.shape[ax] + 1)
        # TODO
        # This is going to lead to trouble with jit since the shape of the
        # coordinate tensor is set according to the shape of the input tensor.
        # Add a test case for this and fix it.
        while coor.ndim < ndim - all_axes[ax]:
            coor = coor[..., None]
        num = (coor * X).sum(axes)
        denom = X.sum(axes)
        out = out.at[..., i].set(num / denom - 1)
        if na_rm is not False:
            out = jnp.where(denom == 0, na_rm, out)
    return out


def cmass_reference_displacement_grid(
    weight: Tensor,
    refs: Tensor,
    axes: Optional[Sequence[int]] = None,
    na_rm: bool = False,
) -> Tensor:
    """
    Displacement of centres of mass from reference points -- grid version.

    See :func:`cmass` for parameter specifications.
    """
    cm = cmass(weight, axes=axes, na_rm=na_rm)
    return cm - refs


def cmass_reference_displacement(
    weight: Tensor,
    refs: Tensor,
    coor: Tensor,
    radius: Optional[float] = None,
) -> Tensor:
    """
    Displacement of centres of mass from reference points -- explicit
    coordinate version.

    See :func:`cmass_coor` for parameter specifications.
    """
    cm = cmass_coor(weight, coor, radius=radius)
    return cm - refs


def cmass_coor(
    X: Tensor,
    coor: Tensor,
    radius: Optional[float] = None,
) -> Tensor:
    r"""
    Differentiably compute a weight's centre of mass.

    :Dimension: **Input :** :math:`(*, W, L)`
                    ``*`` denotes any number of preceding dimensions, W
                    denotes number of weights (e.g., regions of an atlas),
                    and L denotes number of locations (e.g., voxels).
                **coor :** :math:`(*, D, L)`
                    D denotes the dimension of the embedding space of the
                    locations.

    Parameters
    ----------
    X : Tensor
        Weight whose centre of mass is to be computed.
    coor : Tensor
        Coordinates corresponding to each column (location/voxel) in X.
    radius : float or None (default None)
        If this is not None, then the computed centre of mass is projected
        onto a sphere with the specified radius.

    Returns
    -------
    Tensor
        Tensor containing the coordinates of the centre of mass of each row of
        input X. Coordinates are ordered as in the second-to-last axis of
        ``coor``.
    """
    num = (X[..., None, :, :] * coor[..., None, :]).sum(-1)
    denom = X.sum(-1)
    if radius is not None:
        cmass_euc = num / denom
        return radius * cmass_euc / jnp.linalg.norm(cmass_euc, ord=2, axis=-2)
    return num / denom


def diffuse(
    X: Tensor,
    coor: Tensor,
    norm: Any = 2,
    floor: float = 0,
    radius: Optional[float] = None,
) -> Tensor:
    r"""
    Compute a compactness score for a weight.

    The compactness is defined as

    :math:`\mathbf{1}^\intercal\left(A \circ \left\|C - \frac{AC}{A\mathbf{1}} \right\|_{cols} \right)\mathbf{1}`

    :Dimension: **Input :** :math:`(*, W, L)`
                    ``*`` denotes any number of preceding dimensions, W
                    denotes number of weights (e.g., regions of an atlas),
                    and L denotes number of locations (e.g., voxels).
                **coor :** :math:`(*, D, L)`
                    D denotes the dimension of the embedding space of the
                    locations.

    Parameters
    ----------
    X : Tensor
        Weight for which the compactness score is to be computed.
    coor : Tensor
        Coordinates corresponding to each column (location/voxel) in X.
    norm
        Indicator of the type of norm to use for the distance function.
    floor : float (default 0)
        Any points closer to the centre of mass than the floor are assigned
        a compactness score of 0.
    radius : float or None (default None)
        If this is not None, then the centre of mass and distances are
        computed on a sphere with the specified radius.

    Returns
    -------
    float
        Measure of each weight's compactness about its centre of mass.
    """
    cm = cmass_coor(X, coor, radius=radius)
    if radius is None:
        dist = cm[..., None] - coor[..., None, :]
        dist = jnp.linalg.norm(dist, ord=2, axis=-3)
    else:
        dist = spherical_geodesic(
            coor.swapaxes(-1, -2), cm.swapaxes(-1, -2), r=radius
        ).swapaxes(-1, -2)
    dist = jnp.maximum(dist - floor, 0)
    return (X * dist).mean(-1)
