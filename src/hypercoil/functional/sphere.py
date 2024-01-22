# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Operations supporting spherical coordinate systems.
"""
from __future__ import annotations
from functools import partial
from typing import Callable, Optional

import jax.numpy as jnp

from ..engine import Tensor


# TODO: Switch to using kernel.gaussian_kernel instead of this everywhere
def kernel_gaussian(x: Tensor, scale: float = 1) -> Tensor:
    """
    An example of an isotropic kernel. Zero-centered Gaussian kernel with
    specified scale parameter.
    """
    return jnp.exp(-((x / scale) ** 2) / 2) / (scale * (2 * jnp.pi) ** 0.5)


def sphere_to_normals(coor: Tensor, r: float = 1) -> Tensor:
    r"""
    Convert spherical coordinates from latitude/longitude format to normal
    vector format. Note this only works for 2-spheres as of now.

    :Dimension: **Input :** :math:`(*, 2)`
                **Output :** :math:`(*, 3)`

    Parameters
    ----------
    coor : Tensor
        Tensor containing 2-tuple coordinates indicating the latitude and
        longitude of each point.
    r : float (default 1)
        Radius of the sphere.

    Returns
    -------
    coor : Tensor
        Tensor containing 3-tuple coordinates indicating zero-centred x, y, and
        z values of each point at radius r.
    """
    lat, lon = coor.swapaxes(-2, -1)
    cos_lat = jnp.cos(lat)
    x = r * cos_lat * jnp.cos(lon)
    y = r * cos_lat * jnp.sin(lon)
    z = r * jnp.sin(lat)
    return jnp.stack((x, y, z), axis=-1)


def sphere_to_latlong(coor: Tensor) -> Tensor:
    r"""
    Convert spherical coordinates from normal vector format to latitude/
    longitude format. Note this only works for 2-spheres as of now.

    :Dimension: **Input :** :math:`(*, 3)`
                **Output :** :math:`(*, 2)`

    Parameters
    ----------
    coor : Tensor
        Tensor containing 3-tuple coordinates indicating x, y, and z values of
        each point on a sphere whose centre is the origin.

    Returns
    -------
    coor : Tensor
        Tensor containing 2-tuple coordinates indicating the latitude and
        longitude of each point.
    """
    x, y, z = coor.swapaxes(-2, -1)
    # R = jnp.sqrt((coor[0] ** 2).sum())
    # lat = jnp.arcsin(z / R)
    lat = jnp.arctan2(z, jnp.sqrt(x**2 + y**2))
    lon = jnp.arctan2(y, x)
    return jnp.stack((lat, lon), axis=-1)


def spherical_geodesic(
    X: Tensor,
    Y: Optional[Tensor] = None,
    r: float = 1,
) -> Tensor:
    r"""
    Geodesic great-circle distance between two sets of spherical coordinates
    formatted as normal vectors.

    This is not a haversine distance, although the result is identical. Please
    ensure that input vectors are expressed as normals and not as latitude/
    longitude pairs. Because this uses a cross-product in the computation, it
    works only with 2-spheres.

    :Dimension: **X :** :math:`(*, N_X, 3)`
                **Y :** :math:`(*, N_Y, 3)`
                **Output :** :math:`(*, N_X, N_Y)`

    Parameters
    ----------
    X : Tensor
        Tensor containing coordinates on a sphere formatted as surface-normal
        vectors in Euclidean coordinates. Distances are computed between each
        coordinate in X and each coordinate in Y.
    Y : Tensor or None (default X)
        As X. If a second tensor is not provided, then distances are computed
        between every pair of points in X.
    r : float
        Radius of the sphere. We could just get this from X or Y, but we
        don't.

    Returns
    -------
    dist : Tensor
        Tensor containing pairwise great-circle distances between each
        coordinate in X and each coordinate in Y.
    """
    if Y is None:
        Y = X
    if X.shape[-1] != 3:
        raise ValueError('X must have shape (*, N, 3)')
    if Y.shape[-1] != 3:
        raise ValueError('Y must have shape (*, N, 3)')
    X = X[..., None, :]
    Y = Y[..., None, :, :]
    X, Y = jnp.broadcast_arrays(X, Y)
    crXY = jnp.cross(X, Y, axis=-1)
    num = jnp.sqrt((crXY**2).sum(-1))
    denom = (X * Y).sum(-1)
    dist = jnp.arctan2(num, denom)
    dist = jnp.where(
        dist < 0,
        dist + jnp.pi,
        dist,
    )
    return dist * r


def spatial_conv(
    data: Tensor,
    coor: Tensor,
    kernel: Callable = kernel_gaussian,
    metric: Callable = spherical_geodesic,
    max_bin: int = 10000,
    truncate: Optional[float] = None,
) -> Tensor:
    r"""
    Convolve data on a manifold with an isotropic kernel.

    Currently, this works by taking a dataset, a list of coordinates associated
    with each point in the dataset, an isotropic kernel, and a distance metric.
    It proceeds as follows:

    1. Using the provided metric, compute the distance between each pair of
       coordinates.
    2. Evaluate the isotropic kernel at each computed distance. Use this value
       to operationalise the loading weight of every coordinate on every other
       coordinate.
    3. Use the loading weights to perform a matrix product and obtain the
       kernel-convolved dataset.

    :Dimension: **data :** :math:`(*, N, C)`
                    `*` denotes any number of intervening dimensions, `C`
                    denotes the number of data channels, and `N` denotes the
                    number of data observations per channel.
                **coor :** :math:`(*, N, D)`
                    `D` denotes the dimension of the space in which the data
                    are embedded.
                **Output :** :math:`(*, C, N)`

    Parameters
    ----------
    data : Tensor
        Tensor containing data observations, which might be arrayed into
        channels.
    coor : Tensor
        Tensor containing the spatial coordinates associated with each
        observation in `data`.
    kernel : callable
        Function that maps a distance to a weight. Typically, shorter distances
        correspond to larger weights.
    metric : callable
        Function that takes as parameters two :math:`(*, N_i, D)` tensors
        containing coordinates and returns the pairwise distance between each
        pair of tensors. In Euclidean space, this could be the L2 norm; in
        spherical space, this could be the great-circle distance.
    max_bin : int
        Maximum number of points to include in a distance computation. If you
        run out of memory, try decreasing this.
    truncate : float or None (default None)
        Maximum distance at which data points can be convolved together.

    Returns
    -------
    data_conv : Tensor
        The input data convolved with the kernel. Each channel is convolved
        completely separately as of now.
    """
    start = 0
    end = min(start + max_bin, data.shape[-2])
    data_conv = jnp.zeros_like(data)
    while start < data.shape[-2]:
        # TODO: Won't work if the array is more than 2D.
        coor_block = coor[start:end, :]
        dist = metric(coor_block, coor)
        weight = kernel(dist)
        if truncate is not None:
            weight = jnp.where(dist > truncate, 0, weight)
        data_conv = data_conv.at[start:end, :].set(weight @ data)
        start = end
        end += max_bin
    return data_conv


def spherical_conv(
    data: Tensor,
    coor: Tensor,
    scale: float = 1,
    r: float = 1,
    max_bin: int = 10000,
    truncate: Optional[float] = None,
):
    r"""
    Convolve data on a 2-sphere with an isotropic Gaussian kernel.

    This is implemented in pretty much the dumbest possible way, but it works.
    Here is a likely more efficient method that requires Lie groups or some
    such thing:
    https://openreview.net/pdf?id=Hkbd5xZRb

    See :func:`spatial_conv` for implementation details.

    :Dimension: **data :** :math:`(*, C, N)`
                    `*` denotes any number of intervening dimensions, `C`
                    denotes the number of data channels, and `N` denotes the
                    number of data observations per channel.
                **coor :** :math:`(*, N, D)`
                    `D` denotes the dimension of the space in which the data
                    are embedded.
                **Output :** :math:`(*, C, N)`

    Parameters
    ----------
    data : Tensor
        Tensor containing data observations, which might be arrayed into
        channels.
    coor : Tensor
        Tensor containing the spatial coordinates associated with each
        observation in `data`.
    scale : float (default 1)
        Scale parameter of the Gaussian kernel.
    r : float (default 1)
        Radius of the sphere.
    max_bin : int
        Maximum number of points to include in a distance computation. If you
        run out of memory, try decreasing this.
    truncate : float or None (default None)
        Maximum distance at which data points can be convolved together.

    Returns
    -------
    data_conv : Tensor
        The input data convolved with the kernel. Each channel is convolved
        completely separately as of now.
    """
    kernel = partial(kernel_gaussian, scale=scale)
    metric = partial(spherical_geodesic, r=r)
    return spatial_conv(
        data=data,
        coor=coor,
        kernel=kernel,
        metric=metric,
        max_bin=max_bin,
        truncate=truncate,
    )


def _euc_dist(X, Y=None):
    """
    Euclidean L2 norm metric.
    """
    if Y is None:
        Y = X
    X = X[..., None, :]
    Y = Y[..., None, :, :]
    X, Y = jnp.broadcast_arrays(X, Y)
    return jnp.sqrt(((X - Y) ** 2).sum(-1))


def euclidean_conv(
    data: Tensor,
    coor: Tensor,
    scale: float = 1,
    max_bin: int = 10000,
    truncate: Optional[float] = None,
):
    """
    Spatial convolution using the standard L2 metric and a Gaussian kernel.

    See :func:`spatial_conv` for implementation details.
    """
    kernel = partial(kernel_gaussian, scale=scale)
    return spatial_conv(
        data=data,
        coor=coor,
        kernel=kernel,
        metric=_euc_dist,
        max_bin=max_bin,
        truncate=truncate,
    )


def rotation_matrix(axis: Tensor, angle: Tensor) -> Tensor:
    cpix = jnp.asarray((
        (
            (0, 0, 0),
            (0, 0, -1),
            (0, 1, 0),
        ),
        (
            (0, 0, 1),
            (0, 0, 0),
            (-1, 0, 0),
        ),
        (
            (0, -1, 0),
            (1, 0, 0),
            (0, 0, 0),
        ),
    ))
    cpax = cpix @ axis
    return (
        jnp.eye(3) +
        jnp.sin(angle) * cpax +
        (1 - jnp.cos(angle)) * cpax @ cpax
    )


def rotate_axis_angle(src: Tensor, axis: Tensor, angle: Tensor) -> Tensor:
    rmat = rotation_matrix(axis, angle)
    return src @ rmat.T


def icosphere(
    subdivisions: int = 0,
    target: Optional[Tensor] = None,
    secondary: Optional[Tensor] = None,
) -> Tensor:
    """
    Return coordinates for vertices of a unit icosphere.

    :Dimension: **Output :** :math:`(N, 3)`

    Parameters
    ----------
    subdivisions : int
        Number of subdivisions to perform. For example, a value of 1 will
        return the coordinates for a 12-vertex icosphere.
    target : Tensor or None (default None)
        If provided, the coordinates of the first vertex of the icosphere will be
        rotated to align with the provided target coordinate.
    secondary : Tensor or None (default None)
        If provided, the coordinates of the second vertex of the icosphere will
        be rotated to align with the provided secondary coordinate.

    Returns
    -------
    coor : Tensor
        Tensor containing the coordinates of the vertices of the icosphere.
    """
    # 1. Define icosahedron vertices, edges, and faces
    phi = (1 + 5 ** 0.5) / 2
    vertices = jnp.asarray((
        (1, phi, 0),
        (-1, phi, 0),
        (1, -phi, 0),
        (-1, -phi, 0),
        (phi, 0, 1),
        (phi, 0, -1),
        (-phi, 0, 1),
        (-phi, 0, -1),
        (0, 1, phi),
        (0, -1, phi),
        (0, 1, -phi),
        (0, -1, -phi),
    ))
    edges = jnp.asarray((
        (0, 1), (0, 4), (0, 5),
        (0, 8), (0, 10), (1, 6),
        (1, 7), (1, 8), (1, 10),
        (2, 3), (2, 4), (2, 5),
        (2, 9), (2, 11), (3, 6),
        (3, 7), (3, 9), (3, 11),
        (4, 5), (4, 8), (4, 9),
        (5, 10), (5, 11), (6, 7),
        (6, 8), (6, 9), (7, 10),
        (7, 11), (8, 9), (10, 11),
    ))
    faces = jnp.asarray((
        (0, 1, 8),
        (0, 1, 10),
        (0, 4, 5),
        (0, 4, 8),
        (0, 5, 10),
        (1, 6, 7),
        (1, 6, 8),
        (1, 7, 10),
        (2, 3, 9),
        (2, 3, 11),
        (2, 4, 5),
        (2, 4, 9),
        (2, 5, 11),
        (3, 6, 7),
        (3, 6, 9),
        (3, 7, 11),
        (4, 8, 9),
        (6, 8, 9),
        (5, 10, 11),
        (7, 10, 11),
    ))
    # 2. Obtain unique convex combinations
    K = jnp.eye(3, dtype=int)
    for _ in range(subdivisions):
        K = (K + jnp.eye(3, dtype=int)[:, None, :]).reshape(-1, 3)
    K = jnp.unique(K, axis=0)
    K = K / (subdivisions + 1)

    # 3. Compute interior points and edge points
    interior_mask = (K * (K - 1)).all(axis=-1)
    key = jnp.arange(1, subdivisions + 1) / (subdivisions + 1)
    edge_points = jnp.einsum(
        'is,eid->esd',
        jnp.stack((key, 1 - key)),
        vertices[edges],
    ).reshape(-1, 3)
    interior_points = jnp.einsum(
        'sf,ifd->isd',
        K[interior_mask, ...],
        vertices[faces],
    ).reshape(-1, 3)

    # 4. Extrude onto sphere
    icosphere = jnp.concatenate((vertices, edge_points, interior_points))
    icosphere = icosphere / jnp.linalg.norm(icosphere, axis=-1, keepdims=True)

    # 5. Rotate into alignment
    if target is not None:
        axis = jnp.cross(icosphere[0], target)
        angle = jnp.arctan2(
            jnp.linalg.norm(axis),
            icosphere[0] @ target,
        )
        icosphere = rotate_axis_angle(icosphere, axis=axis, angle=angle)
        if secondary is not None:
            axis = target
            icosangle = jnp.arctan(2)
            secondary = secondary / jnp.linalg.norm(secondary, keepdims=True)
            secondary = rotate_axis_angle(
                secondary,
                axis=jnp.cross(axis, secondary),
                angle=icosangle - jnp.arccos(axis @ secondary)
            )
            src = icosphere[1] - axis * (icosphere[1] @ axis)
            tgt = secondary - axis * (secondary @ axis)
            src = src / jnp.linalg.norm(src, keepdims=True)
            tgt = tgt / jnp.linalg.norm(tgt, keepdims=True)
            icosphere = rotate_axis_angle(
                icosphere,
                axis=axis,
                angle=-jnp.arctan2(
                    jnp.linalg.norm(jnp.cross(src, tgt)),
                    src @ tgt,
                ),
            )

    return icosphere
