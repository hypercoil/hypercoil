# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spherical convolution
~~~~~~~~~~~~~~~~~~~~~
Convolve data on a spherical manifold with an isotropic kernel.
"""
import torch
from functools import partial


def kernel_gaussian(x, scale=1):
    """
    An example of an isotropic kernel. Zero-centered Gaussian kernel with
    specified scale parameter.
    """
    # Written for consistency with scipy's gaussian_filter1d.
    # I do not know where the constant 4 comes from in the denominator.
    # Constants shouldn't matter much for our purposes anyway...
    return (
        torch.exp(-((x / scale) ** 2) / 2) /
        (4 * scale * (2 * torch.pi) ** 0.5)
    )


def sphere_to_normals(coor, r=1):
    """
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
    lat, long = coor.T
    coor = torch.empty((coor.shape[0], 3))
    coor[:, 0] = r * torch.cos(lat) * torch.cos(long)
    coor[:, 1] = r * torch.cos(lat) * torch.sin(long)
    coor[:, 2] = r * torch.sin(lat)
    return coor


def sphere_to_latlong(coor):
    """
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
    x, y, z = coor.T
    R = torch.sqrt((coor[0] ** 2).sum())
    coor = torch.empty((coor.shape[0], 2))
    coor[:, 0] = torch.atan2(z, torch.sqrt(x ** 2 + y ** 2))
    coor[:, 1] = torch.atan2(y, x)
    return coor


def spherical_geodesic(X, Y=None, r=1):
    """
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
        Radius of the sphere. We could just get this from X or Y, but we don't.

    Returns
    -------
    dist : Tensor
        Tensor containing pairwise great-circle distances between each
        coordinate in X and each coordinate in Y.
    """
    if not isinstance(Y, torch.Tensor):
        Y = X
    X = X.unsqueeze(-2)
    Y = Y.unsqueeze(-3)
    X, Y = torch.broadcast_tensors(X, Y)
    crXY = torch.cross(X, Y, dim=-1)
    num = (crXY ** 2).sum(-1).sqrt()
    denom = torch.sum(X * Y, dim=-1)
    dist = torch.atan2(num, denom)
    dist[dist < 0] += torch.pi
    return r * dist


def spatial_conv(data, coor, kernel=kernel_gaussian, metric=spherical_geodesic,
                 max_bin=10000, truncate=None):
    """
    Convolve data on a manifold with an isotropic kernel.

    This is implemented in pretty much the dumbest possible way, but it works.
    One day when I am better at maths, I might figure out a more efficient way
    to do this.

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
    end = min(start + max_bin, data.shape[-1])
    data_conv = torch.zeros_like(data)
    while start < data.shape[-1]:
        coor_block = coor[start:end, :]
        dist = metric(coor_block, coor)
        weight = kernel(dist)
        if truncate is not None:
            weight[dist > truncate] = 0
        data_conv[start:end, :] = weight @ data
        start = end
        end += max_bin
    return data_conv


def spherical_conv(data, coor, scale=1, r=1, max_bin=10000, truncate=None):
    """
    Convolve data on a 2-sphere with a Gaussian kernel.

    This is implemented in pretty much the dumbest possible way, but it works.
    Here is a likely more efficient method that requires Lie groups or some
    such thing:
    https://openreview.net/pdf?id=Hkbd5xZRb

    Please see `spatial_conv` for implementation details.

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
    return spatial_conv(data=data, coor=coor, kernel=kernel,
                        metric=metric, max_bin=max_bin, truncate=truncate)


def _euc_dist(X, Y=None):
    """
    Euclidean L2 norm metric, for testing/illustrative purposes only.
    """
    if not isinstance(Y, torch.Tensor):
        Y = X
    X = X.unsqueeze(-2)
    Y = Y.unsqueeze(-3)
    X, Y = torch.broadcast_tensors(X, Y)
    return torch.sqrt(((X- Y) ** 2).sum(-1))


def euclidean_conv(data, coor, scale=1, max_bin=10000):
    """
    Spatial convolution using the standard L2 metric and a Gaussian kernel.
    Please don't use this function. For testing/illustrative purposes only.
    """
    kernel = partial(kernel_gaussian, scale=scale)
    return spatial_conv(data=data, coor=coor, kernel=kernel,
                        metric=_euc_dist, max_bin=max_bin)
