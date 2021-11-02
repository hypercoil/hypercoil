# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spherical convolution
~~~~~~~~~~~~~~~~~~~~~
Convolve data on a spherical manifold with a kernel.
"""
import torch
from functools import partial


def kernel_gaussian(x, scale=1):
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
    longitude pairs.
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


def spatial_conv(data, coor, kernel=kernel_gaussian,
                 metric=spherical_geodesic, max_bin=10000):
    """
    Convolve data on a manifold with a kernel.
    """
    start = 0
    end = min(start + max_bin, data.shape[-1])
    data_conv = torch.zeros_like(data)
    while start < data.shape[-1]:
        coor_block = coor[start:end, :]
        dist = metric(coor_block, coor)
        weight = kernel(dist)
        data_conv[start:end, :] = weight @ data
        start = end
        end += max_bin
    return data_conv


def spherical_conv(data, coor, scale=1, r=1, max_bin=10000):
    kernel = partial(kernel_gaussian, scale=scale)
    metric = partial(spherical_geodesic, r=r)
    return spatial_conv(data=data, coor=coor, kernel=kernel,
                        metric=metric, max_bin=max_bin)


def euc_dist(X, Y=None):
    if not isinstance(Y, torch.Tensor):
        Y = X
    X = X.unsqueeze(-2)
    Y = Y.unsqueeze(-3)
    X, Y = torch.broadcast_tensors(X, Y)
    return torch.sqrt(((X- Y) ** 2).sum(-1))


def euclidean_conv(data, coor, scale=1, max_bin=10000):
    kernel = partial(kernel_gaussian, scale=scale)
    return spatial_conv(data=data, coor=coor, kernel=kernel,
                        metric=euc_dist, max_bin=max_bin)
