# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for spatial convolution on certain manifold-y things.
"""
import pytest
import torch
from functools import partial
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import haversine_distances
from hypercoil.functional.sphere import *


class TestSpherical:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.n = 5
        self.c = 6
        self.data = torch.zeros((self.n, self.c))
        self.data[:self.n, :self.n] = torch.eye(self.n)
        self.data[:, self.n:] = torch.rand((self.n, self.c - self.n))
        self.dist_euc = torch.linspace(0, 1, self.n)
        self.coor_euc = torch.zeros((self.n, 3))
        self.coor_euc[:, 1] = self.dist_euc
        self.coor_sph = torch.tensor([
            [torch.pi / 2, 0],          # north pole
            [0, 0],                     # equator / prime meridian junction
            [0, -torch.pi / 2],         # equator, 90 W
            [-torch.pi / 4, torch.pi],  # 45 S, 180th meridian
            [-torch.pi / 2, 0]          # south pole
        ])
        self.coor_sph_norm = torch.tensor([
            [0, 0, 1],                  # north pole
            [1, 0, 0],                  # equator / prime meridian junction
            [0, -1, 0],                 # equator, 90 W
            [-2 ** 0.5 / 2, 0, -2 ** 0.5 / 2],
            [0, 0, -1]                  # south pole
        ])
        # Truncated distance mask at pi / 2 radians.
        self.truncated = torch.ones((self.n, self.n))
        self.truncated[0, 3] = 0
        self.truncated[0, 4] = 0
        self.truncated[1, 3] = 0
        self.truncated = torch.minimum(self.truncated, self.truncated.T)
        # random spherical coors
        self.coor_sph_rand = torch.rand(self.n, 3)
        self.coor_sph_rand /= torch.norm(self.coor_sph_rand, dim=0, p=2)

    def test_gauss_kernel(self):
        scale = 0.5
        n = torch.distributions.normal.Normal(loc=0, scale=scale)
        ker_ref = lambda x: torch.exp(n.log_prob(x))
        ker = partial(kernel_gaussian, scale=scale)
        out = ker(self.dist_euc) / ker_ref(self.dist_euc)
        assert torch.allclose(out, out.max())

    def test_spherical_conversion(self):
        normals = sphere_to_normals(self.coor_sph)
        assert torch.allclose(normals, self.coor_sph_norm, atol=1e-6)
        restore = sphere_to_latlong(normals)
        # Discard meaningless longitudes at poles
        restore[0, 1] = 0
        restore[4, 1] = 0
        # Account for equivalence between pi and -pi longitude
        restore_equiv = restore
        restore_equiv[3, 1] *= -1
        assert (torch.allclose(restore, self.coor_sph, atol=1e-6) or
                torch.allclose(restore_equiv, self.coor_sph, atol=1e-6))
        # And check that it's still the same forward after multiple
        # applications.
        normals = sphere_to_normals(
            sphere_to_latlong(sphere_to_normals(restore))
        )
        assert torch.allclose(normals, self.coor_sph_norm, atol=1e-6)

    def test_spherical_geodesic(self):
        normals = sphere_to_normals(self.coor_sph)
        out = spherical_geodesic(normals)
        ref = haversine_distances(self.coor_sph)
        assert torch.allclose(out, torch.FloatTensor(ref), atol=1e-6)
        latlong = sphere_to_latlong(self.coor_sph_rand)
        out = spherical_geodesic(self.coor_sph_rand)
        ref = haversine_distances(latlong)
        assert torch.allclose(out, torch.FloatTensor(ref), atol=1e-6)

    def test_spatial_convolution(self):
        scale = 0.5
        out = euclidean_conv(
            data=self.data,
            coor=self.coor_euc,
            scale=scale
        )
        ref = gaussian_filter1d(
            input=self.data,
            sigma=scale * (self.n - 1),
            axis=0,
            mode='constant',
            truncate=16,
        )
        assert torch.allclose(out, torch.Tensor(ref), atol=1e-4)

    def test_spherical_convolution(self):
        """
        WARNING: Correctness is not tested.
        """
        scale = 3
        out = spherical_conv(
            data=self.data,
            coor=sphere_to_normals(self.coor_sph),
            scale=scale
        )
        out = spherical_conv(
            data=self.data,
            coor=self.coor_sph_rand,
            scale=scale
        )
        # truncation test
        out = spherical_conv(
            data=self.data,
            coor=sphere_to_normals(self.coor_sph),
            scale=scale,
            truncate=(torch.pi / 2)
        )
        out = (out[:, :self.n] == 0).float()
        assert torch.allclose(
            out + self.truncated,
            torch.ones((self.n, self.n))
        )
