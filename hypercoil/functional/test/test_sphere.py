# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for spatial convolution on certain manifold-y things.
"""
import pytest
import numpy as np
import jax.numpy as jnp
from distrax import Normal
from functools import partial
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import haversine_distances
from hypercoil.functional.sphere import *


class TestSpherical:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.n = 5
        self.c = 6
        self.data = np.zeros((self.n, self.c))
        self.data[:self.n, :self.n] = np.eye(self.n)
        self.data[:, self.n:] = np.random.rand(self.n, self.c - self.n)
        self.dist_euc = np.linspace(0, 1, self.n)
        self.coor_euc = np.zeros((self.n, 3))
        self.coor_euc[:, 1] = self.dist_euc
        self.coor_sph = np.array([
            [np.pi / 2, 0],          # north pole
            [0, 0],                  # equator / prime meridian junction
            [0, -np.pi / 2],         # equator, 90 W
            [-np.pi / 4, np.pi],     # 45 S, 180th meridian
            [-np.pi / 2, 0]          # south pole
        ])
        self.coor_sph_norm = np.array([
            [0, 0, 1],                  # north pole
            [1, 0, 0],                  # equator / prime meridian junction
            [0, -1, 0],                 # equator, 90 W
            [-2 ** 0.5 / 2, 0, -2 ** 0.5 / 2],
            [0, 0, -1]                  # south pole
        ])
        # Truncated distance mask at pi / 2 radians.
        self.truncated = np.ones((self.n, self.n))
        self.truncated[0, 3] = 0
        self.truncated[0, 4] = 0
        self.truncated[1, 3] = 0
        self.truncated = np.minimum(self.truncated, self.truncated.T)
        # random spherical coors
        self.coor_sph_rand = np.random.rand(self.n, 3)
        self.coor_sph_rand /= np.linalg.norm(self.coor_sph_rand, axis=0, ord=2)

    def test_gauss_kernel(self):
        scale = 0.5
        n = Normal(loc=0, scale=scale)
        ker_ref = lambda x: jnp.exp(n.log_prob(x))
        ker = partial(kernel_gaussian, scale=scale)
        out = ker(self.dist_euc) / ker_ref(self.dist_euc)
        assert np.allclose(out, out.max())

    def test_spherical_conversion(self):
        normals = sphere_to_normals(self.coor_sph)
        assert np.allclose(normals, self.coor_sph_norm, atol=1e-6)
        restore = sphere_to_latlong(normals)
        # Discard meaningless longitudes at poles
        restore = restore.at[0, 1].set(0)
        restore = restore.at[4, 1].set(0)
        # Account for equivalence between pi and -pi longitude
        restore_equiv = restore
        restore_equiv = restore_equiv.at[3, 1].set(restore_equiv[3, 1] * -1)
        assert (np.allclose(restore, self.coor_sph, atol=1e-6) or
                np.allclose(restore_equiv, self.coor_sph, atol=1e-6))
        # And check that it's still the same forward after multiple
        # applications.
        normals = sphere_to_normals(
            sphere_to_latlong(sphere_to_normals(restore))
        )
        assert np.allclose(normals, self.coor_sph_norm, atol=1e-6)

    def test_spherical_geodesic(self):
        normals = sphere_to_normals(self.coor_sph)
        out = spherical_geodesic(normals)
        ref = haversine_distances(self.coor_sph)
        assert np.allclose(out, ref, atol=1e-6)
        latlong = sphere_to_latlong(self.coor_sph_rand)
        out = spherical_geodesic(self.coor_sph_rand)
        ref = haversine_distances(latlong)
        assert np.allclose(out, ref, atol=1e-6)

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
        out = out / out.max()
        ref = ref / ref.max()
        assert np.allclose(out, ref, atol=1e-4)

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
            truncate=(np.pi / 2)
        )
        out = (out[:, :self.n] == 0).astype(float)
        assert np.allclose(
            out + self.truncated,
            np.ones((self.n, self.n))
        )
