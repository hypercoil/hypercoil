# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for base initialisers
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from numpyro.distributions import Normal
from hypercoil.functional.utils import Tensor
from hypercoil.init.base import (
    DistributionInitialiser, ConstantInitialiser, IdentityInitialiser
)
from hypercoil.init.deltaplus import DeltaPlusInitialiser
from hypercoil.init.dirichlet import DirichletInitialiser
from hypercoil.init.laplace import LaplaceInitialiser
from hypercoil.init.mapparam import MappedLogits, _to_jax_array
from hypercoil.init.semidefinite import (
    TangencyInitialiser,
    SPDEuclideanMean, SPDGeometricMean, SPDHarmonicMean, SPDLogEuclideanMean
)
from hypercoil.init.sylo import SyloInitialiser
from hypercoil.init.toeplitz import ToeplitzInitialiser


class TestBaseInit:

    def test_distribution_init(self):
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(key=key, in_features=2, out_features=3)
        model = DistributionInitialiser.init(
            model, distribution=Normal(loc=100, scale=0.1), key=key)
        assert np.abs(model.weight - 100).mean() < 0.1

        model = DistributionInitialiser.init(
            model,
            distribution=Normal(loc=0.5, scale=0.01),
            mapper=MappedLogits,
            key=key
        )
        assert np.abs(_to_jax_array(model.weight) - 0.5).mean() < 0.05
        assert np.abs(model.weight.original - 0).mean() < 0.1

    def test_constant_init(self):
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(key=key, in_features=2, out_features=3)
        model = ConstantInitialiser.init(model, value=1.)
        assert np.all(model.weight == 1)

    def test_identity_init(self):
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(key=key, in_features=5, out_features=5)
        model = IdentityInitialiser.init(model, scale=-1., shift=1.)
        assert np.all(model.weight == ~np.eye(5, dtype=bool))

    def test_deltaplus_init(self):
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(key=key, in_features=3, out_features=5)
        model = DeltaPlusInitialiser.init(
            model, loc=(0,), scale=2, var=0.01, key=key)
        assert np.all(np.abs(model.weight[..., 0] - 2) < 0.05)
        assert np.var(model.weight[..., 1:]) < 0.05

    def test_dirichlet_init(self):
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(key=key, in_features=3, out_features=5)
        model = DirichletInitialiser.init(
            model, concentration=(1e8,), num_classes=5, axis=0, key=key)
        assert np.allclose(_to_jax_array(model.weight), 1 / 5, atol=1e-4)
        assert np.allclose(model.weight.original, np.log(1 / 5), atol=1e-2)

    def test_laplace_init(self):
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(key=key, in_features=3, out_features=5)
        model = LaplaceInitialiser.init(
            model, loc=(0, 0), normalise='max',
            var=0., excl_axis=(0,), key=key)
        assert np.all(model.weight[:, 0] == 1)
        assert np.all(model.weight[:, [0]] >= model.weight)
        model = LaplaceInitialiser.init(
            model, normalise='max', var=0., key=key)
        assert np.unravel_index(
            model.weight.argmax(),
            model.weight.shape
        ) == (2, 1)
        assert (
            model.weight[1, 1] ==
            model.weight[2, 0] ==
            model.weight[2, 2] ==
            model.weight[3, 1]
        )

    def test_tangency_init(self):
        key = jax.random.PRNGKey(0)
        pkey, dkey = jax.random.split(key)
        model = eqx.nn.Conv2d(
            key=key, in_channels=1, out_channels=4, kernel_size=10)
        init_data = jax.random.uniform(dkey, shape=(5, 1, 10, 10))
        init_data = init_data @ init_data.swapaxes(-2, -1)
        init_spec = [
            SPDEuclideanMean(),
            SPDHarmonicMean(),
            SPDLogEuclideanMean(psi=1e-3),
            SPDGeometricMean(psi=1e-3),
        ]
        model = TangencyInitialiser.init(
            model, init_data=init_data, mean_specs=init_spec, key=pkey)

        L = np.linalg.eigvalsh(model.weight)
        assert (L > 0).all()

    def test_sylo_init(self):
        #TODO: replace this with an actual sylo module after it's
        #      translated to jax
        #TODO: add tests for a deep network: verify that the distribution of
        #      activations remains reasonably bounded under these
        #      initialisation schemes
        class SyloModule(eqx.Module):
            weight: tuple[Tensor, Tensor]
            def __init__(self, key, in_channels, out_channels,
                         dim_L, dim_R, rank):
                self.weight = (
                    jnp.zeros((out_channels, in_channels, dim_L, rank)),
                    jnp.zeros((out_channels, in_channels, dim_R, rank)),
                )
        class SyloModulePSD(eqx.Module):
            weight: Tensor
            def __init__(self, key, in_channels, out_channels, dim, rank):
                self.weight = jnp.zeros(
                    (out_channels, in_channels, dim, rank))

        key = jax.random.PRNGKey(0)
        model = SyloModule(key=jax.random.PRNGKey(0), in_channels=2,
                           out_channels=3, dim_L=5, dim_R=10, rank=3)
        model = SyloInitialiser.init(model, key=key)
        assert isinstance(model.weight, tuple)
        assert len(model.weight) == 2
        assert model.weight[0].shape == (3, 2, 5, 3)
        assert model.weight[1].shape == (3, 2, 10, 3)
        assert np.all(model.weight[0] != 0)
        assert np.all(model.weight[1] != 0)

        model = SyloModulePSD(key=jax.random.PRNGKey(0), in_channels=2,
                              out_channels=3, dim=10, rank=3)
        model = SyloInitialiser.init(model, key=key, psd=True)
        assert isinstance(model.weight, jax.Array)
        assert model.weight.shape == (3, 2, 10, 3)
        assert np.all(model.weight != 0)

    def test_toeplitz_init(self):
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(key=key, in_features=3, out_features=5)
        r = np.random.rand(3,)
        c = np.random.rand(3,)
        fill_value = 1.
        model = ToeplitzInitialiser.init(
            model, r=r, c=c, fill_value=fill_value, key=key)
        assert model.weight[2, 0] == model.weight[3, 1] == model.weight[4, 2]
        assert (
            model.weight[3, 0] == model.weight[4, 1] ==
            model.weight[4, 0] == fill_value)
