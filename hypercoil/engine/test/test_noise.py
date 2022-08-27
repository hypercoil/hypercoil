# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for noise sources
"""
import jax
import distrax
import numpy as np
from hypercoil.engine.noise import (
    StochasticSource, refresh,
    ScalarIIDAddStochasticTransform,
    ScalarIIDMulStochasticTransform,
    TensorIIDAddStochasticTransform,
    TensorIIDMulStochasticTransform,
    OuterProduct, Diagonal, MatrixExponential,
)


class TestNoise:

    def test_stochastic_sources(self):
        tr = (
            (0, 1),
            3,
            (1, (StochasticSource(key=jax.random.PRNGKey(23243)),)),
            17,
            (3, 11)
        )
        key_0 = tr[2][1][0].key
        key_1 = refresh(tr)[2][1][0].key
        key_2 = refresh(tr, code=1)[2][1][0].key
        assert np.all(key_0 == key_2)
        assert not np.all(key_0 == key_1)

    def test_scalar_iid_noise(self):
        key = jax.random.PRNGKey(9832)
        distr = distrax.Normal(0, 1)

        src = ScalarIIDAddStochasticTransform(
            distribution=distr,
            key=key
        )
        data = np.zeros((10, 10, 100))
        out = src(data)
        assert not np.allclose(data, out)
        assert np.abs(out.mean()) < 0.05
        assert np.abs(out.std() - 1) < 0.05

        src = ScalarIIDAddStochasticTransform(
            distribution=distr,
            sample_axes=(0,),
            inference=True,
            key=key
        )
        data = np.zeros((3, 4, 5))
        out = src(data)
        assert np.all(out == data) # inference mode
        out = src.inject(data, key=key)
        # test sample axis selection
        assert np.allclose(out.mean((1, 2), keepdims=True), out)

    def test_scalar_iid_dropout(self):
        key = jax.random.PRNGKey(9832)
        distr = distrax.Bernoulli(probs=0.5)

        src = ScalarIIDMulStochasticTransform(
            distribution=distr,
            key=key
        )
        data = np.ones((3, 4, 5))
        out = src(data)
        assert np.logical_or(
            np.isclose(out, 0, atol=1e-6),
            np.isclose(out, 2, atol=1e-6)
        ).all()

    def test_tensor_iid_noise(self):
        key = jax.random.PRNGKey(9832)
        mu = np.random.randn(5)
        sigma = np.random.randn(5, 5)
        sigma = sigma @ sigma.T
        distr = distrax.MultivariateNormalFullCovariance(
            loc=mu,
            covariance_matrix=sigma
        )

        src = TensorIIDAddStochasticTransform(
            distribution=distr,
            event_axes=(-2,),
            key=key
        )
        data = np.zeros((100, 5, 100))
        out = src(data)
        assert np.all(np.abs(
            out.mean((0, -1)) - mu
        ) < 0.05)
        assert np.all(np.abs(
            np.cov(out.swapaxes(-1, -2).reshape((-1, 5)).T) - sigma
        ) < 0.25)

    def test_tensor_iid_dropout(self):
        key = jax.random.PRNGKey(9832)
        alpha = [1] * 5
        distr = distrax.Dirichlet(alpha)

        # No idea why you would ever do this, but here we go
        src = TensorIIDMulStochasticTransform(
            distribution=distr,
            event_axes=(-2,),
            key=key
        )
        data = np.ones((100, 5, 100))
        out = src(data)
        assert np.isclose(out.mean(), 1)

    def test_lowrank_distr(self):
        key = jax.random.PRNGKey(9832)
        inner_distr = distrax.Normal(3, 1)
        distr = OuterProduct(
            src_distribution=inner_distr,
            rank=2,
            multiplicity=10,
        )

        out, lp = distr.sample_and_log_prob(
            seed=key,
            sample_shape=(3, 5)
        )
        assert out.shape == (3, 5, 10, 10)
        assert np.isnan(lp).all()

        std = OuterProduct.rescale_std_for_normal(
            std=3, rank=2, matrix_dim=100
        )
        inner_distr = distrax.Normal(0, std)
        distr = OuterProduct(
            src_distribution=inner_distr,
            rank=2,
            multiplicity=100,
        )
        out = distr.sample(seed=key, sample_shape=(10, 100))
        assert np.abs(out.std(axis=(-2, -1)).mean() - 3) < 0.05

        inner_distr = distrax.Bernoulli(probs=0.3)
        distr = OuterProduct(
            src_distribution=inner_distr,
            multiplicity=100,
        )
        src = TensorIIDMulStochasticTransform(
            distribution=distr,
            event_axes=(0, -1),
            key=key
        )
        data = np.ones((100, 100, 10, 100))
        out = jax.jit(src.__call__)(data)
        assert np.abs(out.mean() - 1) < 0.05

    def test_diagonal_distr(self):
        key = jax.random.PRNGKey(9832)
        inner_distr = distrax.Normal(3, 1)
        distr = Diagonal(
            src_distribution=inner_distr,
            multiplicity=10,
        )

        out, lp = distr.sample_and_log_prob(
            seed=key,
            sample_shape=(3, 5)
        )
        assert out.shape == (3, 5, 10, 10)
        assert lp.shape == (3, 5, 10, 10)
        assert ((out == 0) == (lp == 0)).all()

        inner_distr = distrax.Bernoulli(probs=0.3)
        distr = Diagonal(
            src_distribution=inner_distr,
            multiplicity=100,
        )
        src = TensorIIDMulStochasticTransform(
            distribution=distr,
            event_axes=(0, -1),
            key=key
        )
        data = np.ones((100, 100, 10, 100))
        out = jax.jit(src.__call__)(data)
        assert np.abs(np.diagonal(out, axis1=0, axis2=-1).mean() - 1) < 0.05

    def test_expm_distr(self):
        key = jax.random.PRNGKey(9832)
        inner_inner_distr = distrax.Normal(3, 1)
        inner_distr = Diagonal(
            src_distribution=inner_inner_distr,
            multiplicity=10,
        )
        distr = MatrixExponential(
            src_distribution=inner_distr,
        )
        out, lp = distr.sample_and_log_prob(
            seed=key,
            sample_shape=(3, 5)
        )
        assert out.shape == (3, 5, 10, 10)
        assert lp.shape == (3, 5, 10, 10)

        src = TensorIIDAddStochasticTransform(
            distribution=distr,
            event_axes=(0, -1),
            key=key
        )
        data = np.zeros((10, 100, 100, 10))
        out = jax.jit(src.__call__)(data)
        assert (out >= 0).all()
