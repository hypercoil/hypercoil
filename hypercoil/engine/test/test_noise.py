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
)


#TODO: There are many missing tests for noise and dropout sources.
# We should have at minimum an injection test for each source
# on CPU and CUDA that also verifies each source works given an input
# of a reasonable but nontrivial shape.


# def lr_std_mean(dim=100, rank=None, var=0.05, iter=1000):
#     lrns = LowRankNoiseSource(rank=rank, var=var)
#     return torch.Tensor(
#         [lrns.sample([dim]).std() for _ in range(iter)
#     ]).mean()


class TestNoise:

    # @pytest.fixture(autouse=True)
    # def setup_class(self):
    #     self.atol = 1e-3
    #     self.rtol = 1e-4
    #     self.approx = lambda out, ref: np.isclose(
    #         out, ref, atol=self.atol, rtol=self.rtol)

    # def test_lr_std(self):
    #     out = lr_std_mean()
    #     ref = 0.05
    #     assert self.approx(out, ref)
    #     out = lr_std_mean(var=0.2)
    #     ref = 0.2
    #     assert self.approx(out, ref)
    #     out = lr_std_mean(var=0.03, rank=7)
    #     ref = 0.03
    #     assert self.approx(out, ref)

    # def test_spsd_spsd(self):
    #     spsdns = SPSDNoiseSource()
    #     out = spsdns.sample([100])
    #     assert torch.allclose(out, out.T, atol=1e-5)
    #     L = torch.linalg.eigvalsh(out)
    #     # ignore effectively-zero eigenvalues
    #     L[torch.abs(L) < 1e-4] = 0
    #     assert L.min() >= 0
    #     assert torch.all(L >= 0)

    # def test_band_correction(self):
    #     bds = BandDropoutSource()
    #     out = bds.sample([100]).sum()
    #     ref = bds.bandmask.sum()
    #     assert torch.abs((out - ref) / ref) <= 0.2

    # def test_scalar_iid_noise(self):
    #     sz = torch.Size([3, 8, 1, 21, 1])
    #     inp = torch.rand(sz)
    #     sins = UnstructuredNoiseSource()
    #     out = sins(inp)
    #     assert out.size() == sz

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
        ) < 0.2)

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
