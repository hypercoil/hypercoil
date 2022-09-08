# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for cone-tangent projection layers
"""
import jax
import equinox as eqx
from hypercoil.nn import (
    TangentProject,
    BatchTangentProject,
)
from hypercoil.init.semidefinite import (
    SPDEuclideanMean, SPDGeometricMean, SPDHarmonicMean, SPDLogEuclideanMean
)


class TestTanProject:
    def test_tangent_project(self):
        key = jax.random.PRNGKey(0)
        mkey, dkey = jax.random.split(key)

        init_data = jax.random.uniform(dkey, shape=(5, 10, 10))
        init_data = init_data @ init_data.swapaxes(-2, -1)
        init_spec = [
            SPDEuclideanMean(),
            SPDHarmonicMean(),
            SPDLogEuclideanMean(psi=1e-3),
            SPDGeometricMean(psi=1e-3),
        ]

        model = TangentProject.from_specs(
            init_spec, init_data, recondition=1e-5, std=0, key=mkey
        )
        assert model.weight.shape == (4, 10, 10)
        assert (model.weight[0] == init_data.mean(0)).all()

        model = TangentProject.from_specs(
            init_spec, init_data, recondition=1e-5, std=0.6, key=mkey
        )
        out = eqx.filter_jit(model)(init_data, key=key)
        assert out.shape == (5, 4, 10, 10)
        out = eqx.filter_jit(model)(init_data, dest='cone', key=key)
        assert out.shape == (5, 4, 10, 10)

    def test_batch_tangent_project(self):
        key = jax.random.PRNGKey(0)
        mkey, dkey = jax.random.split(key)

        init_data = jax.random.uniform(dkey, shape=(5, 10, 10))
        init_data = init_data @ init_data.swapaxes(-2, -1)
        init_spec = [
            SPDEuclideanMean(),
            SPDHarmonicMean(),
            SPDLogEuclideanMean(psi=1e-3),
            SPDGeometricMean(psi=1e-3),
        ]

        model = BatchTangentProject.from_specs(
            init_spec, init_data, recondition=1e-5, std=0, key=mkey
        )
        assert model.default_weight.shape == (4, 10, 10)
        assert (model.default_weight[0] == init_data.mean(0)).all()
        f = eqx.filter_jit(model)
        out, weight = f(init_data, key=key)
        assert out.shape == (5, 4, 10, 10)
        assert weight.shape == model.default_weight.shape
        out, weight = f(init_data, weight=weight, key=key)
        assert out.shape == (5, 4, 10, 10)
        assert weight.shape == model.default_weight.shape
        out, weight = f(init_data, weight=weight, dest='cone', key=key)
        assert out.shape == (5, 4, 10, 10)
        assert weight.shape == model.default_weight.shape
