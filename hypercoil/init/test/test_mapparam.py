# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for mapped parameters
"""
import pytest
import jax
import numpy as np
import jax.numpy as jnp
import equinox as eqx

from hypercoil.init.mapparam import (
    _to_jax_array, Clip, Renormalise,
    IdentityMappedParameter, AffineMappedParameter,
    AmplitudeTanhMappedParameter, TanhMappedParameter,
    MappedLogits, NormSphereParameter,
    ProbabilitySimplexParameter,
    AmplitudeProbabilitySimplexParameter,
    OrthogonalParameter
)


class TestMappedParameters:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        A = np.array([-1.1, -0.5, 0, 0.5, 1, 7])[None, ...]
        self.A = self._linear_with_weight(A)
        C = (
            np.array([-1.1, -0.5, 0, 0.5, 1, 7]) +
            np.array([-0.7, -0.2, 1, 1, 0, -5]) * 1j
        )[None, ...]
        self.C = self._linear_with_weight(C)
        AA = np.array([
            [2., 2., 2., 1., 0.],
            [0., 1., 1., 1., 2.]
        ])
        self.AA = self._linear_with_weight(AA)
        ampl_CC = np.array([
            [2, 2, 2, 1, 0, 0],
            [0, 1, 1, 1, 2, 0]
        ])
        phase_CC = np.array([
            [-0.7, -0.2, 1, 1, 0, -5],
            [-1.1, -0.5, 0, 0.5, 1, 7]
        ])
        CC = ampl_CC * jnp.exp(phase_CC * 1j)
        self.CC = self._linear_with_weight(CC)
        Z = np.random.rand(5, 3, 4, 4)

    def _linear_with_weight(self, W):
        key = jax.random.PRNGKey(0) # not relevant for us
        model = eqx.nn.Linear(
            in_features=W.shape[-1],
            out_features=W.shape[-2],
            key=key
        )
        return eqx.tree_at(
            lambda m: m.weight,
            model,
            replace=W
        )

    def test_clip(self):
        A = np.array([-0.7, 0.3, 1.2])
        out = Clip().apply(A, bound=(-float('inf'), 1))
        ref = np.array([-0.7, 0.3, 1])
        assert np.allclose(out, ref)

    def test_norm(self):
        A = np.array([-0.5, 0, 0.5])
        out = Renormalise().apply(A, bound=(0, 1))
        ref = np.array([0, 0.25, 0.5])
        assert np.allclose(out, ref)

    def test_identity(self):
        mapper = IdentityMappedParameter(self.A)
        out = mapper.preimage_map(self.A.weight)
        ref = self.A.weight
        assert np.allclose(out, ref)
        out = mapper.image_map(self.A.weight)
        ref = self.A.weight
        assert np.allclose(out, ref)

    def test_linear(self):
        mapper = AffineMappedParameter(self.A, loc=-3, scale=2)
        out = mapper.preimage_map(self.A.weight)
        ref = (self.A.weight + 3) / 2
        assert np.allclose(out, ref)
        out = mapper.image_map(self.A.weight)
        ref = self.A.weight * 2 - 3
        assert np.allclose(out, ref)

    def test_atanh(self):
        mapper = TanhMappedParameter(self.A, scale=2)
        out = mapper.preimage_map(self.A.weight)
        ref = jnp.arctanh(self.A.weight / 2)
        ref = ref.at[self.A.weight < mapper.image_bound[0]].set(
            mapper.preimage_bound[0])
        ref = ref.at[ref < mapper.preimage_bound[0]].set(
            mapper.preimage_bound[0])
        ref = ref.at[self.A.weight > mapper.image_bound[1]].set(
            mapper.preimage_bound[1])
        ref = ref.at[ref > mapper.preimage_bound[1]].set(
            mapper.preimage_bound[1])
        assert np.allclose(out, ref)
        out = mapper.image_map(self.A.weight)
        ref = jnp.tanh(self.A.weight) * 2
        assert np.allclose(out, ref)

    def test_aatanh(self):
        mapper = AmplitudeTanhMappedParameter(self.C, scale=2)
        out = mapper.preimage_map(self.C.weight)
        ampl, phase = jnp.abs(self.C.weight), jnp.angle(self.C.weight)
        ref = jnp.arctanh(ampl / 2)
        ref = ref.at[ampl < mapper.image_bound[0]].set(
            mapper.preimage_bound[0])
        ref = ref.at[ref < mapper.preimage_bound[0]].set(
            mapper.preimage_bound[0])
        ref = ref.at[ampl > mapper.image_bound[1]].set(
            mapper.preimage_bound[1])
        ref = ref.at[ref > mapper.preimage_bound[1]].set(
            mapper.preimage_bound[1])
        ref = ref * jnp.exp(phase * 1j)
        assert np.allclose(out, ref)
        out = mapper.image_map(self.C.weight)
        ampl, phase = jnp.abs(self.C.weight), jnp.angle(self.C.weight)
        ref = jnp.tanh(ampl) * 2
        ref = ref * jnp.exp(phase * 1j)
        assert np.allclose(out, ref)

    def test_logit(self):
        mapper = MappedLogits(self.A, scale=2)
        out = mapper.preimage_map(self.A.weight)
        ref = jax.scipy.special.logit(self.A.weight / 2)
        ref = ref.at[self.A.weight < mapper.image_bound[0]].set(
            mapper.preimage_bound[0])
        ref = ref.at[ref < mapper.preimage_bound[0]].set(
            mapper.preimage_bound[0])
        ref = ref.at[self.A.weight > mapper.image_bound[1]].set(
            mapper.preimage_bound[1])
        ref = ref.at[ref > mapper.preimage_bound[1]].set(
            mapper.preimage_bound[1])
        assert np.allclose(out, ref)
        out = mapper.image_map(self.A.weight)
        ref = jax.nn.sigmoid(self.A.weight) * 2
        assert np.allclose(out, ref)

    def test_unitsphere(self):
        X = jax.random.normal(key=jax.random.PRNGKey(8439), shape=(4, 8))
        X = self._linear_with_weight(X)
        d = NormSphereParameter(X)
        assert np.allclose((d.image_map(X.weight) ** 2).sum(-1), 1)
        d = NormSphereParameter(X, axis=-2)
        assert np.allclose((d.image_map(X.weight) ** 2).sum(0), 1)
        d = NormSphereParameter(X, norm=1, axis=-2)
        assert np.allclose(jnp.abs(d.image_map(X.weight)).sum(0), 1)
        d = NormSphereParameter(X, norm=jnp.eye(8))
        assert np.allclose((d.image_map(X.weight) ** 2).sum(-1), 1)
        d = NormSphereParameter(X, norm=jnp.eye(4), axis=-2)
        assert np.allclose((d.image_map(X.weight) ** 2).sum(0), 1)

    def test_multilogit(self):
        mapper = ProbabilitySimplexParameter(self.AA, axis=-1)
        out = mapper.preimage_map(self.AA.weight)
        r_in = jnp.array(self.AA.weight)
        r_in = r_in.at[r_in < mapper.image_bound[0]].set(mapper.image_bound[0])
        r_in = r_in.at[r_in > mapper.image_bound[1]].set(mapper.image_bound[1])
        ref = jnp.log(r_in)
        assert np.allclose(out, ref)
        out = mapper.image_map(out)
        ref = self.AA.weight / self.AA.weight.sum(-1, keepdims=True)
        assert np.allclose(out, ref, atol=1e-2)

    def test_amultilogit(self):
        mapper = AmplitudeProbabilitySimplexParameter(self.CC, axis=0)
        out = mapper.preimage_map(self.CC.weight)
        ampl, phase = jnp.abs(self.CC.weight), jnp.angle(self.CC.weight)
        ampl = ampl.at[ampl < mapper.image_bound[0]].set(mapper.image_bound[0])
        ampl = ampl.at[ampl > mapper.image_bound[1]].set(mapper.image_bound[1])
        ref = jnp.log(ampl)
        ref = ref * jnp.exp(phase * 1j)
        assert np.allclose(out, ref)
        out = mapper.image_map(self.CC.weight)
        ampl, phase = jnp.abs(self.CC.weight), jnp.angle(self.CC.weight)
        ampl = jax.nn.softmax(ampl, 0)
        ref = ampl * jnp.exp(phase * 1j)
        assert np.allclose(out, ref)

    def test_orthogonal(self):
        X = jax.random.normal(key=jax.random.PRNGKey(8439), shape=(8, 8))
        X = self._linear_with_weight(X)
        X = OrthogonalParameter.embed(X)
        out = _to_jax_array(X.weight)
        out = out.T @ out
        assert np.allclose(out, jnp.eye(8), atol=1e-5)

    def test_softmax_mapper_full_loop(self):
        @eqx.filter_jit
        @eqx.filter_grad
        def loss_fn(model, x, y):
            pred_y = jax.vmap(model)(x)
            return jax.numpy.mean((y - pred_y) ** 2)

        @eqx.filter_grad
        def loss_ref(model, x, y):
            predict = lambda x: jax.nn.softmax(
                model.weight.original, -1) @ x + model.bias
            pred_y = jax.vmap(predict)(x)
            return jax.numpy.mean((y - pred_y) ** 2)

        key = jax.random.PRNGKey(0)
        mkey, xkey, ykey = jax.random.split(key, 3)
        batch_size, in_size, out_size = 32, 20, 10
        model = eqx.nn.Linear(
            in_features=in_size, out_features=out_size, key=mkey)

        mapper = ProbabilitySimplexParameter(model, axis=-1)
        model_mapped = eqx.tree_at(
            lambda m: m.__getattribute__(mapper.param_name),
            model,
            replace=mapper
        )
        model_mapped_2 = ProbabilitySimplexParameter.embed(model, axis=-1)
        assert np.allclose(model_mapped.weight.original,
                           model_mapped_2.weight.original)

        assert np.allclose(jnp.sum(_to_jax_array(model_mapped.weight), -1), 1)

        x = jax.random.normal(key=xkey, shape=(batch_size, in_size))
        y = jax.random.normal(key=ykey, shape=(batch_size, out_size))

        for epoch in range(10):
            grads = loss_fn(model_mapped, x, y)
            grads_ref = loss_ref(model_mapped, x, y)
            assert np.allclose(
                grads.weight.original, grads_ref.weight.original)
            param_old = model_mapped.weight.original

            model_mapped = eqx.apply_updates(model_mapped, grads)
            assert not np.allclose(model_mapped.weight.original, param_old)

