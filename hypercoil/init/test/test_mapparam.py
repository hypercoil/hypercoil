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

from hypercoil.init.mapparam import AffineMappedParameter, IdentityMappedParameter, MappedParameter, Clip, Renormalise


class TestMappedParameters:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        key = jax.random.PRNGKey(0)
        k0, k1 = jax.random.split(key)
        A = np.array([-1.1, -0.5, 0, 0.5, 1, 7])
        self.A = eqx.nn.Linear(
            in_features=A.shape[-1],
            out_features=1,
            key=k0
        )
        self.A = eqx.tree_at(
            lambda m: m.weight,
            self.A,
            replace=A
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
    
    def test_softmax_mapper(self):
        @eqx.filter_jit
        @eqx.filter_grad
        def loss_fn(model, x, y):
            pred_y = jax.vmap(model)(x)
            return jax.numpy.mean((y - pred_y) ** 2)

        @eqx.filter_grad
        def loss_ref(model, x, y):
            predict = lambda x: jax.nn.softmax(model.weight.original, -1) @ x + model.bias
            pred_y = jax.vmap(predict)(x)
            return jax.numpy.mean((y - pred_y) ** 2)

        key = jax.random.PRNGKey(0)
        mkey, xkey, ykey = jax.random.split(key, 3)
        batch_size, in_size, out_size = 32, 20, 10
        model = eqx.nn.Linear(in_features=in_size, out_features=out_size, key=mkey)
        model = eqx.tree_at(lambda m: m.weight, model, replace_fn=jnp.abs)

        mapper = MappedParameter(model)
        model_mapped = eqx.tree_at(
            lambda m: m.__getattribute__(mapper.param_name),
            model,
            replace=mapper
        )
        model_mapped_2 = MappedParameter.embed(model)
        assert np.allclose(model_mapped.weight.original, model_mapped_2.weight.original)

        x = jax.random.normal(key=xkey, shape=(batch_size, in_size))
        y = jax.random.normal(key=ykey, shape=(batch_size, out_size))

        for epoch in range(10):
            print(epoch)
            grads = loss_fn(model_mapped, x, y)
            grads_ref = loss_ref(model_mapped, x, y)
            assert np.allclose(grads.weight.original, grads_ref.weight.original)
            param_old = model_mapped.weight.original

            model_mapped = eqx.apply_updates(model_mapped, grads)
            assert not np.allclose(model_mapped.weight.original, param_old)

