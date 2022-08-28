# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for noise sources
"""
import jax
import jax.numpy as jnp
import distrax
import equinox as eqx
import numpy as np
from hypercoil.engine.noise import (
    StochasticSource, refresh, StochasticParameter,
    ScalarIIDAddStochasticTransform,
    ScalarIIDMulStochasticTransform,
    TensorIIDAddStochasticTransform,
    TensorIIDMulStochasticTransform,
    EigenspaceReconditionTransform,
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
        key = jax.random.PRNGKey(392)
        mkey, skey, tkey = jax.random.split(key, 3)
        mu = jax.random.normal(key=mkey, shape=(5,))
        sigma = jax.random.normal(key=skey, shape=(5, 5))
        sigma = sigma @ sigma.T
        distr = distrax.MultivariateNormalFullCovariance(
            loc=mu,
            covariance_matrix=sigma
        )

        src = TensorIIDAddStochasticTransform(
            distribution=distr,
            event_axes=(-2,),
            key=tkey
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

    def test_eigenspace_recondition(self):
        key = jax.random.PRNGKey(9832)
        psi = 0.01
        src = EigenspaceReconditionTransform(
            psi=psi,
            matrix_dim=10,
            key=key,
        )
        data = np.random.randn(100, 10, 2)
        # Inputs clearly singular and degenerate
        data = data @ data.swapaxes(-1, -2)
        out = src(data)
        L_in = jnp.linalg.eigvalsh(data)
        L = jnp.linalg.eigvalsh(out)

        # Test nonsingularity
        assert (L > 0).all()
        assert not (L_in > 0).all()

        # And nondegeneracy
        diffs = jnp.abs(L[:, None, :] - L[..., None])
        diffs = (diffs + np.eye(10)).min()
        diffs_in = jnp.abs(L_in[:, None, :] - L_in[..., None])
        diffs_in = (diffs_in + np.eye(10)).min()
        assert diffs > (1e3 * diffs_in)

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

    def test_stochastic_parameter_loop(self):

        @eqx.filter_jit
        def fwd(model, x):
            return jax.vmap(model)(x)

        @eqx.filter_jit
        @eqx.filter_grad
        def loss_fn(model, x, y):
            pred_y = jax.vmap(model)(x)
            return jax.numpy.mean((y - pred_y) ** 2)

        key = jax.random.PRNGKey(0)
        mkey, skey, xkey, ykey = jax.random.split(key, 4)

        batch_size, in_size, out_size = 32, 5, 3
        model = eqx.nn.Linear(
            in_features=in_size, out_features=out_size, key=mkey)
        model = eqx.tree_at(
            lambda m: m.weight,
            model,
            replace=jnp.zeros_like(model.weight)
        )

        transform = ScalarIIDAddStochasticTransform(
            distribution=distrax.Normal(0, 1),
            key=skey,
        )
        model = StochasticParameter.wrap(
            model, transform=transform
        )

        x = jax.random.normal(key=xkey, shape=(batch_size, in_size))
        y = jax.random.normal(key=ykey, shape=(batch_size, out_size))

        out_prev = None
        for epoch in range(10):
            out = fwd(model, x) # jax.vmap(model)(x)
            if out_prev is not None:
                assert not np.allclose(out, out_prev)
            out_prev = out
            model = refresh(model)

        model_update = model
        for epoch in range(10):
            grads = loss_fn(model_update, x, y)
            prev_weight = model_update.weight.__jax_array__()
            model_update = eqx.apply_updates(model_update, grads)
            model_update = refresh(model_update)
            assert not np.allclose(model_update.weight.__jax_array__(), prev_weight)
            # We check to ensure the distribution is unchanged because we have
            # to treat it as a static field, and updating it would require
            # recompiling.
            assert (
                model_update.weight.transform.distribution is
                model.weight.transform.distribution
            )

    #TODO: revisit or delete as deepmind/distrax#193 is resolved
    # def test_minimal_fail(self):
    #     import jax, distrax

    #     minimal = distrax.Normal(0, 1)
    #     null = jax.tree_util.tree_map(lambda _: None, minimal)

    #     print(null)

    #     from jax._src.lib import pytree
    #     print(pytree.flatten(minimal)[-1])
    #     print(pytree.flatten(null)[-1])

    #     jax.tree_util.tree_map(
    #         lambda l, r: l,
    #         minimal,
    #         null
    #     )


    #     from jax._src.lib import pytree
    #     print(pytree.flatten(minimal)[-1])
    #     print(pytree.flatten(null)[-1])
    #     treedef = pytree.flatten(minimal)[-1]
    #     treedef.flatten_up_to(null)

        # minimal = (distrax.Normal(0, 1),)
        # mask_tree = jax.tree_util.tree_map(lambda _: True, minimal)

        # print(mask_tree[0].tree_flatten())
        # print(minimal[0].tree_flatten())

        # print(jax.tree_flatten(minimal[0].__dict__))
        # print(jax.tree_util.tree_flatten(minimal[0].__dict__))

        # jax.tree_util.tree_map(
        #     lambda mask, x: x if bool(mask) != False else None,
        #     mask_tree,
        #     minimal)
        # import jax; import distrax; from distrax._src.utils.jittable import _is_jax_data
        # distr = distrax.Normal(0, 1)
        # leaves, treedef = jax.tree_util.tree_flatten(distr.__dict__)
        # switch = list(map(_is_jax_data, leaves))
        # children = [leaf if s else None for leaf, s in zip(leaves, switch)]
        # metadata = [None if s else leaf for leaf, s in zip(leaves, switch)]
        # flat = children, (metadata, switch, treedef)



        # minimal = (distrax.Normal(0, 1),)

        # @eqx.filter_jit(args=(lambda x: isinstance(x, distrax.Distribution),))
        # def noop(x):
        #     return x

        # noop(minimal)

        # @jax.tree_util.register_pytree_node_class
        # class JaxDataTree(distrax._src.utils.jittable.Jittable):
        #     def __init__(self, param0, param1):
        #         self.param0 = param0
        #         self.param1 = param1

        #     def tree_flatten(self):
        #         leaves, treedef = jax.tree_util.tree_flatten(self.__dict__)
        #         switch = list(map(_is_jax_data, leaves))
        #         children = [leaf if s else None for leaf, s in zip(leaves, switch)]
        #         metadata = [None if s else leaf for leaf, s in zip(leaves, switch)]
        #         print('flatten: ', children, metadata)
        #         return children, (metadata, switch, treedef)

        #     @classmethod
        #     def tree_unflatten(cls, aux_data, children):
        #         metadata, switch, treedef = aux_data
        #         leaves = [j if s else p for j, p, s in zip(children, metadata, switch)]
        #         print('unflatten: ', children, metadata)
        #         obj = object.__new__(cls)
        #         obj.__dict__ = jax.tree_util.tree_unflatten(treedef, leaves)
        #         return obj

        # tr = JaxDataTree(None, 11)
        # trmask = jax.tree_util.tree_map(lambda _: True, tr)

        # tr = JaxDataTree(jnp.zeros((3, 2)), 11)
        # trmask = jax.tree_util.tree_map(lambda _: True, tr)

        # assert 0



        # class PatchedNormal(distrax.Normal):
        #     def tree_flatten(self):
        #         leaves, treedef = jax.tree_util.tree_flatten(self.__dict__)
        #         print(leaves)
        #         print(treedef)
        #         return leaves, (treedef,)

        #     @classmethod
        #     def tree_unflatten(cls, aux_data, children):
        #         treedef, = aux_data
        #         obj = object.__new__(cls)
        #         obj.__dict__ = jax.tree_util.tree_unflatten(treedef, children)
        #         return obj

        # patched = (PatchedNormal(0, 1),)

        # jax.tree_util.tree_flatten(eqx.partition(patched, eqx.is_array)[-1])

        # @eqx.filter_jit
        # def noop(x):
        #     return x

        # noop(patched)
