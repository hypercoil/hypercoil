# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for loss functions.
"""
import jax
import jax.numpy as jnp
from functools import partial
from scipy.stats import entropy as entropy_ref
from scipy.spatial.distance import jensenshannon as js_ref
from hypercoil.functional import linear_kernel
from hypercoil.loss.functional import (
    norm_scalarise, selfwmean, vnorm_scalarise, sum_scalarise, mean_scalarise,
    meansq_scalarise, wmean_scalarise, selfwmean_scalarise, wmean,
    identity, constraint_violation, unilateral_loss, hinge_loss,
    det_gram, log_det_gram,
    entropy, entropy_logit, equilibrium, equilibrium_logit,
    kl_divergence, kl_divergence_logit, js_divergence, js_divergence_logit,
)


class TestLossFunction:
    def test_wmean(self):
        z = jnp.array([[
            [1., 4., 2.],
            [0., 9., 1.],
            [4., 6., 7.]],[
            [0., 9., 1.],
            [4., 6., 7.],
            [1., 4., 2.]
        ]])
        w = jnp.ones_like(z)
        assert jnp.allclose(wmean(z, w), jnp.mean(z))
        w = jnp.array([1., 0., 1.])
        assert jnp.all(wmean(z, w, axis=1) == jnp.array([
            [(1 + 4) / 2, (4 + 6) / 2, (2 + 7) / 2],
            [(0 + 1) / 2, (9 + 4) / 2, (1 + 2) / 2]
        ]))
        assert jnp.all(wmean(z, w, axis=2) == jnp.array([
            [(1 + 2) / 2, (0 + 1) / 2, (4 + 7) / 2],
            [(0 + 1) / 2, (4 + 7) / 2, (1 + 2) / 2]
        ]))
        w = jnp.array([
            [1., 0., 1.],
            [0., 1., 1.]
        ])
        assert jnp.all(wmean(z, w, axis=(0, 1)) == jnp.array([
            [(1 + 4 + 4 + 1) / 4, (4 + 6 + 6 + 4) / 4, (2 + 7 + 7 + 2) / 4]
        ]))
        assert jnp.all(wmean(z, w, axis=(0, 2)) == jnp.array([
            [(1 + 2 + 9 + 1) / 4, (0 + 1 + 6 + 7) / 4, (4 + 7 + 4 + 2) / 4]
        ]))

        loss = jax.jit(wmean_scalarise(identity, axis=(0, 1)))
        out = loss(z, scalarisation_weight=w)
        assert jnp.all(out == jnp.array([
            [(1 + 4 + 4 + 1) / 4, (4 + 6 + 6 + 4) / 4, (2 + 7 + 7 + 2) / 4]
        ]).mean())

    def test_selfwmean(self):
        key = jax.random.PRNGKey(0)
        X = jnp.array([
            [-100, -100, 0, -100, -100],
            [0, -100, -100, -100, -100],
            [-100, -100, -100, -100., 0]
        ])
        Y = jax.random.normal(key=key, shape=(3, 5))

        assert jnp.isclose(selfwmean(X, softmax_axis=-1), 0)
        assert not jnp.isclose(selfwmean(Y, softmax_axis=-1), 0)

        loss = jax.jit(selfwmean_scalarise(identity, axis=None, softmax_axis=-1))
        assert jnp.isclose(loss(X), 0)

    def test_normed_losses(self):
        X = jnp.array([
            [-1, 2, 0, 2, 1],
            [0, 1, 0, 0, -1],
            [3, -1, 0, -2, 0]
        ])

        L0 = jax.jit(vnorm_scalarise(p=0, axis=None))
        assert L0(X) == 9
        L0 = norm_scalarise(p=0, axis=0)(X)
        assert jnp.isclose(L0, (X != 0).sum() / 5)
        L0 = norm_scalarise(p=0, axis=-1)(X)
        assert jnp.isclose(L0, (X != 0).sum() / 3)

        L1 = jax.jit(vnorm_scalarise(p=1, axis=None))
        assert L1(X) == 14
        L1 = norm_scalarise(p=1, axis=0)(X)
        assert jnp.isclose(L1, jnp.abs(X).sum() / 5)
        L1 = norm_scalarise(p=1, axis=-1)(X)
        assert jnp.isclose(L1, jnp.abs(X).sum() / 3)

        L2 = jax.jit(vnorm_scalarise(p=2, axis=None))
        assert L2(X) == jnp.sqrt((X ** 2).sum())
        L2 = norm_scalarise(p=2, axis=0)(X)
        assert jnp.isclose(L2, jnp.sqrt((X ** 2).sum(0)).mean())
        L2 = norm_scalarise(p=2, axis=-1)(X)
        assert jnp.isclose(L2, jnp.sqrt((X ** 2).sum(-1)).mean())

    def test_unilateral_loss(self):
        X = jnp.array([
            [-1, 2, 0, 2, 1],
            [0, 1, 0, 0, -1],
            [3, -1, 0, -2, 0]
        ])

        uL0 = jax.jit(vnorm_scalarise(unilateral_loss, p=0, axis=None))
        assert uL0(X) == 5
        assert uL0(-X) == 4
        uL1 = jax.jit(vnorm_scalarise(unilateral_loss, p=1, axis=None))
        assert uL1(X) == 9
        assert uL1(-X) == 5
        uL2 = jax.jit(vnorm_scalarise(unilateral_loss, p=2, axis=None))
        assert uL2(X) == jnp.sqrt(19)
        assert uL2(-X) == jnp.sqrt(7)

    def test_constraint_violation_loss(self):
        X = jnp.array([
            [1., -1.],
            [-2., 1.],
            [-1., 2.],
            [0., 0.]
        ])
        cjit = partial(jax.jit, static_argnames=('constraints',))

        constraints = (identity,)
        f = cjit(vnorm_scalarise(constraint_violation, p=1, axis=None))
        g = cjit(vnorm_scalarise(constraint_violation, p=0, axis=None))
        h = jax.jit(vnorm_scalarise(unilateral_loss, p=1, axis=None))
        j = vnorm_scalarise(constraint_violation, p=1, axis=None)
        assert f(X, constraints=constraints) == 4
        assert f(X, constraints=constraints) == h(X)

        constraints = (
            lambda x: x @ jnp.array([[1.], [1.]]),
        )
        assert f(X, constraints=constraints) == 1
        assert j(X, constraints=constraints,
                 broadcast_against_input=True) == 2

        constraints = (
            lambda x: x @ jnp.array([[0.], [1.]]),
            lambda x: x @ jnp.array([[1.], [0.]]),
        )
        assert g(X, constraints=constraints) == 3

        constraints = (
            lambda x: jnp.array([1., 1., 1., 1.]) @ x,
        )
        assert f(X, constraints=constraints) == 2

        constraints = (
            lambda x: x @ jnp.array([[-1.], [1.]]),
        )
        assert f(X, constraints=constraints) == 6

    def test_hinge_loss(self):
        Y = jnp.array([-1., 1., 1, -1., 1.])
        Y_hat_0 = jnp.array([0, 0, 0, 0, 0])
        Y_hat_1 = jnp.array([1, 1, 1, 1, 1])
        Y_hat_minus1 = jnp.array([-1, -1, -1, -1, -1])

        hinge = jax.jit(sum_scalarise(hinge_loss))
        assert hinge(Y, Y_hat_0) == 5
        assert hinge(Y, Y_hat_1) == 4
        assert hinge(Y, Y_hat_minus1) == 6

    def test_det_loss(self):
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (2, 10, 5))
        dgjit = partial(jax.jit, static_argnames=('op', 'psi', 'xi',))

        det_loss = dgjit(sum_scalarise(det_gram))
        logdet_loss = dgjit(sum_scalarise(log_det_gram))

        assert jnp.isclose(det_loss(X, op=linear_kernel), 0)
        # Really, this is infinite. But due to numerical issues, it's not.
        assert logdet_loss(X, op=linear_kernel) > 100
        assert logdet_loss(X, op=linear_kernel, psi=1) < 1
        assert logdet_loss(X, op=linear_kernel, psi=1, xi=1, key=key) < 1
        out0 = det_loss(X, op=linear_kernel, psi=1)
        out1 = det_loss(X, op=linear_kernel, psi=1, xi=1,
                        key=jax.random.PRNGKey(1))
        assert out1 < -1
        assert out0 < out1

        X = jnp.eye(10)
        assert jnp.isclose(det_loss(X, op=linear_kernel), -1)
        assert jnp.isclose(logdet_loss(X, op=linear_kernel), 0)

    def test_entropy_loss(self):
        key = jax.random.PRNGKey(0)
        distr_logits = jax.random.normal(key, (5, 10))
        distr = jax.random.uniform(key, (5, 10))
        distr = distr / distr.sum(-1, keepdims=True)

        entropy_loss = jax.jit(mean_scalarise(entropy))
        entropy_logit_loss = jax.jit(mean_scalarise(entropy_logit))

        out = entropy_loss(distr)
        ref = entropy_ref(distr, axis=1).mean()
        assert jnp.isclose(out, ref)

        out = entropy_logit_loss(distr_logits)
        ref = entropy_ref(jax.nn.softmax(distr_logits, axis=-1), axis=1).mean()
        assert jnp.isclose(out, ref)

    def test_kl_loss(self):
        key = jax.random.PRNGKey(0)
        keyP, keyQ = jax.random.split(key)

        P_logits = jax.random.normal(keyP, (5, 10))
        P = jax.random.uniform(keyP, (5, 10))
        P = P / P.sum(-1, keepdims=True)

        Q_logits = jax.random.normal(keyQ, (5, 10))
        Q = jax.random.uniform(keyQ, (5, 10))
        Q = Q / Q.sum(-1, keepdims=True)

        kl_loss = jax.jit(mean_scalarise(kl_divergence))
        kl_logit_loss = jax.jit(mean_scalarise(kl_divergence_logit))

        out = kl_loss(P, Q)
        ref = entropy_ref(P, Q, axis=1).mean()
        assert jnp.isclose(out, ref, atol=1e-4)

        out = kl_logit_loss(P_logits, Q_logits)
        ref = entropy_ref(jax.nn.softmax(P_logits, axis=-1),
                          jax.nn.softmax(Q_logits, axis=-1),
                          axis=1).mean()
        assert jnp.isclose(out, ref, atol=1e-4)

    def test_js_loss(self):
        key = jax.random.PRNGKey(0)
        keyP, keyQ = jax.random.split(key)

        P_logits = jax.random.normal(keyP, (5, 10))
        P = jax.random.uniform(keyP, (5, 10))
        P = P / P.sum(-1, keepdims=True)

        Q_logits = jax.random.normal(keyQ, (5, 10))
        Q = jax.random.uniform(keyQ, (5, 10))
        Q = Q / Q.sum(-1, keepdims=True)

        js_loss = jax.jit(mean_scalarise(js_divergence))
        js_logit_loss = jax.jit(mean_scalarise(js_divergence_logit))

        out = js_loss(P, Q)
        ref = (js_ref(P, Q, axis=1) ** 2).mean()
        assert jnp.isclose(out, ref)

        out = js_logit_loss(P_logits, Q_logits)
        ref = (js_ref(jax.nn.softmax(P_logits, axis=-1),
                      jax.nn.softmax(Q_logits, axis=-1),
                      axis=-1) ** 2).mean()
        assert jnp.isclose(out, ref)

    def test_equilibrium_loss(self):
        key = jax.random.PRNGKey(0)
        base = jnp.ones((5, 10))
        noise = jax.random.normal(key, (5, 10))

        equilibrium_loss = jax.jit(meansq_scalarise(equilibrium))

        out0 = equilibrium_loss(base)
        out1 = equilibrium_loss(base + 1e-1 * noise)
        out2 = equilibrium_loss(base + 1e0 * noise)
        out3 = equilibrium_loss(base + 1e1 * noise)
        assert out0 < out1 < out2 < out3
        assert jnp.isclose(out0, 0)
