# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for loss modules.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from hypercoil.functional.kernel import gaussian_kernel

from hypercoil.loss.functional import (
    identity, difference, constraint_violation, unilateral_loss, hinge_loss,
    smoothness, bimodal_symmetric, det_gram, log_det_gram,
    entropy, entropy_logit, kl_divergence, kl_divergence_logit,
    js_divergence, js_divergence_logit,
    bregman_divergence, bregman_divergence_logit,
    equilibrium, equilibrium_logit, second_moment, second_moment_centred,
)
from hypercoil.loss.scalarise import (
    mean_scalarise, sum_scalarise, meansq_scalarise, vnorm_scalarise,
)
from hypercoil.loss.nn import (
    MSELoss, NormedLoss, ConstraintViolationLoss, UnilateralLoss, HingeLoss,
    SmoothnessLoss, BimodalSymmetricLoss, GramDeterminantLoss,
    GramLogDeterminantLoss, EntropyLoss, EntropyLogitLoss,
    KLDivergenceLoss, KLDivergenceLogitLoss, JSDivergenceLoss,
    JSDivergenceLogitLoss, BregmanDivergenceLoss, BregmanDivergenceLogitLoss,
    EquilibriumLoss, EquilibriumLogitLoss, SecondMomentLoss,
    SecondMomentCentredLoss,
)


class TestLossModule:

    #TODO: Currently we're just regression testing the loss modules against
    #      the loss functionals. Correctness tests should be placed in the
    #      functional tests, and these tests should be used to check that the
    #      modules are correctly wrapping the functionals.
    def test_loss_modules(self):

        key = jax.random.PRNGKey(578423)

        key_x = jax.random.split(key, 1)[0]
        X = jax.random.normal(key_x, (10, 10))
        key_y = jax.random.split(key_x, 1)[0]
        Y = jax.random.normal(key_y, (10, 10))

        out = eqx.filter_jit(MSELoss())(X, Y)
        ref = meansq_scalarise(difference)(X, Y)
        assert out == ref

        out = eqx.filter_jit(NormedLoss(p=2, axis=-1))(X)
        ref = vnorm_scalarise(identity, p=2, axis=-1)(X)
        assert out == ref

        out = eqx.filter_jit(ConstraintViolationLoss(
            constraints=(lambda x: jnp.eye(10) @ x,),
            broadcast_against_input=True))(X)
        ref = mean_scalarise(constraint_violation)(
            X, constraints=(lambda x: jnp.eye(10) @ x,),
            broadcast_against_input=True)
        assert out == ref

        out = eqx.filter_jit(UnilateralLoss())(X)
        ref = mean_scalarise(unilateral_loss)(X)
        assert out == ref

        out = eqx.filter_jit(HingeLoss())(X, Y)
        ref = sum_scalarise(hinge_loss)(X, Y)
        assert out == ref

        out = eqx.filter_jit(
            SmoothnessLoss(n=2, pad_value=-1, axis=-2))(X)
        ref = vnorm_scalarise(
            smoothness, p=1, axis=-1)(X, n=2, pad_value=-1, axis=-2)
        # Not sure why this is inexact.
        assert jnp.isclose(out, ref, atol=1e-4)

        out = eqx.filter_jit(BimodalSymmetricLoss(modes=(-1, 1)))(X)
        ref = mean_scalarise(bimodal_symmetric)(X, modes=(-1, 1))
        assert out == ref

        S = jax.random.normal(key, (10, 100))

        out = eqx.filter_jit(GramLogDeterminantLoss(
            op=gaussian_kernel, psi=0.001, xi=0.001)
        )(S, key=jax.random.PRNGKey(47))
        ref = mean_scalarise(log_det_gram)(
            S, op=gaussian_kernel, psi=0.001, xi=0.001,
            key=jax.random.PRNGKey(47)
        )
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(GramDeterminantLoss())(S)
        ref = mean_scalarise(det_gram)(S)
        assert jnp.isclose(out, ref)

        S, T = jnp.abs(X), jnp.abs(Y)
        S = S / S.sum(axis=-2, keepdims=True)
        T = T / T.sum(axis=-2, keepdims=True)
        out = eqx.filter_jit(EntropyLoss(axis=-2))(S)
        ref = mean_scalarise(entropy)(S, axis=-2)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(EntropyLogitLoss(axis=-2))(X)
        ref = mean_scalarise(entropy_logit)(X, axis=-2)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(KLDivergenceLoss(axis=-2))(S, T)
        ref = mean_scalarise(kl_divergence)(S, T, axis=-2)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(KLDivergenceLogitLoss(axis=-2))(X, Y)
        ref = mean_scalarise(kl_divergence_logit)(X, Y, axis=-2)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(JSDivergenceLoss(axis=-2))(S, T)
        ref = mean_scalarise(js_divergence)(S, T, axis=-2)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(JSDivergenceLogitLoss(axis=-2))(X, Y)
        ref = mean_scalarise(js_divergence_logit)(X, Y, axis=-2)
        assert jnp.isclose(out, ref)

        f = lambda x: jnp.linalg.norm(x, axis=-1) ** 2
        out = eqx.filter_jit(BregmanDivergenceLoss(f=f, f_dim=1))(S, T)
        ref = mean_scalarise(bregman_divergence)(S, T, f=f, f_dim=1)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(BregmanDivergenceLogitLoss(f=f, f_dim=1))(X, Y)
        ref = mean_scalarise(bregman_divergence_logit)(X, Y, f=f, f_dim=1)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(EquilibriumLoss())(S)
        ref = mean_scalarise(equilibrium)(S)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(
            EquilibriumLogitLoss(level_axis=-2, prob_axis=-1))(X)
        ref = mean_scalarise(
            equilibrium_logit)(X, level_axis=-2, prob_axis=-1)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(
            SecondMomentLoss(standardise=True, skip_normalise=True))(X, Y)
        ref = mean_scalarise(
            second_moment)(X, Y, standardise=True, skip_normalise=True)
        assert jnp.isclose(out, ref)

        Z = Y @ X
        out = eqx.filter_jit(SecondMomentCentredLoss(
            standardise_data=True,
            standardise_mu=True,
            skip_normalise=True
        ))(X, Y, Z)
        ref = mean_scalarise(second_moment_centred)(
            X, Y, Z,
            standardise_data=True,
            standardise_mu=True,
            skip_normalise=True
        )
        assert jnp.isclose(out, ref)
