# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for loss modules.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from hypercoil.functional.kernel import gaussian_kernel, linear_distance
from hypercoil.functional.matrix import diag_embed

from hypercoil.loss.functional import (
    identity, difference, constraint_violation, unilateral_loss, hinge_loss,
    smoothness, bimodal_symmetric, det_gram, log_det_gram,
    entropy, entropy_logit, kl_divergence, kl_divergence_logit,
    js_divergence, js_divergence_logit,
    bregman_divergence, bregman_divergence_logit,
    equilibrium, equilibrium_logit, second_moment, second_moment_centred,
    batch_corr, qcfc, reference_tether, interhemispheric_tether, compactness,
    dispersion, multivariate_kurtosis, connectopy, modularity, eigenmaps,
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
    SecondMomentCentredLoss, BatchCorrelationLoss, QCFCLoss, ReferenceTetherLoss,
    InterhemisphericTetherLoss, CompactnessLoss, DispersionLoss,
    MultivariateKurtosis, ConnectopyLoss, ModularityLoss, EigenmapsLoss,
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
        ref = meansq_scalarise()(difference)(X, Y)
        assert out == ref

        out = eqx.filter_jit(NormedLoss(p=2, axis=-1))(X)
        ref = mean_scalarise(inner=vnorm_scalarise(p=2, axis=-1))()(X)
        assert out == ref

        out = eqx.filter_jit(ConstraintViolationLoss(
            constraints=(lambda x: jnp.eye(10) @ x,),
            broadcast_against_input=True))(X)
        ref = mean_scalarise()(constraint_violation)(
            X, constraints=(lambda x: jnp.eye(10) @ x,),
            broadcast_against_input=True)
        assert out == ref

        out = eqx.filter_jit(UnilateralLoss())(X)
        ref = mean_scalarise()(unilateral_loss)(X)
        assert out == ref

        out = eqx.filter_jit(HingeLoss())(X, Y)
        ref = sum_scalarise()(hinge_loss)(X, Y)
        assert out == ref

        out = eqx.filter_jit(
            SmoothnessLoss(n=2, pad_value=-1, axis=-2))(X)
        ref = vnorm_scalarise(p=1, axis=None)(smoothness)(
            X, n=2, pad_value=-1, axis=-2)
        # Not sure why this is inexact.
        assert jnp.isclose(out, ref, atol=1e-4)

        out = eqx.filter_jit(BimodalSymmetricLoss(modes=(-1, 1)))(X)
        ref = mean_scalarise()(bimodal_symmetric)(X, modes=(-1, 1))
        assert out == ref

        S = jax.random.normal(key, (10, 100))

        out = eqx.filter_jit(GramLogDeterminantLoss(
            op=gaussian_kernel, psi=0.001, xi=0.001)
        )(S, key=jax.random.PRNGKey(47))
        ref = mean_scalarise()(log_det_gram)(
            S, op=gaussian_kernel, psi=0.001, xi=0.001,
            key=jax.random.PRNGKey(47)
        )
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(GramDeterminantLoss())(S)
        ref = mean_scalarise()(det_gram)(S)
        assert jnp.isclose(out, ref)

        S, T = jnp.abs(X), jnp.abs(Y)
        S = S / S.sum(axis=-2, keepdims=True)
        T = T / T.sum(axis=-2, keepdims=True)
        out = eqx.filter_jit(EntropyLoss(axis=-2))(S)
        ref = mean_scalarise()(entropy)(S, axis=-2)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(EntropyLogitLoss(axis=-2))(X)
        ref = mean_scalarise()(entropy_logit)(X, axis=-2)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(KLDivergenceLoss(axis=-2))(S, T)
        ref = mean_scalarise()(kl_divergence)(S, T, axis=-2)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(KLDivergenceLogitLoss(axis=-2))(X, Y)
        ref = mean_scalarise()(kl_divergence_logit)(X, Y, axis=-2)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(JSDivergenceLoss(axis=-2))(S, T)
        ref = mean_scalarise()(js_divergence)(S, T, axis=-2)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(JSDivergenceLogitLoss(axis=-2))(X, Y)
        ref = mean_scalarise()(js_divergence_logit)(X, Y, axis=-2)
        assert jnp.isclose(out, ref)

        f = lambda x: jnp.linalg.norm(x, axis=-1) ** 2
        out = eqx.filter_jit(BregmanDivergenceLoss(f=f, f_dim=1))(S, T)
        ref = mean_scalarise()(bregman_divergence)(S, T, f=f, f_dim=1)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(BregmanDivergenceLogitLoss(f=f, f_dim=1))(X, Y)
        ref = mean_scalarise()(bregman_divergence_logit)(X, Y, f=f, f_dim=1)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(EquilibriumLoss())(S)
        ref = meansq_scalarise()(equilibrium)(S)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(
            EquilibriumLogitLoss(level_axis=-2, prob_axis=-1))(X)
        ref = mean_scalarise()(equilibrium_logit)(
            X, level_axis=-2, prob_axis=-1)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(
            SecondMomentLoss(standardise=True, skip_normalise=True))(X, Y)
        ref = mean_scalarise()(second_moment)(
            X, Y, standardise=True, skip_normalise=True)
        assert jnp.isclose(out, ref)

        Z = Y @ X
        out = eqx.filter_jit(SecondMomentCentredLoss(
            standardise_data=True,
            standardise_mu=True,
            skip_normalise=True
        ))(X, Y, Z)
        ref = mean_scalarise()(second_moment_centred)(
            X, Y, Z,
            standardise_data=True,
            standardise_mu=True,
            skip_normalise=True
        )
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(FunctionalHomogeneityLoss(
            skip_normalise=True, use_geom_mean=True
        ))(X, jnp.abs(Y))
        ref = mean_scalarise()(functional_homogeneity)(
            X, jnp.abs(Y), skip_normalise=True, use_geom_mean=True
        )
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(PointHomogeneityLoss())(X, jnp.abs(Y), NH)
        ref = mean_scalarise()(point_homogeneity)(X, jnp.abs(Y), NH)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(PointSimilarityLoss())(jnp.abs(Y), NH)
        ref = mean_scalarise()(point_similarity)(jnp.abs(Y), NH)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(PointAgreementLoss(
            rescale_result=True
        ))(X, jnp.abs(Y), NH)
        ref = mean_scalarise()(point_agreement)(
            X, jnp.abs(Y), NH, rescale_result=True
        )
        assert jnp.isclose(out, ref)

        N = Y.sum(-1)
        out = eqx.filter_jit(
            BatchCorrelationLoss(tol='auto', tol_sig=0.1, abs=True))(X, N)
        ref = mean_scalarise()(batch_corr)(
            X, N, tol='auto', tol_sig=0.1, abs=True)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(
            QCFCLoss(tol='auto', tol_sig=0.1, abs=False))(X, N)
        ref = mean_scalarise()(qcfc)(
            X, N, tol='auto', tol_sig=0.1, abs=False)
        assert jnp.isclose(out, ref)

        key_ld, key_rd, key_r = jax.random.split(key_x, 3)
        key_lc, key_rc = jax.random.split(key_y)
        coor_ref = jax.random.uniform(key_r, (3, 5))
        data_lh = jax.random.uniform(key_ld, (5, 100))
        coor_lh = jax.random.uniform(key_lc, (3, 100))
        data_rh = jax.random.uniform(key_rd, (5, 100))
        coor_rh = jax.random.uniform(key_rc, (3, 100))
        ref = mean_scalarise()(reference_tether)(
            data_lh, ref=coor_ref, coor=coor_lh)
        out = eqx.filter_jit(
            ReferenceTetherLoss(ref=coor_ref, coor=coor_lh))(data_lh)
        assert jnp.isclose(out, ref)
        out = eqx.filter_jit(
            ReferenceTetherLoss(coor=coor_lh))(data_lh, ref=coor_ref)
        assert jnp.isclose(out, ref)
        out = eqx.filter_jit(
            ReferenceTetherLoss())(data_lh, ref=coor_ref, coor=coor_lh)
        assert jnp.isclose(out, ref)

        ref = mean_scalarise()(interhemispheric_tether)(
            data_lh, data_rh, coor_lh, coor_rh)
        out = eqx.filter_jit(
            InterhemisphericTetherLoss(lh_coor=coor_lh, rh_coor=coor_rh)
        )(data_lh, data_rh)
        assert jnp.isclose(out, ref)
        out = eqx.filter_jit(
            InterhemisphericTetherLoss()
        )(data_lh, data_rh, lh_coor=coor_lh, rh_coor=coor_rh)
        assert jnp.isclose(out, ref)

        ref = mean_scalarise()(compactness)(
            data_lh, coor_lh, norm='inf', floor=0.05)
        out = eqx.filter_jit(CompactnessLoss(
            norm='inf', floor=0.05))(data_lh, coor_lh)
        assert jnp.isclose(out, ref)

        ref = mean_scalarise()(dispersion)(
            coor_lh.T, metric=linear_distance)
        out = eqx.filter_jit(DispersionLoss(
            metric=linear_distance))(coor_lh.T)
        assert jnp.isclose(out, ref)

        U = X @ X.swapaxes(-2, -1)
        out = eqx.filter_jit(MultivariateKurtosis(
            l2=0.01, dimensional_scaling=True))(U)
        ref = mean_scalarise()(multivariate_kurtosis)(
            U, l2=0.01, dimensional_scaling=True)
        assert jnp.isclose(out, ref)

        key_d, key_a, key_t, key_o = jax.random.split(key_y, 4)

        Q = jax.random.normal(key_d, shape=(20, 4))
        A = jax.random.normal(key_a, shape=(3, 20, 20))
        D = diag_embed(A.sum(-1))
        theta = jax.random.normal(key_t, shape=(4, 4))
        omega = jax.random.normal(key_o, shape=(20, 20))
        theta = theta @ theta.swapaxes(-2, -1)
        omega = omega @ omega.swapaxes(-2, -1)

        def affinity(X, omega):
            return linear_distance(X, theta=omega)

        out = eqx.filter_jit(ConnectopyLoss(
            theta=theta, omega=omega, affinity=affinity))(Q, A, D)
        ref = mean_scalarise()(connectopy)(
            Q, A, D, theta=theta, omega=omega, affinity=affinity)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(ModularityLoss(
            theta=theta, gamma=0.13, exclude_diag=True))(Q, A, D)
        ref = mean_scalarise()(modularity)(
            Q, A, D, theta=theta, gamma=0.13, exclude_diag=True)
        assert jnp.isclose(out, ref)

        out = eqx.filter_jit(EigenmapsLoss(theta=theta, omega=omega))(Q, A)
        ref = mean_scalarise()(eigenmaps)(Q, A, theta=theta, omega=omega)
        assert jnp.isclose(out, ref)
