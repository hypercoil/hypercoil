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
from hypercoil.functional import diag_embed, linear_kernel, relaxed_modularity
from hypercoil.loss.functional import (
    identity, constraint_violation, unilateral_loss, hinge_loss,
    det_gram, log_det_gram, smoothness, bimodal_symmetric,
    entropy, entropy_logit, equilibrium, equilibrium_logit, bregman_divergence,
    kl_divergence, kl_divergence_logit, js_divergence, js_divergence_logit,
    second_moment, _second_moment_impl, second_moment_centred, batch_corr, qcfc,
    reference_tether, interhemispheric_tether, compactness, dispersion,
    multivariate_kurtosis, modularity, eigenmaps,
)
from hypercoil.loss.scalarise import (
    sum_scalarise, mean_scalarise, meansq_scalarise,
    norm_scalarise, vnorm_scalarise,
)


class TestLossFunction:

    def test_normed_losses(self):
        X = jnp.array([
            [-1, 2, 0, 2, 1],
            [0, 1, 0, 0, -1],
            [3, -1, 0, -2, 0]
        ])

        L0 = jax.jit(vnorm_scalarise(p=0, axis=None)())
        assert L0(X) == 9
        L0 = mean_scalarise(inner=norm_scalarise(p=0, axis=0))()(X)
        assert jnp.isclose(L0, (X != 0).sum() / 5)
        L0 = mean_scalarise(inner=norm_scalarise(p=0, axis=-1))()(X)
        assert jnp.isclose(L0, (X != 0).sum() / 3)

        L1 = jax.jit(vnorm_scalarise(p=1, axis=None)())
        assert L1(X) == 14
        L1 = mean_scalarise(inner=norm_scalarise(p=1, axis=0))()(X)
        assert jnp.isclose(L1, jnp.abs(X).sum() / 5)
        L1 = mean_scalarise(inner=norm_scalarise(p=1, axis=-1))()(X)
        assert jnp.isclose(L1, jnp.abs(X).sum() / 3)

        L2 = jax.jit(vnorm_scalarise(p=2, axis=None)())
        assert L2(X) == jnp.sqrt((X ** 2).sum())
        L2 = mean_scalarise(inner=norm_scalarise(p=2, axis=0))()(X)
        assert jnp.isclose(L2, jnp.sqrt((X ** 2).sum(0)).mean())
        L2 = mean_scalarise(inner=norm_scalarise(p=2, axis=-1))()(X)
        assert jnp.isclose(L2, jnp.sqrt((X ** 2).sum(-1)).mean())

    def test_unilateral_loss(self):
        X = jnp.array([
            [-1, 2, 0, 2, 1],
            [0, 1, 0, 0, -1],
            [3, -1, 0, -2, 0]
        ])

        uL0 = jax.jit(vnorm_scalarise(p=0, axis=None)(unilateral_loss))
        assert uL0(X) == 5
        assert uL0(-X) == 4
        uL1 = jax.jit(vnorm_scalarise(p=1, axis=None)(unilateral_loss))
        assert uL1(X) == 9
        assert uL1(-X) == 5
        uL2 = jax.jit(vnorm_scalarise(p=2, axis=None)(unilateral_loss))
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
        f = cjit(vnorm_scalarise(p=1, axis=None)(constraint_violation))
        g = cjit(vnorm_scalarise(p=0, axis=None)(constraint_violation))
        h = jax.jit(vnorm_scalarise(p=1, axis=None)(unilateral_loss))
        j = vnorm_scalarise(p=1, axis=None)(constraint_violation)
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

        hinge = jax.jit(sum_scalarise()(hinge_loss))
        assert hinge(Y, Y_hat_0) == 5
        assert hinge(Y, Y_hat_1) == 4
        assert hinge(Y, Y_hat_minus1) == 6

    def test_smoothness_loss(self):
        key = jax.random.PRNGKey(0)
        X = jnp.array([
            [0, 0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6, 0.7]
        ])
        y0 = 0.2828426957130432
        y1 = 0.17320507764816284
        sjit = partial(jax.jit, static_argnames=('axis',))

        smoothness_loss = sjit(
            mean_scalarise(inner=vnorm_scalarise(axis=0))(smoothness))
        assert jnp.isclose(smoothness_loss(X, axis=0), y0)
        smoothness_loss = sjit(
            mean_scalarise(inner=vnorm_scalarise(axis=-1))(smoothness))
        assert jnp.isclose(smoothness_loss(X, axis=-1), y1)

    def test_bimodal_symmetric_loss(self):
        X1 = jnp.array([
            [0.2, 0, 1, 0.7, 1],
            [1.2, 0, 0.8, -0.2, 0],
            [0, 1, 0.3, 0, 1]
        ])
        X2 = jnp.array(
            [0.8, 0.5, 0.1]
        )
        y1 = 0.58309518948453
        y2 = .65

        symbm_loss = jax.jit(vnorm_scalarise(axis=None)(bimodal_symmetric))
        assert jnp.isclose(symbm_loss(X1), y1)
        symbm_loss = jax.jit(
            vnorm_scalarise(p=1, axis=None)(bimodal_symmetric),
            static_argnames=('modes',))
        assert jnp.isclose(symbm_loss(X2, modes=(0.95, 0.05)), y2)

    def test_det_loss(self):
        key = jax.random.PRNGKey(0)
        X = jax.random.normal(key, (2, 10, 5))
        dgjit = partial(jax.jit, static_argnames=('op', 'psi', 'xi',))

        det_loss = dgjit(sum_scalarise()(det_gram))
        logdet_loss = dgjit(sum_scalarise()(log_det_gram))

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

        entropy_loss = jax.jit(mean_scalarise()(entropy))
        entropy_logit_loss = jax.jit(mean_scalarise()(entropy_logit))

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

        kl_loss = jax.jit(mean_scalarise()(kl_divergence))
        kl_logit_loss = jax.jit(mean_scalarise()(kl_divergence_logit))

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

        js_loss = jax.jit(mean_scalarise()(js_divergence))
        js_logit_loss = jax.jit(mean_scalarise()(js_divergence_logit))

        out = js_loss(P, Q)
        ref = (js_ref(P, Q, axis=1) ** 2).mean()
        assert jnp.isclose(out, ref)

        out = js_logit_loss(P_logits, Q_logits)
        ref = (js_ref(jax.nn.softmax(P_logits, axis=-1),
                      jax.nn.softmax(Q_logits, axis=-1),
                      axis=-1) ** 2).mean()
        assert jnp.isclose(out, ref)

    def test_bregman_divergence(self):
        #TODO: This needs many, many more test cases.
        key = jax.random.PRNGKey(0)
        key_x, key_y = jax.random.split(key)

        X = jax.random.normal(key_x, (3, 10,))
        Y = jax.random.normal(key_y, (3, 10,))

        bregman_loss = jax.jit(mean_scalarise()(bregman_divergence),
                               static_argnames=('f', 'f_dim'))
        f = lambda x: jnp.linalg.norm(x, axis=-1) ** 2

        out = bregman_loss(X, Y, f=f, f_dim=1)
        ref = ((X - Y) ** 2).sum(-1).mean()
        assert jnp.isclose(out, ref, atol=1e-4)

    def test_equilibrium_loss(self):
        key = jax.random.PRNGKey(0)
        base = jnp.ones((5, 10))
        noise = jax.random.normal(key, (5, 10))

        equilibrium_loss = jax.jit(meansq_scalarise()(equilibrium))

        out0 = equilibrium_loss(base)
        out1 = equilibrium_loss(base + 1e-1 * noise)
        out2 = equilibrium_loss(base + 1e0 * noise)
        out3 = equilibrium_loss(base + 1e1 * noise)
        assert out0 < out1 < out2 < out3
        assert jnp.isclose(out0, 0)

    def test_second_moment(self):
        n_groups = 3
        n_channels = 10
        n_observations = 20

        key = jax.random.PRNGKey(0)

        src = jnp.zeros((n_channels,), dtype=int)
        src = src.at[(n_channels // 2):].set(1)
        weight = 0.5 * jnp.eye(2)[src].swapaxes(-2, -1)
        data = jax.random.normal(key=key, shape=(n_channels, n_observations))

        loss = jax.jit(mean_scalarise()(second_moment),
                       static_argnames=('standardise'))
        out = loss(data, weight, standardise=False)
        ref = jnp.stack((
            data[:(n_channels // 2), :].var(-2, ddof=0),
            data[(n_channels // 2):, :].var(-2, ddof=0)
        ))
        assert jnp.isclose(out, ref.mean())

        mu = weight @ data / weight.sum(-1, keepdims=True)
        out = _second_moment_impl(data, weight, mu).squeeze()
        ref = jnp.stack((
            data[:(n_channels // 2), :].var(-2, ddof=0),
            data[(n_channels // 2):, :].var(-2, ddof=0)
        ))
        assert jnp.allclose(out, ref)

        out = _second_moment_lowmem_impl(data, weight, mu).squeeze()
        ref = _second_moment_impl(data, weight, mu).squeeze()
        assert jnp.allclose(out, ref)

    def test_second_moment_centre_equivalence(self):
        n_groups = 3
        n_channels = 10
        n_observations = 20

        key = jax.random.PRNGKey(0)
        key_d, key_w = jax.random.split(key)

        data = jax.random.normal(key_d, shape=(n_channels, n_observations))
        weight = jax.random.normal(key_w, shape=(n_groups, n_channels))
        mu = weight @ data / weight.sum(-1, keepdims=True)

        loss0 = jax.jit(mean_scalarise()(second_moment))
        loss1 = jax.jit(mean_scalarise()(second_moment_centred))

        ref = loss0(data, weight)
        out = loss1(data, weight, mu)
        assert jnp.allclose(out, ref)

    def test_functional_homogeneity(self):
        N_INSTANCES = 3
        N_PARCELS = 10
        N_VOXELS = 100
        N_TIMEPOINTS = 50
        def corrcoef_triu(x):
            return jnp.corrcoef(x)[jnp.triu_indices(x.shape[0], k=1)]

        parcellation = jax.random.randint(
            jax.random.PRNGKey(0),
            shape=(N_VOXELS,),
            minval=0,
            maxval=N_PARCELS,
        )
        ts = jax.random.normal(
            jax.random.PRNGKey(1),
            shape=(N_INSTANCES, N_VOXELS, N_TIMEPOINTS),
        )
        parcellation = jnp.eye(N_PARCELS)[parcellation].astype(bool)
        fh = jnp.asarray([
            jax.vmap(corrcoef_triu)(ts[..., parcel, :]).mean(-1)
            for parcel in parcellation.T
        ]).T
        out = functional_homogeneity(
            ts,
            parcellation.astype(float),
            standardise=True,
        )
        assert jnp.allclose(out, fh, rtol=1e-4)
        w = parcellation.sum(0)
        ref = (w * fh).sum(-1) / w.sum(-1)
        out = functional_homogeneity(
            ts,
            parcellation.astype(float),
            standardise=True,
            use_schaefer=True,
        )
        assert jnp.allclose(out, ref)

    def test_batch_corr(self):
        n_batch = (100, 1000)
        gt_shared = (0.1, 0.5, 0.9)
        n_channels = 10
        n_observations = 20

        key = jax.random.PRNGKey(0)
        key_d, key_q = jax.random.split(key)

        batch_corr_loss = jax.jit(
            mean_scalarise()(batch_corr),
            static_argnames=('tol', 'abs')
        )
        qcfc_loss = jax.jit(
            mean_scalarise()(batch_corr),
            static_argnames=('tol', 'abs')
        )

        for n in n_batch:
            base = jnp.linspace(0.8, 1.2, n)
            data_noise = jax.random.normal(
                key_d, shape=(n, n_channels, n_observations))
            q_noise = jax.random.normal(key_q, shape=(n,))
            data = 0.1 * data_noise + 0.9 * base[..., None, None]
            out0, out1 = [], []
            for c in gt_shared:
                q = c * base + (1 - c) * q_noise
                out = qcfc(data, q, tol='auto')
                assert out.shape == (n_channels * n_observations, 1)
                out0.append(batch_corr_loss(data, q))
                out1.append(qcfc_loss(data, q, tol='auto'))
                assert out0 > out1
            for i in range(len(out0) - 1):
                assert out0[i] < out0[i + 1]
                assert out1[i] < out1[i + 1]

    def test_spatial_losses(self):
        #TODO: test spatial losses. Deferring everything except smoke tests
        #      for now.
        key = jax.random.PRNGKey(0)
        key_d, key_c = jax.random.split(key)

        key_r = jax.random.split(key_c, 1)[0]
        data = jax.random.uniform(key_d, (5, 100))
        coor = jax.random.uniform(key_c, (3, 100))
        coor_ref = jax.random.uniform(key_r, (3, 5))
        jax.jit(mean_scalarise()(reference_tether))(data, coor_ref, coor)

        key_ld, key_rd = jax.random.split(key_d)
        key_lc, key_rc = jax.random.split(key_c)
        data_lh = jax.random.uniform(key_ld, (5, 100))
        coor_lh = jax.random.uniform(key_lc, (3, 100))
        data_rh = jax.random.uniform(key_rd, (5, 100))
        coor_rh = jax.random.uniform(key_rc, (3, 100))
        jax.jit(mean_scalarise()(interhemispheric_tether))(
            data_lh, data_rh, coor_lh, coor_rh)

        data = jax.random.uniform(key_d, (5, 100))
        coor = jax.random.uniform(key_c, (3, 100))
        jax.jit(mean_scalarise()(compactness))(data, coor)

        coor = jax.random.uniform(key_c, (3, 5))
        jax.jit(mean_scalarise()(dispersion))(coor.T,)

    def test_mvkurtosis_expected_value(self):
        key = jax.random.PRNGKey(0)

        mvk_loss = jax.jit(mean_scalarise()(multivariate_kurtosis),
                           static_argnames=('dimensional_scaling'))
        mvks_loss = partial(mvk_loss, dimensional_scaling=True)

        dims = (5, 10, 20, 50, 100)
        for d in dims:
            ref = -d * (d + 2)
            ts = jax.random.normal(key=key, shape=(10, d, 2000))
            out = mvk_loss(ts)
            assert jnp.isclose(out, ref, rtol=0.05)
            out = mvks_loss(ts)
            assert jnp.isclose(out, -1, atol=0.01)

    def test_connectopies(self):
        #TODO: Correctness tests for connectopies
        key = jax.random.PRNGKey(0)
        key_d, key_a = jax.random.split(key)

        Q = jax.random.normal(key_d, shape=(20, 4))
        A = jax.random.normal(key_a, shape=(3, 20, 20))

        modularity_loss = jax.jit(mean_scalarise()(modularity))
        out = modularity_loss(Q, A, gamma=5) / 2
        ref = -relaxed_modularity(A, Q, gamma=5).mean()
        assert jnp.allclose(out, ref)

        jax.jit(mean_scalarise()(eigenmaps))(Q, A)
