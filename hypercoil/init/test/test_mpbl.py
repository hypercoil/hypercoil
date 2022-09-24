# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for "bipartite lattice" initialisation using the maximum potential
propagation algorithm.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pkg_resources import resource_filename as pkgrf
from hypercoil.functional.matrix import sym2vec
from hypercoil.init.mpbl import (
    maximum_potential_bipartite_lattice
)
from hypercoil.examples.synthetic.scripts.sylo import synthesise_lowrank_block


class TestMPBL:
    
    def plot_orig_and_compressed(self, orig, C_left, C_right, path):
        compr = C_left @ orig @ C_right.T
        recon = C_left.T @ compr @ C_right
        scale = jnp.linalg.lstsq(
            sym2vec(recon).reshape(-1, 1),
            sym2vec(orig).reshape(-1, 1)
        )[0]
        resid = orig - scale * recon

        quantile = 0.95
        lim_o = jnp.quantile(jnp.abs(orig), quantile)
        lim_c = jnp.quantile(jnp.abs(compr), quantile)
        lim_r = jnp.quantile(jnp.abs(recon), quantile)

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Matrix compression: MPBL algorithm', fontsize=16)
        ax[0][0].imshow(orig, cmap='RdBu_r', vmin=-lim_o, vmax=lim_o)
        ax[0][0].set_title('Original')
        ax[0][0].set_xticks([])
        ax[0][0].set_yticks([])
        ax[0][1].imshow(compr, cmap='RdBu_r', vmin=-lim_c, vmax=lim_c)
        ax[0][1].set_title('Compressed')
        ax[0][1].set_xticks([])
        ax[0][1].set_yticks([])
        ax[1][0].imshow(recon, cmap='RdBu_r', vmin=-lim_r, vmax=lim_r)
        ax[1][0].set_title('Reconstructed')
        ax[1][0].set_xticks([])
        ax[1][0].set_yticks([])
        ax[1][1].imshow(resid, cmap='RdBu_r', vmin=-lim_o, vmax=lim_o)
        ax[1][1].set_title('Residual')
        ax[1][1].set_xticks([])
        ax[1][1].set_yticks([])
        results = pkgrf(
            'hypercoil',
            'results/'
        )
        fig.savefig(f'{results}/mpbl_{path}.png', bbox_inches='tight')

    def test_mpbl_symmetric(self):
        A = synthesise_lowrank_block(100, key=jax.random.PRNGKey(0))
        A = jnp.asarray(A)
        orig = A
        C2, _, u_prop = maximum_potential_bipartite_lattice(
            potentials=orig,
            n_out=50,
            order=4,
            iters=1,
            key=jax.random.PRNGKey(0)
        )
        self.plot_orig_and_compressed(
            orig, C2, C2, 'factor02'
        )
        C5, _, u_prop = maximum_potential_bipartite_lattice(
            potentials=u_prop,
            n_out=20,
            order=1,
            iters=1,
            key=jax.random.PRNGKey(0)
        )
        self.plot_orig_and_compressed(
            orig, C5 @ C2, C5 @ C2, 'factor05'
        )
        C10, _, _ = maximum_potential_bipartite_lattice(
            potentials=u_prop,
            n_out=10,
            order=1,
            iters=2,
            key=jax.random.PRNGKey(0)
        )
        self.plot_orig_and_compressed(
            orig, C10 @ C5 @ C2, C10 @ C5 @ C2, 'factor10'
        )

    def test_mpbl_asymmetric(self):
        # Note: this is a very simple example, and the results are not
        # particularly good. This is because the algorithm was not designed
        # with asymmetric matrices in mind. It is not clear how to improve
        # this, but it is worth investigating at some point.
        A = jax.random.normal(
            jax.random.PRNGKey(0),
            shape=(10, 10)
        )
        C_left = jnp.arange(100)[..., None] @ jnp.arange(10)[None, ...]
        C_right = jnp.arange(50)[..., None] @ jnp.arange(10)[None, ...]
        A = C_left @ A @ C_right.T
        potentials = (A @ A.T, A.T @ A)
        (C2L, C2R), _, _ = maximum_potential_bipartite_lattice(
            potentials=potentials,
            objective=A,
            n_out=(50, 25),
            order=4,
            iters=1,
            key=jax.random.PRNGKey(0)
        )
        self.plot_orig_and_compressed(
            A, C2L, C2R, 'asymfactor02'
        )
