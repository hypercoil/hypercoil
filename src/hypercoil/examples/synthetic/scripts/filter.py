# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Filter synthesis
~~~~~~~~~~~~~~~~
Synthesise some simple ground truth datasets for testing filter learning.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from functools import reduce
from typing import Optional, Sequence, Tuple
from jax.numpy.fft import rfft, irfft
from hypercoil.engine.paramutil import PyTree, Tensor, _to_jax_array
from hypercoil.functional import complex_decompose
from hypercoil.examples.synthetic.scripts.mix import synthesise_mixture


APPROX_ORTHO_THRESH = 0.1


def synthesise_across_bands(
    bands: Sequence[Tuple[float, float]],
    time_dim: int = 1000,
    observed_dim: int = 7,
    latent_dim: int = 100,
    *,
    key: 'jax.random.PRNGKey',
):
    """
    Synthesise a dataset that has a different correlation structure in
    different frequency bands.

    Parameters
    ----------
    bands : list(tuple)
        Frequency bands into which structured, correlated data is injected.
        The data in each band will have a different structure. Denote as
        (high pass, low pass) as a fraction of Nyquist.
    time_dim : int (default 1000)
        Number of time points per signal.
    observed_dim : int (default 7)
        Number of observed signals to return.
    latent_dim : int (default 100)
        Number of latent signals to synthesise.
    seed : int (default None)
        Seed for RNG.
    """
    key_s, key_l, key_m = jax.random.split(key, 3)
    sources = jax.random.normal(key_s, (observed_dim, time_dim))
    local = jax.random.normal(key_l, (observed_dim, time_dim))
    key_m = jax.random.split(key_m, len(bands))
    mixtures = [
        synthesise_mixture(
            time_dim=time_dim,
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            subject_dim=1,
            include_local=True,
            local_scale=0.25,
            lp=band[1],
            hp=band[0],
            return_mix_matrix=True,
            key=k
        )
        for band, k in zip(bands, key_m)
    ]
    sources_filt, mixtures = list(zip(*mixtures))
    #sources_filt = [bp_signals(sources, band, time_dim) for band in bands]
    # Verify approximate orthogonality
    Z = jnp.stack(sources_filt)
    for i in range(len(bands)):
        cc = jnp.corrcoef(Z[:, i, :].squeeze())
        assert jnp.all(
            cc[jnp.triu_indices_from(cc, 1)] < APPROX_ORTHO_THRESH), (
                'Specified frequency bands are not approximately '
                'orthogonal'
            )
    # Fill unused bands with uncorrelated noise
    bandfill = bs_signals(local, bands, time_dim)
    signal = Z.sum(0) + bandfill
    # Extract the true states (according to shared variance).
    statevar = [jnp.corrcoef(m) for m in mixtures]
    # for i, s in enumerate(sources_filt):
    #     plt.figure()
    #     plt.plot(jnp.abs(jnp.fft.rfft(s)).T)
    #     plt.savefig(f'/tmp/source_{i}.png')
    #     plt.figure()
    #     plt.imshow(jnp.corrcoef(s))
    #     plt.savefig(f'/tmp/corr_{i}.png')
    #     plt.figure()
    #     plt.imshow(statevar[i])
    #     plt.savefig(f'/tmp/statevar_{i}.png')
    return signal, statevar, bands


def bs_signals(
    sources: Tensor,
    bands: Sequence[Tuple[float, float]],
    n: int,
) -> Tensor:
    """
    Fill frequency bands outside the specified list using the specified
    sources.
    """
    sources_fft = rfft(sources, n=n)
    n_bins = sources_fft.shape[-1]
    for band in bands:
        hp, lp = band
        hp = int(jnp.floor(hp * n_bins))
        lp = int(jnp.ceil(lp * n_bins))
        sources_fft = sources_fft.at[:, hp:lp].set(0)
    sources_filt = irfft(sources_fft, n=n)
    return ((sources_filt.T - sources_filt.T.mean(0)) /
            sources_filt.T.std(0)).T


def collate_observed_signals(
    sources_filt: Tensor,
    bandfill: Tensor,
) -> Tensor:
    collate = lambda a, b: a + b
    signal = reduce(collate, sources_filt) + bandfill
    return signal


def plot_frequency_partition(
    bands: Tensor,
    filter: PyTree,
    save: Optional[str] = None,
) -> None:
    """
    Plotting utility when learning a partition over frequencies.
    """
    weight = _to_jax_array(filter.weight)
    freq_dim = weight.shape[-1]
    plt.figure(figsize=(12, 8))
    for (hp, lp) in bands:
        plt.axvline(hp, ls=':', color='grey')
        plt.axvline(lp, ls=':', color='grey')
    # Omit the last weight. Here we assume it corresponds to a rejection band.
    ampl = complex_decompose(weight)[0][:-1]
    for s in ampl:
        plt.plot(jnp.linspace(0, 1, freq_dim), s)
        plt.ylim([0, 1])
        plt.xlim([0, 1])
    if save:
        plt.savefig(save, bbox_inches='tight')


def plot_mvkurtosis(
    fftfilter: PyTree,
    weight: Tensor,
    input: Tensor,
    bands: Sequence[Tuple[float, float]],
    nu: float = 1,
    l2: float = 0.01,
    save: Optional[str] = None,
) -> None:
    from hypercoil.loss.nn import MultivariateKurtosis
    freq_dim = fftfilter.dim
    freq = jnp.linspace(0, 1, freq_dim)
    out = jnp.zeros(freq_dim)
    mvk = MultivariateKurtosis(nu=nu, l2=l2)
    for hp, lp in bands:
        arr = jnp.zeros_like(_to_jax_array(fftfilter.weight))
        hp = int(freq_dim * hp)
        lp = int(freq_dim * lp)
        arr = arr.at[:, hp:lp].set(1.)
        fftfilter = eqx.tree_at(lambda m: m.weight, fftfilter, arr)
        out = out.at[hp:lp].set(-mvk(fftfilter(input)))
    plt.figure(figsize=(12, 8))
    plt.plot(freq, (out / out.max()), c='#999999')
    plt.plot(freq, weight.squeeze(), c='#7722CC')
    plt.legend(['MV Kurtosis', 'Transfer'])
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.xlabel('Frequency')
    if save:
        plt.savefig(save, bbox_inches='tight')
