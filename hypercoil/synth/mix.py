# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Signal mixture synthesis
~~~~~~~~~~~~~~~~~~~~~~~~
Data synthesis using a linear signal mixture.
"""
import numpy as np
from scipy.fft import rfft, irfft
from scipy.stats import poisson


def synth_slow_signals(
        signal_dim=100,
        time_dim=200,
        subject_dim=1,
        lp=0.3,
        hp=0.0,
        seed=None
    ):
    """
    Synthesise slow signals.

    Parameters
    ----------
    signal_dim : int (default 100)
        Number of signals to synthesise.
    time_dim : int (default 200)
        Number of time points per signal.
    subject_dim : int (default 1)
        Number of subject time series to synthesise.
    lp : float (default 0.3)
        Fraction of lowest frequencies to spare from obliteration.
    seed : int (default None)
        Seed for RNG.

    Returns a numpy array with shape:
    subject_dim x signal_dim x time_dim
    """
    np.random.seed(seed)
    sources = np.random.rand(subject_dim, signal_dim, time_dim)
    sources_freq = rfft(sources, n=time_dim, axis=-1)
    freq_dim = time_dim // 2 + 1
    #print(freq_dim, sources_freq.shape[-1])
    #assert(freq_dim == sources_freq.shape[-1])
    sources_freq[:, :, 0] = 0
    sources_freq[:, :, (int(lp * freq_dim)):] = 0
    sources_freq[:, :, :(int(hp * freq_dim))] = 0
    sources_filt = irfft(sources_freq, n=time_dim, axis=-1)
    return (
        (sources_filt.T - sources_filt.T.mean(0)) /
        sources_filt.T.std(0)
    ).squeeze().T


def mix_data_01(ts, mixture_dim=9, return_mix_matrix=False, seed=None):
    """
    Simple linear mixture.

    Mix the provided latent signals as a linear combination, such
    that each latent signal is weighted randomly and with equal
    probability either 0, -1, or +1. There is a chance (a very low
    chance, hopefully, depending on the signal dimensions specified)
    that a new signal is created completely empty (with all 0s).

    Parameters
    ----------
    ts : np array
        Latent signals to be mixed. The last two dimensions should
        be the number of signals and the number of observations.
    mixture_dim : int (default 9)
        Number of new observed signals to be created as linear
        combinations of the latent signals.
    return_mix_matrix : bool (default False)
        Indicates whether the mixture matrix should be returned
        in addition to the signal mixture.
    """
    np.random.seed(seed)
    signal_dim = ts.shape[-2]
    mix = np.random.randint(-1, 2, (mixture_dim, signal_dim))
    if return_mix_matrix:
        return (mix @ ts), mix
    return mix @ ts


def mix_card_probs_poisson_normalised(latent_dim, mu=1, loc=1):
    """
    Use when creating a linear mixture matrix that maps latent
    signals to observed signals. We call the number of latent
    signals that are combined to produce each observed signal the
    'cardinality' of that observed signal.

    Operationalise the probability distribution over possible
    cardinalities following a Poisson with the specified parameters.
    """
    base = poisson.pmf(np.arange(latent_dim), mu=mu, loc=loc)
    return base / base.sum()


def choose_mix_card(latent_dim, observed_dim, probs):
    """
    Use when creating a linear mixture matrix that maps latent
    signals to observed signals. We call the number of latent
    signals that are combined to produce each observed signal the
    'cardinality' of that observed signal.

    Given the specified probability distribution over possible
    signal cardinalities, sample a cardinality for each observed
    signal.
    """
    return np.random.choice(latent_dim, size=(observed_dim,), p=probs)


def create_mixture_matrix(
        observed_dim,
        latent_dim,
        mix_probs=None
    ):
    """
    Create a linear mixture matrix that maps latent signals to
    observed signals.

    Parameters
    ----------
    observed_dim : int
        Number of observed signals.
    latent_dim : int
        Number of latent signals.
    mix_probs : np array (default None)
        Vector specifying the probability that each possible number of
        latent signals is mapped to an observed signal. The dimension
        of this vector should equal the latent_dim. If none is specified,
        the `mix_card_probs_poisson_normalised` function is used to
        instantiate a distribution. For example, the vector
        [0, 0.1, 0.5, 0.3, 0.1]
        corresponds to a probability of 0.1 that each observed signal is
        created from only a single latent signal, 0.5 that it is created
        as a linear combination of 2 latent signals, etc. The probability
        corresponding to 0 should always be 0, unless you have a specific
        reason otherwise.
    """
    mask = np.zeros((observed_dim, latent_dim))
    mix_probs = mix_probs or mix_card_probs_poisson_normalised(
        latent_dim, mu=(max(1, latent_dim // 3))
    )
    mix_card = choose_mix_card(latent_dim, observed_dim, mix_probs)
    for i, n_signals in enumerate(mix_card):
        idx = np.random.permutation(latent_dim)[:n_signals]
        mask[i, idx] = 1
    weights = np.random.randn(observed_dim, latent_dim)
    state = mask * weights
    return state / np.abs(state).sum(-1, keepdims=True)


def mix_data(mixture, sources, local=None, local_scale=0.25):
    """
    Apply the specified mixture matrix to linearly recombine the latent
    sources into a set of observed signals.

    Parameters
    ----------
    mixture : np array
        Mixture matrix. The final two dimensions should correspond to
        the number of observed signals and the number of latent signals.
    sources : np array
        Latent signals. The final two dimensions should correspond to
        the number of latent signals and the time dimension.
    local : np array or None
        If this is an array, then it should contain the unique, local
        component of each observed signal. The final two dimensions
        should thus correspond to the number of observed signals and
        the time dimension.
    local_scale : float (default 0.25)
        If local signals are provided, this specifies the scaling to
        apply to randomly sampled mixture weights for the local
        signals.
    """
    src_signals = mixture @ sources
    local_signals = 0
    if local is not None:
        local_weights = local_scale * np.random.randn(*local.shape[:-1], 1)
        local_signals = local_weights * local
    return src_signals + local_signals


def synthesise_mixture(
        time_dim=200,
        observed_dim=9,
        latent_dim=100,
        subject_dim=1,
        include_local=False,
        local_scale=0.25,
        lp=0.3,
        hp=0.0,
        seed=0,
        mixture=None,
        return_mix_matrix=False
    ):
    """
    Synthesise data as a linear mixture.

    Parameters
    ----------
    time_dim : int (default 200)
        Number of time points per signal.
    observed_dim : int (default 10)
        Number of observed signals to return.
    latent_dim : int (default 100)
        Number of latent signals to synthesise.
    subject_dim : int (default 1)
        Number of subject time series to synthesise.
    include_local : bool (default False)
        If true, then each observed signal mixture is combined with a
        unique, local signal component.
    local_scale : float (default 0.25)
        If `include_local` is true, this specifies the scaling to
        apply to randomly sampled mixture weights for the local
        signals.
    lp : float (default 0.3)
        Fraction of lowest frequencies to spare from obliteration.
    seed : int (default None)
        Seed for RNG.
    mixture : np array or None (default None)
        Mixture matrix. If not specified, one will be synthesised.
    return_mix_matrix : bool (default False)
        Indicates whether the mixture matrix should be returned
        in addition to the signal mixture.
    """
    np.random.seed(seed)
    sources = synth_slow_signals(
        signal_dim=latent_dim,
        time_dim=time_dim,
        subject_dim=subject_dim,
        lp=lp,
        hp=hp,
        seed=seed
    )
    if include_local:
        localseed = seed
        if localseed is not None: localseed += 1
        local = synth_slow_signals(
            signal_dim=observed_dim,
            time_dim=time_dim,
            subject_dim=subject_dim,
            lp=lp,
            hp=hp,
            seed=localseed
        )
    else:
        local = None
    if mixture is None:
        mixture = create_mixture_matrix(
            observed_dim=observed_dim,
            latent_dim=latent_dim
        )
    signals = mix_data(
        mixture=mixture,
        sources=sources,
        local=local,
        local_scale=local_scale
    )
    if return_mix_matrix:
        return signals, mixture
    return signals
