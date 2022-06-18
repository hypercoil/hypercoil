# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Synthesise data matching spectral and covariance properties of a reference.
"""
import torch
from ..data.functional import normalise


def match_spectra(signal, reference, use_mean=False, frequencies=False):
    """
    Rescale the frequency components of an input signal to match the
    amplitude spectrum of a reference signal.

    This operation assumes real-valued input and always returns real-valued
    output.

    :Dimension: **signal :** :math:`(D, N)`
                    D denotes dimension of the multivariate signal and N
                    denotes the number of observations per channel.
                **reference :** :math:`(D, N)` or :math:`(D, F)`
                    As above. F denotes the frequency dimension of the signal,
                    e.g., N / 2 + 1. Expected dimension depends on the Boolean
                    value passed to ``frequencies``.

    Parameters
    ----------
    signal : tensor
        Input signal.
    reference : tensor
        Reference signal.
    use_mean : bool (default False)
        Indicates that, instead of matching the spectrum of the corresponding
        channel in the reference signal, all channels are matched to the mean
        spectrum across all reference channels.
    frequencies : bool (default False)
        If True, indicates that the ``reference`` provided is a spectrum in
        the frequency domain. Otherwise, the spectrum is computed from the
        provided reference.
    """
    if not frequencies:
        signal = torch.fft.rfft(signal)
        reference = torch.fft.rfft(reference)
    ampl_ref = reference.abs()
    if use_mean:
        ampl_ref = ampl_ref.mean(0)
    matched = signal * ampl_ref
    ampl_matched = matched.abs()
    matched = matched * ampl_ref.mean() / ampl_matched.mean()
    return torch.fft.irfft(matched)


def match_covariance(signal, reference, cov=False):
    r"""
    Project a multivariate signal so that its covariance matches a reference
    covariance.

    This uses the procedure described in Laumann et al. (2017),
    "On the Stability of BOLD fMRI Correlations". Specifically, given an
    input multivariate time series :math:`\mathbf{T}` and a reference
    covariance matrix :math:`\Sigma`, the procedure first computes the
    eigendecomposition of the real symmetric :math:`\Sigma` and then forms a
    projected time series :math:`\widetilde{\mathbf{T}}`.

    .. math::

        \Sigma = Q \Lambda Q^\intercal

        \widetilde{\mathbf{T}} = Q \Lambda^{\frac{1}{2}} \mathbf{T}

    The basic assumption of the procedure is that the input time series
    :math:`\mathbf{T}` (containing :math:`n` observations) is drawn from a
    process such that in expectation

    :math:`\frac{1}{n} \mathbf{E}[\mathbf{T} \mathbf{T}^\intercal] = \mathbf{I}`

    where :math:`\mathbf{I}` is the identity matrix. This assumption holds
    _inter alia_ when :math:`\mathbf{T}` is drawn i.i.d. from a standard
    normal distribution.

    .. math::

        \frac{1}{n} \mathbf{E}[\widetilde{\mathbf{T}} \widetilde{\mathbf{T}}^\intercal]

        = \frac{1}{n} \mathbf{E}[Q \Lambda^{\frac{1}{2}} \mathbf{T} \mathbf{T}^\intercal \Lambda^{\frac{1}{2}} Q^\intercal]

        = Q \Lambda^{\frac{1}{2}} \left(\frac{1}{n} \mathbf{E}[\mathbf{T} \mathbf{T}^\intercal] \right) \Lambda^{\frac{1}{2}} Q^\intercal

        = Q \Lambda Q^\intercal = \Sigma

    :Dimension: **signal :** :math:`(D, N)`
                    D denotes dimension of the multivariate signal and N
                    denotes the number of observations per channel.
                **reference :** :math:`(D, N)` or :math:`(D, D)`
                    As above. Depends on the Boolean value passed to ``cov``.

    Parameters
    ----------
    signal : tensor
        Input signal :math:`\mathbf{T}` to project so that its covariance is
        matched to the ``reference``.
    reference : tensor
        Either a reference signal whose covariance :math:`\Sigma` is to be
        matched, or the covariance matrix itself. See ``cov``.
    cov : bool (default False)
        If True, indicates that the ``reference`` provided is a covariance
        matrix. Otherwise, the covariance matrix is computed from the provided
        reference.

    Returns
    -------
    tensor
        :math:`\widetilde{\mathbf{T}}`, ``signal`` projected to match the
        ``reference`` covariance.
    """
    if not cov:
        reference = reference.cov()
    L, Q = torch.linalg.eigh(reference)
    L_sqrt = L.sqrt()
    return Q @ (L_sqrt.unsqueeze(-1) * signal)


def match_reference(signal, reference, use_mean=False):
    """
    Match both the
    :func:`spectrum <match_spectra>` and
    :func:`covariance <match_covariance>`
    of a reference signal.

    :Dimension: **signal :** :math:`(D, N)`
                    D denotes dimension of the multivariate signal and N
                    denotes the number of observations per channel.
                **reference :** :math:`(D, N)`
                    As above.

    Parameters
    ----------
    signal : tensor
        Input signal.
    reference : tensor
        Reference signal.
    use_mean : bool (default False)
        Indicates that, instead of matching the spectrum of the corresponding
        channel in the reference signal, all channels are matched to the mean
        spectrum across all reference channels.

    See also
    --------
    :func:`synthesise_matched`
    :func:`match_cov_and_spectrum`
    """
    matched = match_spectra(signal, reference, use_mean=use_mean)
    matched = normalise(matched)
    return match_covariance(signal=matched, reference=reference)


def match_cov_and_spectrum(signal, spectrum, cov):
    """
    Transform an input signal to match a given
    :func:`spectrum <match_spectra>` and
    :func:`covariance matrix <match_covariance>`.

    :Dimension: **signal :** :math:`(D, N)`
                    D denotes dimension of the multivariate signal and N
                    denotes the number of observations per channel.
                **spectrum :** :math:`(D, F)`
                    D as above. F denotes the frequency dimension of the
                    signal, e.g., N / 2 + 1.
                **cov :** :math:`(D, D)`
                    As above.
    Parameters
    ----------
    signal : tensor
        Input signal.
    spectrum : tensor
        Spectrum to be matched.
    cov : tensor
        Covariance matrix to be matched.

    See also
    --------
    :func:`synthesise_from_cov_and_spectrum`
    :func:`match_reference`
    """
    matched = match_spectra(
        signal=torch.fft.rfft(signal),
        reference=spectrum,
        frequencies=True
    )
    matched = normalise(matched)
    return match_covariance(
        signal=matched,
        reference=cov,
        cov=True
    )


def synthesise_matched(reference, use_mean=False):
    """
    Create a synthetic signal matched in spectrum and covariance to a
    reference.

    The synthetic signal will be drawn from a stationary Gaussian process.
    After sampling i.i.d. from a standard normal distribution, the synthetic
    data are :func:`transformed <match_reference>` to match the reference.

    :Dimension: **reference :** :math:`(D, N)`
                    D denotes dimension of the multivariate signal and N
                    denotes the number of observations per channel.
                **output :** :math:`(D, N)`
                    As above.

    Parameters
    ----------
    reference : tensor
        Reference signal.
    use_mean : bool (default False)
        Indicates that, instead of matching the spectrum of the corresponding
        channel in the reference signal, all channels are matched to the mean
        spectrum across all reference channels.

    Returns
    -------
    tensor
        Synthetic data matched in spectrum and covariance to the reference.

    See also
    --------
    :func:`synthesise_from_cov_and_spectrum`
    :func:`match_reference`
    """
    synth = torch.randn_like(reference)
    return match_reference(
        signal=synth,
        reference=reference,
        use_mean=use_mean
    )


def synthesise_from_cov_and_spectrum(spectrum, cov, dtype=None, device=None):
    """
    Create a synthetic signal matched in spectrum and covariance to
    references.

    The synthetic signal will be drawn from a stationary Gaussian process.
    After sampling i.i.d. from a standard normal distribution, the synthetic
    data are :func:`transformed <match_cov_and_spectrum>` to match the
    references.

    :Dimension: **spectrum :** :math:`(D, F)`
                    D denotes dimension of the multivariate signal. F denotes
                    the frequency dimension of the signal.
                **cov :** :math:`(D, D)`
                    As above.
                **output :** :math:`D, N`
                    D as above. N is the time dimension of the signal, i.e.,
                    2 * (F - 1).

    Parameters
    ----------
    spectrum : tensor
        Spectrum to be matched.
    cov : tensor
        Covariance matrix to be matched.
    dtype : ``torch.dtype`` (default None)
        Data type of the synthetic dataset.
    device : ``torch.device`` (default None)
        Device on which the synthetic dataset is instantiated.

    See also
    --------
    :func:`synthesise_matched`
    :func:`match_cov_and_spectrum`
    """
    n_ts = cov.size(-1)
    n_obs = 2 * (spectrum.size(-1) - 1)
    synth = torch.randn((n_ts, n_obs), dtype=dtype, device=device)
    return match_cov_and_spectrum(
        signal=synth,
        spectrum=spectrum,
        cov=cov
    )
