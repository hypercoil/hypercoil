# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Batch-dimension correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regularisations to penalise a correlation across the batch dimension.
One example is QC-FC.
"""
import torch
from functools import partial
from ..functional import pairedcorr
from .base import ReducingRegularisation


def auto_tol(batch_size, significance=0.1, tails=2, dtype=None, device=None):
    r"""
    Automatically set the tolerance for batch-dimension correlations based on
    a significance level.

    From the t-value associated with the specified significance level, the
    tolerance is computed as

    :math:`r_{tol} = \sqrt{\frac{t^2}{N - 2 - t^2}}`

    Parameters
    ----------
    batch_size : int
        Number of observations in the batch.
    significance : float in (0, 1) (default 0.1)
        Significance level at which the tolerance should be computed.
    tails : 1 or 2 (default 2)
        Number of tails for the t-test.
    """
    import numpy as np
    from scipy.stats import t
    tsq = t.ppf(q=(1 - significance / tails), df=(batch_size - 2)) ** 2
    return torch.tensor(
        np.sqrt(tsq / (batch_size - 2 + tsq)), dtype=dtype, device=device
    )


def batch_corr(X, N, tol=0, tol_sig=0.1):
    """
    Correlation over the batch dimension.

    Parameters
    ----------
    X : tensor
        Tensor block containing measures to be correlated with those in `N`.
    N : tensor
        Vector of measures with which the measures in `X` are to be
        correlated.
    tol : nonnegative float or `'auto'` (default 0)
        Tolerance for correlations. Only correlation values above `tol` are
        counted. If this is set to `'auto'`, a tolerance is computed for the
        batch size given the significance level in `tol_sig`.
    tol_sig : float in (0, 1)
        Significance level for correlation tolerance. Used only if `tol` is
        set to `'auto'`.

    Returns
    -------
    tensor
        Absolute correlation of each vector in `X` with `N`, after
        thresholding at `tol`. Note that, if you want the original
        correlations back, you will have to add `tol` to any nonzero
        correlations.
    """
    batch_size = X.shape[0]
    batchcorr = pairedcorr(
        X.transpose(0, -1).reshape(-1, batch_size),
        N
    )
    if tol == 'auto':
        tol = auto_tol(batch_size, significance=tol_sig,
                       dtype=batchcorr.dtype, device=batchcorr.device)
    return torch.maximum(
        batchcorr.abs() - tol,
        torch.tensor(0, dtype=batchcorr.dtype, device=batchcorr.device)
    )


class BatchCorrelation(ReducingRegularisation):
    def __init__(self, nu=1, reduction=None, tol=0, tol_sig=0.1):
        reduction = reduction or torch.mean
        reg = partial(batch_corr, tol=tol, tol_sig=tol_sig)
        super(BatchCorrelation, self).__init__(
            nu=nu,
            reduction=reduction,
            reg=reg
        )

    def forward(self, data, measure):
        return self.nu * self.reduction(self.reg(data, measure))


def qcfc_loss(FC, QC, tol=0, tol_sig=0.1):
    return batch_corr(X=FC, N=QC, tol=tol, tol_sig=tol_sig)


class QCFC(BatchCorrelation):
    pass
