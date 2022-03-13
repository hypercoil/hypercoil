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
from .norm import ReducingRegularisation


def auto_tol(batch_size, significance=0.1, tails=2):
    import numpy as np
    from scipy.stats import t
    tsq = t.ppf(q=(1 - significance / tails), df=(batch_size - 2)) ** 2
    return torch.tensor(np.sqrt(tsq / (batch_size - 2 + tsq)))


def batch_corr(X, N, tol=0, tol_sig=0.1):
    batch_size = X.shape[0]
    batchcorr = pairedcorr(
        X.transpose(0, -1).reshape(-1, batch_size),
        N
    )
    if tol == 'auto':
        tol = auto_tol(batch_size, significance=tol_sig)
    return torch.maximum(
        batchcorr.abs() - tol,
        torch.tensor(0)
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
