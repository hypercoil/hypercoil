# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Batch-dimension correlation
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Regularisations to penalise a correlation across the batch dimension.
One example is QC-FC.
"""
from torch import mean
from functools import partial
from ..functional import pairedcorr
from .norm import ReducingRegularisation


def batch_corr(X, N, tol=0):
    batch_size = X.shape[0]
    secondordercorr = pairedcorr(
        X.transpose(0, -1).reshape(-1, batch_size),
        N
    )
    return torch.maximum(
        secondordercorr.abs() - tol,
        torch.tensor(0)
    )


class BatchCorrelation(ReducingRegularisation):
    def __init__(self, nu=1, reduction=None, tol=0):
        reduction = reduction or mean
        reg = partial(batch_corr, tol=tol)
        super(BatchCorrelation, self).__init__(
            nu=nu,
            reduction=reduction,
            reg=reg
        )

    def forward(self, data, measure):
        return self.nu * self.reduction(self.reg(data, measure))


def qcfc_loss(FC, QC, tol=0):
    return batch_corr(X=FC, N=QC, tol=0)


class QCFC(BatchCorrelation):
    pass
