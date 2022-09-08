# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Residualise tensor block via least squares. No parameters here.

.. warning::
    In some cases, we have found that the least-squares fit returned is
    incorrect for reasons that are not clear. (Incorrect results are
    returned by
    ``torch.linalg.lstsq``, although correct results are returned if
    ``torch.linalg.pinv`` is used instead.) Verify that results are
    reasonable when using this operation.
"""
from lib2to3.pgen2.token import OP
from typing import Optional
from equinox import Module
from ..engine import Tensor
from ..functional.resid import residualise


#TODO: assess backprop properties of this approach vs conditional correlation
#TODO: Do we really need this, or can we just use eqx.nn.Lambda? Turns out we
#      can't without making it ugly, so we'll keep this for now.
class Residualise(Module):
    """
    .. warning::
        In some cases, we have found that the least-squares fit returned is
        incorrect for reasons that are not clear. (Incorrect results are
        returned by
        ``torch.linalg.lstsq``, although correct results are returned if
        ``torch.linalg.pinv`` is used instead.) Verify that results are
        reasonable when using this operation.
    """
    rowvar: bool = True
    l2: float = 0.0

    def __call__(
        self,
        Y: Tensor,
        X: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        if mask is not None:
            Y = mask * Y
            X = mask * X
        return residualise(Y=Y, X=X, l2=self.l2, rowvar=self.rowvar)
