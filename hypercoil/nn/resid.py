# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Residualise tensor block via least squares. No parameters here.
"""
from __future__ import annotations
from typing import Literal, Optional

import jax
from equinox import Module

from ..engine import Tensor
from ..functional.resid import document_linreg, residualise


# TODO: assess backprop properties of this approach vs conditional correlation
# TODO: Do we really need this, or can we just use eqx.nn.Lambda? Turns out we
#      can't without making it ugly, so we'll keep this for now.
@document_linreg
class Residualise(Module):
    """
    Residualise a tensor block via ordinary linear least squares.
    \
    {regress_warning}

    Parameters
    ----------\
    {regress_param_spec}
    """

    rowvar: bool = True
    l2: float = 0.0
    return_mode: Literal["residual", "orthogonal"] = "residual"

    def __call__(
        self,
        Y: Tensor,
        X: Tensor,
        mask: Optional[Tensor] = None,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> Tensor:
        if mask is not None:
            Y = mask * Y
            X = mask * X
        return residualise(
            Y=Y,
            X=X,
            l2=self.l2,
            rowvar=self.rowvar,
            return_mode=self.return_mode,
        )
