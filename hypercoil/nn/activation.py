# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Activation function modules.
"""
from __future__ import annotations
from typing import Literal, Optional, Tuple, Union

import jax
import equinox as eqx

from ..engine import Tensor
from ..functional.activation import (
    corrnorm,
    document_corrnorm,
    document_isochoric,
    isochor,
)


@document_corrnorm
class CorrelationNorm(eqx.Module):
    """
    :doc:`Correlation normalisation <hypercoil.functional.activation.corrnorm>`
    module.
    \
    {corrnorm_spec}
    """

    factor: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None
    grad_path: Literal["input", "both"] = "both"

    def __call__(
        self,
        input: Tensor,
        factor: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> Tensor:
        if factor is None:
            factor = self.factor
        return corrnorm(
            input=input,
            factor=factor,
            gradpath=self.grad_path,
        )


@document_isochoric
class Isochor(eqx.Module):
    """
    :doc:`Isochoric normalisation <hypercoil.functional.activation.isochor>`
    module.
    \
    {isochoric_spec}
    """

    volume: float = 1.0
    max_condition: Optional[float] = None
    softmax_temp: Optional[float] = None

    def __call__(
        self,
        input: Tensor,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> Tensor:
        return isochor(
            input=input,
            volume=self.volume,
            max_condition=self.max_condition,
            softmax_temp=self.softmax_temp,
        )
