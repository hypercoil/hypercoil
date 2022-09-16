# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Loss functions as parameterised, callable functional objects.

A loss function is the composition of a score function and a scalarisation
map (which might itself be the composition of different tensor rank reduction
maps). It also includes a multiplier that can be used to scale its
contribution to the overall loss.
"""
import jax
import equinox as eqx
from functools import partial
from types import MappingProxyType
from typing import Callable, Mapping, Optional, Sequence, Union

from ..engine import Tensor
from .functional import (
    identity,
    difference,
    constraint_violation,
    unilateral_loss,
    hinge_loss,
    smoothness,
    bimodal_symmetric,
    det_gram,
    log_det_gram,
    entropy,
    entropy_logit,
    kl_divergence,
    kl_divergence_logit,
    js_divergence,
    js_divergence_logit,
    bregman_divergence,
    equilibrium,
    equilibrium_logit,
    second_moment,
    second_moment_centred,
    batch_corr,
    reference_tether,
    interhemispheric_tether,
    compactness,
    dispersion,
    multivariate_kurtosis,
    connectopy,
    modularity,
    eigenmaps,
)
from .scalarise import (
    mean_scalarise,
    meansq_scalarise,
    vnorm_scalarise,
)


class Loss(eqx.Module):
    name: str
    multiplier: float
    score: Callable
    scalarisation: Callable
    loss: Callable

    def __init__(
        self,
        score: Callable,
        scalarisation: Callable,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):

        if name is None:
            name = ''.join(i.title() for i in score.__name__.split('_'))
            name = f'{name}Loss'
        self.name = name

        self.score = score
        self.scalarisation = scalarisation
        self.multiplier = multiplier
        self.loss = self.scalarisation(self.score)

    def __call__(
        self,
        *pparams,
        key: Optional['jax.random.PRNGKey'] = None,
        **params
    ) -> float:
        return self.multiplier * self.loss(*pparams, **params)


class ParameterisedLoss(Loss):
    params: MappingProxyType

    def __init__(
        self,
        score: Callable,
        scalarisation: Callable,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        params: Optional[Mapping] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):

        super().__init__(
            score=score,
            scalarisation=scalarisation,
            multiplier=multiplier,
            name=name,
            key=key,
        )

        if params is None:
            params = {}
        self.params = MappingProxyType(params.copy())

    def __call__(
        self,
        *pparams,
        key: Optional['jax.random.PRNGKey'] = None,
        **params
    ) -> float:
        return self.multiplier * self.loss(*pparams, **self.params, **params)


class MSELoss(Loss):
    """
    An example of how to compose elements to define a loss function.

    There are probably better implementations of the mean squared error loss
    out there.
    """
    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__

        super().__init__(
            score=difference,
            scalarisation=meansq_scalarise,
            multiplier=multiplier,
            name=name,
            key=key,
        )


class NormedLoss(Loss):
    p: float
    axis: Union[int, Sequence[int]]
    outer_scalarise: Callable

    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        score: Callable = identity,
        *,
        p: float = 2.0,
        axis: Union[int, Sequence[int]] = -1,
        outer_scalarise: Callable = mean_scalarise,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        scalarisation = partial(
            vnorm_scalarise,
            p=p,
            axis=axis,
            outer_scalarise=outer_scalarise,
        )
        super().__init__(
            score=score,
            scalarisation=scalarisation,
            multiplier=multiplier,
            name=name,
            key=key,
        )


class ConstraintViolationLoss(Loss):
    constraints: Sequence[Callable]
    broadcast_against_input: bool

    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        constraints: Sequence[Callable],
        broadcast_against_input: bool = False,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=constraint_violation,
            scalarisation=scalarisation or mean_scalarise,
            multiplier=multiplier,
            name=name,
            key=key,
        )
        self.constraints = constraints
        self.broadcast_against_input = broadcast_against_input

    def __call__(
        self,
        X: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.multiplier * self.loss(
            X,
            constraints=self.constraints,
            broadcast_against_input=self.broadcast_against_input,
        )


class UnilateralLoss(Loss):
    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=unilateral_loss,
            scalarisation=scalarisation or mean_scalarise,
            multiplier=multiplier,
            name=name,
            key=key,
        )


class HingeLoss(Loss):
    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=hinge_loss,
            scalarisation=scalarisation or mean_scalarise,
            multiplier=multiplier,
            name=name,
            key=key,
        )

    def __call__(
        self,
        Y_hat: Tensor,
        Y: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.multiplier * self.loss(Y_hat, Y)


class SmoothnessLoss(Loss):
    n: int
    pad_value: Optional[float]
    axis: int

    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        n: int = 1,
        pad_value: Optional[float] = None,
        axis: int = -1,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=smoothness,
            scalarisation=scalarisation or mean_scalarise,
            multiplier=multiplier,
            name=name,
            key=key,
        )
        
        self.n = n
        self.pad_value = pad_value
        self.axis = axis

    def __call__(
        self,
        X: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.multiplier * self.loss(
            X,
            n=self.n,
            pad_value=self.pad_value,
            axis=self.axis,
        )
