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
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Tuple, Union

from ..engine import Tensor
from ..functional import corr_kernel, spherical_geodesic, linear_distance
from .functional import (
    identity,
    difference,
    constraint_violation,
    unilateral_loss,
    hinge_loss,
    smoothness,
    _bimodal_symmetric_impl,
    det_gram,
    log_det_gram,
    entropy,
    entropy_logit,
    kl_divergence,
    kl_divergence_logit,
    js_divergence,
    js_divergence_logit,
    bregman_divergence,
    bregman_divergence_logit,
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
)
from .scalarise import (
    mean_scalarise,
    meansq_scalarise,
    sum_scalarise,
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
        return self.multiplier * self.loss(*pparams, key=key, **params)


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
        return self.multiplier * self.loss(
            *pparams,
            key=key,
            **self.params,
            **params,
        )


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
    """
    :math:`L_p` norm regulariser.
    """
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
        self.p = p
        self.axis = axis
        self.outer_scalarise = outer_scalarise


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
            key=key,
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
            scalarisation=scalarisation or sum_scalarise,
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
        return self.multiplier * self.loss(Y_hat, Y, key=key)


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
            scalarisation=scalarisation or partial(vnorm_scalarise, p=1),
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
            key=key,
        )


class BimodalSymmetricLoss(Loss):
    modes: Tuple[int, int]
    mean: float
    step: float

    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        modes: Tuple[int, int] = (0, 1),
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=_bimodal_symmetric_impl,
            scalarisation=scalarisation or mean_scalarise,
            multiplier=multiplier,
            name=name,
            key=key,
        )
        self.modes = modes
        self.mean = sum(modes) / 2
        self.step = max(modes) - self.mean

    def __call__(
        self,
        X: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.multiplier * self.loss(
            X,
            mean=self.mean,
            step=self.step,
            key=key,
        )


class _GramDeterminantLoss(Loss):
    op: Callable
    theta: Optional[Tensor]
    psi: float
    xi: float

    def __init__(
        self,
        score: Callable,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        op: Callable = corr_kernel,
        theta: Optional[Tensor] = None,
        psi: float = 0.,
        xi: float = 0.,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=score,
            scalarisation=scalarisation or mean_scalarise,
            multiplier=multiplier,
            name=name,
            key=key,
        )
        self.op = op
        self.theta = theta
        self.psi = psi
        self.xi = xi

    def __call__(
        self,
        X: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.multiplier * self.loss(
            X,
            theta=self.theta,
            op=self.op,
            psi=self.psi,
            xi=self.xi,
            key=key,
        )


class GramDeterminantLoss(_GramDeterminantLoss):

    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        op: Callable = corr_kernel,
        theta: Optional[Tensor] = None,
        psi: float = 0.,
        xi: float = 0.,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=det_gram,
            multiplier=multiplier,
            name=name,
            op=op,
            theta=theta,
            psi=psi,
            xi=xi,
            scalarisation=scalarisation,
            key=key,
        )


class GramLogDeterminantLoss(_GramDeterminantLoss):

    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        op: Callable = corr_kernel,
        theta: Optional[Tensor] = None,
        psi: float = 0.,
        xi: float = 0.,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=log_det_gram,
            multiplier=multiplier,
            name=name,
            op=op,
            theta=theta,
            psi=psi,
            xi=xi,
            scalarisation=scalarisation,
            key=key,
        )


class _InformationLoss(Loss):
    axis: Union[int, Tuple[int, ...]]
    keepdims: bool
    reduce: bool

    def __init__(
        self,
        score: Callable,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=score,
            scalarisation=scalarisation or mean_scalarise,
            multiplier=multiplier,
            name=name,
            key=key,
        )
        self.axis = axis
        self.keepdims = keepdims
        self.reduce = reduce

    def __call__(
        self,
        *pparams,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.multiplier * self.loss(
            *pparams,
            axis=self.axis,
            keepdims=self.keepdims,
            reduce=self.reduce,
            key=key,
        )


class EntropyLoss(_InformationLoss):
    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=entropy,
            multiplier=multiplier,
            name=name,
            axis=axis,
            keepdims=keepdims,
            reduce=reduce,
            scalarisation=scalarisation,
            key=key,
        )


class EntropyLogitLoss(_InformationLoss):
    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=entropy_logit,
            multiplier=multiplier,
            name=name,
            axis=axis,
            keepdims=keepdims,
            reduce=reduce,
            scalarisation=scalarisation,
            key=key,
        )


class KLDivergenceLoss(_InformationLoss):
    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=kl_divergence,
            multiplier=multiplier,
            name=name,
            axis=axis,
            keepdims=keepdims,
            reduce=reduce,
            scalarisation=scalarisation,
            key=key,
        )


class KLDivergenceLogitLoss(_InformationLoss):
    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=kl_divergence_logit,
            multiplier=multiplier,
            name=name,
            axis=axis,
            keepdims=keepdims,
            reduce=reduce,
            scalarisation=scalarisation,
            key=key,
        )


class JSDivergenceLoss(_InformationLoss):
    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=js_divergence,
            multiplier=multiplier,
            name=name,
            axis=axis,
            keepdims=keepdims,
            reduce=reduce,
            scalarisation=scalarisation,
            key=key,
        )


class JSDivergenceLogitLoss(_InformationLoss):
    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=js_divergence_logit,
            multiplier=multiplier,
            name=name,
            axis=axis,
            keepdims=keepdims,
            reduce=reduce,
            scalarisation=scalarisation,
            key=key,
        )


class _BregmanDivergenceLoss(Loss):
    f: Callable
    f_dim: int

    def __init__(
        self,
        score: Callable,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        f: Callable,
        f_dim: int,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            multiplier=multiplier,
            name=name,
            score=score,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.f = f
        self.f_dim = f_dim

    def __call__(
        self,
        X: Tensor,
        Y: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.multiplier * self.loss(
            X,
            Y,
            f=self.f,
            f_dim=self.f_dim,
            key=key,
        )


class BregmanDivergenceLoss(_BregmanDivergenceLoss):
    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        f: Callable,
        f_dim: int,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=bregman_divergence,
            multiplier=multiplier,
            name=name,
            f=f,
            f_dim=f_dim,
            scalarisation=scalarisation,
            key=key,
        )


class BregmanDivergenceLogitLoss(_BregmanDivergenceLoss):
    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        f: Callable,
        f_dim: int,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            score=bregman_divergence_logit,
            multiplier=multiplier,
            name=name,
            f=f,
            f_dim=f_dim,
            scalarisation=scalarisation,
            key=key,
        )


class EquilibriumLoss(Loss):
    level_axis: Union[int, Tuple[int, ...]]
    instance_axes: Union[int, Tuple[int, ...]]
    keepdims: bool

    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        level_axis: Union[int, Tuple[int, ...]] = -1,
        instance_axes: Union[int, Tuple[int, ...]] = (-2, -1),
        keepdims: bool = False,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            multiplier=multiplier,
            name=name,
            score=equilibrium,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.level_axis = level_axis
        self.instance_axes = instance_axes
        self.keepdims = keepdims

    def __call__(
        self,
        X: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.multiplier * self.loss(
            X,
            level_axis=self.level_axis,
            instance_axes=self.instance_axes,
            keepdims=self.keepdims,
            key=key,
        )


class EquilibriumLogitLoss(Loss):
    level_axis: Union[int, Tuple[int, ...]]
    prob_axis: Union[int, Tuple[int, ...]]
    instance_axes: Union[int, Tuple[int, ...]]
    keepdims: bool

    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        level_axis: Union[int, Tuple[int, ...]] = -1,
        prob_axis: Union[int, Tuple[int, ...]] = -2,
        instance_axes: Union[int, Tuple[int, ...]] = (-2, -1),
        keepdims: bool = False,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            multiplier=multiplier,
            name=name,
            score=equilibrium_logit,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.level_axis = level_axis
        self.prob_axis = prob_axis
        self.instance_axes = instance_axes
        self.keepdims = keepdims

    def __call__(
        self,
        X: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.multiplier * self.loss(
            X,
            level_axis=self.level_axis,
            prob_axis=self.prob_axis,
            instance_axes=self.instance_axes,
            keepdims=self.keepdims,
            key=key,
        )


class SecondMomentLoss(Loss):
    standardise: bool
    skip_normalise: bool

    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        standardise: bool = False,
        skip_normalise: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            multiplier=multiplier,
            name=name,
            score=second_moment,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.standardise = standardise
        self.skip_normalise = skip_normalise

    def __call__(
        self,
        X: Tensor,
        weight: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.multiplier * self.loss(
            X,
            weight,
            standardise=self.standardise,
            skip_normalise=self.skip_normalise,
            key=key,
        )


class SecondMomentCentredLoss(Loss):
    standardise_data: bool
    standardise_mu: bool
    skip_normalise: bool

    def __init__(
        self,
        multiplier: float = 1.0,
        name: Optional[str] = None,
        *,
        standardise_data: bool = False,
        standardise_mu: bool = False,
        skip_normalise: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = type(self).__name__
        super().__init__(
            multiplier=multiplier,
            name=name,
            score=second_moment_centred,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.standardise_data = standardise_data
        self.standardise_mu = standardise_mu
        self.skip_normalise = skip_normalise

    def __call__(
        self,
        X: Tensor,
        weight: Tensor,
        mu: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.multiplier * self.loss(
            X,
            weight,
            mu,
            standardise_data=self.standardise_data,
            standardise_mu=self.standardise_mu,
            skip_normalise=self.skip_normalise,
            key=key,
        )
