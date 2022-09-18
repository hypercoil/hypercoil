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
    nu: float
    score: Callable
    scalarisation: Callable
    loss: Callable

    def __init__(
        self,
        score: Callable,
        scalarisation: Callable,
        nu: float = 1.0,
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
        self.nu = nu
        self.loss = self.scalarisation(self.score)

    def __call__(
        self,
        *pparams,
        key: Optional['jax.random.PRNGKey'] = None,
        **params
    ) -> float:
        return self.nu * self.loss(*pparams, key=key, **params)


class ParameterisedLoss(Loss):
    params: MappingProxyType

    def __init__(
        self,
        score: Callable,
        scalarisation: Callable,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        params: Optional[Mapping] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):

        super().__init__(
            score=score,
            scalarisation=scalarisation,
            nu=nu,
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
        return self.nu * self.loss(
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'MSELoss'

        super().__init__(
            score=difference,
            scalarisation=meansq_scalarise,
            nu=nu,
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
        nu: float = 1.0,
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
            nu=nu,
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        constraints: Sequence[Callable],
        broadcast_against_input: bool = False,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'ConstraintViolation'
        super().__init__(
            score=constraint_violation,
            scalarisation=scalarisation or mean_scalarise,
            nu=nu,
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
        return self.nu * self.loss(
            X,
            constraints=self.constraints,
            broadcast_against_input=self.broadcast_against_input,
            key=key,
        )


class UnilateralLoss(Loss):
    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'UnilateralLoss'
        super().__init__(
            score=unilateral_loss,
            scalarisation=scalarisation or mean_scalarise,
            nu=nu,
            name=name,
            key=key,
        )


class HingeLoss(Loss):
    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'HingeLoss'
        super().__init__(
            score=hinge_loss,
            scalarisation=scalarisation or sum_scalarise,
            nu=nu,
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
        return self.nu * self.loss(Y_hat, Y, key=key)


class SmoothnessLoss(Loss):
    n: int
    pad_value: Optional[float]
    axis: int

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        n: int = 1,
        pad_value: Optional[float] = None,
        axis: int = -1,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'Smoothness'
        super().__init__(
            score=smoothness,
            scalarisation=scalarisation or partial(vnorm_scalarise, p=1),
            nu=nu,
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
        return self.nu * self.loss(
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        modes: Tuple[int, int] = (0, 1),
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'BimodalSymmetric'
        super().__init__(
            score=_bimodal_symmetric_impl,
            scalarisation=scalarisation or mean_scalarise,
            nu=nu,
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
        return self.nu * self.loss(
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        op: Callable = corr_kernel,
        theta: Optional[Tensor] = None,
        psi: float = 0.,
        xi: float = 0.,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        super().__init__(
            score=score,
            scalarisation=scalarisation or mean_scalarise,
            nu=nu,
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
        return self.nu * self.loss(
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        op: Callable = corr_kernel,
        theta: Optional[Tensor] = None,
        psi: float = 0.,
        xi: float = 0.,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'GramDeterminant'
        super().__init__(
            score=det_gram,
            nu=nu,
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        op: Callable = corr_kernel,
        theta: Optional[Tensor] = None,
        psi: float = 0.,
        xi: float = 0.,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'GramLogDeterminant'
        super().__init__(
            score=log_det_gram,
            nu=nu,
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        super().__init__(
            score=score,
            scalarisation=scalarisation or mean_scalarise,
            nu=nu,
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
        return self.nu * self.loss(
            *pparams,
            axis=self.axis,
            keepdims=self.keepdims,
            reduce=self.reduce,
            key=key,
        )


class EntropyLoss(_InformationLoss):
    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'Entropy'
        super().__init__(
            score=entropy,
            nu=nu,
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'EntropyLogit'
        super().__init__(
            score=entropy_logit,
            nu=nu,
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'KLDivergence'
        super().__init__(
            score=kl_divergence,
            nu=nu,
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'KLDivergenceLogit'
        super().__init__(
            score=kl_divergence_logit,
            nu=nu,
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'JSDivergence'
        super().__init__(
            score=js_divergence,
            nu=nu,
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        axis: Union[int, Tuple[int, ...]] = -1,
        keepdims: bool = False,
        reduce: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'JSDivergenceLogit'
        super().__init__(
            score=js_divergence_logit,
            nu=nu,
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        f: Callable,
        f_dim: int,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        super().__init__(
            nu=nu,
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
        return self.nu * self.loss(
            X,
            Y,
            f=self.f,
            f_dim=self.f_dim,
            key=key,
        )


class BregmanDivergenceLoss(_BregmanDivergenceLoss):
    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        f: Callable,
        f_dim: int,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'BregmanDivergence'
        super().__init__(
            score=bregman_divergence,
            nu=nu,
            name=name,
            f=f,
            f_dim=f_dim,
            scalarisation=scalarisation,
            key=key,
        )


class BregmanDivergenceLogitLoss(_BregmanDivergenceLoss):
    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        f: Callable,
        f_dim: int,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'BregmanDivergenceLogit'
        super().__init__(
            score=bregman_divergence_logit,
            nu=nu,
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        level_axis: Union[int, Tuple[int, ...]] = -1,
        instance_axes: Union[int, Tuple[int, ...]] = (-2, -1),
        keepdims: bool = False,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'Equilibrium'
        super().__init__(
            nu=nu,
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
        return self.nu * self.loss(
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        level_axis: Union[int, Tuple[int, ...]] = -1,
        prob_axis: Union[int, Tuple[int, ...]] = -2,
        instance_axes: Union[int, Tuple[int, ...]] = (-2, -1),
        keepdims: bool = False,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'EquilibriumLogit'
        super().__init__(
            nu=nu,
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
        return self.nu * self.loss(
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        standardise: bool = False,
        skip_normalise: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'SecondMoment'
        super().__init__(
            nu=nu,
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
        return self.nu * self.loss(
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
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        standardise_data: bool = False,
        standardise_mu: bool = False,
        skip_normalise: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'SecondMomentCentred'
        super().__init__(
            nu=nu,
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
        return self.nu * self.loss(
            X,
            weight,
            mu,
            standardise_data=self.standardise_data,
            standardise_mu=self.standardise_mu,
            skip_normalise=self.skip_normalise,
            key=key,
        )


class BatchCorrelationLoss(Loss):
    tol: float
    tol_sig: float
    abs: bool

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        tol: Union[float, Literal['auto']] = 0,
        tol_sig: float = 0.1,
        abs: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'BatchCorrelation'
        super().__init__(
            nu=nu,
            name=name,
            score=batch_corr,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.tol = tol
        self.tol_sig = tol_sig
        self.abs = abs

    def __call__(
        self,
        X: Tensor,
        N: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.nu * self.loss(
            X,
            N,
            tol=self.tol,
            tol_sig=self.tol_sig,
            abs=self.abs,
            key=key,
        )


class QCFCLoss(BatchCorrelationLoss):
    def __call__(
        self,
        FC: Tensor,
        QC: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return super().__call__(
            X=FC,
            N=QC,
            key=key,
        )


class ReferenceTetherLoss(Loss):
    ref: Optional[Tensor]
    coor: Optional[Tensor]
    radius: float

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        ref: Optional[Tensor] = None,
        coor: Optional[Tensor] = None,
        radius: Optional[float] = 100.,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'ReferenceTether'
        super().__init__(
            nu=nu,
            name=name,
            score=reference_tether,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.ref = ref
        self.coor = coor
        self.radius = radius

    def __call__(
        self,
        X: Tensor,
        ref: Optional[Tensor] = None,
        coor: Optional[Tensor] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        if ref is None: ref = self.ref
        if coor is None: coor = self.coor
        return self.nu * self.loss(
            X,
            ref=ref,
            coor=coor,
            radius=self.radius,
            key=key,
        )


class InterhemisphericTetherLoss(Loss):
    lh_coor: Optional[Tensor]
    rh_coor: Optional[Tensor]
    radius: float

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        lh_coor: Optional[Tensor] = None,
        rh_coor: Optional[Tensor] = None,
        radius: Optional[float] = 100.,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'InterhemisphericTether'
        super().__init__(
            nu=nu,
            name=name,
            score=interhemispheric_tether,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.lh_coor = lh_coor
        self.rh_coor = rh_coor
        self.radius = radius

    def __call__(
        self,
        lh: Tensor,
        rh: Tensor,
        lh_coor: Optional[Tensor] = None,
        rh_coor: Optional[Tensor] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        if lh_coor is None: lh_coor = self.lh_coor
        if rh_coor is None: rh_coor = self.rh_coor
        return self.nu * self.loss(
            lh=lh,
            rh=rh,
            lh_coor=lh_coor,
            rh_coor=rh_coor,
            radius=self.radius,
            key=key,
        )


class CompactnessLoss(Loss):
    coor: Optional[Tensor]
    norm: Union[int, float, Literal['inf']]
    floor: float
    radius: float

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        coor: Optional[Tensor] = None,
        radius: Optional[float] = 100.,
        norm: Union[int, float, Literal['inf']] = 2,
        floor: float = 0.0,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'Compactness'
        super().__init__(
            nu=nu,
            name=name,
            score=compactness,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.coor = coor
        self.norm = norm
        self.floor = floor
        self.radius = radius

    def __call__(
        self,
        X: Tensor,
        coor: Optional[Tensor] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        if coor is None: coor = self.coor
        return self.nu * self.loss(
            X,
            coor=coor,
            norm=self.norm,
            floor=self.floor,
            radius=self.radius,
            key=key,
        )


class DispersionLoss(Loss):
    metric: Callable

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        metric: Callable = spherical_geodesic,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'Dispersion'
        super().__init__(
            nu=nu,
            name=name,
            score=dispersion,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.metric = metric

    def __call__(
        self,
        X: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.nu * self.loss(
            X,
            metric=self.metric,
            key=key,
        )


class MultivariateKurtosis(Loss):
    l2: float
    dimensional_scaling: bool

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        l2: float = 0.0,
        dimensional_scaling: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'MultivariateKurtosis'
        super().__init__(
            nu=nu,
            name=name,
            score=multivariate_kurtosis,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.l2 = l2
        self.dimensional_scaling = dimensional_scaling

    def __call__(
        self,
        X: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        return self.nu * self.loss(
            X,
            l2=self.l2,
            dimensional_scaling=self.dimensional_scaling,
            key=key,
        )


class ConnectopyLoss(Loss):
    theta: Optional[Any]
    omega: Optional[Any]
    dissimilarity: Callable
    affinity: Optional[Callable]
    progressive_theta: bool

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        theta: Optional[Any] = None,
        omega: Optional[Any] = None,
        dissimilarity: Optional[Callable] = None,
        affinity: Optional[Callable] = None,
        scalarisation: Optional[Callable] = None,
        progressive_theta: bool = False,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'Connectopy'
        super().__init__(
            nu=nu,
            name=name,
            score=connectopy,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.theta = theta
        self.omega = omega
        self.dissimilarity = dissimilarity or linear_distance
        self.affinity = affinity
        self.progressive_theta = progressive_theta

    def __call__(
        self,
        Q: Tensor,
        A: Tensor,
        D: Optional[Tensor] = None,
        theta: Optional[Any] = None,
        omega: Optional[Any] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        if theta is None: theta = self.theta
        if omega is None: omega = self.omega
        return self.nu * self.loss(
            Q=Q,
            A=A,
            D=D,
            theta=theta,
            omega=omega,
            dissimilarity=self.dissimilarity,
            affinity=self.affinity,
            progressive_theta=self.progressive_theta,
            key=key,
        )


class ModularityLoss(Loss):
    theta: Optional[Tensor]
    gamma: float
    exclude_diag: bool

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        theta: Optional[Tensor] = None,
        gamma: float = 1.0,
        exclude_diag: bool = True,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'Modularity'
        super().__init__(
            nu=nu,
            name=name,
            score=modularity,
            scalarisation=scalarisation or mean_scalarise,
            key=key,
        )
        self.theta = theta
        self.gamma = gamma
        self.exclude_diag = exclude_diag

    def __call__(
        self,
        Q: Tensor,
        A: Tensor,
        D: Optional[Tensor] = None,
        theta: Optional[Tensor] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> float:
        if theta is None: theta = self.theta
        return self.nu * self.loss(
            Q=Q,
            A=A,
            D=D,
            theta=theta,
            gamma=self.gamma,
            exclude_diag=self.exclude_diag,
            key=key,
        )
