# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Loss functions as parameterised, callable functional objects.
"""
import jax
import equinox as eqx
from functools import partial
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Tuple, Union

from ..engine import NestedDocParse, Tensor
from ..functional import corr_kernel, spherical_geodesic, linear_distance
from .functional import (
    document_batch_correlation,
    document_bimodal_symmetric,
    document_bregman,
    document_connectopy,
    document_constraint_violation,
    document_entropy,
    document_equilibrium,
    document_gramian_determinant,
    document_mv_kurtosis,
    document_second_moment,
    document_smoothness,
    document_spatial_loss,
)
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


def document_loss(default_scalarise: str = "mean"):
    param_spec = """
    Parameters
    ----------
    name: str
        Designated name of the loss function. It is not required that this be
        specified, but it is recommended to ensure that the loss function can
        be identified in the context of a reporting utilities. If not
        explicitly specified, the name will be inferred from the class name
        and the name of the scoring function.
    nu: float
        Loss strength multiplier. This is a scalar multiplier that is applied
        to the loss value before it is returned. This can be used to
        modulate the relative contributions of different loss functions to
        the overall loss value. It can also be used to implement a
        schedule for the loss function, by dynamically adjusting the
        multiplier over the course of training."""
    score_spec = """
    score: Callable
        The scoring function to be used to compute the loss value. This
        function should take a single argument, which is a tensor of
        arbitrary shape, and return a score value for each (potentially
        multivariate) observation in the tensor."""
    scalarisation_spec = f"""
    scalarisation: Callable
        The scalarisation function to be used to aggregate the values
        returned by the scoring function. This function should take a
        single argument, which is a tensor of arbitrary shape, and return
        a single scalar value. By default, the {default_scalarise}
        scalarisation is used."""

    fmt = NestedDocParse(
        param_spec=param_spec,
        score_spec=score_spec,
        scalarisation_spec=scalarisation_spec,
    )

    def _doc_transform(cls):
        cls.__doc__ = cls.__doc__.format_map(fmt)
        return cls
    return _doc_transform


@document_loss()
class Loss(eqx.Module):
    """
    Base class for loss functions.

    A loss function is the composition of a score function and a scalarisation
    map (which might itself be the composition of different tensor rank
    reduction maps). It also includes a multiplier that can be used to scale
    its contribution to the overall loss. The multiplier is specified using
    the ``nu`` parameter.

    The API vis-a-vis dimension reduction is subject to change. We will likely
    make scalarisations more flexible with regard to both compositionality and
    the number/specification of dimensions they reduce to.
    \
    {param_spec}\
    {score_spec}\
    {scalarisation_spec}
    """

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

    def __repr__(self) -> str:
        return f'[ν = {self.nu}]{self.name}'


@document_loss()
class ParameterisedLoss(Loss):
    """
    Extensible class for loss functions with simple parameterisations.

    This class is intended to be used as a base class for loss functions that
    have a simple parameterisation, i.e. a fixed set of parameters that are
    passed to the scoring function. The parameters are specified using the
    ``params`` argument, which should be a mapping from parameter names to
    values. Note that the class is immutable, so the parameters cannot be
    changed after the class has been instantiated.
    \
    {param_spec}\
    {score_spec}\
    {scalarisation_spec}
    params: Mapping[str, Any]
        A mapping from parameter names to values. These will be passed to the
        scoring function when the loss function is called.
    """

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


@document_loss()
class MSELoss(Loss):
    """
    Mean squared error loss function.

    An example of how to compose elements to define a loss function. The
    score function is the difference between the input and the target, and
    the scalarisation function is the mean of squared values.

    There are probably better implementations of the mean squared error loss
    out there.
    \
    {param_spec}
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
            scalarisation=meansq_scalarise(),
            nu=nu,
            name=name,
            key=key,
        )

    def __call__(
        self,
        Y: Tensor,
        Y_hat: Tensor,
        key: Optional['jax.random.PRNGKey'] = None
    ) -> float:
        return super().__call__(X=Y, Y=Y_hat)


@document_loss()
class NormedLoss(Loss):
    """
    :math:`L_p` norm regulariser.

    An example of how to compose elements to define a loss function. By
    default, this function flattens the input tensor and computes the
    :math:`L_2` norm of the resulting vector. The dimensions to be flattened
    and the norm order can be specified using the ``axis`` and ``p`` arguments
    respectively. If the norm is computed over only a subset of axes, the
    remaining axes can be further reduced by specifying a scalarisation
    function using the ``outer_scalarise`` argument. By default, the outer
    scalarisation function is the mean function. Setting this to an identity
    function will result in a loss function that returns a vector of values
    for each observation.
    \
    {param_spec}\
    {score_spec}\
    p: float
        The order of the norm to be computed. If ``p = 1``, the function
        computes the :math:`L_1` Manhattan / city block norm. If ``p = 2``,
        the function computes the :math:`L_2` Euclidean norm. If ``p = inf``,
        the function computes the :math:`L_\infty` maximum norm.
    axis: Optional[Union[int, Sequence[int]]]
        The axes to be flattened. If ``None``, all axes are flattened.
    outer_scalarise: Optional[Callable]
        The scalarisation function to be applied to any dimensions that are
        not flattened (i.e., those not specified in ``axis``). If ``None``,
        the mean function is used. If ``axis`` is ``None``, this argument is
        ignored. To return a vector of values for each observation, explicitly
        set this to an identity function.
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
        axis: Union[int, Sequence[int]] = None,
        outer_scalarise: Callable = mean_scalarise,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        scalarisation = outer_scalarise(
            inner=vnorm_scalarise(p=p, axis=axis)
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


@document_constraint_violation
@document_loss()
class ConstraintViolationLoss(Loss):
    """
    Loss function for constraint violations.
    \
    {long_description}
    \
    {param_spec}\
    {constraint_violation_spec}\
    {scalarisation_spec}
    """

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
            scalarisation=scalarisation or mean_scalarise(),
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


@document_constraint_violation
@document_loss()
class UnilateralLoss(Loss):
    """
    Loss function corresponding to a single soft nonpositivity constraint.
    \
    {long_description_unil}
    \
    {param_spec}\
    {scalarisation_spec}
    """

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
            scalarisation=scalarisation or mean_scalarise(),
            nu=nu,
            name=name,
            key=key,
        )


@document_constraint_violation
@document_loss(default_scalarise='sum')
class HingeLoss(Loss):
    """
    Hinge loss function.
    \
    {long_description_hinge}
    \
    {param_spec}\
    {scalarisation_spec}
    """

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
            scalarisation=scalarisation or sum_scalarise(),
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


@document_smoothness
@document_loss(default_scalarise='L1 norm')
class SmoothnessLoss(Loss):
    """
    Smoothness loss function.
    \
    {long_description}
    \
    {param_spec}\
    {smoothness_spec}\
    {scalarisation_spec}
    """

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
            scalarisation=scalarisation or vnorm_scalarise(p=1, axis=None),
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


@document_bimodal_symmetric
@document_loss()
class BimodalSymmetricLoss(Loss):
    """
    Loss based on the minimum distance from either of two modes.
    \
    {long_description}
    \
    {param_spec}\
    {bimodal_symmetric_spec}\
    {scalarisation_spec}
    """

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
            scalarisation=scalarisation or mean_scalarise(),
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
    """Base implementation for Gram determinant loss functions."""

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
            scalarisation=scalarisation or mean_scalarise(),
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


@document_gramian_determinant
@document_loss()
class GramDeterminantLoss(_GramDeterminantLoss):
    """
    Loss based on the determinant of the Gram matrix.
    \
    {long_description}
    \
    {param_spec}\
    {det_gram_spec}\
    {scalarisation_spec}
    """

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


@document_gramian_determinant
@document_loss()
class GramLogDeterminantLoss(_GramDeterminantLoss):
    """
    Loss based on the log-determinant of the Gram matrix.
    \
    {long_description}
    {ultra_long_description}
    \
    {param_spec}\
    {det_gram_spec}\
    {scalarisation_spec}
    """

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
    """Base implementation for loss functions based on information theoretic
    quantities."""

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
            scalarisation=scalarisation or mean_scalarise(),
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


@document_entropy
@document_loss()
class EntropyLoss(_InformationLoss):
    """
    Loss based on the entropy of a categorical distribution.

    This operates on probability tensors. For a version that operates on
    logits, see :class:`EntropyLogitLoss`.
    \
    {entropy_long_description}
    \
    {param_spec}\
    {entropy_spec}\
    {scalarisation_spec}
    """
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


@document_entropy
@document_loss()
class EntropyLogitLoss(_InformationLoss):
    """
    Loss based on the entropy of a categorical distribution.

    This operates on logit tensors. For a version that operates on
    probabilities, see :class:`EntropyLoss`.
    \
    {entropy_long_description}
    \
    {param_spec}\
    {entropy_spec}\
    {scalarisation_spec}
    """
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


@document_entropy
@document_loss()
class KLDivergenceLoss(_InformationLoss):
    """
    Loss based on the Kullback-Leibler divergence between two categorical
    distributions.

    This operates on probability tensors. For a version that operates on
    logits, see :class:`KLDivergenceLogitLoss`.
    \
    {kl_long_description}
    \
    {param_spec}\
    {kl_spec}\
    {scalarisation_spec}
    """
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


@document_entropy
@document_loss()
class KLDivergenceLogitLoss(_InformationLoss):
    """
    Loss based on the Kullback-Leibler divergence between two categorical
    distributions.

    This operates on logit tensors. For a version that operates on
    probabilities, see :class:`KLDivergenceLoss`.
    \
    {kl_long_description}
    \
    {param_spec}\
    {kl_spec}\
    {scalarisation_spec}
    """
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


@document_entropy
@document_loss()
class JSDivergenceLoss(_InformationLoss):
    """
    Loss based on the Jensen-Shannon divergence between two categorical
    distributions.

    This operates on probability tensors. For a version that operates on
    logits, see :class:`JSDivergenceLogitLoss`.
    \
    {js_long_description}
    \
    {param_spec}\
    {js_spec}\
    {scalarisation_spec}
    """
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


@document_entropy
@document_loss()
class JSDivergenceLogitLoss(_InformationLoss):
    """
    Loss based on the Jensen-Shannon divergence between two categorical
    distributions.

    This operates on logit tensors. For a version that operates on
    probabilities, see :class:`JSDivergenceLoss`.
    \
    {js_long_description}
    \
    {param_spec}\
    {js_spec}\
    {scalarisation_spec}
    """
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
    """
    Base class for Bregman divergence losses.
    """

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
            scalarisation=scalarisation or mean_scalarise(),
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


@document_bregman
@document_loss()
class BregmanDivergenceLoss(_BregmanDivergenceLoss):
    """
    Loss based on the Bregman divergence between two categorical
    distributions.

    This operates on unmapped tensors. For a version that operates on logits
    logits, see :class:`BregmanDivergenceLogitLoss`.
    \
    {long_description}
    \
    {param_spec}\
    {bregman_spec}\
    {scalarisation_spec}
    """

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


@document_bregman
@document_loss()
class BregmanDivergenceLogitLoss(_BregmanDivergenceLoss):
    """
    Loss based on the Bregman divergence between two categorical
    distributions.

    This operates on logits. For a version that operates on unmapped
    probabilities, see :class:`BregmanDivergenceLoss`.
    \
    {long_description}
    \
    {param_spec}\
    {bregman_spec}\
    {scalarisation_spec}
    """
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


@document_equilibrium
@document_loss(default_scalarise='mean square')
class EquilibriumLoss(Loss):
    """
    Mass equilibrium loss.

    This loss operates on unmapped mass tensors. For a version that operates
    on logits, see :class:`EquilibriumLogitLoss`.
    \
    {long_description}
    \
    {ultra_long_description}
    \
    {param_spec}\
    {equilibrium_spec}\
    {scalarisation_spec}
    """

    level_axis: Union[int, Tuple[int, ...]]
    instance_axes: Union[int, Tuple[int, ...]]

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        level_axis: Union[int, Tuple[int, ...]] = -1,
        instance_axes: Union[int, Tuple[int, ...]] = (-2, -1),
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'Equilibrium'
        super().__init__(
            nu=nu,
            name=name,
            score=equilibrium,
            scalarisation=scalarisation or meansq_scalarise(),
            key=key,
        )
        self.level_axis = level_axis
        self.instance_axes = instance_axes

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
            key=key,
        )


@document_equilibrium
@document_loss()
class EquilibriumLogitLoss(Loss):
    """
    Mass equilibrium loss.

    This loss operates on logits. For a version that operates on unmapped
    mass tensors, see :class:`EquilibriumLoss`.
    \
    {long_description}
    \
    {ultra_long_description}
    \
    {param_spec}\
    {equilibrium_spec}\
    {scalarisation_spec}
    """

    level_axis: Union[int, Tuple[int, ...]]
    prob_axis: Union[int, Tuple[int, ...]]
    instance_axes: Union[int, Tuple[int, ...]]

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        level_axis: Union[int, Tuple[int, ...]] = -1,
        prob_axis: Union[int, Tuple[int, ...]] = -2,
        instance_axes: Union[int, Tuple[int, ...]] = (-2, -1),
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'EquilibriumLogit'
        super().__init__(
            nu=nu,
            name=name,
            score=equilibrium_logit,
            scalarisation=scalarisation or mean_scalarise(),
            key=key,
        )
        self.level_axis = level_axis
        self.prob_axis = prob_axis
        self.instance_axes = instance_axes

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
            key=key,
        )


@document_second_moment
@document_loss()
class SecondMomentLoss(Loss):
    """
    Second moment loss.
    \
    {long_description}
    \
    {ultra_long_description}
    \
    {param_spec}\
    {std_spec_nomean}\
    {second_moment_spec}\
    {scalarisation_spec}
    """

    standardise: bool
    skip_normalise: bool

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        standardise: bool = False,
        skip_normalise: bool = False,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'SecondMoment'
        super().__init__(
            nu=nu,
            name=name,
            score=second_moment,
            scalarisation=scalarisation or mean_scalarise(),
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


@document_second_moment
@document_loss()
class SecondMomentCentredLoss(Loss):
    """
    Second moment loss centred on a precomputed mean.
    \
    {long_description}
    \
    {ultra_long_description}
    \
    {param_spec}\
    {std_spec_nomean}\
    {second_moment_spec}\
    {scalarisation_spec}
    """

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
        skip_normalise: bool = False,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'SecondMomentCentred'
        super().__init__(
            nu=nu,
            name=name,
            score=second_moment_centred,
            scalarisation=scalarisation or mean_scalarise(),
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


@document_batch_correlation
@document_loss()
class BatchCorrelationLoss(Loss):
    """
    Batch correlation loss.
    \
    {param_spec}\
    {batch_correlation_spec}\
    {scalarisation_spec}
    """

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
            scalarisation=scalarisation or mean_scalarise(),
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


@document_batch_correlation
@document_loss()
class QCFCLoss(BatchCorrelationLoss):
    """
    QC-FC loss.
    \
    {param_spec}\
    {batch_correlation_spec}\
    {scalarisation_spec}
    """
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


@document_spatial_loss
@document_loss()
class ReferenceTetherLoss(Loss):
    """
    Loss function penalising distance from a tethered reference point.
    \
    {param_spec}\
    {spatial_loss_spec}\
    {scalarisation_spec}
    """

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
            scalarisation=scalarisation or mean_scalarise(),
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


@document_spatial_loss
@document_loss()
class InterhemisphericTetherLoss(Loss):
    """
    Loss function penalising distance between matched parcels or objects on
    opposite hemispheres.
    \
    {interhemispheric_long_description}
    \
    {param_spec}\
    {interhemispheric_tether_spec}\
    {scalarisation_spec}
    """

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
            scalarisation=scalarisation or mean_scalarise(),
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


@document_spatial_loss
@document_loss()
class CompactnessLoss(Loss):
    """
    Loss function penalising distances between locations in a mass and the
    centroid of that mass.
    \
    {compactness_long_description}
    \
    {param_spec}\
    {compactness_spec}\
    {scalarisation_spec}
    """

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
            scalarisation=scalarisation or mean_scalarise(),
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


@document_spatial_loss
@document_loss()
class DispersionLoss(Loss):
    """
    Loss function penalising proximity between vectors.
    \
    {dispersion_long_description}

    {param_spec}\
    {dispersion_spec}\
    {scalarisation_spec}
    """

    metric: Callable

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        metric: Callable = linear_distance,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'Dispersion'
        super().__init__(
            nu=nu,
            name=name,
            score=dispersion,
            scalarisation=scalarisation or mean_scalarise(),
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


@document_mv_kurtosis
@document_loss()
class MultivariateKurtosis(Loss):
    """
    Multivariate kurtosis loss for a time series.
    \
    {mv_kurtosis_long_description}
    \
    {param_spec}\
    {mv_kurtosis_spec}\
    {scalarisation_spec}
    """

    l2: float
    dimensional_scaling: bool

    def __init__(
        self,
        nu: float = 1.0,
        name: Optional[str] = None,
        *,
        l2: float = 0.0,
        dimensional_scaling: bool = False,
        scalarisation: Optional[Callable] = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if name is None: name = 'MultivariateKurtosis'
        super().__init__(
            nu=nu,
            name=name,
            score=multivariate_kurtosis,
            scalarisation=scalarisation or mean_scalarise(),
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


@document_connectopy
@document_loss()
class ConnectopyLoss(Loss):
    """
    Generalised connectopic functional, for computing different kinds of
    connectopic maps.
    \
    {connectopy_long_description}
    \
    {param_spec}\
    {connectopy_spec}\
    {prog_theta_spec}\
    {scalarisation_spec}
    """

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
            scalarisation=scalarisation or mean_scalarise(),
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


@document_connectopy
@document_loss()
class ModularityLoss(Loss):
    """\
    {modularity_long_description}
    \
    {param_spec}\
    {connectopy_spec}\
    {modularity_spec}\
    {scalarisation_spec}
    """

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
            scalarisation=scalarisation or mean_scalarise(),
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
