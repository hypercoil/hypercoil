# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Score functions and functionals for loss functions.

A loss function is the composition of a score function and a scalarisation
map (which might itself be the composition of different tensor rank reduction
maps.)
"""
import jax
import jax.numpy as jnp
from distrax._src.utils.math import mul_exp
from functools import partial, reduce
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Union

from ..engine import Tensor, vmap_over_outer
from ..functional import (
    coaffiliation, corr_kernel, cmass_coor, graph_laplacian, linear_distance,
    modularity_matrix, pairedcorr, precision, recondition_eigenspaces,
    spherical_geodesic, sym2vec,
)
from ..functional.cmass import cmass_reference_displacement, diffuse


# Trivial score functions ----------------------------------------------------


def identity(
    X: Tensor,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Identity function.
    """
    return X


def zero(
    X: Tensor,
    *,
    broadcast: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Zero function.
    """
    if broadcast:
        return jnp.zeros_like(X)
    return 0.


def difference(
    X: Tensor,
    Y: Tensor,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Difference score function.
    """
    return X - Y


# Constraint violation penalties ---------------------------------------------


def constraint_violation(
    X: Tensor,
    *,
    constraints: Sequence[Callable[[Tensor], Tensor]],
    broadcast_against_input: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Constraint violation score function.

    This loss uses a set of constraint functions and evaluates them on its
    input. If a constraint evaluates to 0 or less, then it is considered to
    be satisfied and no penalty is applied. Otherwise, a score is returned
    in proportion to the maximum violation of any constraint.

    For example, using the identity constraint penalises only positive
    elements (equivalent to :func:`unilateral_loss`), while ``lambda x: -x``
    penalises only negative elements.
    ``lambda x : tensor([1, 3, 0, -2]) @ x - 2`` applies the specified affine
    function as a constraint.

    .. warning::
        Because of broadcasting rules, the results of constraint computations
        are not necessarily the same shape as the input, and the output of
        this function will be the same shape as the largest constraint. This
        might lead to unexpected scaling of different constraints, and so the
        ``broadcast_against_input`` option is provided to broadcast all
        outputs against the input shape. In the future, we might add an option
        that normalises each constraint violation by the number of elements
        in the output.

    Parameters
    ----------
    X : Tensor
        Input tensor.
    constraints : Sequence[Callable[[Tensor], Tensor]]
        Iterable containing constraint functions.
    broadcast_against_input : bool, optional (default: ``False``)
        If ``True``, broadcast all constraint outputs against the input shape.

    Returns
    -------
    Tensor
        Maximum constraint violation score for each element.
    """
    broadcast = broadcast_against_input
    constraints = (partial(zero, broadcast=broadcast),) + tuple(constraints)
    if key is not None:
        return reduce(jnp.maximum, (c(X, key=key) for c in constraints))
    return reduce(jnp.maximum, (c(X) for c in constraints))


def unilateral_loss(
    X: Tensor,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Unilateral loss function.

    This loss penalises only positive elements of its input. It is a special
    case of :func:`constraint_violation` with the identity constraint.
    """
    return constraint_violation(X, constraints=(identity,))


def hinge_loss(
    Y_hat: Tensor,
    Y: Tensor,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Hinge loss function.

    This is the loss function used in support vector machines. It is a special
    case of :func:`constraint_violation` or :func:`unilateral_loss` where the
    inputs are transformed according to the following:

    .. math::

        1 - Y \hat{Y}
    """
    score = 1 - Y * Y_hat
    return unilateral_loss(score)


# Smoothness -----------------------------------------------------------------


def smoothness(
    X: Tensor,
    *,
    n: int = 1,
    #pad_value: Optional[Union[float, Literal['initial']]] = None,
    pad_value: Optional[float] = None,
    axis: int = -1,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Smoothness score function.

    This loss penalises large or sudden changes in the input tensor. It is
    currently a thin wrapper around ``jax.numpy.diff``.
    """
    # if pad_value == 'initial':
    #     axis = standard_axis_number(axis, X.ndim)
    #     pad_value = X[(slice(None),) * axis + (0,)]
    return jnp.diff(X, n=n, axis=axis, prepend=pad_value)


# Bimodal symmetric ----------------------------------------------------------


def _bimodal_symmetric_impl(
    X: Tensor,
    *,
    mean: float,
    step: float,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Bimodal symmetric score function. This parameterisation is used internally
    by :func:`bimodal_symmetric` and the version in ``loss.nn``.
    """
    return jnp.abs(jnp.abs(X - mean) - step)


def bimodal_symmetric(
    X: Tensor,
    *,
    modes: Tuple[int, int] = (0, 1),
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Bimodal symmetric score function.

    This function returns a score equal to the absolute difference between
    each element of the input tensor and whichever of the two specified modes
    is closer.
    """
    mean = sum(modes) / 2
    step = max(modes) - mean
    return _bimodal_symmetric_impl(X, mean=mean, step=step, key=key)


# Gramian determinants -------------------------------------------------------


def document_gramian_determinant(func: Callable) -> Callable:
    long_description = """
    This function computes the determinant of the Gram matrix of the input
    tensor, defined according to the kernel function ``op``. The kernel
    function should always be a positive semi-definite function, and
    additional arguments are provided to ensure a non-singular (i.e.,
    strictly positive definite) matrix."""
    det_gram_spec = r"""
    Parameters
    ----------
    X : Tensor
        Input tensor.
    theta : Tensor, optional (default: ``None``)
        Kernel parameter tensor. If ``None``, then the kernel is assumed to
        be isotropic.
    op : Callable, optional (default: :func:`corr_kernel`)
        Kernel function. By default, the Pearson correlation kernel is used.
    psi : float, optional (default: ``0.``)
        Kernel regularisation parameter. If ``psi > 0``, then the kernel
        matrix is regularised by adding ``psi`` to the diagonal. This can be
        used to ensure that the matrix is strictly positive definite.
    xi : float, optional (default: ``0.``)
        Kernel regularisation parameter. If ``xi > 0``, then the kernel
        matrix is regularised by stochastically adding samples from a uniform
        distribution with support :math:`\psi - \xi, \xi` to the diagonal.
        This can be used to ensure that the matrix does not have degenerate
        eigenvalues. If ``xi > 0``, then ``psi`` must also be greater than
        ``xi`` and a key must be provided.
    key: PRNGKey, optional (default: ``None``)
        Random number generator key. This is only required if ``xi > 0``.

    Returns
    -------
    Tensor
        Gramian determinant score for each set of observations."""

    func.__doc__ = func.__doc__.format(
        long_description=long_description,
        det_gram_spec=det_gram_spec,
    )
    return func


@document_gramian_determinant
def det_gram(
    X: Tensor,
    theta: Optional[Tensor] = None,
    *,
    op: Optional[Callable] = corr_kernel,
    psi: Optional[float] = 0.,
    xi: Optional[float] = 0.,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Gramian determinant score function.
    \
    {long_description}\
    {det_gram_spec}
    """
    Z = op(X, theta=theta)
    if xi > 0:
        Z = recondition_eigenspaces(Z, psi=psi, xi=xi, key=key)
    elif psi > 0:
        Z = Z + psi * jnp.eye(Z.shape[-1])
    return -jnp.linalg.det(Z)


@document_gramian_determinant
def log_det_gram(
    X: Tensor,
    theta: Optional[Tensor] = None,
    *,
    op: Optional[Callable] = corr_kernel,
    psi: Optional[float] = 0.,
    xi: Optional[float] = 0.,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Gramian log-determinant score function.
    \
    {long_description}\
    {det_gram_spec}
    """
    Z = op(X, theta=theta)
    if xi > 0:
        Z = recondition_eigenspaces(Z, psi=psi, xi=xi, key=key)
    elif psi > 0:
        Z = Z + psi * jnp.eye(Z.shape[-1])
    _, logdet = jnp.linalg.slogdet(Z)
    return -logdet


# Information and entropy ----------------------------------------------------


def document_entropy(func: Callable) -> Callable:
    entropy_spec = """
    Parameters
    ----------
    X : Tensor
        Input tensor containing probabilities or logits for each category.
    axis : int or sequence of ints, optional (default: ``-1``)
        Axis or axes over which to compute the entropy.
    keepdims : bool, optional (default: ``True``)
        As in ``jax.numpy.sum``.
    reduce : bool, optional (default: ``True``)
        If this is False, then the unsummed probability-weighted surprise is
        computed for each element of the input tensor. Otherwise, the entropy
        is computed over the specified axis or axes.

    Returns
    -------
    Tensor
        Entropy score for each set of observations."""

    kl_spec = r"""
    Adapted from ``distrax``.

    .. note::

        The KL divergence is not symmetric, so this function returns
        :math:`KL(P || Q)`. For a symmetric measure, see
        :func:`js_divergence`.

    .. math::

        KL(P || Q) = \sum_{x \in \mathcal{X}}^n P_x \log \frac{P_x}{Q_x}

    Parameters
    ----------
    P : Tensor
        Input tensor parameterising the first categorical distribution.
    Q : Tensor
        Input tensor parameterising the second categorical distribution.
    axis : int or sequence of ints, optional (default: ``-1``)
        Axis or axes over which to compute the KL divergence.
    keepdims : bool, optional (default: ``True``)
        As in ``jax.numpy.sum``.
    reduce : bool, optional (default: ``True``)
        If this is False, then the unsummed KL divergence is computed for
        each element of the input tensor. Otherwise, the KL divergence is
        computed over the specified axis or axes.

    Returns
    -------
    Tensor
        KL divergence between the two distributions."""

    js_spec = r"""

    .. math::

        JS(P || Q) = \frac{1}{2} KL(P || M) + \frac{1}{2} KL(Q || M)

    Parameters
    ----------
    P : Tensor
        Input tensor parameterising the first categorical distribution.
    Q : Tensor
        Input tensor parameterising the second categorical distribution.
    axis : int or sequence of ints, optional (default: ``-1``)
        Axis or axes over which to compute the JS divergence.
    keepdims : bool, optional (default: ``True``)
        As in ``jax.numpy.sum``.
    reduce : bool, optional (default: ``True``)
        If this is False, then the unsummed JS divergence is computed for
        each element of the input tensor. Otherwise, the JS divergence is
        computed over the specified axis or axes.

    Returns
    -------
    Tensor
        JS divergence between the two distributions."""

    func.__doc__ = func.__doc__.format(
        entropy_spec=entropy_spec,
        kl_spec=kl_spec,
        js_spec=js_spec,
    )
    return func


@document_entropy
def entropy(
    X: Tensor,
    *,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Entropy of a categorical distribution or set of categorical distributions.

    This function operates on probability tensors. For a version that operates
    on logits, see :func:`entropy_logit`.
    \
    {entropy_spec}
    """
    eps = jnp.finfo(X.dtype).eps
    entropy = -X * jnp.log(X + eps)
    if not reduce:
        return entropy
    return entropy.sum(axis, keepdims=keepdims)


@document_entropy
def entropy_logit(
    X: Tensor,
    *,
    temperature: float = 1.,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Project logits in the input matrix onto the probability simplex, and then
    compute the entropy of the resulting categorical distribution.

    This function operates on logit tensors. For a version that operates on
    probabilities, see :func:`entropy`.
    \
    {entropy_spec}
    """
    probs = jax.nn.softmax(X / temperature, axis=axis)
    return entropy(probs, axis=axis, keepdims=keepdims, reduce=reduce)


@document_entropy
def kl_divergence(
    P: Tensor,
    Q: Tensor,
    *,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Kullback-Leibler divergence between two categorical distributions.

    This function operates on probability tensors. For a version that operates
    on logits, see :func:`kl_divergence_logit`.
    \
    {kl_spec}
    """
    eps = jnp.finfo(P.dtype).eps
    P = jnp.log(P + eps)
    Q = jnp.log(Q + eps)
    kl_div = mul_exp(P - Q, P)
    if not reduce:
        return kl_div
    return kl_div.sum(axis=axis, keepdims=keepdims)


@document_entropy
def kl_divergence_logit(
    P: Tensor,
    Q: Tensor,
    *,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Kullback-Leibler divergence between two categorical distributions.

    This function operates on logits. For a version that operates on
    probabilities, see :func:`kl_divergence`.
    \
    {kl_spec}
    """
    P = jax.nn.log_softmax(P, axis=axis)
    Q = jax.nn.log_softmax(Q, axis=axis)
    kl_div = mul_exp(P - Q, P)
    if not reduce:
        return kl_div
    return kl_div.sum(axis=axis, keepdims=keepdims)


@document_entropy
def js_divergence(
    P: Tensor,
    Q: Tensor,
    *,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Jensen-Shannon divergence between two categorical distributions.

    This function operates on probability tensors. For a version that operates
    on logits, see :func:`js_divergence_logit`.
    \
    {js_spec}
    """
    M = 0.5 * (P + Q)
    js_div = (kl_divergence(P, M, reduce=False) +
              kl_divergence(Q, M, reduce=False)) / 2
    if not reduce:
        return js_div
    return js_div.sum(axis=axis, keepdims=keepdims)


@document_entropy
def js_divergence_logit(
    P: Tensor,
    Q: Tensor,
    *,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Jensen-Shannon divergence between two categorical distributions.

    This function operates on logits. For a version that operates on
    probabilities, see :func:`js_divergence`.
    \
    {js_spec}
    """
    prob_axis = axis
    if prob_axis is None:
        prob_axis = -1
    P = jax.nn.softmax(P, prob_axis)
    Q = jax.nn.softmax(Q, prob_axis)
    return js_divergence(P, Q, axis=axis, keepdims=keepdims, reduce=reduce)


# Bregman --------------------------------------------------------------------


def document_bregman(func: Callable) -> Callable:

    long_description = """
    This function computes the Bregman divergence between the input tensor
    and the target tensor, induced according to the convex function ``f``."""

    param_spec = """
    Parameters
    ----------
    X : Tensor
        Input tensor.
    Y : Tensor
        Target tensor.
    f : Callable
        Convex function to induce the Bregman divergence.
    f_dim : int
        Dimension of arguments to ``f``.

    Returns
    -------
    Tensor
        Bregman divergence score for each set of observations."""

    func.__doc__ = func.__doc__.format(
        long_description=long_description,
        param_spec=param_spec,
    )
    return func


def _bregman_divergence_impl(
    X: Tensor,
    Y: Tensor,
    *,
    f: Callable,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Bregman divergence score function for a single pair of distributions or
    observations.
    """
    df = jax.grad(f)
    return (f(Y) - f(X)) - df(X).ravel() @ (Y - X).ravel()


@document_bregman
def bregman_divergence(
    X: Tensor,
    Y: Tensor,
    *,
    f: Callable,
    f_dim: int,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Bregman divergence score function.
    \
    {long_description}

    For a version of this function that operates on logits, see
    :func:`bregman_divergence_logit`.
    \
    {param_spec}
    """
    f = vmap_over_outer(partial(_bregman_divergence_impl, f=f), f_dim)
    return f((X, Y))


@document_bregman
def bregman_divergence_logit(
    X: Tensor,
    Y: Tensor,
    *,
    f: Callable,
    f_dim: int,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Bregman divergence score function for logits.
    \
    {long_description}

    This function operates on logits. For the standard version of this
    function, see :func:`bregman_divergence`.
    \
    {param_spec}
    """
    prob_axes = tuple(range(-f_dim, 0))
    P = jax.nn.softmax(X, axis=prob_axes)
    Q = jax.nn.softmax(Y, axis=prob_axes)
    return bregman_divergence(P, Q, f=f, f_dim=f_dim, key=key)


# Equilibrium ----------------------------------------------------------------


def document_equilibrium(func: Callable) -> Callable:

    long_description = """
    The equilibrium scores the deviation of the total weight assigned to each
    parcel or level from the mean weight assigned to each parcel or level. It
    can be used to encourage the model to learn parcels that are balanced in
    size."""

    equilibrium_spec = """
    Parameters
    ----------
    X: Tensor
        A tensor of probabilities (or masses of another kind).
    level_axis: int or sequence of ints, optional
        The axis or axes over which to compute the equilibrium. Within each
        data instance or weight channel, all elements along the specified axis
        or axes should correspond to a single level or parcel. The default is
        -1.
    instance_axes: int or sequence of ints, optional
        The axis or axes corresponding to a single data instance or weight
        channel. This should be a superset of `level_axis`. The default is
        (-1, -2).
    keepdims: bool, optional
        As in :func:`jax.numpy.sum`. The default is True.

    Returns
    -------
    Tensor
        A tensor of equilibrium scores for each parcel or level."""

    func.__doc__ = func.__doc__.format(
        long_description=long_description,
        equilibrium_spec=equilibrium_spec,
    )
    return func


@document_equilibrium
def equilibrium(
    X: Tensor,
    *,
    level_axis: Union[int, Sequence[int]] = -1,
    instance_axes: Union[int, Sequence[int]] = (-1, -2),
    keepdims: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Compute the parcel equilibrium.
    \
    {long_description}

    This function operates on probability tensors. For a version that operates
    on logits, see :func:`equilibrium_logit`.
    \
    {equilibrium_spec}
    """
    parcel = X.mean(level_axis, keepdims=keepdims)
    total = X.mean(instance_axes, keepdims=keepdims)
    return jnp.abs(parcel - total)


@document_equilibrium
def equilibrium_logit(
    X: Tensor,
    *,
    level_axis: Union[int, Sequence[int]] = -1,
    prob_axis: Union[int, Sequence[int]] = -2,
    instance_axes: Union[int, Sequence[int]] = (-1, -2),
    keepdims: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Project logits in the input matrix onto the probability simplex, and then
    compute the parcel equilibrium.
    \
    {long_description}

    This function operates on logits. For a version that operates on
    probabilities, see :func:`equilibrium`.
    \
    {equilibrium_spec}
    """
    probs = jax.nn.softmax(X, axis=prob_axis)
    return equilibrium(
        probs,
        level_axis=level_axis,
        instance_axes=instance_axes,
        keepdims=keepdims,
    )


# Second moments -------------------------------------------------------------


def document_second_moment(func: Callable) -> Callable:

    long_description = r"""
    Given an input matrix :math:`T` and a weight matrix :math:`A`, the second
    moment is computed as

    :math:`\left[ A \circ \left (T - \frac{AT}{A\mathbf{1}} \right )^2  \right] \frac{\mathbf{1}}{A \mathbf{1}}`

    The term :math:`\frac{AT}{A\mathbf{1}}` can also be precomputed and passed
    as the `mu` argument to the :func:`second_moment_centred` function. If the
    mean is already known, it is more efficient to use that function.
    Otherwise, the :func:`second_moment` function will compute the mean
    internally."""

    pparam_spec = """
    Parameters
    ----------
    X: Tensor
        A tensor of observations.
    weight: Tensor
        A tensor of weights."""

    param_spec = """
    skip_normalise: bool, optional
        If True, do not include normalisation by the sum of the weights in the
        computation. In practice, this seems to work better than computing the
        actual second moment. Instead of computing the second moment, this
        corresponds to computed a weighted mean squared error about the mean.
        The default is False.

    Returns
    -------
    Tensor
        Tensor of second moments."""

    std_spec_nomean = """
    standardise: bool, optional
        If True, z-score the input matrix before computing the second moment.
        The default is False."""

    std_spec_mean = """
    standardise_data: bool, optional
        If True, z-score the input matrix before computing the second moment.
        The default is False.
    standardise_mu: bool, optional
        If True, z-score the mean matrix ``mu`` before computing the second
        moment. The default is False."""

    func.__doc__ = func.__doc__.format(
        long_description=long_description,
        pparam_spec=pparam_spec,
        param_spec=param_spec,
        std_spec_nomean=std_spec_nomean,
        std_spec_mean=std_spec_mean,
    )
    return func

def _second_moment_impl(
    X: Tensor,
    weight: Tensor,
    mu: Tensor,
    *,
    skip_normalise: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Core computation for second-moment loss.
    """
    weight = jnp.abs(weight)[..., None]
    if skip_normalise:
        normfac = 1
    else:
        normfac = weight.sum(-2)
    diff = X[..., None, :, :] - mu[..., None, :]
    sigma = ((diff * weight) ** 2).sum(-2) / normfac
    return sigma


@document_second_moment
def second_moment(
    X: Tensor,
    weight: Tensor,
    *,
    standardise: bool = False,
    skip_normalise: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Compute the second moment of a dataset.
    \
    {long_description}
    \
    {pparam_spec}\
    {std_spec_nomean}\
    {param_spec}
    """
    if standardise:
        X = (X - X.mean(-1, keepdims=True)) / X.std(-1, keepdims=True)
    mu = (weight @ X / weight.sum(-1, keepdims=True))
    return _second_moment_impl(
        X, weight, mu, skip_normalise=skip_normalise, key=key)


@document_second_moment
def second_moment_centred(
    X: Tensor,
    weight: Tensor,
    mu: Tensor,
    *,
    standardise_data: bool = False,
    standardise_mu: bool = False,
    skip_normalise: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Compute the second moment of a dataset about a specified mean.
    \
    {long_description}
    \
    {pparam_spec}\
    {std_spec_mean}\
    {param_spec}
    """
    if standardise_data:
        X = (X - X.mean(-1, keepdims=True)) / X.std(-1, keepdims=True)
    if standardise_mu:
        mu = (mu - mu.mean(-1)) / mu.std(-1)
    return _second_moment_impl(
        X, weight, mu, skip_normalise=skip_normalise, key=key)


# Batch correlation ----------------------------------------------------------


def auto_tol(
    batch_size: int,
    significance: float = 0.1,
    tails: int = 2,
) -> float:
    r"""
    Automatically set the tolerance for batch-dimension correlations based on
    a significance level.

    From the t-value associated with the specified significance level, the
    tolerance is computed as

    :math:`r_{tol} = \sqrt{\frac{t^2}{N - 2 - t^2}}`

    .. warning::

        The tolerance computed corresponds to an uncorrected p-value. If
        multiple tests are performed, it might be necessary to use a more
        sophisticated correction.

    Parameters
    ----------
    batch_size : int
        Number of observations in the batch.
    significance : float in (0, 1) (default 0.1)
        Significance level at which the tolerance should be computed.
    tails : 1 or 2 (default 2)
        Number of tails for the t-test.

    Returns
    -------
    float
        Tolerance for batch-dimension correlations.
    """
    import numpy as np
    from scipy.stats import t
    tsq = t.ppf(q=(1 - significance / tails), df=(batch_size - 2)) ** 2
    return jnp.sqrt(tsq / (batch_size - 2 + tsq))


def batch_corr(
    X: Tensor,
    N: Tensor,
    *,
    tol: Union[float, Literal['auto']] = 0,
    tol_sig: float = 0.1,
    abs: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Correlation over the batch dimension.

    Parameters
    ----------
    X : tensor
        Tensor block containing measures to be correlated with those in ``N``.
    N : tensor
        Vector of measures with which the measures in ``X`` are to be
        correlated.
    tol : nonnegative float or ``'auto'`` (default 0)
        Tolerance for correlations. Only correlation values above ``tol`` are
        counted. If this is set to ``'auto'``, a tolerance is computed for the
        batch size given the significance level in ``tol_sig``.
    tol_sig : float in (0, 1)
        Significance level for correlation tolerance. Used only if ``tol`` is
        set to ``'auto'``.
    abs : bool (default True)
        Use the absolute value of correlations. If this is being used as a loss
        function, the model's weights will thus be updated to shrink all
        batchwise correlations toward zero.

    Returns
    -------
    tensor
        Absolute correlation of each vector in ``X`` with ``N``, after
        thresholding at `tol`. Note that, if you want the original
        correlations back, you will have to add ``tol`` to any nonzero
        correlations.
    """
    batch_size = X.shape[0]
    batchcorr = pairedcorr(
        X.swapaxes(0, -1).reshape(-1, batch_size),
        jnp.atleast_2d(N)
    )
    if tol == 'auto':
        tol = auto_tol(batch_size, significance=tol_sig)

    batchcorr_thr = jnp.maximum(jnp.abs(batchcorr) - tol, 0)
    if abs:
        return batchcorr_thr
    else:
        return jnp.sign(batchcorr) * batchcorr_thr


def qcfc(
    fc: Tensor,
    qc: Tensor,
    *,
    tol: Union[float, Literal['auto']] = 0,
    tol_sig: float = 0.1,
    abs: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Alias for :func:`batch_corr`. Quality control-functional connectivity
    correlation.
    """
    return batch_corr(fc, qc, tol=tol, tol_sig=tol_sig, abs=abs, key=key)


# Distance-based losses ------------------------------------------------------


def reference_tether(
    X: Tensor,
    ref: Tensor,
    coor: Tensor,
    *,
    radius: Optional[float] = 100.,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Distance of centres of mass from tethered reference points.

    This can potentially be used, for instance, to adapt an existing atlas to
    a new dataset.

    Parameters
    ----------
    X : Tensor
        Tensor block containing the data to be tethered. Each row is a
        collection of masses assigned to the single object or parcel, and each
        column denotes the distribution of masses across objects or parcels at
        a single spatial location.
    ref : Tensor
        Coordinates of the reference points to which the centres of mass of
        the objects in ``X`` should be tethered. Each object or parcel should
        have a single corresponding reference point.
    coor : Tensor
        Coordinates of the spatial locations in each of the columns of ``X``.
    radius : float or None (default 100)
        Radius of the spherical manifold on which the coordinates are
        located. If this is specified as None, it is assumed that the
        coordinates are in Euclidean space.

    Returns
    -------
    Tensor
        Distance of the centres of mass of the objects in ``X`` from the
        corresponding reference points in ``ref``.
    """
    return cmass_reference_displacement(
        weight=X,
        refs=ref,
        coor=coor,
        radius=radius,
    )


def interhemispheric_tether(
    lh: Tensor,
    rh: Tensor,
    lh_coor: Tensor,
    rh_coor: Tensor,
    *,
    radius: float = 100.,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Distance of centres of mass of left-hemisphere parcels from corresponding
    right-hemisphere parcels. This can be used to promote a symmetric atlas.

    .. note::

        It is assumed that the first coordinate in ``lh_coor`` and ``rh_coor``
        is the x-coordinate, and that the left hemisphere is on the opposite
        side of the x-axis from the right hemisphere.

    Parameters
    ----------
    lh : Tensor
        Tensor block containing the data for the left hemisphere. Each row is a
        collection of masses assigned to the single object or parcel, and each
        column denotes the distribution of masses across objects or parcels at
        a single spatial location.
    rh : Tensor
        Tensor block containing the data for the right hemisphere.
    lh_coor : Tensor
        Coordinates of the spatial locations in each of the columns of ``lh``.
    rh_coor : Tensor
        Coordinates of the spatial locations in each of the columns of ``rh``.
    radius : float (default 100)
        Radius of the spherical manifold on which the coordinates are
        located.

    Returns
    -------
    Tensor
        Distance of the centres of mass of the objects in ``lh`` from the
        corresponding objects in ``rh``.
    """
    ipsilateral_weight = rh
    ipsilateral_coor = rh_coor
    contralateral_ref = cmass_coor(X=lh, coor=lh_coor, radius=radius)
    contralateral_ref = contralateral_ref.at[0, :].set(
        -contralateral_ref[0, :])
    return cmass_reference_displacement(
        weight=ipsilateral_weight,
        refs=contralateral_ref,
        coor=ipsilateral_coor,
        radius=radius,
    )


def compactness(
    X: Tensor,
    coor: Tensor,
    *,
    radius: Optional[float] = 100.,
    norm: Union[int, float, Literal['inf']] = 2,
    floor: float = 0.,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Distance of masses in each object from the centre of mass of that object.

    A diffuse object will have a high score, while a compact object will
    have a low score.

    Parameters
    ----------
    X : Tensor
        Tensor block containing the data to be evaluated. Each row is a
        collection of masses assigned to the single object or parcel, and each
        column denotes the distribution of masses across objects or parcels at
        a single spatial location.
    coor : Tensor
        Coordinates of the spatial locations in each of the columns of ``X``.
    radius : float or None (default 100)
        Radius of the spherical manifold on which the coordinates are
        located. If this is specified as None, it is assumed that the
        coordinates are in a space induced by the ``norm`` parameter.
    norm : int, float or 'inf' (default 2)
        The norm to use to calculate the distance between the centres of mass
        and the masses. Ignored if ``radius`` is specified; in this case, the
        spherical geodesic distance is used.
    floor : float (default 0)
        Minimum distance to be penalised. This can be used to avoid penalising
        masses that are already very close to the centre of mass.

    Returns
    -------
    Tensor
        Distance of the masses in each object from the centre of mass of that
        object.
    """
    return diffuse(X=X, coor=coor, norm=norm, floor=floor, radius=radius)


def dispersion(
    X: Tensor,
    *,
    metric: Callable = spherical_geodesic,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Dispersion of the centres of mass of objects in a tensor block.

    .. note::
        This operates on precomputed centres of mass.

    Parameters
    ----------
    X : Tensor
        Tensor block containing the data to be evaluated. Each row contains
        the centre of mass of a single object or parcel.
    metric : Callable
        Function to calculate the distance between centres of mass. This
        should take either one or two arguments, and should return the
        distance between each pair of observations.

    Returns
    -------
    Tensor
        Dispersion of the centres of mass of objects in a tensor block.
    """
    return sym2vec(-metric(X.swapaxes(-2, -1)))


# Multivariate kurtosis ------------------------------------------------------


def multivariate_kurtosis(
    ts: Tensor,
    *,
    l2: float = 0.,
    dimensional_scaling: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Multivariate kurtosis of a time series.

    This is the multivariate kurtosis following Mardia, as used by Laumann
    and colleagues in the setting of functional connectivity. It is equal to
    the mean of the squared Mahalanobis norm of each time point (as
    parameterised by the inverse covariance of the multivariate time series).

    Parameters
    ----------
    ts : Tensor
        Multivariate time series to be evaluated. Each row is a channel or
        variable, and each column is a time point.
    l2 : float (default 0)
        L2 regularisation to be applied to the covariance matrix to ensure
        that it is invertible.
    dimensional_scaling : bool (default False)
        The expected value of the multivariate kurtosis for a normally
        distributed, stationary process of infinite duration with d channels
        (or variables) is :math:`d (d + 2)`. Setting this to true normalises
        for the process dimension by dividing the obtained kurtosis by
        :math:`d (d + 2)`. This has no effect in determining the optimum.

    Returns
    -------
    Tensor
        Multivariate kurtosis of the input time series.
    """
    if dimensional_scaling:
        d = ts.shape[-2]
        denom = d * (d + 2)
    else:
        denom = 1
    prec = precision(ts, l2=l2)[..., None, :, :]
    ts = ts.swapaxes(-1, -2)[..., None, :]
    maha = (ts @ prec @ ts.swapaxes(-1, -2)).squeeze()
    return -(maha ** 2).mean(-1) / denom


# Connectopies ---------------------------------------------------------------


def document_connectopy(func: Callable) -> Callable:

    param_spec = """
    :Dimension: **Q :** :math:`(D, C)`
                    D denotes the number of vertices in the affinity matrix
                    and C denotes the number of proposed maps.
                **A :** :math:`(D, D)`
                    As above.
                **D :** :math:`(D, D)`
                    As above.
                **theta :** :math:`(C)` or :math:`(C, C)`
                    As above.

    Parameters
    ----------
    Q : tensor
        Proposed connectopies or maps.
    A : tensor
        Affinity matrix.
    D : tensor or None (default None)
        If this argument is provided, then the affinity matrix is first
        transformed as :math:`D A D^\intercal`. For instance, setting D to
        a diagonal matrix whose entries are the reciprocal of the square root
        of vertex degrees corresponds to learning eigenmaps of a normalised
        graph Laplacian.
    theta : tensor, float, or None (default None)
        Parameterisation of the pairwise dissimilarity function.
    omega : tensor, float, or None (default None)
        Optional parameterisation of the affinity function, if one is
        provided.
    dissimilarity : callable
        Function to compute dissimilarity between latent coordinates induced
        by the proposed connectopies. By default, the square of the L2
        distance is used. The callable must accept ``Q`` and ``theta`` as
        arguments. (``theta`` may be unused.)
    affinity : callable or None (default None)
        If an affinity function is provided, then the image of argument A
        under this function is the affinity matrix. Otherwise, argument A is
        the affinity matrix."""

    return_spec = """
    Returns
    -------
    Tensor
        Connectopic functional value."""

    objective = r""":math:`\mathbf{1}^\intercal \left( \mathbf{A} \circ S_\theta(\mathbf{Q}) \right) \mathbf{1}`"""

    func.__doc__ = func.__doc__.format(
        param_spec=param_spec,
        return_spec=return_spec,
        objective=objective,
    )
    return func


@document_connectopy
def connectopy(
    Q: Tensor,
    A: Tensor,
    D: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
    omega: Optional[Tensor] = None,
    *,
    dissimilarity: Optional[Callable] = None,
    affinity: Optional[Callable] = None,
    progressive_theta: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Generalised connectopic functional, for computing different kinds of
    connectopic maps.

    .. admonition:: Connectopic functional

        Given an affinity matrix A, the connectopic functional is the
        objective

        {objective}

        for a pairwise function S. The default pairwise function is the square
        of the L2 distance. The columns of the Q that minimises the objective
        are the learned connectopic maps.

    .. warning::
        If you're using this for a well-characterised connectopic map with a
        closed-form or algorithmically optimised solution, such as Laplacian
        eigenmaps or many forms of community detection, then in most cases you
        would be better off directly computing exact maps rather than using this
        functional to approximate them.

        Because this operation attempts to learn all of the maps that jointly
        minimise the objective in a single shot rather than using iterative
        projection, it is more prone to misalignment than a projective approach
        for eigendecomposition-based maps.

    .. danger::
        Note that ``connectopy`` is often insufficient on its own as a loss.
        It should be combined with appropriate constraints, for instance to
        ensure the learned maps are zero-centred and orthogonal.
    \
    {param_spec}
    progressive_theta : bool (default False)
        When this is True, a ``theta`` is generated such that the last map in
        ``Q`` has a weight of 1, the second-to-last has a weight of 2, and so
        on. This can be used to encourage the last column to correspond to the
        least important connectopic map and the first column to correspond to
        the most important connectopic map.
    \
    {return_spec}
    """
    if progressive_theta:
        n_maps = Q.shape[-1]
        theta = jnp.arange(n_maps, 0, -1)
    if dissimilarity is None:
        dissimilarity = linear_distance
    if affinity is not None:
        A = affinity(A, omega=omega)
    if D is not None:
        A = D @ A @ D.swapaxes(-2, -1)
    H = dissimilarity(Q, theta=theta)
    return (H * A).sum((-2, -1))


@document_connectopy
def modularity(
    Q: Tensor,
    A: Tensor,
    D: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
    *,
    gamma: float = 1.,
    exclude_diag: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Modularity functional.

    The connectopies that minimise the modularity functional define a
    community structure for the graph.
    \
    {param_spec}
    gamma : float (default 1.)
        Modularity parameter. Takes the place of the ``omega`` argument in
        ``connectopy``.
    exclude_diag : bool (default True)
        If True, then the diagonal of the affinity matrix is set to zero.
    \
    {return_spec}
    """
    def dissimilarity(Q, theta):
        return coaffiliation(Q, L=theta, normalise=True,
                             exclude_diag=exclude_diag)

    def affinity(A, omega):
        return modularity_matrix(A, gamma=omega, normalise=True)

    return connectopy(
        Q=Q,
        A=A,
        D=D,
        theta=theta,
        omega=gamma,
        dissimilarity=dissimilarity,
        affinity=affinity,
        key=key,
    )


@document_connectopy
def eigenmaps(
    Q: Tensor,
    A: Tensor,
    theta: Optional[Tensor] = None,
    *,
    normalise: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Laplacian eigenmaps functional.

    .. warning::

        This function is provided as an illustrative example of how to
        parameterise the connectopy functional. It is not recommended for
        practical use, because it is incredibly inefficient and numerically
        unstable. Instead, use the ``laplacian_eigenmaps`` function from
        ``hypercoil.functional``.
    \
    {param_spec}
    normalise : bool (default True)
        If True, then the graph Laplacian is normalised by the vertex degrees.
        Takes the place of the ``D`` and ``omega`` arguments in
        ``connectopy``.
    \
    {return_spec}
    """
    def dissimilarity(Q, theta):
        return linear_distance(Q, theta=theta)

    def affinity(A, omega):
        return graph_laplacian(A, normalise=omega)

    return connectopy(
        Q=Q,
        A=A,
        theta=theta,
        omega=normalise,
        dissimilarity=dissimilarity,
        affinity=affinity,
        key=key,
    )
