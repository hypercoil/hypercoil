# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Score functions and functionals for loss functions.

A loss function is the composition of a score function and a scalarisation
map (which might itself be the composition of different tensor rank reduction
maps.)
"""
from __future__ import annotations
from functools import partial, reduce
from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp

from ..engine import NestedDocParse, Tensor, vmap_over_outer
from ..functional import (
    cmass_coor,
    coaffiliation,
    corr_kernel,
    graph_laplacian,
    linear_distance,
    modularity_matrix,
    pairedcorr,
    precision,
    recondition_eigenspaces,
    sym2vec,
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
    return 0.0


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


def document_constraint_violation(func):
    long_description = """
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
        in the output."""
    long_description_unil = """
    This loss penalises only positive elements of its input. It is a special
    case of :func:`constraint_violation` with the identity constraint."""
    long_description_hinge = r"""
    This is the loss function used in support vector machines. It is a special
    case of :func:`constraint_violation` or :func:`unilateral_loss` where the
    inputs are transformed according to the following:

    .. math::

        1 - Y \hat{Y}"""
    constraint_violation_spec = """
    constraints : Sequence[Callable[[Tensor], Tensor]]
        Iterable containing constraint functions.
    broadcast_against_input : bool, optional (default: ``False``)
        If ``True``, broadcast all constraint outputs against the input shape."""
    fmt = NestedDocParse(
        long_description=long_description,
        long_description_unil=long_description_unil,
        long_description_hinge=long_description_hinge,
        constraint_violation_spec=constraint_violation_spec,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
    return func


@document_constraint_violation
def constraint_violation(
    X: Tensor,
    *,
    constraints: Sequence[Callable[[Tensor], Tensor]],
    broadcast_against_input: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Constraint violation score function.
    \
    {long_description}

    Parameters
    ----------
    X : Tensor
        Input tensor.\
    {constraint_violation_spec}

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


@document_constraint_violation
def unilateral_loss(
    X: Tensor,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Unilateral loss function.
    \
    {long_description_unil}
    """
    return constraint_violation(X, constraints=(identity,))


@document_constraint_violation
def hinge_loss(
    Y_hat: Tensor,
    Y: Tensor,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Hinge loss function.
    \
    {long_description_hinge}
    """
    score = 1 - Y * Y_hat
    return unilateral_loss(score)


# Smoothness -----------------------------------------------------------------


def document_smoothness(func):
    long_description = """
    This loss penalises large or sudden changes in the input tensor. It is
    currently a thin wrapper around ``jax.numpy.diff``.

    .. warning::

        This function returns both positive and negative values, and so
        should probably not be used with a scalarisation map like
        ``mean_scalarise`` or ``sum_scalarise``. Instead, maps like
        ``meansq_scalarise`` or ``vnorm_scalarise`` with either the ``p=1``
        or ``p=inf`` options might be more appropriate."""
    smoothness_spec = """
    n: int, optional (default: 1)
        Number of times to differentiate using the backwards differences
        method.
    axis : int, optional (default: -1)
        Axis defining the slice of the input tensor over which differences
        are computed
    pad_value : float, optional (default: None)
        Arguments to ``jnp.diff``. Values to prepend to the input along the
        specified axis before computing the difference."""
    fmt = NestedDocParse(
        long_description=long_description,
        smoothness_spec=smoothness_spec,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
    return func


@document_smoothness
def smoothness(
    X: Tensor,
    *,
    n: int = 1,
    # pad_value: Optional[Union[float, Literal['initial']]] = None,
    pad_value: Optional[float] = None,
    axis: int = -1,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Smoothness score function.
    \
    {long_description}

    Parameters
    ----------
    X : Tensor
        Input tensor.\
    {smoothness_spec}
    """
    # if pad_value == 'initial':
    #     axis = standard_axis_number(axis, X.ndim)
    #     pad_value = X[(slice(None),) * axis + (0,)]
    return jnp.diff(X, n=n, axis=axis, prepend=pad_value)


# Bimodal symmetric ----------------------------------------------------------


def document_bimodal_symmetric(func):
    long_description = """
    This function returns a score equal to the absolute difference between
    each element of the input tensor and whichever of the two specified modes
    is closer. Penalising this quantity can be used to concentrate weights at
    two modes, for instance 0 and 1 or -1 and 1."""
    bimodal_symmetric_spec = """
    modes : tuple(float, float) (default (0, 1))
        Modes of the loss."""
    fmt = NestedDocParse(
        long_description=long_description,
        bimodal_symmetric_spec=bimodal_symmetric_spec,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
    return func


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


@document_bimodal_symmetric
def bimodal_symmetric(
    X: Tensor,
    *,
    modes: Tuple[int, int] = (0, 1),
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Bimodal symmetric score function.
    \
    {long_description}

    Parameters
    ----------
    X : Tensor
        Input tensor.\
    {bimodal_symmetric_spec}
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
    ultra_long_description = """
    .. admonition:: Log-det-Gram

        The log-det-Gram loss among a set of vectors :math:`X` is defined as
        the negative log-determinant of the Gram matrix of those vectors.

        :math:`-\log \det \mathbf{K}(X)`

        .. image:: ../_images/determinant.svg
            :width: 250
            :align: center

        Penalising the negative log-determinant of a Gram matrix can
        promote a degree of independence among the vectors being correlated.

    One example of the log-det-Gram loss is the log-det-corr loss, which
    penalises the negative log-determinant of the correlation matrix of a
    set of vectors. This has a number of desirable properties and applications
    outlined below.

    Correlation matrices, which occur frequently in time series analysis, have
    several properties that make them well-suited for loss functions based on
    the Gram determinant.

    First, correlation matrices are positive semidefinite, and accordingly
    their determinants will always be nonnegative. For positive semidefinite
    matrices, the log-determinant is a concave function and accordingly has a
    global maximum that can be identified using convex optimisation methods.

    Second, correlation matrices are normalised such that their determinant
    attains a maximum value of 1. This maximum corresponds to an identity
    correlation matrix, which in turn occurs when the vectors or time series
    input to the correlation are **orthogonal**. Thus, a strong
    determinant-based loss applied to a correlation matrix will seek an
    orthogonal basis of input vectors.

    In the parcellation setting, a weaker log-det-corr loss can be leveraged to
    promote relative independence of parcels. Combined with a
    :ref:`second-moment loss <hypercoil.loss.secondmoment.SecondMoment>`, a
    log-det-corr loss can be interpreted as inducing a clustering: the second
    moment loss favours internal similarity of clusters, while the log-det-corr
    loss favours separation of different clusters.

    .. warning::
        Determinant-based losses use ``jax``'s determinant functionality,
        which itself might use the singular value decomposition in certain
        cases. Differentiation through SVD involves terms whose denominators
        include the differences between pairs of singular values. Thus, if two
        singular values of the input matrix are close together, the gradient
        can become unstable (and undefined if the singular values are
        identical). A simple
        :doc:`matrix reconditioning <hypercoil.functional.matrix.recondition_eigenspaces>`
        procedure is available for all operations involving the determinant to
        reduce the occurrence of degenerate eigenvalues."""
    det_gram_spec = r"""
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
        Random number generator key. This is only required if ``xi > 0``."""
    return_spec = """
    Returns
    -------
    Tensor
        Gramian determinant score for each set of observations."""

    fmt = NestedDocParse(
        long_description=long_description,
        ultra_long_description=ultra_long_description,
        det_gram_spec=det_gram_spec,
        return_spec=return_spec,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
    return func


@document_gramian_determinant
def det_gram(
    X: Tensor,
    theta: Optional[Tensor] = None,
    *,
    op: Optional[Callable] = corr_kernel,
    psi: Optional[float] = 0.0,
    xi: Optional[float] = 0.0,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Gramian determinant score function.
    \
    {long_description}

    Parameters
    ----------
    X : Tensor
        Input tensor.\
    {det_gram_spec}
    \
    {return_spec}
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
    psi: Optional[float] = 0.0,
    xi: Optional[float] = 0.0,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Gramian log-determinant score function.
    \
    {long_description}

    Parameters
    ----------
    X : Tensor
        Input tensor.\
    {det_gram_spec}
    \
    {return_spec}
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
    divergence_param_spec = """
    Parameters
    ----------
    P : Tensor
        Input tensor parameterising the first categorical distribution.
    Q : Tensor
        Input tensor parameterising the second categorical distribution."""

    entropy_spec = """
    axis : int or sequence of ints, optional (default: ``-1``)
        Axis or axes over which to compute the entropy.
    keepdims : bool, optional (default: ``True``)
        As in ``jax.numpy.sum``.
    reduce : bool, optional (default: ``True``)
        If this is False, then the unsummed probability-weighted surprise is
        computed for each element of the input tensor. Otherwise, the entropy
        is computed over the specified axis or axes."""

    kl_spec = """
    axis : int or sequence of ints, optional (default: ``-1``)
        Axis or axes over which to compute the KL divergence.
    keepdims : bool, optional (default: ``True``)
        As in ``jax.numpy.sum``.
    reduce : bool, optional (default: ``True``)
        If this is False, then the unsummed KL divergence is computed for
        each element of the input tensor. Otherwise, the KL divergence is
        computed over the specified axis or axes."""

    js_spec = """
    axis : int or sequence of ints, optional (default: ``-1``)
        Axis or axes over which to compute the JS divergence.
    keepdims : bool, optional (default: ``True``)
        As in ``jax.numpy.sum``.
    reduce : bool, optional (default: ``True``)
        If this is False, then the unsummed JS divergence is computed for
        each element of the input tensor. Otherwise, the JS divergence is
        computed over the specified axis or axes."""

    entropy_long_description = r"""
    .. admonition:: Entropy

        The entropy of a categorical distribution :math:`A` is defined as

        :math:`-\mathbf{1}^\intercal \left(A \circ \log A\right) \mathbf{1}`

        (where :math:`\log` denotes the elementwise logarithm).

        .. image:: ../_images/entropysimplex.svg
            :width: 250
            :align: center

        *Cartoon schematic of the contours of an entropy-like function over
        categorical distributions. The function attains its maximum for the
        distribution in which all outcomes are equiprobable. The function can
        become smaller without bound away from this maximum. The superposed
        triangle represents the probability simplex. By pre-transforming the
        penalised weights to constrain them to the simplex, the entropy
        function is bounded and attains a separate minimum for each
        deterministic distribution.*

        Penalising the entropy promotes concentration of weight into a single
        category. This has applications in problem settings such as
        parcellation, when more deterministic parcel assignments are desired.

    .. warning::
        Entropy is a concave function. Minimising it without constraint
        affords an unbounded capacity for reducing the loss. This is almost
        certainly undesirable. For this reason, it is recommended that some
        constraint be imposed on the input set when placing a penalty on
        entropy. One possibility is using a
        :doc:`probability simplex parameter mapper <hypercoil.init.mapparam.ProbabilitySimplexParameter>`
        to first project the input weights onto the probability simplex."""

    kl_long_description = r"""
    Adapted from ``distrax``.

    .. note::

        The KL divergence is not symmetric, so this function returns
        :math:`KL(P || Q)`. For a symmetric measure, see
        :func:`js_divergence`.

    .. math::

        KL(P || Q) = \sum_{x \in \mathcal{X}}^n P_x \log \frac{P_x}{Q_x}
    """

    js_long_description = r"""
    .. math::

        JS(P || Q) = \frac{1}{2} KL(P || M) + \frac{1}{2} KL(Q || M)
    """

    entropy_return_spec = """
    Returns
    -------
    Tensor
        Entropy score for each set of observations."""

    kl_return_spec = """
    Returns
    -------
    Tensor
        KL divergence between the two distributions."""

    js_return_spec = """
    Returns
    -------
    Tensor
        JS divergence between the two distributions."""

    fmt = NestedDocParse(
        divergence_param_spec=divergence_param_spec,
        entropy_spec=entropy_spec,
        kl_spec=kl_spec,
        js_spec=js_spec,
        entropy_long_description=entropy_long_description,
        kl_long_description=kl_long_description,
        js_long_description=js_long_description,
        entropy_return_spec=entropy_return_spec,
        kl_return_spec=kl_return_spec,
        js_return_spec=js_return_spec,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
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

    Parameters
    ----------
    X : Tensor
        Input tensor containing probabilities or logits for each category.\
    {entropy_spec}
    \
    {entropy_return_spec}
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
    temperature: float = 1.0,
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

    Parameters
    ----------
    X : Tensor
        Input tensor containing probabilities or logits for each category.\
    {entropy_spec}
    \
    {entropy_return_spec}
    """
    probs = jax.nn.softmax(X / temperature, axis=axis)
    return entropy(probs, axis=axis, keepdims=keepdims, reduce=reduce)


def _mul_exp(x: Tensor, logp: Tensor) -> Tensor:
    """
    Shamelessly pilfered from ``distrax``.
    """
    p = jnp.exp(logp)
    x = jnp.where(p == 0, 0.0, x)
    return x * p


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
    {kl_long_description}
    \
    {divergence_param_spec}\
    {kl_spec}
    \
    {kl_return_spec}
    """
    eps = jnp.finfo(P.dtype).eps
    P = jnp.log(P + eps)
    Q = jnp.log(Q + eps)
    kl_div = _mul_exp(P - Q, P)
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
    {kl_long_description}
    \
    {divergence_param_spec}\
    {kl_spec}
    \
    {kl_return_spec}
    """
    P = jax.nn.log_softmax(P, axis=axis)
    Q = jax.nn.log_softmax(Q, axis=axis)
    kl_div = _mul_exp(P - Q, P)
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
    {js_long_description}
    \
    {divergence_param_spec}\
    {js_spec}
    \
    {js_return_spec}
    """
    M = 0.5 * (P + Q)
    js_div = (
        kl_divergence(P, M, reduce=False) + kl_divergence(Q, M, reduce=False)
    ) / 2
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
    {js_long_description}
    \
    {divergence_param_spec}\
    {js_spec}
    \
    {js_return_spec}
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

    bregman_spec = """
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

    fmt = NestedDocParse(
        long_description=long_description,
        bregman_spec=bregman_spec,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
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
    {bregman_spec}
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
    {bregman_spec}
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

    ultra_long_description = r"""
    Loss functions to favour equal weight across one dimension of a tensor
    whose slices are masses.

    .. admonition:: Equilibrium

        The equilibrium loss of a mass tensor :math:`A` is defined as

        :math:`\mathbf{1}^\intercal \left[\left(A \mathbf{1}\right) \circ \left(A \mathbf{1}\right) \right]`

        Th equilibrium loss has applications in the context of parcellation
        tensors. A parcellation tensor is one whose rows correspond to features
        (e.g., voxels, time points, frequency bins, or network nodes) and whose
        columns correspond to parcels. Element :math:`i, j` in this tensor
        accordingly indexes the assignment of feature :math:`j` to parcel
        :math:`i`. Examples of parcellation tensors might include atlases that
        map voxels to regions or affiliation matrices that map graph vertices
        to communities. It is often desirable to constrain feature-parcel
        assignments to :math:`[0, k]` for some :math:`k` and ensure that the
        sum over each feature's assignment is always :math:`k`. (Otherwise, the
        unnormalised loss could be improved by simply shrinking all weights.)
        For this reason, we can either normalise the loss or situate the
        parcellation tensor in the probability simplex using a
        :doc:`multi-logit (softmax) domain mapper <hypercoil.init.mapparam.ProbabilitySimplexParameter>`.

        The equilibrium loss attains a minimum when parcels are equal in their
        total weight. It has a trivial and uninteresting minimum where all
        parcel assignments are equiprobable for all features. Other minima,
        which might be of greater interest, occur where each feature is
        deterministically assigned to a single parcel. These minima can be
        favoured by using the equilibrium in conjunction with a penalty on the
        :doc:`entropy <hypercoil.loss.entropy>`.
    """

    equilibrium_spec = """
    level_axis: int or sequence of ints, optional
        The axis or axes over which to compute the equilibrium. Within each
        data instance or weight channel, all elements along the specified axis
        or axes should correspond to a single level or parcel. The default is
        -1.
    prob_axis: int or sequence of ints, optional
        The axis or axes over which to compute the probabilities (logit version
        only). The default is -2. In general the union of ``level_axis`` and
        ``prob_axis`` should be the same as ``instance_axes``.
    instance_axes: int or sequence of ints, optional
        The axis or axes corresponding to a single data instance or weight
        channel. This should be a superset of `level_axis`. The default is
        (-1, -2).
    keepdims: bool, optional
        As in :func:`jax.numpy.sum`. The default is True."""

    input_spec = """
    Parameters
    ----------
    X: Tensor
        A tensor of probabilities (or masses of another kind)."""

    return_spec = """
    Returns
    -------
    Tensor
        A tensor of equilibrium scores for each parcel or level."""

    fmt = NestedDocParse(
        long_description=long_description,
        ultra_long_description=ultra_long_description,
        equilibrium_spec=equilibrium_spec,
        input_spec=input_spec,
        return_spec=return_spec,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
    return func


@document_equilibrium
def equilibrium(
    X: Tensor,
    *,
    level_axis: Union[int, Sequence[int]] = -1,
    instance_axes: Union[int, Sequence[int]] = (-1, -2),
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Compute the parcel equilibrium.
    \
    {long_description}

    This function operates on probability tensors. For a version that operates
    on logits, see :func:`equilibrium_logit`.
    \
    {input_spec}\
    {equilibrium_spec}
    \
    {return_spec}
    """
    parcel = X.mean(level_axis, keepdims=True)
    total = X.mean(instance_axes, keepdims=True)
    return jnp.abs(parcel - total)


@document_equilibrium
def equilibrium_logit(
    X: Tensor,
    *,
    level_axis: Union[int, Sequence[int]] = -1,
    prob_axis: Union[int, Sequence[int]] = -2,
    instance_axes: Union[int, Sequence[int]] = (-1, -2),
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
    {input_spec}\
    {equilibrium_spec}
    \
    {return_spec}
    """
    probs = jax.nn.softmax(X, axis=prob_axis)
    return equilibrium(
        probs,
        level_axis=level_axis,
        instance_axes=instance_axes,
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

    ultra_long_description = r"""
    Regularise the second moment, e.g. to favour a dimension reduction mapping
    that is internally homogeneous.

    .. admonition:: Second Moment

        Second moment losses are based on a reduction of the second moment
        quantity

        :math:`\left[ A \circ \left (T - \frac{AT}{A\mathbf{1}} \right )^2  \right] \frac{\mathbf{1}}{A \mathbf{1}}`

        where the division operator is applied elementwise with broadcasting
        and the difference operator is applied via broadcasting. The
        broadcasting operations involved in the core computation -- estimating
        a weighted mean and then computing the weighted sum of squares about
        that mean -- are illustrated in the below cartoon.

        .. image:: ../_images/secondmomentloss.svg
            :width: 300
            :align: center

        *Illustration of the most memory-intensive stage of loss computation.
        The lavender tensor represents the weighted mean, the blue tensor the
        original observations, and the green tensor the weights (which might
        correspond to a dimension reduction mapping such as a parcellation).*

    .. note::
        In practice, we've found that using the actual second moment loss often
        results in large and uneven parcels. Accordingly, an unnormalised
        extension of the second moment (which omits the normalisation
        :math:`\frac{1}{A \mathbf{1}}`) is also available. This unnormalised
        quantity is equivalent to the weighted mean squared error about each
        weighted mean. In practice, we've found that this quantity works better
        for most of our use cases.

    .. warning::
        This loss can have a very large memory footprint, because it requires
        computing an intermediate tensor with dimensions equal to the number
        of rows in the linear mapping, multiplied by the number of columns in
        the linear mapping, multiplied by the number of columns in the
        dataset.

        When using this loss to learn a parcellation on voxelwise time series,
        the full computation will certainly be much too large to fit in GPU
        memory. Fortunately, because much of the computation is elementwise, it
        can be broken down along multiple axes without affecting the result.
        This tensor slicing is implemented automatically in the
        :doc:`ReactiveTerminal <hypercoil.engine.terminal.ReactiveTerminal>`
        class. Use extreme caution with ``ReactiveTerminals``, as improper use
        can result in destruction of the computational graph."""

    pparam_spec = """
    Parameters
    ----------
    X: Tensor
        A tensor of observations.
    weight: Tensor
        A tensor of weights."""

    second_moment_spec = """
    skip_normalise: bool, optional
        If True, do not include normalisation by the sum of the weights in the
        computation. In practice, this seems to work better than computing the
        actual second moment. Instead of computing the second moment, this
        corresponds to computed a weighted mean squared error about the mean.
        The default is False."""

    return_spec = """
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

    fmt = NestedDocParse(
        long_description=long_description,
        ultra_long_description=ultra_long_description,
        pparam_spec=pparam_spec,
        second_moment_spec=second_moment_spec,
        std_spec_nomean=std_spec_nomean,
        std_spec_mean=std_spec_mean,
        return_spec=return_spec,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
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
    sigma = (weight * diff**2).sum(-2) / normfac
    return sigma


def _second_moment_lowmem_impl(
    X: Tensor,
    weight: Tensor,
    mu: Tensor,
    *,
    skip_normalise: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Core computation for second-moment loss, with memory-efficient
    implementation.
    """
    weight = jnp.abs(weight)
    if not skip_normalise:
        weight = weight / weight.sum(-1, keepdims=True)
    Xsq = jnp.einsum('...jt,...ij->...it', X**2, weight)
    musq = jnp.einsum('...it,...ij->...it', mu**2, weight)
    Xmu = jnp.einsum('...it,...jt,...ij->...it', mu, X, weight)
    return Xsq - 2 * Xmu + musq


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
    {second_moment_spec}
    """
    if standardise:
        X = (X - X.mean(-1, keepdims=True)) / X.std(-1, keepdims=True)
    mu = weight @ X / weight.sum(-1, keepdims=True)
    return _second_moment_impl(
        X, weight, mu, skip_normalise=skip_normalise, key=key
    )


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
    {second_moment_spec}
    """
    if standardise_data:
        X = (X - X.mean(-1, keepdims=True)) / X.std(-1, keepdims=True)
    if standardise_mu:
        mu = (mu - mu.mean(-1)) / mu.std(-1)
    return _second_moment_impl(
        X, weight, mu, skip_normalise=skip_normalise, key=key
    )


# Functional homogeneity -----------------------------------------------------


def _homogeneity_prepare_args(
    X: Tensor,
    weight: Tensor,
    *,
    standardise: bool = False,
    skip_normalise: bool = False,
    use_geom_mean: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Prepare arguments for functional homogeneity computation.
    """
    if standardise:
        X = (X - X.mean(-1, keepdims=True)) / X.std(-1, keepdims=True)
    if use_geom_mean:
        weight = jnp.sqrt(weight)
    if not skip_normalise:
        weight = weight / (
            weight.sum(-2, keepdims=True) + jnp.finfo(weight.dtype).eps
        )
    return X, weight


def _point_weights(
    weight: Tensor,
    ref_weight: Tensor,
    neighbourhood: Tensor,
) -> Tensor:
    """
    Compute point weights for a local neighbourhood.
    """
    cmp_weight = weight[..., neighbourhood, :]
    return jnp.einsum('...p,...dp->...d', ref_weight, cmp_weight)


def functional_homogeneity(
    X: Tensor,
    weight: Tensor,
    *,
    standardise: bool = False,
    skip_normalise: bool = False,
    use_geom_mean: bool = False,
    use_schaefer: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Compute the functional homogeneity of a dataset.
    """
    orig_weight = weight
    X, weight = _homogeneity_prepare_args(
        X,
        weight,
        standardise=standardise,
        skip_normalise=skip_normalise,
        use_geom_mean=use_geom_mean,
    )
    fh = jnp.einsum(
        '...ap,...bp,...at,...bt->...p',
        weight, weight, X, X
    ) / X.shape[-1]
    # Correct for the diagonal: we shouldn't count the self-correlation
    parcel_size = orig_weight.sum(-2)
    fh = (fh - (weight ** 2).sum(-2)) * parcel_size / (parcel_size - 1)
    if use_schaefer:
        w = orig_weight.sum(-2)
        fh = (fh * w).sum(-1) / w.sum(-1)
    return fh


# def local_homogeneity(
#     X: Tensor,
#     weight: Tensor,
#     neighbourhood: Tensor,
#     *,
#     standardise: bool = False,
#     skip_normalise: bool = False,
#     use_geom_mean: bool = False,
#     use_schaefer: bool = False,
#     key: Optional['jax.random.PRNGKey'] = None,
# ) -> Tensor:
#     """
#     Compute the functional homogeneity of a dataset within a local
#     neighbourhood.
#     """
#     orig_weight = weight
#     X = X[..., neighbourhood, :]
#     weight = weight[..., neighbourhood, :]
#     unn_weight = weight
#     X, weight = _homogeneity_prepare_args(
#         X,
#         weight,
#         standardise=standardise,
#         skip_normalise=skip_normalise,
#         use_geom_mean=use_geom_mean,
#     )
#     fh = jnp.einsum(
#         '...nap,...nbp,...nat,...nbt->...p',
#         weight, weight, X, X
#     ) / X.shape[-1]
#     # Correct for the diagonal: we shouldn't count the self-correlation
#     parcel_size = unn_weight.sum((-3, -2))
#     fh = (fh - (weight ** 2).sum((-3, -2))) * parcel_size / (parcel_size - 1)
#     if use_schaefer:
#         w = orig_weight.sum(-2)
#         fh = (fh * w).sum(-1) / w.sum(-1)
#     return fh


def point_homogeneity(
    X: Tensor,
    weight: Tensor,
    neighbourhood: Tensor,
    reference: Optional[Tensor] = None,
    *,
    standardise: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Compute the functional homogeneity of a dataset at a set of points.
    """
    X, _ = _homogeneity_prepare_args(
        X,
        weight,
        standardise=standardise,
        skip_normalise=True,
        use_geom_mean=False,
    )
    if reference is not None:
        Xref = X[..., reference, :]
        wref = weight[..., reference, :]
    else:
        Xref = X
        wref = weight
    wpt = _point_weights(weight, wref, neighbourhood)
    fh = jnp.einsum(
        '...na,...nt,...nat->...',
        wpt,
        Xref,
        X[..., neighbourhood, :],
    ) / X.shape[-1] / wpt.sum((-2, -1))
    return fh


def point_similarity(
    weight: Tensor,
    neighbourhood: Tensor,
    reference: Optional[Tensor] = None,
    *,
    rectify_at: Union[float, Literal['auto']] = 'auto',
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Compute the local similarity of a weight assignment at a set of points.

    This loss primarily exists as an ablation for the ``point_agreement``
    function. It is equivalent to the agreement loss with the functional
    homogeneity term removed (i.e., ``kappa=1``).
    """
    if rectify_at == 'auto':
        # This blocks penalisation when the similarity kernel is any better
        # than random.
        rectify_at = 1 / weight.shape[-1]
    if reference is not None:
        ref_weight = weight[..., reference, :]
    else:
        ref_weight = weight
    wpt = _point_weights(weight, ref_weight, neighbourhood)
    return jax.nn.relu(rectify_at - wpt)


def point_agreement(
    X: Tensor,
    weight: Tensor,
    neighbourhood: Tensor,
    reference: Optional[Tensor] = None,
    *,
    kappa: Union[float, Literal['auto']] = 'auto',
    rectify_agreement: float = 1.,
    rectify_boundaries: Union[float, Literal['auto']] = 'auto',
    standardise: bool = False,
    rescale_result: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Compute a homogeneity-derived measure of agreement for a dataset at a set
    of points, penalising boundaries.
    """
    X, _ = _homogeneity_prepare_args(
        X,
        weight,
        standardise=standardise,
        skip_normalise=True,
        use_geom_mean=False,
    )
    if rectify_boundaries == 'auto':
        # This blocks penalisation when the similarity kernel is any better
        # than random.
        rectify_boundaries = 1 / weight.shape[-1]
    if kappa == 'auto':
        kappa = rectify_agreement / (rectify_agreement + rectify_boundaries)
    if reference is not None:
        ref_X = X[..., reference, :]
        ref_weight = weight[..., reference, :]
    else:
        ref_X = X
        ref_weight = weight
    wpt = _point_weights(weight, ref_weight, neighbourhood)
    fh = jnp.einsum(
        '...na,...nt,...nat->...na',
        wpt,
        ref_X,
        X[..., neighbourhood, :],
    ) / X.shape[-1] / wpt.sum((-2, -1))
    result = (
        kappa * jax.nn.relu(rectify_boundaries - wpt) +
        (1 - kappa) * jax.nn.relu(rectify_agreement - fh)
    )
    if rescale_result:
        result = result / (
            kappa * rectify_boundaries + (1 - kappa) * rectify_agreement
        )
    return result


# Batch correlation ----------------------------------------------------------


def document_batch_correlation(func):

    batch_correlation_spec = """
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
        batchwise correlations toward zero."""

    return_spec = """
    Returns
    -------
    tensor
        Absolute correlation of each vector in ``X`` with ``N``, after
        thresholding at `tol`. Note that, if you want the original
        correlations back, you will have to add ``tol`` to any nonzero
        correlations."""

    fmt = NestedDocParse(
        batch_correlation_spec=batch_correlation_spec,
        return_spec=return_spec,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
    return func


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
    from scipy.stats import t

    tsq = t.ppf(q=(1 - significance / tails), df=(batch_size - 2)) ** 2
    return jnp.sqrt(tsq / (batch_size - 2 + tsq))


@document_batch_correlation
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
        correlated.\
    {batch_correlation_spec}
    \
    {return_spec}
    """
    batch_size = X.shape[0]
    batchcorr = pairedcorr(
        X.swapaxes(0, -1).reshape(-1, batch_size),
        jnp.atleast_2d(N),
    )
    if tol == 'auto':
        tol = auto_tol(batch_size, significance=tol_sig)

    batchcorr_thr = jnp.maximum(jnp.abs(batchcorr) - tol, 0)
    if abs:
        return batchcorr_thr
    else:
        return jnp.sign(batchcorr) * batchcorr_thr


@document_batch_correlation
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

    Parameters
    ----------
    FC : tensor
        Tensor block containing functional connectivity measures.
    QC : tensor
        Vector of quality control measures with which the FC measures are to
        be correlated.\
    {batch_correlation_spec}
    \
    {return_spec}
    """
    return batch_corr(fc, qc, tol=tol, tol_sig=tol_sig, abs=abs, key=key)


# Distance-based losses ------------------------------------------------------


def document_spatial_loss(func):
    reference_tether_spec = """
    ref : Tensor
        Coordinates of the reference points to which the centres of mass of
        the objects in ``X`` should be tethered. Each object or parcel should
        have a single corresponding reference point.
    coor : Tensor
        Coordinates of the spatial locations in each of the columns of ``X``.
    radius : float or None (default 100)
        Radius of the spherical manifold on which the coordinates are
        located. If this is specified as None, it is assumed that the
        coordinates are in Euclidean space."""

    interhemispheric_long_description = """
    Displacement of centres of mass in one cortical hemisphere from
    corresponding centres of mass in the other cortical hemisphere.

    .. admonition:: Hemispheric Tether

        The hemispheric tether is defined as

        :math:`\sum_{\ell} \left\| \ell_{LH, centre} - \ell_{RH, centre} \right\|`

        where :math:`\ell` denotes a pair of regions, one in each cortical
        hemisphere.

        .. image:: ../_images/spatialnull.gif
            :width: 500
            :align: center

        `The symmetry of this spatial null model is enforced through a
        moderately strong hemispheric tether.`

    When an atlas is initialised with the same number of parcels in each
    cortical hemisphere compartment, the hemispheric tether can be used to
    approximately enforce symmetry and to enforce analogy between a pair of
    parcels in the two cortical hemispheres.

    .. warning::
        Currently, this loss only works in spherical surface space."""

    interhemispheric_tether_spec = """
    lh_coor : Tensor
        Coordinates of the spatial locations in each of the columns of ``lh``.
    rh_coor : Tensor
        Coordinates of the spatial locations in each of the columns of ``rh``.
    radius : float (default 100)
        Radius of the spherical manifold on which the coordinates are
        located."""

    compactness_spec = """
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
        masses that are already very close to the centre of mass."""

    compactness_long_description = r"""
    .. admonition:: Compactness

        The compactness is defined as

        :math:`\mathbf{1}^\intercal\left(A \circ \left\|C - \frac{AC}{A\mathbf{1}} \right\|_{cols} \right)\mathbf{1}`

        Given a coordinate set :math:`C` for the columns of a weight
        :math:`A`, the compactness measures the weighted average norm of the
        displacement of each of the weight's entries from its row's centre of
        mass. (The centre of mass is expressed above as
        :math:`\frac{AC}{A\mathbf{1}}`).

        .. image:: ../_images/compactloss.gif
            :width: 200
            :align: center

        `In this simulation, the compactness loss is applied with a
        multi-logit domain mapper and without any other losses or
        regularisations. The weights collapse to compact but unstructured
        regions of the field.`

    Penalising this quantity can promote more compact rows (i.e., concentrate
    the weight in each row over columns corresponding to coordinates close to
    the row's spatial centre of mass).

    .. warning::
        This loss can have a large memory footprint, because it requires
        computing an intermediate tensor with dimensions equal to the number
        of rows in the weight, multiplied by the number of columns in the
        weight, multiplied by the dimension of the coordinate space.
    """

    dispersion_spec = """
    metric : Callable
        Function to calculate the distance between centres of mass. This
        should take either one or two arguments, and should return the
        distance between each pair of observations."""

    dispersion_long_description = r"""
    Mutual separation among a set of vectors.

    .. admonition:: Vector dispersion

        The dispersion among a set of vectors :math:`v \in \mathcal{V}` is
        defined as

        :math:`\sum_{i, j} \mathrm{d}\left(v_i - v_j\right)`

        for some measure of separation :math:`\mathrm{d}`. (It is also
        valid to use a reduction other than the sum.)

    This can be used as one half of a clustering loss. Such a clustering loss
    would promote mutual separation among centroids (between-cluster
    separation, imposed by the ``VectorDispersion`` loss) while also promoting
    proximity between observations and their closest centroids (within-cluster
    closeness, for instance using a
    :doc:`norm loss <hypercoil.loss.norm>` or
    :doc:`compactness <hypercoil.loss.cmass.Compactness>`
    if the clusters are associated with spatial coordinates)."""

    fmt = NestedDocParse(
        reference_tether_spec=reference_tether_spec,
        interhemispheric_tether_spec=interhemispheric_tether_spec,
        interhemispheric_long_description=interhemispheric_long_description,
        compactness_spec=compactness_spec,
        compactness_long_description=compactness_long_description,
        dispersion_spec=dispersion_spec,
        dispersion_long_description=dispersion_long_description,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
    return func


@document_spatial_loss
def reference_tether(
    X: Tensor,
    ref: Tensor,
    coor: Tensor,
    *,
    radius: Optional[float] = 100.0,
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
        a single spatial location.\
    {reference_tether_spec}

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


@document_spatial_loss
def interhemispheric_tether(
    lh: Tensor,
    rh: Tensor,
    lh_coor: Tensor,
    rh_coor: Tensor,
    *,
    radius: float = 100.0,
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
        Tensor block containing the data for the right hemisphere.\
    {interhemispheric_tether_spec}

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
        -contralateral_ref[0, :]
    )
    return cmass_reference_displacement(
        weight=ipsilateral_weight,
        refs=contralateral_ref,
        coor=ipsilateral_coor,
        radius=radius,
    )


@document_spatial_loss
def compactness(
    X: Tensor,
    coor: Tensor,
    *,
    norm: Union[int, float, Literal['inf']] = 2,
    floor: float = 0.0,
    radius: Optional[float] = 100.0,
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
        a single spatial location.\
    {compactness_spec}

    Returns
    -------
    Tensor
        Distance of the masses in each object from the centre of mass of that
        object.
    """
    return diffuse(X=X, coor=coor, norm=norm, floor=floor, radius=radius)


@document_spatial_loss
def dispersion(
    X: Tensor,
    *,
    metric: Callable = linear_distance,
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
        the centre of mass of a single object or parcel.\
    {dispersion_spec}

    Returns
    -------
    Tensor
        Dispersion of the centres of mass of objects in a tensor block.
    """
    return sym2vec(-metric(X))


# Multivariate kurtosis ------------------------------------------------------


def document_mv_kurtosis(func):
    mv_kurtosis_spec = """
    l2 : float (default 0)
        L2 regularisation to be applied to the covariance matrix to ensure
        that it is invertible.
    dimensional_scaling : bool (default False)
        The expected value of the multivariate kurtosis for a normally
        distributed, stationary process of infinite duration with d channels
        (or variables) is :math:`d (d + 2)`. Setting this to true normalises
        for the process dimension by dividing the obtained kurtosis by
        :math:`d (d + 2)`. This has no effect in determining the optimum."""

    mv_kurtosis_long_description = """
    This is the multivariate kurtosis following Mardia, as used by Laumann
    and colleagues in the setting of functional connectivity. It is equal to
    the mean of the squared Mahalanobis norm of each time point (as
    parameterised by the inverse covariance of the multivariate time series)."""

    fmt = NestedDocParse(
        mv_kurtosis_spec=mv_kurtosis_spec,
        mv_kurtosis_long_description=mv_kurtosis_long_description,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
    return func


@document_mv_kurtosis
def multivariate_kurtosis(
    ts: Tensor,
    *,
    l2: float = 0.0,
    dimensional_scaling: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    """
    Multivariate kurtosis of a time series.
    \
    {mv_kurtosis_long_description}

    Parameters
    ----------
    ts : Tensor
        Multivariate time series to be evaluated. Each row is a channel or
        variable, and each column is a time point.\
    {mv_kurtosis_spec}

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
    return -(maha**2).mean(-1) / denom


# Connectopies ---------------------------------------------------------------


def document_connectopy(func: Callable) -> Callable:

    connectopy_pparam_spec = """
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
        graph Laplacian."""

    connectopy_spec = """
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

    modularity_spec = """
    gamma : float (default 1.)
        Modularity parameter. Takes the place of the ``omega`` argument in
        ``connectopy``.
    exclude_diag : bool (default True)
        If True, then the diagonal of the affinity matrix is set to zero."""

    prog_theta_spec = """
    progressive_theta : bool (default False)
        When this is True, a ``theta`` is generated such that the last map in
        ``Q`` has a weight of 1, the second-to-last has a weight of 2, and so
        on. This can be used to encourage the last column to correspond to the
        least important connectopic map and the first column to correspond to
        the most important connectopic map."""

    connectopy_long_description = r"""
    .. admonition:: Connectopic functional

        Given an affinity matrix A, the connectopic functional is the
        objective

        :math:`\mathbf{1}^\intercal \left( \mathbf{A} \circ S_\theta(\mathbf{Q}) \right) \mathbf{1}`

        for a pairwise function S. The default pairwise function is the square
        of the L2 distance. The columns of the Q that minimises the objective
        are the learned connectopic maps.

    .. warning::
        If you're using this for a well-characterised connectopic map with a
        closed-form or algorithmically optimised solution, such as Laplacian
        eigenmaps or many forms of community detection, then in most cases you
        would be better off directly computing exact maps rather than using
        this functional to approximate them.

        Because this operation attempts to learn all of the maps that jointly
        minimise the objective in a single shot rather than using iterative
        projection, it is more prone to misalignment than a projective approach
        for eigendecomposition-based maps.

    .. danger::
        Note that a connectopic loss is often insufficient on its own as a
        loss. It should be combined with appropriate projections and
        constraints, for instance to ensure the learned maps are zero-centred
        and orthogonal."""

    modularity_long_description = r"""
    Differentiable relaxation of the Girvan-Newman modularity.

    This relaxation supports non-deterministic assignments of vertices to
    communities and non-assortative linkages between communities. It reverts
    to standard behaviour when the inputs it is provided are standard (i.e.,
    deterministic and associative).

    .. admonition:: Girvan-Newman Modularity Relaxation

        The relaxed modularity loss is defined as the negative sum of all
        entries in the Hadamard (elementwise) product between the modularity
        matrix and the coaffiliation matrix.

        :math:`\mathcal{L}_Q = -\nu_Q \mathbf{1}^\intercal \left( B \circ H \right) \mathbf{1}`

        .. image:: ../_images/modularityloss.svg
            :width: 500
            :align: center

        - The modularity matrix :math:`B` is the difference between the
          observed connectivity matrix :math:`A` and the expected connectivity
          matrix under some null model that assumes no community structure,
          :math:`P`: :math:`B = A - \gamma P`.

          - The community resolution parameter :math:`\gamma` essentially
            determines the scale of the community structure that optimises the
            relaxed modularity loss.

          - By default, we use the Girvan-Newman null model
            :math:`P_{GN} = \frac{A \mathbf{1} \mathbf{1}^\intercal A}{\mathbf{1}^\intercal A \mathbf{1}}`,
            which can be interpreted as the expected weight of connections
            between each pair of vertices if all existing edges are cut and
            then randomly rewired.

          - Note that :math:`A \mathbf{1}` is the in-degree of the adjacency
            matrix and :math:`\mathbf{1}^\intercal A` is its out-degree, and
            the two are transposes of one another for symmetric :math:`A`.
            Also note that the denominator
            :math:`\mathbf{1}^\intercal A \mathbf{1}` is twice the number of
            edges for an undirected graph.)

        - The coaffiliation matrix :math:`H` is calculated as
          :math:`H = C_{in} L C_{out}^\intercal`, where
          :math:`C_{in} \in \mathbb{R}^{(v_{in} \times c)}` and
          :math:`C_{out} \in \mathbb{R}^{(v_{out} \times c)}` are proposed
          assignment weights of in-vertices and out-vertices to communities.
          :math:`L \in \mathbb{R}^{c \times c)}` is the proposed coupling
          matrix among each pair of communities and defaults to identity to
          represent associative structure.

          - Note that, when :math:`C_{in} = C_{out}` is deterministically in
            :math:`\{0, 1\}` and :math:`L = I`, this term reduces to the
            familiar delta-function notation for the true Girvan-Newman
            modularity.

    Penalising this favours a weight that induces a modular community
    structure on the input matrix -- or, an input matrix whose structure
    is reasonably accounted for by the proposed community affiliation
    weights.

    .. warning::
        To conform with the network community interpretation of this loss
        function, parameters representing the community affiliation :math:`C`
        and coupling :math:`L` matrices can be pre-transformed. Mapping the
        community affiliation matrix :math:`C` through a
        :doc:`softmax <hypercoil.init.domain.MultiLogit>`
        function along the community axis lends the affiliation matrix the
        intuitive interpretation of distributions over communities, or a
        quantification of the uncertainty of each vertex's community
        assignment. Similarly, the coupling matrix can be pre-transformed
        through a
        :doc:`sigmoid <hypercoil.init.domain.Logit>`
        to constrain inter-community couplings to :math:`(0, 1)`.
    .. note::
        Because the community affiliation matrices :math:`C` induce
        parcellations, we can regularise them using parcellation losses. For
        instance, penalising the
        :doc:`entropy <hypercoil.loss.entropy>`
        will promote a solution wherein each node's community assignment
        probability distribution is concentrated in a single community.
        Similarly, using parcel
        :doc:`equilibrium <hypercoil.loss.equilibrium>` will favour a solution
        wherein communities are of similar sizes."""

    return_spec = """
    Returns
    -------
    Tensor
        Connectopic functional value."""

    fmt = NestedDocParse(
        connectopy_pparam_spec=connectopy_pparam_spec,
        connectopy_spec=connectopy_spec,
        prog_theta_spec=prog_theta_spec,
        modularity_spec=modularity_spec,
        return_spec=return_spec,
        connectopy_long_description=connectopy_long_description,
        modularity_long_description=modularity_long_description,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
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
    negative_affinity: Optional[Literal[
        'abs',
        'rectify',
        'reciprocal',
        'complement',
    ]] = 'complement',
    progressive_theta: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Generalised connectopic functional, for computing different kinds of
    connectopic maps.
    \
    {connectopy_long_description}
    \
    {connectopy_pparam_spec}\
    {connectopy_spec}\
    {prog_theta_spec}
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
    H = dissimilarity(Q, theta=theta)

    match negative_affinity:
        case 'abs':
            A = jnp.abs(A)
        case 'rectify':
            A = jnp.maximum(A, 0)
        case 'reciprocal':
            neg_idx = A < 0
            A = jnp.where(neg_idx, -A, A)
            H = jnp.where(neg_idx, 1 / (H + jnp.finfo(H.dtype).eps), H)
        case 'complement':
            neg_idx = A < 0
            Hmax = jnp.maximum(
                H.max((-2, -1), keepdims=True),
                1.0,
            )
            A = jnp.where(neg_idx, -A, A)
            H = jnp.where(neg_idx, Hmax - H, H)

    if D is not None:
        A = D @ A @ D.swapaxes(-2, -1)
    return (H * A).sum((-2, -1))


@document_connectopy
def modularity(
    Q: Tensor,
    A: Tensor,
    D: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
    *,
    gamma: float = 1.0,
    exclude_diag: bool = True,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Modularity functional.

    The connectopies that minimise the modularity functional define a
    community structure for the graph.
    \
    {connectopy_pparam_spec}\
    {connectopy_spec}\
    {modularity_spec}
    \
    {return_spec}
    """

    def dissimilarity(Q, theta):
        return -coaffiliation(
            Q,
            L=theta,
            normalise_coaffiliation=True,
            exclude_diag=exclude_diag,
        )

    def affinity(A, omega):
        return modularity_matrix(
            A,
            gamma=omega,
            normalise_modularity=True,
        )

    return connectopy(
        Q=Q,
        A=A,
        D=D,
        theta=theta,
        omega=gamma,
        dissimilarity=dissimilarity,
        affinity=affinity,
        negative_affinity=None,
        key=key,
    )


@document_connectopy
def eigenmaps(
    Q: Tensor,
    A: Tensor,
    theta: Optional[Tensor] = None,
    omega: Optional[Tensor] = None,
    *,
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
    {connectopy_pparam_spec}\
    {connectopy_spec}
    normalise : bool (default True)
        If True, then the graph Laplacian is normalised by the vertex degrees.
        Takes the place of the ``D`` and ``omega`` arguments in
        ``connectopy``.
    \
    {return_spec}
    """
    def dissimilarity(Q, theta):
        Q = Q - Q.mean(-2, keepdims=True)
        Qproj = Q / jnp.linalg.norm(Q, axis=-2, keepdims=True)
        return linear_distance(Qproj, theta=theta)

    def affinity(A, omega):
        A = jax.nn.relu(corr_kernel(A, theta=omega))
        D = A.sum(-1, keepdims=True) ** -1
        return (D @ D.swapaxes(-1, -2)) * A

    return connectopy(
        Q=Q,
        A=A,
        theta=theta,
        omega=omega,
        dissimilarity=dissimilarity,
        affinity=affinity,
        negative_affinity='rectify',
        key=key,
    )
