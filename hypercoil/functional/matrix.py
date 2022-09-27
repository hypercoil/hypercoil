# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Special matrix functions.
"""
import jax
import jax.numpy as jnp
import math
from functools import partial
from typing import Literal, Optional, Tuple
from .utils import conform_mask
from ..engine import Tensor, vmap_over_outer


def cholesky_invert(X: Tensor) -> Tensor:
    """
    Invert a symmetric positive definite matrix using Cholesky decomposition.

    .. warning::
        The input matrix must be symmetric and positive definite. If this is
        not the case, the function will either raise a `LinAlgError` or
        produce undefined results. For positive semidefinite matrices, the
        Moore-Penrose pseudoinverse can be used instead.

    .. admonition::
        This does not appear to be any faster than using the inverse directly.
        In fact, it is almost always slower than ``jnp.linalg.inv``. It's
        retained for historical reasons.

    :Dimension: **Input :** :math:`(*, D, D)`
                    D denotes the row or column dimension of the matrices to
                    be inverted. ``*`` denotes any number of preceding
                    dimensions.
                **Output :** :math:`(*, D, D)`
                    As above.

    Parameters
    ----------
    A : Tensor
        Symmetric positive definite matrix.

    Returns
    -------
    Ainv : Tensor
        Inverse of the input matrix.
    """
    L = jnp.linalg.cholesky(X)
    Li = jnp.linalg.inv(L)
    return Li.swapaxes(-1, -2) @ Li


def symmetric(
    X: Tensor,
    skew: bool = False,
    axes: Tuple[int, int] = (-2, -1)
) -> Tensor:
    """
    Impose symmetry on a tensor block.

    The input tensor block is averaged with its transpose across the slice
    delineated by the specified axes.

    Parameters
    ----------
    X : Tensor
        Input to be symmetrised.
    skew : bool (default False)
        Indicates whether skew-symmetry (antisymmetry) should be imposed on
        the input.
    axes : tuple(int, int) (default (-2, -1))
        Axes that delineate the square slices of the input on which symmetry
        is imposed. By default, symmetry is imposed on the last 2 slices.

    Returns
    -------
    output : Tensor
        Input with symmetry imposed across specified slices.
    """
    if not skew:
        return (X + X.swapaxes(*axes)) / 2
    else:
        return (X - X.swapaxes(*axes)) / 2


#TODO: marking this as an experimental function
#      When it's implemented, we should change symmetric to single dispatch.
#      It unfortunately won't work with our top-k format for sparse tensors
#      without potentially deleting some existing nonzero entries.
# def symmetric_sparse(
#     W: Tensor,
#     edge_index: Tensor,
#     skew: bool = False,
#     n_vertices: Optional[int] = None,
#     divide: bool = True,
#     return_coo: bool = False
# ) -> Tensor:
#     r"""
#     Impose symmetry (undirectedness) on a weight-edge index pair
#     representation of a graph.

#     All edges are duplicated and their source and target vertices reversed.

#     .. note::

#         This operation is differentiable with respect to the weight tensor
#         ``W``.

#     .. note::

#         The weight tensor ``W`` can have any number of batched elements, but
#         all must include the same edges as indexed in ``edge_index``. If
#         certain edges are missing in some of the batched examples but present
#         in others, the missing edges should explicitly be set to 0 in ``W``.

#     :Dimension: **W :** :math:`(*, E)`
#                     ``*`` denotes any number of preceding dimensions. E
#                     denotes the number of nonzero edges.
#                 **edge_index :** :math:`(2, E)`
#                     As above.
#                 **W_out :** :math:`(*, E_{out})`
#                     :math:`E_{out}` denotes the number of nonzero edges after
#                     symmetrisation.
#                 **edge_index :** :math:`(2, E_{out})`
#                     As above.

#     Parameters
#     ----------
#     W : tensor
#         List of weights corresponding to the edges in ``edge_index``
#         (potentially batched).
#     edge_index : ``LongTensor``
#         List of edges corresponding to the provided weights. Each column
#         contains the index of the source vertex and the index of the target
#         vertex for the corresponding weight in ``W``.
#     skew : bool (default False)
#         Indicates whether skew-symmetry (antisymmetry) should be imposed on
#         the input.
#     """
#     if n_vertices is None:
#         n_vertices = edge_index.max() + 1
#     if W.dim() > 1:
#         shape = (n_vertices, n_vertices, *W.shape[:-1])
#         W_in = W.transpose(0, -1)
#     else:
#         shape = (n_vertices, n_vertices)
#         W_in = W
#     base = torch.sparse_coo_tensor(
#         edge_index,
#         W_in,
#         shape
#     )
#     if not skew:
#         out = (base + base.transpose(0, 1)).coalesce()
#     else:
#         out = (base - base.transpose(0, 1)).coalesce()
#     if divide:
#         out = out / 2
#     if return_coo:
#         return out
#     elif W.dim() > 1:
#         return out.values().transpose(-1, 0), out.indices()
#     else:
#         return out.values(), out.indices()


def spd(
    X: Tensor,
    eps: float = 1e-6,
    method: Literal['eig', 'svd'] = 'eig'
) -> Tensor:
    """
    Impose symmetric positive definiteness on a tensor block.

    Each input matrix is first made symmetric. Next, the symmetrised inputs
    are decomposed via diagonalisation or SVD. If the inputs are diagonalised,
    the smallest eigenvalue is identified, and a scaled identity matrix is
    added to the input such that the smallest eigenvalue of the resulting
    matrix is no smaller than a specified threshold. If the inputs are
    decomposed via SVD, then the matrix is reconstituted from the left
    singular vectors (which are in theory identical to the right singular
    vectors up to sign for a symmetric matrix) and the absolute values of the
    eigenvalues. Thus, the maximum reconstruction error is in theory the
    minimum threshold for diagonalisation and the absolute value of the
    smallest negative eigenvalue for SVD.

    Parameters
    ----------
    X : Tensor
        Input to be made positive definite.
    eps : float
        Minimum threshold for the smallest eigenvalue identified in diagonal
        eigendecomposition. If diagonalisation is used to impose positive
        semidefiniteness, then this will be the minimum possible eigenvalue of
        the output. If SVD is used to impose positive semidefiniteness, then
        this is unused.
    method : ``'eig'`` or ``'svd'``
        Method used to ensure that all eigenvalues are positive.

        - ``eig`` denotes that the input matrices are symmetrised and then
          diagonalised. The method returns the symmetrised sum of the input
          and an identity matrix scaled to guarantee no eigenvalue is smaller
          than `eps`.
        - ``svd`` denotes that the input matrices are decomposed via singular
          value decomposition after symmetrisation. The method returns a
          recomposition of the matrix that treats the left singular vectors
          and singular values output from SVD as though they were outputs of
          diagonalisation, thereby guaranteeing that no eigenvalue is smaller
          than the least absolute value among all input eigenvalues. Note that
          this margin is occasionally insufficient to avoid numerical error if
          the same matrix is again decomposed.

    Returns
    -------
    output : Tensor
        Input modified so that each slice is symmetric and positive definite.
    """
    if method == 'eig':
        L = vmap_over_outer(jnp.linalg.eigvalsh, 2)((symmetric(X),))
        lmin = L.min(axis=-1) - eps
        lmin = jnp.minimum(lmin, 0).squeeze()
        return symmetric(
            X - lmin[..., None, None] * jnp.eye(X.shape[-1]))
    elif method == 'svd':
        Q, L, _ = vmap_over_outer(jnp.linalg.svd, 2)((symmetric(X),))
        return symmetric(
            Q @ (L[..., None] * Q.swapaxes(-1, -2)))


def expand_outer(
    L: Tensor,
    R: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    symmetry: Optional[Literal['cross', 'skew']] = None
) -> Tensor:
    r"""
    Multiply out a left and a right generator matrix as an outer product.

    The rank of the output is limited according to the inner dimensions of the
    input generators. This approach can be used to produce a low-rank output
    or to share the generators' parameters across the rows and columns of the
    output matrix.

    :Dimension: **L :** :math:`(*, H, rank)`
                    H denotes the height of the expanded matrix, and rank
                    denotes its maximum rank. ``*`` denotes any number of
                    preceding dimensions.
                **R :** :math:`(*, W, rank)`
                    W denotes the width of the expanded matrix.
                **C :** :math:`(*, rank, rank)`
                    As above.
                **Output :** :math:`(*, H, W)`
                    As above.

    Parameters
    ----------
    L : Tensor
        Left generator of a low-rank matrix (:math:`L R^\intercal`).
    R : Tensor or None (default None)
        Right generator of a low-rank matrix (:math:`L R^\intercal`). If this
        is None, then the output matrix is symmetric :math:`L L^\intercal`.
    C : Tensor or None (default None)
        Coupling term. If this is specified, each outer product expansion is
        modulated by a corresponding coefficient in the coupling matrix.
        Providing a vector is equivalent to providing a diagonal coupling
        matrix. This term can, for instance, be used to toggle between
        positive and negative semidefinite outputs.
    symmetry : ``'cross'``, ``'skew'``, or None (default None)
        Symmetry constraint imposed on the generated low-rank template matrix.

        * ``cross`` enforces symmetry by replacing the initial expansion with
          the average of the initial expansion and its transpose,
          :math:`\frac{1}{2} \left( L R^\intercal + R L^\intercal \right)`
        * ``skew`` enforces skew-symmetry by subtracting from the initial
          expansion its transpose,
          :math:`\frac{1}{2} \left( L R^\intercal - R L^\intercal \right)`
        * Otherwise, no explicit symmetry constraint is imposed. Symmetry can
          also be enforced by passing None for R or by passing the same input
          for R and L. (This approach also guarantees that the output is
          positive semidefinite.)
    """
    if L.ndim == 1:
        L = L[..., None]
    if R is None:
        R = L
    elif R.ndim == 1:
        R = R[..., None]
    if C is None:
        output = L @ R.swapaxes(-2, -1)
    elif C.shape[-1] == C.shape[-2] == L.shape[-1]:
        output = L @ C @ R.swapaxes(-2, -1)
    else:
        output = L @ (C * R.swapaxes(-2, -1))
    #TODO: Unit tests are not hitting this conditional...
    if symmetry == 'cross' or symmetry == 'skew':
        return symmetric(output, skew=(symmetry == 'skew'))
    return output


def recondition_eigenspaces(
    A: Tensor,
    psi : float,
    xi : float,
    key : Tensor
) -> Tensor:
    r"""
    Recondition a positive semidefinite matrix such that it has no zero
    eigenvalues, and all of its eigenspaces have dimension one.

    This reconditioning operation should help stabilise differentiation
    through singular value decomposition.

    This operation modifies the input matrix A following

    :math:`A := A + \left(\psi - \frac{\xi}{2}\right) I + I\mathbf{x}`

    :math:`x_i \sim \mathrm{Uniform}(0, \xi) \forall x_i`

    :math:`\psi > \xi`

    Parameters
    ----------
    A : tensor
        Matrix or matrix block to be reconditioned.
    psi : float
        Reconditioning parameter for ensuring nonzero eigenvalues.
    xi : float
        Reconditioning parameter for ensuring nondegenerate eigenvalues.

    Returns
    -------
    tensor
        Reconditioned matrix or matrix block.
    """
    x = jax.random.uniform(key=key, shape=A.shape, maxval=xi)
    mask = jnp.eye(A.shape[-1])
    return A + (psi - xi + x) * mask


def delete_diagonal(A: Tensor) -> Tensor:
    """
    Delete the diagonal from a block of square matrices. Dimension is inferred
    from the final axis.
    """
    mask = ~jnp.eye(A.shape[-1], dtype=bool)
    return A * mask


def diag_embed(v: Tensor, offset: int = 0) -> Tensor:
    """
    Embed a vector into a diagonal matrix.
    """
    return vmap_over_outer(partial(jnp.diagflat, k=offset), 1)((v,))


def fill_diagonal(A: Tensor, fill: float = 0, offset: int = 0) -> Tensor:
    """
    Fill a selected diagonal in a block of square matrices. Dimension is
    inferred from the final axes.
    """
    dim = A.shape[-2:]
    mask = jnp.ones(max(dim) - abs(offset), dtype=bool)
    mask = diag_embed(mask, offset=offset)
    mask = mask[:dim[0], :dim[1]]
    mask = conform_mask(A, mask, axis=(-2, -1))
    return jnp.where(mask, fill, A)


def toeplitz_2d(
    c: Tensor,
    r: Optional[Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
    fill_value: float = 0
) -> Tensor:
    """
    Construct a 2D Toeplitz matrix from a column and row vector.

    Based on the second method posted by @mattjj and evaluated by
    @blakehechtman here:
    https://github.com/google/jax/issues/1646#issuecomment-1139044324
    Apparently this method underperforms on TPU. But given that TPUs are
    only available for Google, this is probably not a big deal.

    Our method is more flexible in that it admits Toeplitz matrices without
    circulant structure. Our API is also closer to that of the scipy toeplitz
    function. We also support a fill value for the matrix. See
    :func:`toeplitz` for complete API documentation.

    .. note::

        Use the :func:`toeplitz` function for an API that supports any number
        of leading dimensions.
    """
    if r is None:
        r = c
    m_in, n_in = c.shape[-1], r.shape[-1]
    m, n = shape if shape is not None else (m_in, n_in)
    d = max(m, n)
    if (m != n) or (m_in != n_in) or (m_in != d):
        r_arg, c_arg = fill_value * jnp.ones(d), fill_value * jnp.ones(d)
        r_arg = r_arg.at[:n_in].set(r)
        c_arg = c_arg.at[:m_in].set(c)
    else:
        r_arg, c_arg = r, c
    c_arg = jnp.flip(c_arg, -1)

    mask = jnp.zeros(2 * d - 1, dtype=bool)
    mask = mask.at[:(d - 1)].set(True)
    iota = jnp.arange(d)
    def roll(c, r, i, mask):
        rs = jnp.roll(r, i, axis=-1)
        cs = jnp.roll(c, i + 1, axis=-1)
        ms = jnp.roll(mask, i, axis=-1)[-d:]
        return jnp.where(ms, cs, rs)
    f = jax.vmap(roll, in_axes=(None, None, 0, None))
    return f(c_arg, r_arg, iota[..., None], mask)[..., :m, :n]


def toeplitz(
    c: Tensor,
    r: Optional[Tensor] = None,
    shape: Optional[Tuple[int, int]] = None,
    fill_value: float = 0.,
) -> Tensor:
    r"""
    Populate a block of tensors with Toeplitz banded structure.

    .. warning::

        Inputs ``c`` and ``r`` must contain the same first element for
        functionality to match ``scipy.toeplitz``. This is not checked.
        In the event that this is not the case, ``c[0]`` is ignored.
        Note that this is the opposite of ``scipy.toeplitz``.

    :Dimension: **c :** :math:`(C, *)`
                    C denotes the number of elements in the first column whose
                    values are propagated along the matrix diagonals. ``*``
                    denotes any number of additional dimensions.
                **r :** :math:`(R, *)`
                    R denotes the number of elements in the first row whose
                    values are propagated along the matrix diagonals.
                **fill_value :** :math:`(*)`
                    As above.
                **Output :** :math:`(*, C^{*}, R^{*})`
                    :math:`C^{*}` and :math:`{*}` default to C and R unless
                    specified otherwise in the `dim` argument.

    Parameters
    ----------
    c: Tensor
        Tensor of entries in the first column of each Toeplitz matrix. The
        first axis corresponds to a single matrix column; additional
        dimensions correspond to concatenation of Toeplitz matrices into a
        stack or block tensor.
    r: Tensor
        Tensor of entries in the first row of each Toeplitz matrix. The first
        axis corresponds to a single matrix row; additional dimensions
        correspond to concatenation of Toeplitz matrices into a stack or block
        tensor. The first entry in each column should be the same as the first
        entry in the corresponding column of `c`; otherwise, it will be
        ignored.
    dim: 2-tuple of (int, int) or None (default)
        Dimension of each Toeplitz banded matrix in the output block. If this
        is None or unspecified, it defaults to the sizes of the first axes of
        inputs `c` and `r`. Otherwise, the row and column inputs are extended
        until their dimensions equal those specified here. This can be useful,
        for instance, to create a large banded matrix with mostly zero
        off-diagonals.
    fill_value: Tensor or float (default 0)
        Specifies the value that should be used to populate the off-diagonals
        of each Toeplitz matrix if the specified row and column elements are
        extended to conform with the specified `dim`. If this is a tensor,
        then each entry corresponds to the fill value in a different data
        channel. Has no effect if ``dim`` is None.

    Returns
    -------
    out: Tensor
        Block of Toeplitz matrices populated from the specified row and column
        elements.
    """
    return vmap_over_outer(
        partial(toeplitz_2d, shape=shape, fill_value=fill_value), 1
    )((c, r))


def sym2vec(sym: Tensor, offset: int = 1) -> Tensor:
    """
    Convert a block of symmetric matrices into ravelled vector form.

    Ordering in the ravelled form follows the upper triangle of the matrix
    block.

    Parameters
    ----------
    sym : tensor
        Block of tensors to convert. The last two dimensions should be equal,
        with each slice along the final 2 axes being a square, symmetric
        matrix.
    offset : int (default 1)
        Offset from the main diagonal where the upper triangle begins. By
        default, the main diagonal is not included. Set this to 0 to include
        the main diagonal.

    Returns
    -------
    vec : tensor
        Block of ravelled vectors formed from the upper triangles of the input
        `sym`, beginning with the diagonal offset from the main by the input
        `offset`.
    """
    idx = jnp.triu_indices(m=sym.shape[-2], n=sym.shape[-1], k=offset)
    shape = sym.shape[:-2]
    #print(idx, shape)
    vec = sym[..., idx[0], idx[1]]
    return vec.reshape(*shape, -1)


def vec2sym(vec: Tensor, offset: int = 1) -> Tensor:
    """
    Convert a block of ravelled vectors into symmetric matrices.

    The ordering of the input vectors should follow the upper triangle of the
    matrices to be formed.

    Parameters
    ----------
    vec : tensor
        Block of vectors to convert. Input vectors should be of length
        (n choose 2), where n is the number of elements on the offset
        diagonal, plus 1.
    offset : int (default 1)
        Offset from the main diagonal where the upper triangle begins. By
        default, the main diagonal is not included. Set this to 0 to place
        elements along the main diagonal.

    Returns
    -------
    sym : tensor
        Block of symmetric matrices formed by first populating the offset
        upper triangle with the elements from the input `vec`, then
        symmetrising.
    """
    shape = vec.shape[:-1]
    vec = vec.reshape(*shape, -1)
    cn2 = vec.shape[-1]
    side = int(0.5 * (math.sqrt(8 * cn2 + 1) + 1)) + (offset - 1)
    idx = jnp.triu_indices(m=side, n=side, k=offset)
    sym = jnp.zeros((*shape, side, side))
    sym = sym.at[..., idx[0], idx[1]].set(vec)
    sym = sym + sym.swapaxes(-1, -2)
    if offset == 0:
        mask = jnp.eye(side, dtype=bool)
        sym = sym.at[..., mask].set(sym[..., mask] / 2)
    return sym


def squareform(X: Tensor) -> Tensor:
    """
    Convert between symmetric matrix and vector forms.

    .. warning::
        Unlike numpy or matlab implementations, this does not verify a
        conformant input.

    Parameters
    ----------
    X : tensor
        Block of symmetric matrices, in either square matrix or vectorised
        form.

    Returns
    -------
    tensor
        If the input block is in square matrix form, returns it
        :doc:`in vector form <hypercoil.functional.matrix.sym2vec>`.
        If the input block is in vector form, returns it
        :doc:`in square matrix form <hypercoil.functional.matrix.vec2sym>`.
    """
    if (X.shape[-2] == X.shape[-1]
        and jnp.allclose(X, X.swapaxes(-1, -2))):
        return sym2vec(X, offset=1)
    else:
        return vec2sym(X, offset=1)
