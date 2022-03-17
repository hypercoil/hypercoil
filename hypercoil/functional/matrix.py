# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Special matrix functions.
"""
import torch
import math


def invert_spd(A, force_invert_singular=True):
    """
    Invert a symmetric positive definite matrix.

    Currently, this operates by computing the Cholesky decomposition of the
    matrix, inverting the decomposition, and recomposing. If the Cholesky
    decomposition fails because the input is singular, then it instead returns
    the Moore-Penrose pseudoinverse.

    :Dimension: **Input :** :math:`(*, D, D)`
                    D denotes the row or column dimension of the matrices to
                    be inverted. ``*`` denotes any number of preceding
                    dimensions.
                **Output :** :math:`(*, D, D)`
                    As above.

    Parameters
    ----------
    A : Tensor
        Batch of symmetric positive definite matrices.

    Returns
    -------
    Ainv : Tensor
        Inverse or Moore-Penrose pseudoinverse of each matrix in the input
        batch.
    """
    try:
        L = torch.linalg.cholesky(A)
        Li = torch.inverse(L)
        return Li.transpose(-1, -2) @ Li
    except RuntimeError:
        #TODO: getting to this point often means the matrix is singular,
        # so it probably should fail or at least have the option to
        #TODO: Right now we're using the pseudoinverse here. Does this make
        # more sense than trying again with a reconditioned matrix?
        if force_invert_singular:
            return torch.pinverse(A)
            return symmetric(invert_spd(
                recondition_eigenspaces(A, psi=1e-4, xi=1e-5),
                force_invert_singular=False
            ))
        raise


def symmetric(X, skew=False, axes=(-2, -1)):
    """
    Impose symmetry on a tensor block.

    The input tensor block is averaged with its transpose across the slice
    delineated by the specified axes.

    Parameters
    ----------
    X : Tensor
        Input to be symmetrised.
    skew : bool (default False)
        Indicates whether skew-symmetry (antisymmetry) should be imposed on the
        input.
    axes : tuple(int, int) (default (-2, -1))
        Axes that delineate the square slices of the input on which symmetry
        is imposed. By default, symmetry is imposed on the last 2 slices.

    Returns
    -------
    output : Tensor
        Input with symmetry imposed across specified slices.
    """
    if not skew:
        return (X + X.transpose(*axes)) / 2
    else:
        return (X - X.transpose(*axes)) / 2


def spd(X, eps=1e-6, method='eig'):
    """
    Impose symmetric positive definiteness on a tensor block.

    Each input matrix is first made symmetric. Next, the symmetrised inputs are
    decomposed via diagonalisation or SVD. If the inputs are diagonalised, the
    smallest eigenvalue is identified, and a scaled identity matrix is added to
    the input such that the smallest eigenvalue of the resulting matrix is no
    smaller than a specified threshold. If the inputs are decomposed via SVD,
    then the matrix is reconstituted from the left singular vectors (which are
    in theory identical to the right singular vectors up to sign for a
    symmetric matrix) and the absolute values of the eigenvalues. Thus, the
    maximum reconstruction error is in theory the minimum threshold for
    diagonalisation and the absolute value of the smallest negative eigenvalue
    for SVD.

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
          diagonalised. The method returns the symmetrised sum of the input and
          an identity matrix scaled to guarantee no eigenvalue is smaller than
          `eps`.
        - ``svd`` denotes that the input matrices are decomposed via singular
          value decomposition after symmetrisation. The method returns a
          recomposition of the matrix that treats the left singular vectors and
          singular values output from SVD as though they were outputs of
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
        L = torch.linalg.eigvalsh(symmetric(X))
        lmin = L.amin(axis=-1) - eps
        lmin = torch.minimum(lmin, torch.zeros(1)).squeeze()
        return symmetric(X - lmin[..., None, None] * torch.eye(X.size(-1)))
    elif method == 'svd':
        Q, L, _ = torch.svd(symmetric(X))
        return symmetric(Q @ torch.diag_embed(L) @ Q.transpose(-1, -2))


def expand_outer(L, R=None, symmetry=None):
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
                **Output :** :math:`(*, H, W)`
                    As above.

    Parameters
    ----------
    L : Tensor
        Left generator of a low-rank matrix (:math:`L R^\intercal`).
    R : Tensor or None (default None)
        Right generator of a low-rank matrix (:math:`L R^\intercal`). If this
        is None, then the output matrix is symmetric :math:`L L^\intercal`.
    symmetry : ``'cross'``, ``'skew'``, or other (default None)
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
    if R is None:
        R = L
    output = L @ R.transpose(-2, -1)
    #TODO: Unit tests are not hitting this conditional...
    if symmetry == 'cross' or symmetry == 'skew':
        return symmetric(output, skew=symmetry)
    return output


def recondition_eigenspaces(A, psi, xi):
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
    x = xi * torch.rand(A.shape[-1], dtype=A.dtype, device=A.device)
    mask = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return A + (psi - xi + x) * mask


def delete_diagonal(A):
    """
    Delete the diagonal from a block of square matrices. Dimension is inferred
    from the final axis.
    """
    dim = A.shape[-1]
    mask = (~torch.eye(dim, device=A.device, dtype=torch.bool)).type(A.dtype)
    return A * mask


def toeplitz(c, r=None, dim=None, fill_value=0):
    """
    Populate a block of tensors with Toeplitz banded structure.

    :Dimension: **c :** :math:`(C, *)`
                    C denotes the number of elements in the first column whose
                    values are propagated along the matrix diagonals. ``*``
                    denotes any number of additional dimensions.
                **R :** :math:`(R, *)`
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
        first axis corresponds to a single matrix column; additional dimensions
        correspond to concatenation of Toeplitz matrices into a stack or block
        tensor.
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
        extended to conform with the specified `dim`. If this is a tensor, then
        each entry corresponds to the fill value in a different data channel.
        Has no effect if `dim` is None.

    Returns
    -------
    out: Tensor
        Block of Toeplitz matrices populated from the specified row and column
        elements.
    """
    if r is None:
        r = c.conj()
    clen, rlen = c.size(0), r.size(0)
    obj_shp = c.size()[1:]
    if dim is not None and dim is not (clen, rlen):
        r_ = torch.zeros([dim[1], *obj_shp], dtype=c.dtype, device=c.device)
        c_ = torch.zeros([dim[0], *obj_shp], dtype=c.dtype, device=c.device)
        if isinstance(fill_value, torch.Tensor) or fill_value != 0:
            r_ += fill_value
            c_ += fill_value
        r_[:rlen] = r
        c_[:clen] = c
        r, c = r_, c_
    return _populate_toeplitz(c, r, obj_shp)


def _populate_toeplitz(c, r, obj_shp):
    """
    Populate a block of Toeplitz matrices without any preprocessing.

    Thanks to https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/utils/toeplitz.py
    for ideas toward a faster implementation.
    """
    out_shp = c.size(0), r.size(0)
    # return _strided_view_toeplitz(r, c, out_shp)
    X = torch.empty([*out_shp, *obj_shp], dtype=c.dtype, device=c.device)
    for i, val in enumerate(c):
        m = min(i + out_shp[1], out_shp[0])
        for j in range(i, m):
            X[j, j - i] = val
    for i, val in list(enumerate(r))[1:]:
        m = min(i + out_shp[0], out_shp[1])
        for j in range(i, m):
            X[j - i, j] = val
    return X.permute(*range(2, X.dim()), 0, 1)


def _strided_view_toeplitz(r, c, out_shp):
    """
    torch is *not* planning on implementing negative strides anytime soon, so
    this numpy-like code will likely not be usable for the foreseeable future.

    See el3ment's comments and the response here:
    https://github.com/pytorch/pytorch/issues/604

    It's not great practice, but we're keeping it here in case they have a
    change of heart.
    """
    raise NotImplementedError('This operation is not currently supported.')
    vals = torch.cat([c.flip(-1)[:-1], r])
    n = vals.stride(0)
    return torch.as_strided(vals[out_shp[0]:],
                            shape=out_shp,
                            strides=(-n, n))


def sym2vec(sym, offset=1):
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
    idx = torch.triu_indices(*sym.shape[-2:], offset=offset)
    shape = sym.shape[:-2]
    vec = sym[..., idx[0], idx[1]]
    return vec.view(*shape, -1)


def vec2sym(vec, offset=1):
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
    vec = vec.view(*shape, -1)
    cn2 = vec.shape[-1]
    side = int(0.5 * (math.sqrt(8 * cn2 + 1) + 1)) + (offset - 1)
    idx = torch.triu_indices(side, side, offset)
    sym = torch.zeros((*shape, side, side)).type(vec.dtype)
    sym[..., idx[0], idx[1]] = vec
    sym = sym + sym.transpose(-1, -2)
    if offset == 0:
        mask = torch.eye(side).bool()
        sym[..., mask] = sym[..., mask] / 2
    return sym


def squareform(X):
    """
    Convert between symmetric matrix and vector forms.

    Parameters
    ----------
    X : tensor
        Block of symmetric matrices, in either square matrix or vectorised
        form.

    Returns
    -------
    tensor
        If the input block is in square matrix form, returns it in vector
        form. If the input block is in vector form, returns it in square
        matrix form.

    Notes
    -----
    Unlike numpy or matlab implementations, this does not verify a conformant
    input.
    """
    if (X.shape[-2] == X.shape[-1]
        and torch.allclose(X, X.transpose(-1, -2))):
        return sym2vec(X, offset=1)
    else:
        return vec2sym(X, offset=1)
