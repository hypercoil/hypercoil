# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Special matrix functions.
"""
import torch


def invert_spd(A):
    """
    Invert a symmetric positive definite matrix.

    Currently, this operates by computing the Cholesky decomposition of the
    matrix, inverting the decomposition, and recomposing. If the Cholesky
    decomposition fails because the input is singular, then it instead returns
    the Moore-Penrose pseudoinverse.

    Dimension
    ---------
    - Input: :math:`(*, D, D)`
      D denotes the row or column dimension of the matrices to be inverted.
    - Output: :math:`(*, D, D)`

    Parameters
    ----------
    A: Tensor
        Batch of symmetric positive definite matrices.

    Returns
    -------
    Ainv: Tensor
        Inverse or Moore-Penrose pseudoinverse of each matrix in the input
        batch.
    """
    try:
        L = torch.cholesky(A)
        Li = torch.inverse(L)
        return Li.transpose(-1, -2) @ Li
    except RuntimeError:
        return torch.pinverse(A)


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
    method : 'eig' or 'svd'
        Method used to ensure that all eigenvalues are positive.
        - `eig` denotes that the input matrices are symmetrised and then
          diagonalised. The method returns the symmetrised sum of the input and
          an identity matrix scaled to guarantee no eigenvalue is smaller than
          `eps`.
        - `svd` denotes that the input matrices are decomposed via singular
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
        L, _ = torch.symeig(symmetric(X))
        lmin = L.amin(axis=-1) - eps
        lmin = torch.minimum(lmin, torch.zeros(1)).squeeze()
        return symmetric(X - lmin[..., None, None] * torch.eye(X.size(-1)))
    elif method == 'svd':
        Q, L, _ = torch.svd(symmetric(X))
        return symmetric(Q @ torch.diag_embed(L) @ Q.transpose(-1, -2))


def expand_outer(L, R=None, symmetry=None):
    """
    Multiply out a left and a right generator matrix as an outer product.

    The rank of the output is limited according to the inner dimensions of the
    input generators. This approach can be used to produce a low-rank output
    or to share the generators' parameters across the rows and columns of the
    output matrix.

    Dimension
    ---------
    - L: :math:`(*, H, rank)`
      H denotes the height of the expanded matrix, and rank denotes its maximum
      rank.
    - R: :math:`(*, W, rank)`
      W denotes the width of the expanded matrix.
    - Output: :math:`(*, H, W)`

    Parameters
    ----------
    L : Tensor
        Left generator of a low-rank matrix (:math:`L R^\intercal`).
    R : Tensor or None (default None)
        Right generator of a low-rank matrix (:math:`L R^\intercal`). If this
        is None, then the output matrix is symmetric :math:`L L^\intercal`.
    symmetry : 'cross', 'skew', or other (default None)
        Symmetry constraint imposed on the generated low-rank template matrix.
        * `cross` enforces symmetry by replacing the initial expansion with
          the average of the initial expansion and its transpose,
          :math:`\frac{1}{2} \left( L R^\intercal + R L^\intercal \right)`
        * `skew` enforces skew-symmetry by subtracting from the initial
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
    if symmetry == 'cross' or 'skew':
        return symmetric(output, skew(symmetry=='skew'))
    return output


def toeplitz(c, r=None, dim=None, fill_value=0):
    """
    Populate a block of tensors with Toeplitz banded structure.

    Dimension
    ---------
    - c: :math:`(C, *)`
      C denotes the number of elements in the first column whose values are
      propagated along the matrix diagonals. `*` denotes any number of
      additional dimensions.
    - r: :math:`(R, *)`
      R denotes the number of elements in the first row whose values are
      propagated along the matrix diagonals. `*` must be the same as in input
      `c` or compatible via broadcasting.
    - fill_value: :math:`(*)`
    - Output: :math:`(*, C^{*}, R^{*})`
      :math:`C^{*}` and :math:`{*}` default to C and R unless specified
      otherwise in the `dim` argument.

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