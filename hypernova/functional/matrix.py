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
    matrix, inverting the decomposition, and recomposing.

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
    L = torch.cholesky(A)
    Li = torch.pinverse(L)
    return Li.transpose(-1, -2) @ Li


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
