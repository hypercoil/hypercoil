# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
A hideous, disorganised group of utility functions. Hopefully someday they
can disappear altogether or be moved elsewhere, but for now they exist, a sad
blemish.
"""
import torch


def conform_mask(tensor, msk, axis, batch=False):
    """
    Conform a mask or weight for elementwise applying to a tensor.

    There is almost certainly a better way to do this.

    See also
    --------
    :func:`apply_mask`
    """
    #TODO: require axis to be ordered as in `orient_and_conform`
    if batch and tensor.dim() == 1:
        batch = False
    if isinstance(axis, int):
        if not batch:
            shape_pfx = tensor.shape[:axis]
            msk = msk.tile(*shape_pfx, 1)
            return msk
        axis = (axis,)
    if batch:
        axis = (0, *axis)
    msk = msk.squeeze()
    tile = list(tensor.shape)
    shape = [1 for _ in range(tensor.dim())]
    for i, ax in enumerate(axis):
        tile[ax] = 1
        shape[ax] = msk.shape[i]
    msk = msk.view(*shape).tile(*tile)
    return msk


def apply_mask(tensor, msk, axis):
    """
    Mask a tensor along an axis.

    See also
    --------
    :func:`conform_mask`
    """
    shape_pfx = tensor.shape[:axis]
    if axis == -1:
        shape_sfx = ()
    else:
        shape_sfx = tensor.shape[(axis + 1):]
    msk = msk.tile(*shape_pfx, 1)
    return tensor[msk].view(*shape_pfx, -1, *shape_sfx)


def wmean(input, weight, dim=None, keepdim=False):
    """
    Reducing function for reducing losses: weighted mean.
    """
    if dim is None:
        dim = list(range(input.dim()))
    elif isinstance(dim, int):
        dim = (dim,)
    assert weight.dim() == len(dim), (
        'Weight must have as many dimensions as are being reduced')
    retain = [True for _ in range(input.dim())]
    for d in dim:
        retain[d] = False
    for i, d in enumerate(retain):
        if d: weight = weight.unsqueeze(i)
    wtd = (weight * input)
    return wtd.sum(dim, keepdim=keepdim) / weight.sum(dim, keepdim=keepdim)


def selfwmean(input, dim=None, keepdim=False, gradpath='input', softmax=True):
    """
    Self-weighted mean reducing function. Completely untested. Will break and
    probably kill you in the process.
    """
    i = input.clone()
    w = input.clone()
    if softmax:
        w = torch.softmax(w)
    if gradpath == 'input':
        w = w.detach()
    elif gradpath == 'weight':
        i = i.detach()
    return wmean(input=i, weight=w, keepdim=keepdim, gradpath=gradpath)


# torch is actually very, very good at doing this. Looks like we might have
# miscellaneous utilities.
# It's not even continuous, let alone differentiable. Let's not use this.
def threshold(input, threshold, dead=0, leak=0):
    if not isinstance(dead, torch.Tensor):
        dead = torch.tensor(dead, dtype=input.dtype, device=input.device)
    if leak == 0:
        return torch.where(input > threshold, input, dead)
    return torch.where(input > threshold, input, dead + leak * input)


def complex_decompose(complex):
    """
    Decompose a complex-valued tensor into amplitude and phase components.

    :Dimension:
        Each output is of the same shape as the input.

    Parameters
    ----------
    complex : Tensor
        Complex-valued tensor.

    Returns
    -------
    ampl : Tensor
        Amplitude of each entry in the input tensor.
    phase : Tensor
        Phase of each entry in the input tensor, in radians.

    See also
    --------
    :func:`complex_recompose`
    """
    ampl = torch.abs(complex)
    phase = torch.angle(complex)
    return ampl, phase


def complex_recompose(ampl, phase):
    """
    Reconstitute a complex-valued tensor from real-valued tensors denoting its
    amplitude and its phase.

    :Dimension:
        Both inputs must be the same shape (or broadcastable). The
        output is the same shape as the inputs.

    Parameters
    ----------
    ampl : Tensor
        Real-valued array storing complex number amplitudes.
    phase : Tensor
        Real-valued array storing complex number phases in radians.

    Returns
    -------
    complex : Tensor
        Complex numbers formed from the specified amplitudes and phases.

    See also
    --------
    :func:`complex_decompose`
    """
    # TODO : consider using the complex exponential when torch enables it,
    # depending on the gradient properties
    # see here : https://discuss.pytorch.org/t/complex-functions-exp-does- ...
    # not-support-automatic-differentiation-for-outputs-with-complex- ...
    # dtype/98039
    # Supposedly it was updated, but it still isn't working after calling
    # pip install torch --upgrade
    # (old note, might be working now)
    # https://github.com/pytorch/pytorch/issues/43349
    # https://github.com/pytorch/pytorch/pull/47194
    return ampl * (torch.cos(phase) + 1j * torch.sin(phase))
    #return ampl * torch.exp(phase * 1j)


def _promote_nnz_dim(values):
    return values.permute(list(range(values.dim()))[::-1])
    # slightly faster but not as easy to implement
    #return values.permute((*list(range(values.dim()))[1:], 0))


def _demote_nnz_dim(values):
    return _promote_nnz_dim(values)
    #return values.permute((-1, *list(range(values.dim()))[:-1]))


def _conform_dims(A_values, B_values):
    A_shape = A_values.shape[1:]
    B_shape = B_values.shape[1:]
    missing_dims = len(A_shape) - len(B_shape)
    if missing_dims == 0:
        return A_values, B_values
    elif missing_dims > 0:
        B_shape = [1] * missing_dims + list(B_shape)
        return A_values, B_values.view(-1, *B_shape)
    else:
        A_shape = [1] * -missing_dims + list(A_shape)
        return A_values.view(-1, *A_shape), B_values


def sparse_mm(A, B):
    """
    Batched sparse-sparse matrix multiplication.

    .. admonition:: Dimensions and broadcasting

        The dense dimension of each input COO tensor (that is, the dimension
        of the tensor in ``tensor._values()``) is equal to ``(nnz, *)``, where
        ``nnz`` denotes the number of nonzero elements in the tensor. For the
        matrix multiply, array broadcasting can be performed using the
        remaining dense dimensions after the ``nnz`` dimension. For instance,
        an array ``A`` with dense size (20, 3, 5) can be broadcast with an
        array ``B`` that has dense size (35, 5) or (17, 1, 1, 5), but not one
        that has dense size (20, 3). (The first dense axis size, ``nnz``, is
        ignored when broadcasting.)

    :Dimension: **A :** :math:`(M, K, *)`
                    `*` denotes any number of trailing (dense) dimensions,
                    potentially including a batch dimension. Note that the
                    batch dimension must be a dense dimension. M and K denote
                    the sparse dimensions of the input tensor, corresponding
                    to the axes of the matrix to be multiplied. Note that the
                    sparse dimension must be exactly 2.
                **B :** :math:`(K, N, *)`
                    N denotes the outer sparse dimension of the second matrix
                    batch.
                **Output :** :math:`(M, N, *)`
                    As above.

    Parameters
    ----------
    A, B : sparse COO tensor
        Tensors to be multiplied.

    Returns
    -------
    sparse COO tensor
        Sparse product of sparse tensors.

    See: https://github.com/rusty1s/pytorch_sparse/issues/147
    """
    m = A.shape[0]
    n = B.shape[1]
    k = A.shape[1]
    assert B.shape[0] == k, (
        f'Inner matrix dimensions {A.shape[1]} and {B.shape[1]} '
        'must agree')
    #assert A.dense_dim() == B.dense_dim()
    A = A.coalesce()
    B = B.coalesce()
    A_values = A.values()
    B_values = B.values()
    A_indices = A.indices()
    B_indices = B.indices()
    A_values, B_values = _conform_dims(A_values, B_values)
    A_values = _demote_nnz_dim(A_values)
    B_values = _demote_nnz_dim(B_values)
    out_indices, out_values = _sparse_mm(
        A_indices, A_values, B_indices, B_values, m, k, n)
    out_values = _promote_nnz_dim(out_values)
    o = out_values.shape[1:]
    return torch.sparse_coo_tensor(
        indices=out_indices, values=out_values, size=(m, n, *o)
    )


def _sparse_mm(A_indices, A_values, B_indices, B_values, m, k, n):
    if A_values.dim() <= 1:
        out = torch.sparse.mm(
            torch.sparse_coo_tensor(
                indices=A_indices, values=A_values, size=(m, k)),
            torch.sparse_coo_tensor(
                indices=B_indices, values=B_values, size=(k, n)),
        ).coalesce()
        return out.indices(), out.values()
    else:
        if len(A_values) != len(B_values):
            if len(A_values) == 1:
                A_values = [A_values[0]] * len(B_values)
            elif len(B_values) == 1:
                B_values = [B_values[0]] * len(A_values)
            else:
                raise RuntimeError(
                    'Dense dimensions of arrays are incompatible: '
                    f'{A_values.shape} and {B_values.shape}')
        out = [
            _sparse_mm(A_indices, a, B_indices, b, m, k, n)
            for a, b in zip(A_values, B_values)
        ]
        out_indices, out_values = zip(*out)
        return out_indices[0], torch.stack(out_values)


def sparse_rcmul(A, R=None, C=None, coalesce_output=True):
    if R is not None and C is not None:
        R = _rcmul_broadcast(R, A._indices()[0])
        C = _rcmul_broadcast(C, A._indices()[1])
        RC = R * C
    elif R is not None:
        RC = _rcmul_broadcast(R, A._indices()[0])
    elif C is not None:
        RC = _rcmul_broadcast(C, A._indices()[1])
    else:
        raise ValueError(
            'Must specify either row or column multiplier or both')
    out_values = A._values() * RC
    out = torch.sparse_coo_tensor(
        indices=A._indices(),
        values=out_values,
        size=A.size()
    )
    if coalesce_output:
        return out.coalesce()
    return out


def _rcmul_broadcast(tensor, indices):
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    return tensor[indices]


def sparse_reciprocal(A):
    if not A.is_sparse:
        return A.reciprocal()
    coalesce_output = A.is_coalesced()
    values = A._values().reciprocal()
    values[torch.isnan(values)] = 0
    out = torch.sparse_coo_tensor(
        indices=A._indices(),
        values=values,
        size=(A.size())
    )
    if coalesce_output:
        return out.coalesce()
    return out


def _conform_vector_weight(weight):
    if weight.dim() == 1:
        return weight
    if weight.shape[-2] != 1:
        return weight.unsqueeze(-2)
    return weight


def orient_and_conform(input, axis, reference=None, dim=None):
    if isinstance(axis, int):
        axis = (axis,)
    if dim is None and reference is None:
        raise ValueError('Must specify either `reference` or `dim`')
    elif dim is None:
        dim = reference.dim()
    assert len(axis) == input.dim(), (
        'Output orientation axis required for each input dimension')
    shape = [1] * dim
    asgn = [0] * dim
    for size, ax in zip(input.size(), axis):
        shape[ax] = size
        assert sum(asgn[ax:]) == 0, (
            'All axes must be in order. Transpose the input if necessary.')
        asgn[ax] = 1
    return input.view(*shape)
