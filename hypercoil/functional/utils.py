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


def sparse_mm(A, B):
    """
    See: https://github.com/rusty1s/pytorch_sparse/issues/147
    """
    m = A.shape[0]
    n = B.shape[1]
    k = A.shape[1]
    assert B.shape[0] == k
    assert A.dense_dim() == B.dense_dim()
    A = A.coalesce()
    B = B.coalesce()
    A_values = A.values()
    B_values = B.values()
    A_indices = A.indices()
    B_indices = B.indices()
    A_values = A_values.permute(list(range(A_values.dim()))[::-1])
    B_values = B_values.permute(list(range(B_values.dim()))[::-1])
    out_indices, out_values = _sparse_mm(
        A_indices, A_values, B_indices, B_values, m, k, n)
    out_values = out_values.permute(list(range(out_values.dim()))[::-1])
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
        #return torch_sparse.spspmm(
        #    A_indices, A_values, B_indices, B_values, m, k, n)
    else:
        out = [
            _sparse_mm(A_indices, a, B_indices, b, m, k, n)
            for a, b in zip(A_values, B_values)
        ]
        out_indices, out_values = zip(*out)
        #print(out_indices)
        #print(out_values)
        return out_indices[0], torch.stack(out_values)


# def sparse_mm(A, B):
#     """
#     See: https://github.com/rusty1s/pytorch_sparse/issues/147
#     """
#     print(A.shape, B.shape)
#     assert A.dense_dim() == B.dense_dim()
#     if A.dense_dim() == 0:
#         #print(A, B, torch.sparse.mm(A, B))
#         return torch.sparse.mm(A, B)
#     else:
#         A = A.coalesce()
#         B = B.coalesce()
#         A_v = A.values().transpose(0, -1)
#         B_v = B.values().transpose(0, -1)
#         print('values', A_v.shape, B_v.shape)
#         out = [
#             sparse_mm(
#                 torch.sparse_coo_tensor(
#                     indices=A.indices(), values=a.transpose(0, -1),
#                     size=(*A.shape[:A.sparse_dim()], *a.shape[1:])),
#                 torch.sparse_coo_tensor(
#                     indices=B.indices(), values=b.transpose(0, -1),
#                     size=(*B.shape[:B.sparse_dim()], *b.shape[1:]))
#             ).coalesce()
#             for a, b in zip(A_v, B_v)
#         ]
#         out_values = torch.stack([o.values() for o in out]).transpose(0, -1)
#         return torch.sparse_coo_tensor(
#             out[0].indices(),
#             out_values,
#             size=(*out[0].size(), out_values.size(-1))
#         )


def _conform_vector_weight(weight):
    if weight.dim() == 1:
        return weight
    if weight.shape[-2] != 1:
        return weight.unsqueeze(-2)
    return weight
