# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Centre of mass
~~~~~~~~~~~~~~
Differentiably compute a weight's centre of mass.
"""
import torch


def cmass(X, axes=None, na_rm=False):
    """
    Differentiably compute a weight's centre of mass. This can be used to
    regularise the weight so that its centre of mass is close to a provided
    coordinate.

    Dimension
    ---------
    - Input: :math:`(*, k_1, k_2, ..., k_n)`
      `*` denotes any number of batch and intervening dimensions, and the
      `k_i`s are dimensions along which the centre of mass is computed
    - Output: :math:`(*, n)`
      `n` denotes the number of dimensions of each centre of mass vector

    Parameters
    ----------
    X : Tensor
        Tensor containing the weights whose centres of mass are to be
        computed.
    axes : iterable or None (default None)
        Axes of the input tensor that together define each slice of the tensor
        within which a single centre-of-mass vector is computed. If this is
        set to None, then the centre of mass is computed across all axes.
    na_rm : float or False (default False)
        If any single slice of the input tensor has zero mass, then the centre
        of mass within that slice is undefined and populated with NaN. The
        `na_rm` parameter specified how such undefined values are handled. If
        this is False, then NaN values are left intact; if this is a float,
        then NaN values are replaced by the specified float.

    Returns
    -------
    cmass : Tensor
        Centre of mass vectors of each slice from the input tensor. The
        coordinates are ordered according to the specification in `axes`.
    """
    dim = X.size()
    ndim = X.dim()
    all_axes = list(range(ndim))
    if axes is not None:
        axes = [all_axes[ax] for ax in axes]
    else:
        axes = all_axes
    out_dim = [s for ax, s in enumerate(dim) if all_axes[ax] not in axes]
    out_dim += [len(axes)]
    out = torch.zeros(out_dim)
    for i, ax in enumerate(axes):
        coor = torch.arange(1, X.size(ax) + 1)
        while coor.dim() < ndim - all_axes[ax]:
            coor.unsqueeze_(-1)
        num = (coor * X).sum(axes)
        denom = X.sum(axes)
        out[..., i] = num / denom - 1
        if na_rm is not False:
            out[denom == 0, i] = na_rm
    return out
