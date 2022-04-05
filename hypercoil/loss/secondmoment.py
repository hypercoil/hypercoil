# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Regularise the second moment, e.g. to favour a dimension reduction mapping
that is internally homogeneous.

.. admonition:: Second Moment

    Second moment losses are based on a reduction of the second moment
    quantity

    :math:`\left[ A \circ \left (T - \frac{AT}{A\mathbf{1}} \right )^2  \right] \frac{\mathbf{1}}{A \mathbf{1}}`

    where the division operator is applied elementwise with broadcasting and
    the difference operator is applied via broadcasting. The broadcasting
    operations involved in the core computation -- estimating a weighted mean
    and then computing the weighted sum of squares about that mean -- are
    illustrated in the below cartoon.

    .. image:: ../_images/secondmomentloss.svg
        :width: 300
        :align: center

    *Illustration of the most memory-intensive stage of loss computation. The
    lavender tensor represents the weighted mean, the blue tensor the
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

    When using this loss to learn a parcellation on voxelwise time series, the
    full computation will certainly be much too large to fit in GPU memory.
    Fortunately, because much of the computation is elementwise, it can be
    broken down along multiple axes without affecting the result. This tensor
    slicing is implemented automatically in the
    :doc:`ReactiveTerminal <hypercoil.engine.terminal.ReactiveTerminal>`
    class. Use extreme caution with ``ReactiveTerminals``, as improper use can
    result in destruction of the computational graph.
"""
from torch import mean
from .base import ReducingLoss
from functools import partial


def _second_moment(weight, data, mu, skip_normalise=False):
    """
    Core computation for second-moment loss.
    """
    weight = weight.abs().unsqueeze(-1)
    if skip_normalise:
        normfac = 1
    else:
        normfac = weight.sum(-2)
    diff = data.unsqueeze(-3) - mu.unsqueeze(-2)
    sigma = ((diff * weight) ** 2).sum(-2) / normfac
    return sigma


def second_moment(weight, data, standardise=False, skip_normalise=False):
    r"""
    Compute the second moment of a dataset.

    The second moment is computed as

    :math:`\left[ A \circ \left (T - \frac{AT}{A\mathbf{1}} \right )^2  \right] \frac{\mathbf{1}}{A \mathbf{1}}`

    :Dimension: **weight :** :math:`(*, R, V)`
                    ``*`` denotes any number of preceding dimensions, R
                    denotes number of weights (e.g., regions of an atlas),
                    and V denotes number of locations (e.g., voxels).
                **data :** :math:`(*, V, T)`
                    T denotes number of observations at each location (e.g.,
                    number of time points).
    """
    if standardise:
        data = (
            data - data.mean(-1, keepdim=True)) / data.std(-1, keepdim=True)
    mu = (weight @ data / weight.sum(-1, keepdim=True))
    return _second_moment(weight, data, mu, skip_normalise)


def second_moment_centred(weight, data, mu,
                          standardise_data=False,
                          standardise_mu=False,
                          skip_normalise=False):
    r"""
    Compute the second moment of a dataset about a specified mean.
    """
    if standardise_data:
        data = (
            data - data.mean(-1, keepdim=True)) / data.std(-1, keepdim=True)
    if standardise_mu:
        mu = (mu - mu.mean(-1)) / mu.std(-1)
    return _second_moment(weight, data, mu, skip_normalise)


class SecondMoment(ReducingLoss):
    r"""
    Compute the second moment of a dataset/linear mapping pair.

    The second moment loss is computed as a reduction of

    :math:`\left[ A \circ \left (T - \frac{AT}{A\mathbf{1}} \right )^2  \right] \frac{\mathbf{1}}{A \mathbf{1}}`

    Given an input dataset :math:`T` and a linear mapping :math:`A`, the
    second moment loss measures the extent to which :math:`A` tends to map
    similar inputs to the same output.

    Penalising this quantity can thus promote learning a linear mapping whose
    rows load onto similar feature sets.

    .. warning::
        This loss can have a very large memory footprint, because it requires
        computing an intermediate tensor with dimensions equal to the number
        of rows in the linear mapping, multiplied by the number of columns in
        the linear mapping, multiplied by the number of columns in the
        dataset.

    Parameters
    ----------
    nu : float (default 1)
        Loss function weight multiplier.
    reduction : callable (default `torch.mean`)
        Map from a tensor of arbitrary dimension to a scalar. The output of
        the second moment loss is passed into `reduction` to return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu=1, standardise=False, skip_normalise=False,
                 reduction=None, name=None):
        reduction = reduction or mean
        loss = partial(
            second_moment,
            standardise=standardise,
            skip_normalise=skip_normalise
        )
        super(SecondMoment, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name,
        )


class SecondMomentCentred(ReducingLoss):
    r"""
    Compute the second moment of a dataset/linear mapping pair about a
    specified mean.
    """
    def __init__(self, nu=1, standardise_data=False, standardise_mu=False,
                 skip_normalise=False, reduction=None, name=None):
        reduction = reduction or mean
        loss = partial(second_moment_centred,
                       standardise_data=standardise_data,
                       standardise_mu=standardise_mu,
                       skip_normalise=skip_normalise)
        super(SecondMomentCentred, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )
