# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base losses
~~~~~~~~~~~
Base modules for loss functions.
"""
from torch.nn import Module


class Loss(Module):
    """
    Base class for hypercoil loss functions.
    """
    @property
    def extra_repr(self):
        return ()

    def __repr__(self):
        if self.extra_repr:
            s = ', '.join((f'ν = {self.nu}', *self.extra_repr()))
            return f'[{s}]{self.name}'
        return f'[ν = {self.nu}]{self.name}'


class LossApply(Module):
    """
    Callable loss function wrapper that composes the loss with a selector or
    other pretransformation.

    Parameters
    ----------
    loss : callable
        Loss function to apply.
    apply : callable
        Callable that pretransforms the input to the loss. This can be used
        as a selector, so that different entries in a `LossScheme` are applied
        to different combinations of weights, inputs, and outputs. One use
        case is filtering the inputs to a `LossScheme` and forwarding only the
        ones relevant to the loss function at hand.
    """
    def __init__(self, loss, apply=None):
        super(LossApply, self).__init__()
        if apply is None:
            apply = lambda x: x
        self.loss = loss
        self.apply = apply

    def __repr__(self):
        return self.loss.__repr__()

    def forward(self, *args, **kwargs):
        return self.loss(self.apply(*args, **kwargs))


class ReducingLoss(Loss):
    """
    Callable loss function wrapper that composes an objective, which may have
    only an elementwise or slicewise definition, with a reduction operation
    that maps its output to a scalar.

    Example reductions include the mean and sum, as well as tensor norms.

    Parameters
    ----------
    nu : float
        Loss function weight multiplier.
    reduction : callable
        Map from a tensor of arbitrary dimension to a scalar. The output of
        `loss` is passed into `reduction` to return a scalar.
    loss : callable
        Objective criterion, which might not always return a scalar output.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu, reduction, loss, name=None):
        super(ReducingLoss, self).__init__()
        if name is None:
            name = type(self).__name__
        self.nu = nu
        self.reduction = reduction
        self.loss = loss
        self.name = name

    def forward(self, *args, **kwargs):
        return self.nu * self.reduction(self.loss(*args, **kwargs))
