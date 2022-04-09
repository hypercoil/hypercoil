# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base losses
~~~~~~~~~~~
Base modules for loss functions.
"""
from torch.nn import Module
from collections.abc import Mapping
from hypercoil.engine import Sentry, SentryModule
from hypercoil.engine.action import UpdateMultiplier


def identity(*args):
    if len(args) == 1:
        return args[0]
    return args


class LossArgument(Mapping):
    """
    Effectively this is currently little more than a prettified dict.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __setitem__(self, k, v):
        self.__setattr__(k, v)

    def __getitem__(self, k):
        return self.__dict__[k]

    def __delitem__(self, k):
        del self.__dict__[k]

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)


class UnpackingLossArgument(LossArgument):
    """
    LossArgument variant that is automatically unpacked when it is the output
    of an `apply` call.
    """
    pass


class Loss(SentryModule):
    """
    Base class for hypercoil loss functions.
    """
    def __init__(self, nu=1):
        super().__init__()
        if isinstance(nu, Sentry):
            self.nu = nu.base
            nu.register_sentry(self)
            self.register_action(UpdateMultiplier())
        else:
            self.nu = nu

    @property
    def extra_repr(self):
        return ()

    def __repr__(self):
        if self.extra_repr:
            s = ', '.join((f'ν = {self.nu}', *self.extra_repr()))
            return f'[{s}]{self.name}'
        return f'[ν = {self.nu}]{self.name}'


class LossApply(SentryModule):
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
            apply = identity
        self.loss = loss
        self.apply = apply

    def __repr__(self):
        return self.loss.__repr__()

    def register_sentry(self, sentry):
        self.loss.register_sentry(sentry)

    def register_action(self, action):
        self.loss.register_action(action)

    def forward(self, *args, **kwargs):
        applied = self.apply(*args, **kwargs)
        if isinstance(applied, UnpackingLossArgument):
            return self.loss(**applied)
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
        super(ReducingLoss, self).__init__(nu=nu)
        if name is None:
            name = type(self).__name__
        self.reduction = reduction
        self.loss = loss
        self.name = name

    def forward(self, *args, **kwargs):
        out = self.nu * self.reduction(self.loss(*args, **kwargs))
        self.message.update(
            ('NAME', self.name),
            ('LOSS', out.clone().detach().item()),
            ('NU', self.nu)
        )
        for s in self.listeners:
            s._listen(self.message)
        self.message.clear()
        return out
