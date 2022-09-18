# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Module for additively applying a set of several losses or
regularisations to a set of inputs.
"""
import equinox as eqx
from functools import partial
from typing import Any, Callable, Tuple
from ..engine.argument import UnpackingModelArgument as UnpackingLossArgument


def unpack_or_noop(*pparams) -> Any:
    """
    For an argument sequence, unpack if it's a singleton sequence, and
    otherwise return the sequence.
    """
    if len(pparams) == 1:
        return pparams[0]
    return pparams


def apply_and_evaluate(loss: Callable, applied: Any) -> Any:
    if isinstance(applied, UnpackingLossArgument):
        return loss(**applied)
    return loss(applied)


#TODO: add examples of how to use this.
class LossApply(eqx.Module):
    """
    Callable loss function wrapper that composes the loss with a selector or
    other pretransformation.

    Parameters
    ----------
    loss : callable
        Loss function to apply.
    apply : callable
        Callable that pretransforms the input to the loss. This can be used
        as a selector, so that different entries in a
        :doc:`LossScheme <hypercoil.loss.scheme.LossScheme>` are applied
        to different combinations of weights, inputs, and outputs. One use
        case is filtering the inputs to a ``LossScheme`` and forwarding only
        the ones relevant to the loss function at hand.
    """
    loss: Callable
    apply: Callable = unpack_or_noop
    name: str

    def __init__(
        self,
        loss: Callable,
        apply: Callable = unpack_or_noop,
    ):
        self.loss = loss
        self.apply = apply
        self.name = loss.name

    def __call__(
        self,
        *pparams,
        **params
    ) -> float:
        """
        Evaluate the loss applied to the output of the ``apply`` operation.
        """
        applied = self.apply(*pparams, **params)
        return apply_and_evaluate(self.loss, applied)


class LossScheme(eqx.Module):
    loss: Tuple[Callable]
    apply: Callable = unpack_or_noop
    name = 'LossScheme'

    def __add__(self, other):
        return LossScheme(loss=(self.loss + other.loss))

    def __iter__(self):
        return iter(self.loss)

    def __len__(self):
        return len(self.loss)

    def __call__(self, *pparams, **params):
        total_loss = 0
        all_items = {}
        applied = self.apply(*pparams, **params)
        for f in self:
            if isinstance(f, LossScheme):
                acc, items = apply_and_evaluate(f, applied)
            elif isinstance(f, LossApply) and isinstance(f.loss, LossScheme):
                f = partial(f.loss, f.apply) # TODO: this doesn't work. not even close.
                def f_(*pparams, **params):
                    return f.loss(f.apply(*pparams, **params))
                acc, items = apply_and_evaluate(f_, applied)
            else:
                acc = apply_and_evaluate(f, applied)
                items = {f.name: acc}
            total_loss += acc
            all_items.update(items)
        return total_loss, all_items
