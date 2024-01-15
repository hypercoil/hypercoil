# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Module for additively applying a set of several losses or
regularisations to a set of inputs.
"""
from __future__ import annotations
from collections import namedtuple
from typing import Any, Callable, Optional, Tuple

import jax
import equinox as eqx

from ..engine.argument import (
    ModelArgument as LossArgument,  # noqa
)
from ..engine.argument import (
    UnpackingModelArgument as UnpackingLossArgument,
)


LossReturn = namedtuple('LossReturn', ('value', 'nu'))


def unpack_or_noop(*pparams) -> Any:
    """
    For an argument sequence, unpack if it's a singleton sequence, and
    otherwise return the sequence.
    """
    if len(pparams) == 1:
        return pparams[0]
    return pparams


def apply_and_evaluate(
    loss: Callable,
    applied: Any,
    *,
    key: 'jax.random.PRNGKey',
) -> Any:
    if isinstance(applied, UnpackingLossArgument):
        return loss(**applied, key=key)
    return loss(applied, key=key)


# TODO: add examples of how to use this.
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

    def __repr__(self):
        return self.loss.__repr__()

    def __call__(
        self,
        *pparams,
        key: 'jax.random.PRNGKey',
        **params,
    ) -> float:
        """
        Evaluate the loss applied to the output of the ``apply`` operation.
        """
        applied = self.apply(*pparams, **params)
        return apply_and_evaluate(self.loss, applied, key=key)

    @property
    def nu(self) -> float:
        return getattr(self.loss, 'nu', None)

    def losses(self) -> Tuple[Callable, ...]:
        if not (
            isinstance(self.loss, LossScheme) or
            isinstance(self.loss, LossApply)
        ):
            return (self.loss,)
        return self.loss.losses()

    def get_loss(self, name: Optional[str] = None) -> Callable:
        if name is None:
            loss = self.loss
        else:
            cname = getattr(self.loss, 'name', None)
            if name == cname:
                loss = self.loss
            else:
                loss = self.loss.get_loss(name)
        return loss

    def cfg(
        self,
        value: Any,
        where: str = '_nu',
        loss: Optional[str] = None,
    ) -> LossApply:
        loss = self.get_loss(loss)
        return eqx.filter(
            self,
            filter_spec=lambda x: x is getattr(loss, where),
            inverse=True,
            replace=value,
        )

    def step(self, count: Optional[int] = None) -> LossApply:
        new = self
        for loss in self.losses():
            repl = loss.step(count=count)
            new = eqx.filter(
                new,
                filter_spec=lambda x: x is loss,
                inverse=True,
                replace=repl,
                is_leaf=lambda x: (x in new.losses()),
            )
        return new


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

    def __repr__(self) -> str:
        return eqx.tree_pformat(
            self,
            follow_wrapped=True,
            short_arrays=True,
            truncate_leaf=lambda x: False,
            indent=2,
        )

    # TODO: Check if the multiplier for each loss is 0, and if so, don't
    #      evaluate it.
    def __call__(self, *pparams, key='jax.random.PRNGKey', **params):
        total_loss = 0
        all_items = {}
        applied = self.apply(*pparams, **params)
        keys = jax.random.split(key, len(self))
        for f, k in zip(self, keys):
            if isinstance(f, LossScheme):
                acc, items = apply_and_evaluate(f, applied, key=k)
            elif isinstance(f, LossApply) and isinstance(f.loss, LossScheme):

                def f_(*pparams, key: 'jax.random.PRNGKey', **params):
                    return f.loss(f.apply(*pparams, **params), key=key)

                acc, items = apply_and_evaluate(f_, applied, key=k)
            else:
                acc = apply_and_evaluate(f, applied, key=k)
                items = {f.name: LossReturn(acc, f.nu)}
            total_loss += acc
            all_items.update(items)
        return total_loss, all_items

    @staticmethod
    def _unnest(seq: Tuple[Any]) -> Tuple[Any]:
        for item in seq:
            if isinstance(item, tuple):
                yield from LossScheme._unnest(item)
            else:
                yield item

    def losses(self) -> Tuple[Callable, ...]:
        return tuple(LossScheme._unnest(tuple(
            loss if not (
                isinstance(self.loss, LossScheme) or
                isinstance(self.loss, LossApply)
            ) else loss.losses()
            for loss in self.loss
        )))

    def get_loss(self, name: str) -> Callable:
        return [loss for loss in self.losses() if loss.name == name][0]

    def cfg(
        self,
        value: Any,
        loss: str,
        where: str = '_nu',
    ) -> LossApply:
        loss = self.get_loss(loss)
        return eqx.filter(
            self,
            filter_spec=lambda x: x is getattr(loss, where),
            inverse=True,
            replace=value,
        )

    def step(self, count: Optional[int] = None) -> LossApply:
        new = self
        for loss in self.losses():
            repl = loss.step(count=count)
            new = eqx.filter(
                new,
                filter_spec=lambda x: x is loss,
                inverse=True,
                replace=repl,
                is_leaf=lambda x: (x in new.losses()),
            )
        return new
