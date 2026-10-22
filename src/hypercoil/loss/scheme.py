# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Loss schemes: additive application of several losses to a set of inputs.
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
"""Value of one loss in a scheme, and its multiplier ``nu``."""


def unpack_or_noop(*pparams) -> Any:
    """
    Return the single argument if exactly one is given, else all arguments.
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
    """
    Evaluate a loss on the output of an ``apply`` operation.

    Parameters
    ----------
    loss
        Loss function.
    applied
        Output of the ``apply`` operation. An
        :class:`~hypercoil.engine.argument.UnpackingModelArgument` is passed
        as keyword arguments; any other value as one positional argument.
    key
        Pseudo-random number generator key.

    Returns
    -------
    Any
        Output of ``loss``.
    """
    if isinstance(applied, UnpackingLossArgument):
        return loss(**applied, key=key)
    return loss(applied, key=key)


# TODO: add examples of how to use this.
class LossApply(eqx.Module):
    """
    Loss wrapper that applies a selector or other transformation first.

    Parameters
    ----------
    loss
        Loss function to apply.
    apply
        Callable that pretransforms the input to the loss. This can be used
        as a selector, so that different entries in a
        :class:`~hypercoil.loss.scheme.LossScheme` are applied
        to different combinations of weights, inputs, and outputs. One use case
        is filtering the inputs to a :class:`LossScheme` and forwarding only
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
        """Multiplier of the wrapped loss, or ``None`` if it has none."""
        return getattr(self.loss, 'nu', None)

    def losses(self) -> Tuple[Callable, ...]:
        """Return the losses that the wrapper contains, unnested."""
        if not (
            isinstance(self.loss, LossScheme) or
            isinstance(self.loss, LossApply)
        ):
            return (self.loss,)
        return self.loss.losses()

    def get_loss(self, name: Optional[str] = None) -> Callable:
        """Return the wrapped loss, or the contained loss named ``name``."""
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
        """
        Return a copy with an attribute of a contained loss set to a value.

        Parameters
        ----------
        value
            New value.
        where
            Name of the attribute.
        loss
            Name of the contained loss; None selects the wrapped loss.

        Returns
        -------
        LossApply
            Copy with the attribute replaced.
        """
        loss = self.get_loss(loss)
        return eqx.filter(
            self,
            filter_spec=lambda x: x is getattr(loss, where),
            inverse=True,
            replace=value,
        )

    def step(self, count: Optional[int] = None) -> LossApply:
        """Advance each loss schedule one step, or to step ``count``."""
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
    """
    Sum of several losses applied to the same inputs.

    Calling a scheme returns the total loss and a dictionary of
    :data:`LossReturn` values, one per loss, keyed by loss name.

    Parameters
    ----------
    loss
        Losses, :class:`LossApply` wrappers or nested schemes.
    apply
        Callable that pretransforms the inputs before every loss.
    """
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
        """Return the losses of the scheme, with nested tuples flattened."""
        return tuple(LossScheme._unnest(tuple(
            loss if not (
                isinstance(self.loss, LossScheme) or
                isinstance(self.loss, LossApply)
            ) else loss.losses()
            for loss in self.loss
        )))

    def get_loss(self, name: str) -> Callable:
        """Return the first loss of the scheme named ``name``."""
        return [loss for loss in self.losses() if loss.name == name][0]

    def cfg(
        self,
        value: Any,
        loss: str,
        where: str = '_nu',
    ) -> LossApply:
        """
        Return a copy with an attribute of a named loss set to a value.

        Parameters
        ----------
        value
            New value.
        loss
            Name of the loss.
        where
            Name of the attribute.

        Returns
        -------
        LossScheme
            Copy with the attribute replaced.

        Raises
        ------
        IndexError
            If the scheme contains no loss named ``loss``.
        """
        loss = self.get_loss(loss)
        return eqx.filter(
            self,
            filter_spec=lambda x: x is getattr(loss, where),
            inverse=True,
            replace=value,
        )

    def step(self, count: Optional[int] = None) -> LossApply:
        """Advance each loss schedule one step, or to step ``count``."""
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
