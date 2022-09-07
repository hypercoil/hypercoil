# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
IIR filter implemented as RNN. This follows a close adaptation of:
https://github.com/boris-kuz/differentiable_iir_filters

If you use this in published work, you must cite:

Kuznetsov B, Parker JD, Esqueda F (2020) Differentiable IIR filters for
machine learning applications. DAFx2020, Vienna, Austria, September 2020-21.

.. warning::
    A stable backward pass is not yet implemented.
"""
import jax.numpy as jnp
import equinox as eqx
from typing import Optional, Tuple
from ..engine import Tensor
from ..init.iirfilter import IIRFilterSpec


#TODO: mark all code as experimental until the IIR filter module is properly
#      differentiable and the zero-phase filtering is implemented.
class DTDFCell(eqx.Module):
    N: int
    multiplier: int
    b: Tensor
    a: Tensor

    def __init__(
        self,
        init_spec: IIRFilterSpec
    ):

        self.N = init_spec.N
        self.multiplier = 1
        if init_spec.btype in ('bandpass', 'bandstop'):
            self.multiplier = 2

        coefs = jnp.array(init_spec.coefs)
        b = coefs[..., 0, :]
        a = coefs[..., 1, :][..., 1:]
        self.b, self.a = b, a

    def init_states(self, shape: Tuple[int]) -> Tensor:
        return jnp.zeros((*shape, self.multiplier * self.N))

    def __call__(self, input: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        output = input * self.b[..., 0] + v[..., 0][..., None]

        v_new = input * self.b[..., 1:] - output * self.a
        v_new = v_new.at[..., :-1].set(v_new[..., :-1] + v[..., 1:])
        return output.squeeze(-1), v_new


class Spec(object):
    def __init__(self, N):
        self.N = N
        self.btype = 'lowpass'


class DTDF(eqx.Module):
    cell: DTDFCell

    def __init__(
        self,
        spec: IIRFilterSpec,
    ):

        self.cell = DTDFCell(init_spec=spec)

    def __call__(
        self,
        input: Tensor,
        initial_states: Optional[Tensor] = None,
        feature_ax: bool = False
    ):
        if not feature_ax:
            input = input[..., None]
        else:
            raise NotImplementedError('No support for final feature axis yet')
        batch_size = input.shape[0]
        sequence_length = input.shape[-2]

        if initial_states is None:
            states = self.cell.init_states(shape=input.shape[:-2])
        else:
            states = initial_states

        out_sequence = jnp.zeros(input.shape[:-1])
        for s_idx in range(sequence_length):
            out, states = self.cell(
                input[..., s_idx, :],
                states
            )
            out_sequence = out_sequence.at[..., s_idx].set(out)

        if initial_states is None:
            return out_sequence
        else:
            return out_sequence, states


class IIRFilter(DTDF):
    """Currently an alias for DTDF."""


class IIRFiltFilt(IIRFilter):
    def forward(
        self,
        input: Tensor,
        initial_states: Optional[Tensor] = None,
        feature_ax: bool = False
    ):
        raise NotImplementedError('Zero-phase filter not yet implemented')
        out = super()(
            input=input,
            initial_states=initial_states,
            feature_ax=feature_ax
        )
        if not feature_ax:
            out = jnp.flip(out, -1)
        else:
            out = jnp.flip(out, -2)
        out = super()(
            input=out,
            initial_states=initial_states,
            feature_ax=feature_ax
        )
        if not feature_ax:
            return jnp.flip(out, -1)
        return jnp.flip(out, -2)
