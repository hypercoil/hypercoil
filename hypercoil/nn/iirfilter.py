# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
IIR filter implemented as RNN. This follows a close adaptation of:
https://github.com/boris-kuz/differentiable_iir_filters

If you use this in published work, you must cite:

Kuznetsov B, Parker JD, Esqueda F (2020) Differentiable IIR filters for machine
learning applications. DAFx2020, Vienna, Austria, September 2020-21.

.. warning::
    This is not yet implemented.
"""
import torch
import numpy as np
from torch.nn import Module, Parameter
from torch import FloatTensor
from numpy.random import uniform


class DTDFCell(Module):
    def __init__(self, init_spec, dtype=None, device=None):

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(DTDFCell, self).__init__()
        self.N = init_spec.N
        self.multiplier = 1
        if init_spec.btype in ('bandpass', 'bandstop'):
            self.multiplier = 2
        self.b = Parameter(
            torch.zeros(self.multiplier * self.N + 1, **factory_kwargs)
        )
        self.a = Parameter(
            torch.zeros(self.multiplier * self.N, **factory_kwargs)
        )
        self.init_params(init_spec, **factory_kwargs)

    def forward(self, input, v):
        output = input * self.b[0] + v[..., 0].unsqueeze(-1)

        v_new = input * self.b[1:] - output * self.a
        v_new[..., :-1] = v_new[..., :-1] + v[..., 1:]
        return output.squeeze(-1), v_new

    def init_states(self, size, **factory_kwargs):
        v = torch.zeros(
            *size, self.multiplier * self.N,
            **factory_kwargs
        ).to(next(self.parameters()).device)
        return v

    def init_params(self, spec, **factory_kwargs):
        spec.initialise_coefs()
        coefs = torch.tensor(np.array(spec.coefs), **factory_kwargs)
        b = coefs[..., 0, :]
        a = coefs[..., 1, :][..., 1:]
        with torch.no_grad():
            self.b[:] = b
            self.a[:] = a


class Spec(object):
    def __init__(self, N):
        self.N = N
        self.btype = 'lowpass'


class DTDF(Module):
    def __init__(self, spec, device=None, dtype=None):
        raise NotImplementedError
        super(DTDF, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.factory_kwargs = factory_kwargs
        self.cell = DTDFCell(init_spec=spec, **factory_kwargs)

    def forward(self, input, initial_states=None, feature_ax=False):
        if not feature_ax:
            input = input.unsqueeze(-1)
        else:
            raise NotImplementedError('No support for final feature axis yet')
        batch_size = input.shape[0]
        sequence_length = input.shape[-2]

        if initial_states is None:
            states = self.cell.init_states(
                input.shape[:-2],
                **self.factory_kwargs
            )
        else:
            states = initial_states

        out_sequence = torch.zeros(
            input.shape[:-1],
            **self.factory_kwargs
        ).to(input.device)
        for s_idx in range(sequence_length):
            out_sequence[..., s_idx], states = self.cell(
                input[..., s_idx, :],
                states
            )
        out_sequence = out_sequence

        if initial_states is None:
            return out_sequence
        else:
            return out_sequence, states


class IIRFilter(DTDF):
    """Currently an alias for DTDF."""


class IIRFiltFilt(IIRFilter):
    def forward(self, input, initial_states=None, feature_ax=False):
        out = super().forward(
            input=input,
            initial_states=initial_states,
            feature_ax=feature_ax
        )
        if not feature_ax:
            out = out.flip(-1)
        else:
            out = out.flip(-2)
        out = super().forward(
            input=out,
            initial_states=initial_states,
            feature_ax=feature_ax
        )
        if not feature_ax:
            return out.flip(-1)
        return out.flip(-2)
