# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
IIR filter
~~~~~~~~~~
IIR filter implemented as RNN. This follows a close adaptation of:
https://github.com/boris-kuz/differentiable_iir_filters

If you use this in published work, you must cite:

Kuznetsov B, Parker JD, Esqueda F (2020) Differentiable IIR filters for machine
learning applications. DAFx2020, Vienna, Austria, September 2020-21.
"""
import torch
import numpy as np
from torch.nn import Module, Parameter
from torch import FloatTensor
from numpy.random import uniform


class DTDFCell(Module):
    def __init__(self, init_spec):
        super(DTDFCell, self).__init__()
        self.N = init_spec.N
        self.multiplier = 1
        if init_spec.btype in ('bandpass', 'bandstop'):
            self.multiplier = 2
        self.b = Parameter(torch.zeros(self.multiplier * self.N + 1))
        self.a = Parameter(torch.zeros(self.multiplier * self.N))
        self.init_params(init_spec)

    def forward(self, input, v):
        output = input * self.b[0] + v[..., 0].unsqueeze(-1)

        v_new = input * self.b[1:] - output * self.a
        v_new[..., :-1] = v_new[..., :-1] + v[..., 1:]
        return output.squeeze(-1), v_new

    def init_states(self, size):
        v = torch.zeros(*size, self.multiplier * self.N).to(
            next(self.parameters()).device)
        return v

    def init_params(self, spec):
        spec.initialise_coefs()
        coefs = torch.tensor(np.array(spec.coefs))
        b = coefs[..., 0, :]
        a = coefs[..., 1, :][..., 1:]
        rg = self.b.requires_grad
        self.b.requires_grad = False
        self.b[:] = b
        self.b.requires_grad = rg
        rg = self.a.requires_grad
        self.a.requires_grad = False
        self.a[:] = a
        self.a.requires_grad = rg


class Spec(object):
    def __init__(self, N):
        self.N = N
        self.btype = 'lowpass'


class DTDF(Module):
    def __init__(self, spec):
        super(DTDF, self).__init__()
        self.cell = DTDFCell(init_spec=spec)

    def forward(self, input, initial_states=None, feature_ax=False):
        if not feature_ax:
            input = input.unsqueeze(-1)
        else:
            raise NotImplementedError('No support for final feature axis yet')
        batch_size = input.shape[0]
        sequence_length = input.shape[-2]

        if initial_states is None:
            states = self.cell.init_states(input.shape[:-2])
        else:
            states = initial_states

        out_sequence = torch.zeros(input.shape[:-1]).to(input.device)
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
