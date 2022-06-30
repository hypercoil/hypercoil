# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Optimisers with ephemeral, instance-specific parameters.
"""
import torch
from typing import Optional
from torch.nn import Parameter
from torch.optim import (
    Optimizer, SGD, Adam
)


class EphemeralMixin:
    @property
    def ephemeral_state(self):
        state = {}
        if self.ephemeral_index is not None:
            ephemeral = self.param_groups[self.ephemeral_index]['params']
            for p in ephemeral:
                state[p] = self.state[p]
        return state

    def load_ephemeral(self, params, buffers=None):
        if isinstance(params, torch.Tensor):
            params = [params]
        params_ephemeral = {'params' : params}
        params_ephemeral.update(self.params_ephemeral)
        self.param_groups += [params_ephemeral]
        self.ephemeral_index = len(self.param_groups) - 1
        if all([not buffer for buffer in buffers.values()]):
            return
        for i, p in enumerate(params):
            self.state[p] = {}
            for buffer, value in buffers.items():
                print(buffer, value)
                if value is not None:
                    self.state[p][buffer] = value[i]

    def purge_ephemeral(self):
        if self.ephemeral_index is not None:
            ephemeral = self.param_groups[self.ephemeral_index]['params']
            for p in ephemeral:
                if self.state.get(p) is not None:
                    del self.state[p]
            del self.param_groups[self.ephemeral_index]
            self.ephemeral_index = None

    @torch.no_grad()
    def step(self, closure=None, return_ephemeral_state=True):
        super().step(closure=closure)
        if return_ephemeral_state:
            return self.ephemeral_state


class SGDEphemeral(EphemeralMixin, SGD):
    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize=False): #,
                 #foreach: Optional[bool] = None):
        super().__init__(
            params=params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            #foreach=foreach
        )
        self.ephemeral_index = None
        self.params_ephemeral = {
            'lr' : lr,
            'momentum' : momentum,
            'dampening' : dampening,
            'weight_decay' : weight_decay,
            'nesterov' : nesterov,
            'maximize' : maximize
        }

    def load_ephemeral(self, params, momentum_buffers=None):
        buffers = {
            'momentum_buffers': momentum_buffers
        }
        super().load_ephemeral(
            params=params,
            buffers=buffers
        )
