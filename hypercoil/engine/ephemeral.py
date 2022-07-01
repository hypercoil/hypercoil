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
        if self.ephemeral_index is None:
            params_ephemeral = {'params' : params}
            params_ephemeral.update(self.params_ephemeral)
            self.param_groups += [params_ephemeral]
            self.ephemeral_index = len(self.param_groups) - 1
        else:
            self.param_groups[self.ephemeral_index]['params'] += params
        if all([not buffer for buffer in buffers.values()]):
            return
        for i, p in enumerate(params):
            if self.state.get(p) is None:
                self.state[p] = {}
            for buffer, value in buffers.items():
                #TODO: this doesn't allow for the case that we want to
                # override a buffer to `None`
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
                 weight_decay=0, nesterov=False, *, maximize=False,
                 lr_ephemeral=None, momentum_ephemeral=None,
                 dampening_ephemeral=None, weight_decay_ephemeral=None): #,
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
        if lr_ephemeral is None:
            lr_ephemeral = lr
        if momentum_ephemeral is None:
            momentum_ephemeral = momentum
        if dampening_ephemeral is None:
            dampening_ephemeral = dampening
        if weight_decay_ephemeral is None:
            weight_decay_ephemeral = weight_decay
        self.ephemeral_index = None
        self.params_ephemeral = {
            'lr' : lr_ephemeral,
            'momentum' : momentum_ephemeral,
            'dampening' : dampening_ephemeral,
            'weight_decay' : weight_decay_ephemeral,
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


class AdamEphemeral(EphemeralMixin, Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                 weight_decay=0, amsgrad=False, *, maximize=False,
                 lr_ephemeral=None, betas_ephemeral=None,
                 weight_decay_ephemeral=None): #,
                 #foreach: Optional[bool] = None):
        super().__init__(
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad = amsgrad,
            maximize=maximize,
            #foreach=foreach
        )
        if lr_ephemeral is None:
            lr_ephemeral = lr
        if betas_ephemeral is None:
            betas_ephemeral = betas
        if weight_decay_ephemeral is None:
            weight_decay_ephemeral = weight_decay
        self.ephemeral_index = None
        self.params_ephemeral = {
            'lr' : lr_ephemeral,
            'betas' : betas_ephemeral,
            'eps' : eps,
            'weight_decay' : weight_decay_ephemeral,
            'amsgrad' : amsgrad,
            'maximize' : maximize
        }

    def load_ephemeral(self, params, step=None,
                       exp_avg=None, exp_avg_sq=None):
        buffers = {
            'step': step,
            'exp_avg': exp_avg,
            'exp_avg_sq': exp_avg_sq,
        }
        super().load_ephemeral(
            params=params,
            buffers=buffers
        )
