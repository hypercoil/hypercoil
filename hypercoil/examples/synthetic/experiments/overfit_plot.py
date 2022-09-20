# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Overfit model plot
~~~~~~~~~~~~~~~~~~
Plot learning progress and outcome when overfitting a model to a small
dataset.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional
from hypercoil.engine.noise import refresh
from hypercoil.engine.paramutil import PyTree, Tensor
from hypercoil.loss.scheme import LossReturn


def default_plot(model):
    return model.weight


def forward(
    model: PyTree,
    X: Tensor,
    Y: Optional[Tensor] = None,
    *,
    loss: Optional[Callable] = None,
    reg: Callable,
    key: 'jax.random.PRNGKey',
):
    key_m, key_l = jax.random.split(key, 2)
    Y_hat = model(X, key=key_m).squeeze()
    loss_epoch = 0
    loss_meta = {}
    if Y is not None:
        loss_epoch = loss(Y, Y_hat)
        loss_meta = {loss.name: LossReturn(value=loss_epoch, nu=loss.nu)}
    reg_epoch, reg_meta = reg(model, key=key_l)
    return loss_epoch + reg_epoch, {**loss_meta, **reg_meta}


def overfit_and_plot_progress(
        out_fig: str,
        model: PyTree,
        optim_state: PyTree,
        optim: Callable,
        reg: Callable,
        loss: Callable,
        max_epoch: int,
        X: Tensor,
        Y: Tensor,
        target: Tensor,
        log_interval: int = 25,
        value_min: float = 0.1,
        value_max: float = 0.55,
        plot: Callable = default_plot,
        *,
        key: 'jax.random.PRNGKey',
    ):
    plt.figure(figsize=(9, 18))
    plt.subplot(3, 1, 2)
    color = np.array([value_min] * 3)
    incr = (value_max - color) / max_epoch

    losses = [float('inf') for _ in range(max_epoch)]
    for e in range(max_epoch):
        key = jax.random.split(key, 1)[0]
        (total, meta), grad = eqx.filter_jit(eqx.filter_value_and_grad(
            forward,
            has_aux=True,
        ))(model, X, Y, loss=loss, reg=reg, key=key)
        # for k, v in meta.items():
        #     print(f'{k}: {v.value:.4f}')
        if jnp.isinf(total) or jnp.isnan(total):
            raise ValueError('Loss is NaN or Inf')
        losses[e] = total
        updates, optim_state = optim.update(
            eqx.filter(grad, eqx.is_inexact_array),
            optim_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        model = refresh(model)

        if e % log_interval == 0:
            print(f'[ Epoch {e} | Loss {total} ]')
            plt.plot(plot(model).squeeze(),
                     color=(1 - color))
            color = color + incr * log_interval

    plt.plot(plot(model).squeeze(), color='red')
    plt.gca().set_title('Weight over the course of learning')
    plt.subplot(3, 1, 1)
    plt.plot(losses)
    plt.gca().set_title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(3, 1, 3)
    plt.plot(target)
    plt.plot(plot(model).squeeze())
    plt.gca().set_title('Learned and target weights')
    plt.legend(['Target', 'Learned'])
    plt.savefig(out_fig, bbox_inches='tight')
