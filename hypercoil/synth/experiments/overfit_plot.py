# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Overfit model plot
~~~~~~~~~~~~~~~~~~
Plot learning progress and outcome when overfitting a model to a small
dataset.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt


# TODO
# This scheme is bad, since it doesn't allow different regularisations to
# be applied to different parts of the model. So far, we haven't needed this,
# but it's likely that at some point we will.
def default_penalise(model):
    return model.weight


def default_plot(model):
    return model.weight


def overfit_and_plot_progress(
        out_fig, model, optim, reg, loss, max_epoch, X, Y, target,
        log_interval=25, seed=0, value_min=0.1, value_max=0.55,
        penalise=default_penalise, plot=default_plot
    ):
    plt.figure(figsize=(9, 18))
    plt.subplot(3, 1, 2)
    color = np.array([value_min] * 3)
    incr = (value_max - color) / max_epoch

    print('Statistics at train start:')
    for i in range(len(reg)):
        print(f'- {reg[i]}: {reg[i](penalise(model)).detach().item()}')

    # Not sure that we actually need this here...
    torch.manual_seed(seed)
    losses = [float('inf') for _ in range(max_epoch)]
    for e in range(max_epoch):
        Y_hat = model(X).squeeze()
        l = loss(Y, Y_hat) + reg(penalise(model))
        l.backward()
        optim.step()
        model.zero_grad()
        losses[e] = l.item()
        if e % log_interval == 0:
            print(f'[ Epoch {e} | Loss {l} ]')
            plt.plot(plot(model).squeeze().detach().numpy(),
                     color=(1 - color))
            color = color + incr * log_interval
    plt.plot(plot(model).squeeze().detach().numpy(), color='red')
    plt.gca().set_title('Weight over the course of learning')
    plt.subplot(3, 1, 1)
    plt.plot(losses)
    plt.gca().set_title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(3, 1, 3)
    plt.plot(target)
    plt.plot(plot(model).squeeze().detach().numpy())
    plt.gca().set_title('Learned and target weights')
    plt.legend(['Target', 'Learned'])
    plt.savefig(out_fig, bbox_inches='tight')
