#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
SVM experiments
~~~~~~~~~~~~~~~
Simple experiments in classification using a differentiable SVM.
"""
import torch
from matplotlib.pyplot import close
from hypercoil.nn.svm import (
    SVM,
    LinearKernel,
    GaussianKernel,
    PolynomialKernel
)
from hypercoil.synth.svm import (
    generate_data,
    orient_data,
    make_labels,
    plot_prediction,
    plot_training
)
from hypercoil.loss import HingeLoss


def separation_experiment(
    kernel='rbf',
    sigma=3,
    C=1.,
    n=(100, 100),
    d=2,
    mu=None,
    ori=None,
    learnable_parameter='mu',
    learnable_index=0,
    multiclass_type='ovr',
    max_epoch=500,
    lr=0.005,
    lr_decay=0.995,
    log_interval=1,
    seed=None,
    save=None
):
    if mu is None:
        mu = [torch.zeros(d) for _ in n]
    else:
        mu = [torch.FloatTensor(x) for x in mu]
    if ori is None:
        ori = [torch.eye(d) for _ in n]
    else:
        ori = [torch.FloatTensor(x) for x in ori]
    if learnable_parameter == 'mu':
        param = mu[learnable_index]
        coor0s = []
        coor1s = []
    elif learnable_parameter == 'ori':
        param = ori[learnable_index]
        dets = []
        frobs = []
    param.requires_grad = True

    x = generate_data(d=d, n=n, seed=seed)
    X = orient_data(x, mu=mu, ori=ori)

    multiclass = False
    Y = make_labels(n)
    if len(n) > 2:
        multiclass = True

    if kernel == 'rbf':
        K = GaussianKernel(sigma=sigma)
    elif kernel == 'linear':
        K = LinearKernel()

    model = SVM(
        n_observations=sum(n),
        n_classes=2,
        kernel=kernel,
        C=C,
        gamma=(1 / (2 * sigma ** 2)),
        decision_function_shape=multiclass_type
    )

    X = torch.cat(X)
    Y_hat = model(X, Y)
    #raise Exception
    plot_prediction(X, Y, Y_hat, save=f'{save}_epoch-start.png')

    opt = torch.optim.Adam(params=[param], lr=lr)

    losses = []
    loss = HingeLoss()

    for epoch in range(max_epoch):

        X = orient_data(x, mu=mu, ori=ori)
        X = torch.cat(X)
        Y_hat = model(X, Y)
        loss_epoch = loss(Y_hat, model.Y)
        losses += [loss_epoch.detach().item()]
        if learnable_parameter == 'mu':
            coor0s += [param[0].clone().detach().numpy()]
            coor1s += [param[0].clone().detach().numpy()]
        elif learnable_parameter == 'ori':
            det = torch.abs(torch.linalg.det(param))
            dets += [det.detach().item()]
            frob = torch.linalg.matrix_norm(param)
            frobs += [frob.detach().item()]
        loss_epoch.backward()
        opt.step()
        param.grad.zero_()
        opt.param_groups[0]['lr'] *= lr_decay
        if epoch % log_interval == 0:
            plot_prediction(
                X, model.Y, Y_hat,
                save=f'{save}_epoch-{epoch:07}.png',
                plot_confusion=(not multiclass),
                legend=False
            )
            print(f'[ Epoch {epoch} | Loss {loss_epoch} ]')
            close('all')
    if learnable_parameter == 'mu':
        plot_training(
            epoch_vals=(losses, coor0s, coor1s),
            training_legend=('loss', r'$x_0$', r'$x_1$'),
            save=f'{save}_training.png'
        )
    elif learnable_parameter == 'ori':
        plot_training(
            epoch_vals=(losses, dets, frobs),
            training_legend=('loss', '|det|', 'Frob. norm'),
            save=f'{save}_training.png'
        )
    close('all')


def main():
    from hypercoil.synth.experiments.run import run_layer_experiments
    run_layer_experiments('svm')


if __name__ == '__main__':
    main()
