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
    hinge_loss,
    LinearKernel,
    GaussianKernel,
    PolynomialKernel
)
from hypercoil.synth.svm import (
    generate_data,
    orient_data,
    make_labels,
    labels_one_vs_one,
    labels_one_vs_rest,
    plot_prediction,
    plot_training
)


def separation_experiment(
    kernel='rbf',
    sigma=3,
    C=1.,
    n=(100, 100),
    d=2,
    mu=None,
    ori=None,
    learnable_parameter='mu',
    learnable_idx=0,
    multiclass_type='one_vs_rest',
    max_epoch=500,
    lr=0.005,
    lr_decay=0.995,
    log_interval=1,
    seed=None,
    save=None
):
    if mu is None:
        mu = [torch.zeros(d) for _ in n]
    if ori is None:
        ori = [torch.eye(d) for _ in n]
    if learnable_parameter == 'mu':
        param = mu[learnable_idx]
        coor0s = []
        coor1s = []
    elif learnable_parameter == 'ori':
        param = ori[learnable_idx]
        dets = []
        frobs = []
    param.requires_grad = True

    x = generate_data(d=d, n=n, seed=seed)
    X = orient_data(x, mu=mu, ori=ori)

    multiclass = False
    Y = make_labels(n)
    if len(n) > 2:
        multiclass = True
        # This is going to be moved into the SVM module.
        if multiclass_type == 'one_vs_rest':
            Y = labels_one_vs_rest(Y)
        elif multiclass_type == 'one_vs_one':
            Y = labels_one_vs_one(Y)

    if kernel == 'rbf':
        kernel = GaussianKernel(sigma=sigma)
    elif kernel == 'linear':
        kernel = LinearKernel()

    model = SVM(
        n=sum(n),
        K=kernel
    )

    X = torch.cat(X)
    Y_hat = model(X, Y)
    plot_prediction(X, Y, Y_hat, save=f'{save}_epoch-start.png')

    opt = torch.optim.Adam(params=[param], lr=lr)

    losses = []

    for epoch in range(max_epoch):

        X = orient_data(x, mu=mu, ori=ori)
        X = torch.cat(X)
        Y_hat = model(X, Y)
        loss = hinge_loss(Y_hat, Y.squeeze())
        losses += [loss.detach().item()]
        if learnable_parameter == 'mu':
            coor0s += [param[0].clone().detach().numpy()]
            coor1s += [param[0].clone().detach().numpy()]
        elif learnable_parameter == 'ori':
            det = torch.abs(torch.linalg.det(param))
            dets += [det.detach().item()]
            frob = torch.linalg.matrix_norm(param)
            frobs += [frob.detach().item()]
        loss.backward()
        #print(model.symsqker.grad)
        #print(model.ker.grad)
        #print(O.grad)
        opt.step()
        opt.param_groups[0]['lr'] *= lr_decay
        if epoch % log_interval == 0:
            plot_prediction(
                X, Y, Y_hat,
                save=f'{save}_epoch-{epoch:07}.png',
                plot_confusion=(not multiclass),
                legend=False
            )
            print(f'[ Epoch {epoch} | Loss {loss} ]')
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


if __name__ == '__main__':
    import os
    import hypercoil
    results = os.path.abspath(f'{hypercoil.__file__}/../results')

    print('\n-----------------------------------------')
    print('Experiment 1: Linear A')
    print('-----------------------------------------')
    os.makedirs(f'{results}/svm_expt-linear0', exist_ok=True)
    separation_experiment(
        kernel='linear',
        C=1.,
        n=(100, 100),
        d=2,
        mu=[torch.tensor([1., -1.]), torch.ones(2)],
        ori=[torch.tensor([[-2., 1.], [1., -2.]]) for _ in range(2)],
        learnable_parameter='mu',
        learnable_idx=0,
        max_epoch=101,
        lr=0.05,
        lr_decay=0.995,
        log_interval=1,
        seed=0,
        save=f'{results}/svm_expt-linear0/svm_expt-linear0'
    )

    print('\n-----------------------------------------')
    print('Experiment 1: Linear B')
    print('-----------------------------------------')
    os.makedirs(f'{results}/svm_expt-linear1', exist_ok=True)
    separation_experiment(
        kernel='linear',
        C=1.,
        n=(100, 100),
        d=2,
        mu=[torch.tensor([1., -1.]), torch.ones(2)],
        ori=[torch.tensor([[1., -2.], [-2., 1.]]) for _ in range(2)],
        learnable_parameter='mu',
        learnable_idx=0,
        max_epoch=101,
        lr=0.05,
        lr_decay=0.995,
        log_interval=1,
        seed=0,
        save=f'{results}/svm_expt-linear1/svm_expt-linear1'
    )

    print('\n-----------------------------------------')
    print('Experiment 3: Radial Collapse')
    print('-----------------------------------------')
    os.makedirs(f'{results}/svm_expt-rbfcollapse', exist_ok=True)
    separation_experiment(
        kernel='rbf',
        sigma=1,
        C=0.1,
        n=(100, 100),
        d=2,
        mu=[torch.ones(2) for _ in range(2)],
        ori=[2 * torch.eye(2), 4 * torch.eye(2)],
        learnable_parameter='ori',
        learnable_idx=0,
        max_epoch=501,
        lr=0.005,
        lr_decay=0.995,
        log_interval=1,
        seed=0,
        save=f'{results}/svm_expt-rbfcollapse/svm_expt-rbfcollapse'
    )

    print('\n-----------------------------------------')
    print('Experiment 4: Radial Expansion')
    print('-----------------------------------------')
    os.makedirs(f'{results}/svm_expt-rbfexpand', exist_ok=True)
    separation_experiment(
        kernel='rbf',
        sigma=3,
        C=0.1,
        n=(100, 100),
        d=2,
        mu=[torch.ones(2) for _ in range(2)],
        ori=[5 * torch.eye(2), 3 * torch.eye(2)],
        learnable_parameter='ori',
        learnable_idx=0,
        max_epoch=501,
        lr=0.02,
        lr_decay=0.995,
        log_interval=1,
        seed=0,
        save=f'{results}/svm_expt-rbfexpand/svm_expt-rbfexpand'
    )
