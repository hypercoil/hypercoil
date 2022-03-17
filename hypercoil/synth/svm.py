# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
SVM data synthesis
~~~~~~~~~~~~~~~~~~
Synthesise some simple ground truth datasets for testing the differentiable
SVM.
"""
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generate_data(d=2, n=(100, 100), seed=None):
    """
    Generate data. Note that this does not orient the data.

    Note: Code essentially follows
    https://ecomunsing.com/build-your-own-support-vector-machine

    Parameters
    ----------
    d : int
        Number of features.
    n : iterable(int)
        Number of examples per class.
    """
    if seed is not None: torch.manual_seed(seed)
    return [torch.randn(m, d) for m in n]


def orient_data(x, mu=None, ori=None):
    """
    Orient data.

    Parameters
    ----------
    x : iterable(tensor)
        Dataset of examples for each class.
    mu : iterable(tensor)
        Bias (displacement) vectors for each class.
    ori : iterable(tensor)
        Orientation vectors for each class.

    Note: Code essentially follows
    https://ecomunsing.com/build-your-own-support-vector-machine
    """
    d = x[0].shape[-1]
    if mu is None:
        mu = [torch.zeros(d) for _ in x]
    if ori is None:
        ori = [torch.eye(d) for _ in x]
    return [
        mu_i + x_i @ ori_i
        for (x_i, mu_i, ori_i) in zip(x, mu, ori)
    ]


def make_labels(n=(100, 100)):
    if len(n) == 2:
        return torch.cat((
            torch.ones((n[0], 1)),
            -torch.ones((n[1], 1))
        ))
    return torch.cat([
        label * torch.ones((count, 1), dtype=torch.int64)
        for label, count in enumerate(n)
    ])


def labels_one_vs_rest(labels):
    uniq = labels.unique()
    return [
        2 * (labels == u).float() - 1
        for u in uniq
    ]


def labels_one_vs_one(labels):
    uniq = labels.unique()
    uniqlist = uniq.tolist()
    label_combinations = product(uniqlist, uniqlist)
    return [
        (labels == label1).float() - (labels == label2).float()
        for label1, label2 in label_combinations
        if label1 != label2
    ]


def plot_prediction(
    X, Y, Y_hat,
    plot_confusion=False,
    plot_dims=(0, 1),
    legend=True,
    save=None
):
    df_dict = {
        'y': Y.detach().squeeze(),
        'y_hat': Y_hat.detach().squeeze()
    }
    for dim, vec in enumerate(X.t()):
        df_dict[f'x_{dim}'] = vec.detach().squeeze()
    df = pd.DataFrame(df_dict)

    fig_params = (10, 2)
    if plot_confusion:
        fig_params = (15, 3)


    plt.figure(figsize=(fig_params[0], 5))
    plt.subplot(1, fig_params[1], 1)
    plt.title('True')
    sns.scatterplot(
        x=f'x_{plot_dims[0]}',
        y=f'x_{plot_dims[1]}',
        hue='y',
        data=df,
        legend=legend
    )
    plt.subplot(1, fig_params[1], 2)
    plt.title('Predicted')
    sns.scatterplot(
        x=f'x_{plot_dims[0]}',
        y=f'x_{plot_dims[1]}',
        hue='y_hat',
        data=df,
        legend=legend
    )
    if plot_confusion:
        conf = confusion(Y, Y_hat)
        plt.subplot(1, fig_params[1], 3)
        sns.barplot(
            x='category',
            y='count',
            hue='hue',
            palette='bone',
            data=conf
        )
        plt.ylim(0, 100)
        plt.legend(loc='upper left')
    plt.title('Results')
    if save:
        plt.savefig(save, bbox_inches='tight')


def confusion(Y, Y_hat):
    #TODO: get this working as one-vs-rest in the multiclass setting
    pred = (2 * ((Y_hat > 0).float() - 0.5)).numpy()
    tr = (Y.squeeze().numpy() == pred)
    fa = ~tr
    pos = (Y.squeeze().numpy() == -1)
    neg = ~pos
    tp = np.logical_and(tr, pos).sum()
    tn = np.logical_and(tr, neg).sum()
    fp = np.logical_and(fa, pos).sum()
    fn = np.logical_and(fa, neg).sum()
    return pd.DataFrame({
        'category': ['Positive', 'Negative', 'Positive', 'Negative'],
        'count': [tp, tn, fp, fn],
        'hue': [True, True, False, False]
    })


def plot_training(epoch_vals, training_legend, save=None):
    plt.figure(figsize=(10, 4))
    for v in epoch_vals:
        plt.plot(np.abs(v) / np.max(np.abs(v)))
    plt.legend(training_legend)
    if save:
        plt.savefig(save, bbox_inches='tight')
