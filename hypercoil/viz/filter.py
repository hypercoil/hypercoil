# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Filter scaling visualisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Visualisation for filter scaling experiments. Eventually we'll want to adapt
this into something more general.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hypercoil.engine import Sentry
from hypercoil.functional.domainbase import complex_decompose


class StreamPlot(Sentry):
    def __init__(self, stream, objective='partition'):
        self.stream = stream
        self.objective = objective

    def __call__(self, save=None):
        weight = torch.stack([
            f.model.weight.detach() for f in self.stream.fftfilter
        ]).transpose(-1, -2)
        weight, _ = complex_decompose(weight)
        if self.objective == 'partition':
            n_bands = weight.shape[-1] - 1
            if n_bands == 2:
                to_plot = weight.tile()
            elif n_bands == 3:
                to_plot = weight.transpose(0, -1)[:n_bands].transpose(0, -1)
                cmap = None
            else:
                to_plot = weight.argmax(-1).squeeze()
        else:
            to_plot = weight.squeeze()
            cmap = 'magma'
        print(to_plot.shape)
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        ax.imshow(to_plot, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if save is not None:
            fig.savefig(save, bbox_inches='tight')
