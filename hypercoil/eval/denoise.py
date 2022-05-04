# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Denoising model evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluation of a proposed denoising model on (previously unseen) data.
"""
import torch
from hypercoil.functional import pairedcorr, sym2vec, vec2sym
from hypercoil.loss.batchcorr import qcfc_loss, auto_tol
from hypercoil.viz.qcfc import QCFCPlot


class VarianceExplained:
    def __call__(self, confmodel, confounds, names):
        sol = torch.linalg.lstsq(
            confmodel.transpose(-1, -2),
            confounds.transpose(-1, -2)
        ).solution
        ##TODO: computing the off-diag elements is wasteful
        varexp = (torch.diagonal(pairedcorr(
            sol.transpose(-1, -2) @ confmodel, confounds
        ), dim1=-2, dim2=-1) ** 2).t()
        return {name : v.tolist() for name, v in zip(names, varexp)}


class QCFCEdgewise:
    def __call__(self, fc, qc):
        return qcfc_loss(FC=sym2vec(fc), QC=qc)


class DenoisingEval:
    def __init__(
        self, confound_names, evaluate_qcfc=True, evaluate_varexp=True,
        plot_result=True, atlas=None, significance=0.05):
        if evaluate_varexp:
            self.varexp = VarianceExplained()
        else:
            self.varexp = False
        if evaluate_qcfc:
            self.qcfc = QCFCEdgewise()
        else:
            self.qcfc = False
        if plot_result:
            self.plotter = QCFCPlot()
            self.atlas = atlas
            self.significance = significance
        else:
            self.plotter = False

        self.results = {}

    def evaluate(self, connectomes, model, confounds, qc, save=None):
        ##TODO: enable partial regression
        if self.varexp:
            self.results['variance_explained'] = self.varexp(
                confmodel=model,
                confounds=confounds,
                names=self.confound_names
            )
        if self.qcfc:
            qcfc = self.qcfc(connectomes, qc)
            if self.plotter:
                n = connectomes.size(0)
                self.plotter(
                    qcfc=vec2sym(qcfc),
                    n=n,
                    significance=self.significance,
                    save=save
                )
            thresh = auto_tol(batch_size=n, significance=self.significance)
            self.results['qcfc'] = {
                'number_sig_edges' : (qcfc > thresh).sum(),
                'abs_med_corr' : qcfc.abs().median(),
                'edges' : qcfc.tolist()
            }
