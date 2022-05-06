# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Denoising model evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~
Evaluation of a proposed denoising model on (previously unseen) data.
"""
import torch
import templateflow.api as tflow
from hypercoil.nn import AtlasLinear
from hypercoil.init import CortexSubcortexCIfTIAtlas
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
        return qcfc_loss(FC=sym2vec(fc), QC=qc, tol=0, abs=False)


class DenoisingEval:
    def __init__(
        self, confound_names, evaluate_qcfc=True, evaluate_varexp=True,
        plot_result=True, atlas=None, significance=0.05):
        self.confound_names = confound_names
        if evaluate_varexp:
            self.varexp = VarianceExplained()
        else:
            self.varexp = False
        if evaluate_qcfc:
            self.qcfc = QCFCEdgewise()
        else:
            self.qcfc = False
        if plot_result:
            self.atlas = self.cfg_atlas(atlas)
            self.plotter = QCFCPlot(self.atlas)
            self.significance = significance
        else:
            self.plotter = False

        self.results = {}

    def cfg_atlas(self, atlas_path):
        ##TODO: hard coding surface here for now.
        atlas = CortexSubcortexCIfTIAtlas(
            ref_pointer=atlas_path,
            mask_L=tflow.get(
                template='fsLR',
                hemi='L',
                desc='nomedialwall',
                density='32k'),
            mask_R=tflow.get(
                template='fsLR',
                hemi='R',
                desc='nomedialwall',
                density='32k'),
            clear_cache=False,
            dtype=torch.float
        )
        lin = AtlasLinear(atlas)
        return lin

    def evaluate(self, connectomes, model, confounds, qc, save=None):
        ##TODO: enable partial regression
        if self.varexp:
            self.results['variance_explained'] = self.varexp(
                confmodel=model,
                confounds=confounds,
                names=self.confound_names
            )
        if self.qcfc:
            qcfc = self.qcfc(connectomes, qc).squeeze()
            if self.plotter:
                n = connectomes.size(0)
                self.plotter(
                    qcfc=vec2sym(qcfc),
                    n=n,
                    significance=self.significance,
                    save=f'{save}.png'
                )
            thresh = auto_tol(batch_size=n, significance=self.significance)
            self.results['qcfc'] = {
                'number_sig_edges' : (qcfc > thresh).sum().item(),
                'abs_med_corr' : qcfc.abs().median().item(),
                'edges' : qcfc.squeeze().tolist()
            }
