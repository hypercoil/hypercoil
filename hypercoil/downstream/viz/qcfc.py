# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
QC-FC plots with `netplotbrain`.
"""
import torch
import netplotbrain
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nb
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from templateflow import api as tflow
from hypercoil.engine import Sentry
from hypercoil.functional.cmass import cmass_coor
from hypercoil.functional.matrix import sym2vec
from hypercoil.functional.sphere import spherical_geodesic
from hypercoil.loss.batchcorr import auto_tol


class QCFCPlot(Sentry):
    def __init__(self, atlas):
        super().__init__()
        self.module = atlas
        self.atlas = atlas.atlas

    def get_vol_coordinates(self, atlas):
        coors_surf = np.empty(
            (atlas.atlas.decoder['_all'].max() + 1, 3)) * float('nan')
        coors_compartment = {}
        for hemisphere in ('L', 'R'):
            compartment = f'cortex_{hemisphere}'
            decoder = atlas.atlas.decoder[compartment]
            compartment_mask = atlas.atlas.compartments[compartment]
            coors = atlas.atlas.coors[compartment_mask[self.atlas.mask]]
            maps = atlas.weight[compartment]
            cmass = cmass_coor(maps, coors.T, radius=100)
            dist = spherical_geodesic(cmass.T, coors)
            indices = dist.argmin(-1).numpy()

            template = tflow.get(
                template='fsLR',
                density='32k',
                hemi=hemisphere,
                suffix='midthickness',
                desc=None,
                space=None
            )
            surf = nb.load(template).darrays[0].data
            coors_surf[decoder] = surf[indices]
        coors_surf = coors_surf[~np.isnan(coors_surf.sum(1))]
        nodes_surf = pd.DataFrame({
            'x': coors_surf[:, 0],
            'y': coors_surf[:, 1],
            'z': coors_surf[:, 2]
        })
        return nodes_surf

    def threshold_edges(self, edges, n, significance):
        edges = edges.cpu()
        thresh = auto_tol(batch_size=n, significance=significance)
        thresholded = edges.numpy()
        thresholded[np.abs(edges) <= thresh] = 0
        return thresholded

    def fit_line(self, df):
        predictors = np.stack([
            df.distance,
            np.ones_like(df.distance)
        ], -2).T
        sol, _, _, _ = np.linalg.lstsq(
            predictors,
            df.qcfc,
            rcond=None
        )
        x = np.stack([
            np.linspace(df.distance.min(), df.distance.max(), 10),
            np.ones(10)
        ])
        y = (x.T @ sol).squeeze()
        return x[0], y

    def __call__(self, qcfc, n, significance=0.1, save=None):
        nodes_surf = self.get_vol_coordinates(self.module)
        dist = pdist(nodes_surf.values)
        vec = pd.DataFrame({
            'qcfc' : sym2vec(qcfc.cpu().detach()).numpy().squeeze(),
            'distance' : dist.squeeze()
        })

        fig = plt.figure(figsize=(33, 8))
        ax = fig.add_subplot(141, projection='3d')

        thresholded = self.threshold_edges(
            qcfc.squeeze().detach().clone(), n=n, significance=significance)
        #print((thresholded > 0).sum(), thresholded.shape)

        netplotbrain.plot(template='MNI152NLin2009cAsym',
                          templatestyle='surface',
                          title='',
                          fig=fig,
                          ax=ax,
                          view='R',
                          nodes=nodes_surf,
                          nodetype='circles',
                          nodecolor='#3f3f3f',
                          edgecolor='#0078c6',
                          nodescale=100,
                          nodealpha=0.9,
                          edgealpha=0.7,
                          edges=(10 * thresholded))

        ax = fig.add_subplot(142)
        # using blue for positive to match overall appearance
        sns.heatmap(
            -thresholded, center=0, square=True,
            vmin=-0.4, vmax=0.4, cbar=False,
            xticklabels=False, yticklabels=False
        )
        plt.xticks([])
        plt.yticks([])

        box = dict(
            boxstyle="round,pad=0.3",
            fc="#00000077",
            lw=0
        )
        axlabel_params = {
            'size': 'xx-large',
            'fontfamily': ('FuturaAeterna', 'Futura', 'sans-serif'),
        }
        annot_params = {
            'xycoords': 'axes fraction',
            'ha': 'center',
            'va': 'center',
            'c': 'white',
            'bbox': box,
            **axlabel_params
        }

        ax = fig.add_subplot(143)
        mincorr, maxcorr = vec.qcfc.min(), vec.qcfc.max()
        mindist, maxdist = vec.distance.min(), vec.distance.max()
        sns.kdeplot(
           data=vec, x='qcfc', fill=True, ax=ax,
           alpha=.5, linewidth=0,
        )
        plt.xlim(mincorr, maxcorr)
        plt.xticks([])
        plt.yticks([])
        plt.annotate(
            f'{mincorr:.2f}',
            xy=(0.05, 0.5),
            rotation='vertical',
            **annot_params
        )
        plt.annotate(
            f'{maxcorr:.2f}',
            xy=(0.95, 0.5),
            rotation='vertical',
            **annot_params
        )
        plt.ylabel('', **axlabel_params)
        plt.xlabel('Correlations', **axlabel_params)
        plt.axvline(0, color='black', linewidth=4)

        ax = fig.add_subplot(144)
        fit_x, fit_y = self.fit_line(vec)
        sns.kdeplot(
            data=vec, x='distance', y='qcfc',
            fill=True, ax=ax
        )
        plt.xlim(mindist, maxdist)
        plt.ylim(mincorr, maxcorr)
        plt.xticks([])
        plt.yticks([])
        plt.annotate(
            f'{mindist:.2f}',
            xy=(0.05, 0.5),
            rotation='vertical',
            **annot_params
        )
        plt.annotate(
            f'{maxdist:.2f}',
            xy=(0.95, 0.5),
            rotation='vertical',
            **annot_params
        )
        plt.annotate(
            f'{maxcorr:.2f}',
            xy=(0.5, 0.95),
            rotation='horizontal',
            **annot_params
        )
        plt.annotate(
            f'{mincorr:.2f}',
            xy=(0.5, 0.05),
            rotation='horizontal',
            **annot_params
        )
        plt.axhline(0, color='black', linewidth=4)
        plt.plot(fit_x, fit_y, color='red', linewidth=4)
        plt.ylabel('Correlations', **axlabel_params)
        plt.xlabel('Distance', **axlabel_params)
        fig.tight_layout()
        if save is not None:
            plt.savefig(f'{save}', bbox_inches='tight')