# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain surface plots with `surfplot`.
These currently require a custom patch of `surfplot`.
"""
import torch
import surfplot
import numpy as np
import nibabel as nb
import matplotlib
import matplotlib.pyplot as plt
import templateflow.api as tflow
from hypercoil.engine import Sentry
from hypercoil.functional.cmass import cmass_coor
from hypercoil.functional.sphere import spherical_geodesic
from hypercoil.neuro.const import fsLR
from hypercoil.viz.surfutils import _CMapFromSurfMixin


POLE_DECODER = {
    'cortex_L': np.array([
        'medial', 'anterior', 'dorsal',
        'lateral', 'posterior', 'ventral'
    ]),
    'cortex_R': np.array([
        'lateral', 'anterior', 'dorsal',
        'medial', 'posterior', 'ventral'
    ]),
}

POLES = torch.tensor([
    [100., 0., 0.],
    [0., 100., 0.],
    [0., 0., 100.],
    [-100., 0., 0.],
    [0., -100., 0.],
    [0., 0., -100.],
])


VIEWS = {
    'dorsal' : {
        'views' : 'dorsal',
        'size' : (250, 300),
        'zoom' : 3
    },
    'ventral' : {
        'views' : 'ventral',
        'size' : (250, 300),
        'zoom' : 3,
        'flip' : True
    },
    'posterior' : {
        'views' : 'posterior',
        'size' : (300, 300),
        'zoom' : 3
    },
    'anterior' : {
        'views' : 'anterior',
        'size' : (300, 300),
        'zoom' : 3,
        'flip' : True
    },
    'medial' : {
        'views' : 'medial',
        'size' : (900, 300),
        'zoom' : 1.8,
    },
    'lateral' : {
        'views' : 'lateral',
        'size' : (900, 300),
        'zoom' : 1.8,
    },
}


class fsLRSurfacePlot(Sentry):
    def __init__(self, atlas):
        super().__init__()
        self.module = atlas
        self.atlas = atlas.atlas
        coor_query = fsLR().TFLOW_COOR_QUERY
        coor_query.update(suffix='veryinflated')
        self.lh, self.rh = (
            tflow.get(**coor_query, hemi='L'),
            tflow.get(**coor_query, hemi='R')
        )
        self.dim_lh = nb.load(self.lh).darrays[0].dims[0]
        self.dim_rh = nb.load(self.rh).darrays[0].dims[0]
        self.dim = self.dim_lh + self.dim_rh

        self.data_mask = {
            'cortex_L' : torch.ones_like(self.atlas.mask),
            'cortex_R' : torch.ones_like(self.atlas.mask),
            'all' : torch.ones_like(self.atlas.mask)
        }
        self.cmap_mask = {
            'cortex_L' : self.atlas.compartments['cortex_L'].clone(),
            'cortex_R' : self.atlas.compartments['cortex_R'].clone(),
            'all' : self.atlas.mask.clone()
        }
        self.data_mask['cortex_L'][self.dim_lh:] = False
        self.data_mask['cortex_R'][:self.dim_lh] = False
        self.data_mask['cortex_R'][self.dim:] = False
        self.data_mask['all'][self.dim:] = False
        self.cmap_mask['cortex_L'][self.dim_lh:] = False
        self.cmap_mask['cortex_R'][:self.dim_lh] = False
        self.cmap_mask['cortex_R'][self.dim:] = False
        self.cmap_mask['all'][self.dim:] = False
        self.cortical_mask = (
            self.atlas.compartments['cortex_L'] |
            self.atlas.compartments['cortex_R']
        )
        self.coor_all = self.atlas.coors[self.cortical_mask[self.atlas.mask]]

    def drop_null(self, null=0):
        subset = (self.data != null)
        return self.data[subset], self.coor_all[subset]


class fsLRAtlasParcels(
    _CMapFromSurfMixin,
    fsLRSurfacePlot
):
    def __call__(self, cmap, views=('lateral', 'medial'), save=None):
        offscreen = False
        if save is not None:
            matplotlib.use('agg')
            offscreen = True
        data = torch.zeros_like(self.atlas.mask, dtype=torch.long)
        for compartment in ('cortex_L', 'cortex_R'):
            mask = self.atlas.compartments[compartment]
            labels = self.module.weight[compartment].argmax(0).detach()
            compartment_data = (
                self.atlas.decoder[compartment][labels]
            )
            compartment_data[self.module.weight[compartment].sum(0) == 0] = 0
            data[mask] = compartment_data
        self.data = data[self.cmap_mask['all']].cpu()
        labels = self.atlas.decoder['_all']
        cmap = self._select_cmap(cmap=cmap, labels=labels)
        self.data = data[self.data_mask['all']].cpu().numpy()

        for view in views:
            view_args = VIEWS[view]
            p = surfplot.Plot(
                surf_lh=self.lh,
                surf_rh=self.rh,
                brightness=1,
                **view_args
            )
            p.offscreen = offscreen
            p.add_layer(
                self.data.astype('long')[:self.dim],
                cmap=cmap,
                color_range=(1, len(cmap.colors)),
                cbar=None
            )
            fig = p.build()
            fig.set_dpi(200)
            if save is not None:
                plt.savefig(f'{save}_view-{view}.png',
                            dpi=1000,
                            bbox_inches='tight')
                plt.close('all')
            else:
                fig.show()


class fsLRAtlasMaps(fsLRSurfacePlot):
    def plot_nodemaps(self, figs, max_per_batch=21, scale=(2, 2),
                      nodes_per_row=3, start_batch=0, stop_batch=None,
                      save=None):
        # This ridiculous-looking hack is necessary to ensure the first
        # figure is saved with the correct proportions.
        plt.figure()
        plt.close('all')

        n_figs = len(figs)
        n_batches = int(np.ceil(n_figs / max_per_batch))

        figs_plotted = 0
        figs_remaining = n_figs
        batch_index = start_batch
        if stop_batch is None:
            stop_batch = n_batches
        stop_batch = min(stop_batch, n_batches)

        while batch_index < stop_batch:
            start_fig = batch_index * max_per_batch
            figs_remaining = n_figs - start_fig
            figs_per_batch = min(max_per_batch, figs_remaining)
            stop_fig = start_fig + figs_per_batch

            n_rows = int(np.ceil(figs_per_batch / nodes_per_row))
            figsize = (6 * nodes_per_row, 3 * n_rows)


            fig, ax = plt.subplots(
                n_rows,
                nodes_per_row,
                figsize=figsize,
                num=1,
                clear=True
            )
            batch_figs = figs[start_fig:stop_fig]
            for index, (name, f) in enumerate(batch_figs):
                i = index // nodes_per_row
                j = index % nodes_per_row
                f._check_offscreen()
                x = f.to_numpy(transparent_bg=True, scale=(scale))
                ax[i, j].imshow(x)
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                ax[i, j].set_title(f'Node {name}')

            if save:
                plt.tight_layout()
                plt.savefig(f'{save}_batch-{batch_index}.png',
                            dpi=300,
                            bbox_inches='tight')
                fig.clear()
                plt.close('all')
            else:
                fig.show()
            batch_index += 1

    def __call__(self, cmap='Blues', color_range=(0, 1),
                 max_per_batch=21, stop_batch=None, save=None):
        offscreen = False
        if save is not None:
            matplotlib.use('agg')
            offscreen = True
        figs = [None for _ in range(self.atlas.decoder['_all'].max() + 1)]
        for compartment in ('cortex_L', 'cortex_R'):
            map = self.module.weight[compartment]
            decoder = self.atlas.decoder[compartment]
            compartment_mask = self.atlas.compartments[compartment]
            coor = self.atlas.coors[compartment_mask[self.atlas.mask]].t()
            cmasses = cmass_coor(map, coor, radius=100)
            closest_poles = spherical_geodesic(
                cmasses.t(),
                POLES.to(device=cmasses.device, dtype=cmasses.dtype)
            ).argsort(-1)[:, :3].cpu()
            closest_poles = POLE_DECODER[compartment][closest_poles.numpy()]
            if compartment == 'cortex_L':
                surf_lh = self.lh
                surf_rh = None
            elif compartment == 'cortex_R':
                surf_lh = None
                surf_rh = self.rh
            for node, views, name in zip(map, closest_poles, decoder):
                data = torch.zeros_like(
                    self.atlas.mask,
                    dtype=map.dtype
                )
                data[compartment_mask] = node.detach()
                data = data[self.data_mask[compartment]].cpu().numpy()
                p = surfplot.Plot(
                    surf_lh=surf_lh,
                    surf_rh=surf_rh,
                    brightness=1,
                    views=views.tolist(),
                    zoom=1.25,
                    size=(400, 200)
                )
                p.offscreen = offscreen
                p.add_layer(
                    data[:self.dim],
                    cmap=cmap,
                    cbar=None,
                    color_range=color_range
                )
                figs[name] = p.render()
        figs = [(i, f) for i, f in enumerate(figs) if f is not None]

        n_figs = len(figs)
        batches_per_run = 5
        if stop_batch is None:
            total_batches = int(np.ceil(n_figs / max_per_batch))
        else:
            total_batches = stop_batch
        self.plot_nodemaps(figs=figs, save=save,
                           stop_batch=total_batches)
