# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spatial-only nulls
~~~~~~~~~~~~~~~~~~
Learning a null parcellation from spatial regularisers alone.
"""
import torch
import templateflow.api as tflow
from pkg_resources import resource_filename as pkgrf
from hypercoil.functional import spherical_geodesic, cmass_coor
from hypercoil.loss import (
    LossScheme,
    LossApply,
    LossArgument,
    UnpackingLossArgument,
    Compactness,
    Entropy,
    Equilibrium,
    HemisphericTether,
    VectorDispersion
)
from hypercoil.nn import AtlasLinear
from hypercoil.init.atlas import DirichletInitSurfaceAtlas
from hypercoil.engine.terminal import ReactiveTerminal
from hypercoil.viz.surf import (
    fsLRAtlasParcels,
    fsLRAtlasMaps
)


max_epoch = 1001
log_interval = 25
lr = 0.1
atlas_path = '/mnt/pulsar/Downloads/atlases/glasser.nii'
modal_cmap = pkgrf(
    'hypercoil',
    'viz/resources/cmap_modal.nii'
)
network_cmap = pkgrf(
    'hypercoil',
    'viz/resources/cmap_network.nii'
)


dirichlet = DirichletInitSurfaceAtlas(
    cifti_template=atlas_path,
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
    compartment_labels={
        'cortex_L': 200,
        'cortex_R': 200,
        'subcortex': 0
    },
    conc=100.,
    dtype=torch.float,
    device='cuda:0'
)
atlas = AtlasLinear(dirichlet, mask_input=False,
                    domain=dirichlet.init['_all'].domain,
                    dtype=torch.float, device='cuda:0')

lh_coor = atlas.coors[dirichlet.compartments['cortex_L'][atlas.mask]].t()
rh_coor = atlas.coors[dirichlet.compartments['cortex_R'][atlas.mask]].t()

opt = torch.optim.Adam(params=atlas.parameters(), lr=lr)
loss = LossScheme((
    LossScheme((
        Entropy(nu=0.0005, name='LeftHemisphereEntropy'),
        Equilibrium(nu=500, name='LeftHemisphereEquilibrium'),
        Compactness(nu=0.2, radius=100, coor=lh_coor,
                    name='LeftHemisphereCompactness')
    ), apply=lambda arg: arg.lh),
    LossScheme((
        Entropy(nu=0.0005, name='RightHemisphereEntropy'),
        Equilibrium(nu=500, name='RightHemisphereEquilibrium'),
        Compactness(nu=0.2, radius=100, coor=rh_coor,
                    name='RightHemisphereCompactness')
    ), apply=lambda arg: arg.rh),
    LossApply(
        HemisphericTether(nu=1, name='InterHemisphericTether'),
        apply=lambda arg: UnpackingLossArgument(
            lh=arg.lh,
            rh=arg.rh,
            lh_coor=arg.lh_coor,
            rh_coor=arg.rh_coor)
    ),
    LossApply(
        VectorDispersion(nu=10, metric=spherical_geodesic,
                         name='LeftHemisphereDispersion'),
        apply=lambda arg: cmass_coor(arg.lh, arg.lh_coor, 100).t()),
    LossApply(
        VectorDispersion(nu=10, metric=spherical_geodesic,
                         name='RightHemisphereDispersion'),
        apply=lambda arg: cmass_coor(arg.rh, arg.rh_coor, 100).t()),
))
"""
terminal_L = ReactiveTerminal(
    loss=Compactness(nu=1, radius=100, coor=lh_coor,
                     name='LeftHemisphereCompactness'),
    slice_target='X',
    slice_axis=-2,
    max_slice=10
)
terminal_R = ReactiveTerminal(
    loss=Compactness(nu=1, radius=100, coor=rh_coor,
                     name='RightHemisphereCompactness'),
    slice_target='X',
    slice_axis=-2,
    max_slice=1
)
"""


for epoch in range(max_epoch):
    arg = LossArgument(
        lh=atlas.weight['cortex_L'],
        rh=atlas.weight['cortex_R'],
        lh_coor=lh_coor, rh_coor=rh_coor
    )
    """
    arg_L = {'X': arg.lh}
    arg_R = {'X': arg.rh}
    out = terminal_L(arg_L)
    print(f'{terminal_L.loss} : {out}')
    out = terminal_R(arg_R)
    print(f'{terminal_R.loss} : {out}')
    """
    out = loss(arg, verbose=(epoch % log_interval == 0))
    print(f'[ Epoch {epoch} | Total loss {out} ]\n')
    out.backward()
    opt.step()
    opt.zero_grad()

    if epoch % log_interval == 0:
        views = ('medial', 'lateral')
        results = '/tmp/parc-spatial200'
        plotter = fsLRAtlasParcels(atlas)
        plotter(
            cmap=modal_cmap,
            views=views,
            save=f'{results}/epoch-{epoch:08}'
        )


all_views = (
    'dorsal', 'ventral',
    'posterior', 'anterior',
    'medial', 'lateral'
)
results = '/tmp'
plotter = fsLRAtlasParcels(atlas)
plotter(
    cmap=modal_cmap,
    views=all_views,
    save=f'{results}/parcellation_cmap-modal'
)
plotter(
    cmap=network_cmap,
    views=all_views,
    save=f'{results}/parcellation_cmap-network'
)
plotter = fsLRAtlasMaps(atlas)
plotter(save=f'{results}/parcellation_maps')
