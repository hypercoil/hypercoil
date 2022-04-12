# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spatial nulls
~~~~~~~~~~~~~
Create a null distribution of spatial-only parcellations.
"""
import os
import torch
import nibabel as nb
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


ref_pointer = pkgrf(
    'hypercoil',
    'viz/resources/nullexample.nii'
)
atlas = CortexSubcortexCIfTIAtlas(
    ref_pointer='/tmp/spatialnulls/parcellation000000/parcellation000000.nii',
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
modal_cmap = pkgrf(
    'hypercoil',
    'viz/resources/cmap_modal.nii'
)
all_views = (
    'dorsal', 'ventral',
    'posterior', 'anterior',
    'medial', 'lateral'
)
plotter = fsLRAtlasParcels(lin)
plotter(
    cmap=modal_cmap,
    views=all_views,
    save='/tmp/regressiontestatlasspatial'
)


max_epoch = 2001
log_interval = 25
distribution_size = 200
n_labels_min = 50
n_labels_max = 501
lr = 0.1
atlas_path = '/mnt/pulsar/Downloads/atlases/glasser.nii'
modal_cmap = pkgrf(
    'hypercoil',
    'viz/resources/cmap_modal.nii'
)
surf_views = (
    'medial', 'lateral'
)


for n_labels in range(n_labels_min, n_labels_max):
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
            'cortex_L': n_labels,
            'cortex_R': n_labels,
            'subcortex': 0
        },
        conc=100.,
        dtype=torch.float,
        device='cuda:0'
    )
    atlas = AtlasLinear(dirichlet, mask_input=False,
                        domain=dirichlet.init['_all'].domain,
                        dtype=torch.float, device='cuda:0')
    opt = torch.optim.Adam(params=atlas.parameters(), lr=lr)

    lh_coor = atlas.coors[dirichlet.compartments['cortex_L'][atlas.mask]].t()
    rh_coor = atlas.coors[dirichlet.compartments['cortex_R'][atlas.mask]].t()


    loss = LossScheme((
        LossScheme((
            Entropy(nu=0.0005, name='LeftHemisphereEntropy'),
            Equilibrium(nu=500, name='LeftHemisphereEquilibrium'),
            Compactness(nu=2, radius=100, coor=lh_coor,
                        name='LeftHemisphereCompactness')
        ), apply=lambda arg: arg.lh),
        LossScheme((
            Entropy(nu=0.0005, name='RightHemisphereEntropy'),
            Equilibrium(nu=500, name='RightHemisphereEquilibrium'),
            Compactness(nu=2, radius=100, coor=rh_coor,
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


    for epoch in range(max_epoch):
        arg = LossArgument(
            lh=atlas.weight['cortex_L'],
            rh=atlas.weight['cortex_R'],
            lh_coor=lh_coor, rh_coor=rh_coor
        )
        out = loss(arg, verbose=(epoch % log_interval == 0))
        print(f'[ Epoch {epoch} | Total loss {out} ]\n')
        out.backward()
        opt.step()
        opt.zero_grad()

    results = f'/tmp/spatialnulls/parcellation_labels-{n_labels:06}'
    os.makedirs(results, exist_ok=True)
    plotter = fsLRAtlasParcels(atlas)
    plotter(
        cmap=modal_cmap,
        views=surf_views,
        save=f'{results}/parcellation_labels-{n_labels:06}_cmap-modal'
    )
    mask_L = dirichlet.compartments['cortex_L'][atlas.mask].unsqueeze(0).cpu()
    mask_R = dirichlet.compartments['cortex_R'][atlas.mask].unsqueeze(0).cpu()
    results_L = atlas.weight['cortex_L'].argmax(0).cpu() + 1
    results_R = atlas.weight['cortex_R'].argmax(0).cpu() + 1 + n_labels
    dirichlet.ref.get_fdata()[mask_L] = results_L
    dirichlet.ref.get_fdata()[mask_R] = results_R
    new_cifti = nb.Cifti2Image(
        dirichlet.ref.get_fdata(),
        header=dirichlet.ref.header,
        nifti_header=dirichlet.ref.nifti_header
    )
    new_cifti.to_filename(f'{results}/parcellation_labels-{n_labels:06}.nii')
