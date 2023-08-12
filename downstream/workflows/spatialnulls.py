#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spatial nulls
~~~~~~~~~~~~~
Create a null distribution of spatial-only parcellations.
"""
import os
import click
import torch
import numpy as np
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
from hypercoil.viz.surf import fsLRAtlasParcels


def create_nulls(
    out_root,
    atlas_template,
    label_counts,
    distribution_size,
    device,
    lr=0.5,
    entropy_nu=0.0005,
    equilibrium_nu=500,
    compactness_nu=2,
    dispersion_nu=10,
    tether_nu=1,
    max_epoch=2001,
    log_interval=25
):
    modal_cmap = pkgrf(
        'hypercoil',
        'viz/resources/cmap_modal.nii'
    )
    surf_views = (
        'medial', 'lateral'
    )

    for n_labels in label_counts:
        print(f'[ Null parcellation | Label count : {n_labels} ]\n')
        for iter in range(distribution_size):
            print(f'[ Null parcellation | parcellation : {iter} ]\n')
            atlas_basename = (
                f'desc-spatialnull_res-{n_labels:06}_iter-{iter:06}'
            )
            atlas_name = f'{atlas_basename}_atlas'
            dirichlet = DirichletInitSurfaceAtlas(
                cifti_template=atlas_template,
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
                name=atlas_name,
                conc=100.,
                dtype=torch.float,
                device='cuda:0'
            )
            atlas = AtlasLinear(dirichlet, mask_input=False,
                                domain=dirichlet.init['_all'].domain,
                                dtype=torch.float, device='cuda:0')
            opt = torch.optim.Adam(params=atlas.parameters(), lr=lr)

            lh_coor = (
                atlas.coors[dirichlet.compartments['cortex_L'][atlas.mask]].t()
            )
            rh_coor = (
                atlas.coors[dirichlet.compartments['cortex_R'][atlas.mask]].t()
            )


            loss = LossScheme((
                LossScheme((
                    Entropy(nu=entropy_nu,
                            name='LeftHemisphereEntropy'),
                    Equilibrium(nu=equilibrium_nu,
                                name='LeftHemisphereEquilibrium'),
                    Compactness(nu=compactness_nu, radius=100, coor=lh_coor,
                                name='LeftHemisphereCompactness')
                ), apply=lambda arg: arg.lh),
                LossScheme((
                    Entropy(nu=entropy_nu,
                            name='RightHemisphereEntropy'),
                    Equilibrium(nu=equilibrium_nu,
                                name='RightHemisphereEquilibrium'),
                    Compactness(nu=compactness_nu, radius=100, coor=rh_coor,
                                name='RightHemisphereCompactness')
                ), apply=lambda arg: arg.rh),
                LossApply(
                    HemisphericTether(nu=tether_nu,
                                      name='InterHemisphericTether'),
                    apply=lambda arg: UnpackingLossArgument(
                        lh=arg.lh,
                        rh=arg.rh,
                        lh_coor=arg.lh_coor,
                        rh_coor=arg.rh_coor)
                ),
                LossApply(
                    VectorDispersion(nu=dispersion_nu,
                                     metric=spherical_geodesic,
                                     name='LeftHemisphereDispersion'),
                    apply=lambda arg: cmass_coor(arg.lh, arg.lh_coor, 100).t()
                ),
                LossApply(
                    VectorDispersion(nu=dispersion_nu,
                                     metric=spherical_geodesic,
                                     name='RightHemisphereDispersion'),
                    apply=lambda arg: cmass_coor(arg.rh, arg.rh_coor, 100).t()
                ),
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

            results = f'{out_root}/{atlas_name}'
            os.makedirs(results, exist_ok=True)
            plotter = fsLRAtlasParcels(atlas)
            plotter(
                cmap=modal_cmap,
                views=surf_views,
                save=f'{results}/{atlas_basename}_cmap-modal'
            )
            mask_L = dirichlet.compartments['cortex_L'][atlas.mask]
            mask_R = dirichlet.compartments['cortex_R'][atlas.mask]
            mask_L = mask_L.unsqueeze(0).cpu()
            mask_R = mask_R.unsqueeze(0).cpu()
            results_L = atlas.weight['cortex_L'].argmax(0).cpu() + 1
            results_R = atlas.weight['cortex_R'].argmax(0).cpu() + 1 + n_labels
            dataobj = dirichlet.ref.get_fdata().astype(np.int32)
            dataobj[mask_L] = results_L
            dataobj[mask_R] = results_R
            new_cifti = nb.Cifti2Image(
                dataobj=dataobj,
                header=dirichlet.ref.header,
                nifti_header=dirichlet.ref.nifti_header
            )
            new_cifti.to_filename(f'{results}/{atlas_name}.nii')


@click.command()
@click.option('-o', '--out-root', required=True, type=str)
@click.option('-a', '--atlas-template', default=None, type=str)
@click.option('-l', '--labels', 'n_labels', default=200, type=int)
@click.option('-s', '--distribution-size', default=100, type=int)
@click.option('-c', '--device', default='cuda:0', type=str)
@click.option('--res-start', default=None, type=int)
@click.option('--res-stop', default=None, type=int)
@click.option('--res-step', default=None, type=int)
@click.option('--lr', default=0.1, type=float)
@click.option('--entropy-nu', default=0.0005, type=float)
@click.option('--equilibrium-nu', default=500., type=float)
@click.option('--compactness-nu', default=2., type=float)
@click.option('--dispersion-nu', default=10., type=float)
@click.option('--tether-nu', default=1., type=float)
@click.option('--max-epoch', default=2001, type=int)
@click.option('--log-interval', default=25, type=int)
def main(out_root, atlas_template, n_labels, distribution_size, device,
         res_start, res_stop, res_step, lr, entropy_nu, equilibrium_nu,
         compactness_nu, dispersion_nu, tether_nu, max_epoch, log_interval):
    if res_start is not None:
        label_counts = list(range(res_start, res_stop, res_step))
        distribution_size = 1
    else:
        label_counts = [n_labels]
    if atlas_template is None:
        atlas_template = pkgrf(
            'hypercoil',
            'viz/resources/nullexample.nii'
        )
    create_nulls(
        out_root=out_root,
        atlas_template=atlas_template,
        label_counts=label_counts,
        distribution_size=distribution_size,
        device=device,
        lr=lr,
        entropy_nu=entropy_nu,
        equilibrium_nu=equilibrium_nu,
        compactness_nu=compactness_nu,
        dispersion_nu=dispersion_nu,
        tether_nu=tether_nu,
        max_epoch=max_epoch,
        log_interval=log_interval
    )


if __name__ == '__main__':
    main()
