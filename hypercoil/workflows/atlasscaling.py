# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Spatiotemporal atlas
~~~~~~~~~~~~~~~~~~~~
Atlas combining spatial regularisers with temporal loss.
"""
import torch
import templateflow.api as tflow
from functools import partial
from pkg_resources import resource_filename as pkgrf
from hypercoil.functional import (
    spherical_geodesic,
    cmass_coor,
    residualise
)
from hypercoil.engine import (
    Epochs,
    MultiplierSigmoidSchedule,
    LossArchive
)
from hypercoil.loss import (
    LossScheme,
    LossApply,
    LossArgument,
    UnpackingLossArgument,
    Compactness,
    Entropy,
    Equilibrium,
    HemisphericTether,
    VectorDispersion,
    SecondMoment,
    LogDetCorr
)
from hypercoil.nn import AtlasLinear
from hypercoil.init.atlas import DirichletInitSurfaceAtlas
from hypercoil.engine.terminal import ReactiveTerminal
from hypercoil.viz.surf import (
    fsLRAtlasParcels,
    fsLRAtlasMaps
)
from hypercoil.data.functional import window_map, identity
from hypercoil.data.transforms import (
    Compose,
    PolynomialDetrend,
    Normalise,
    NaNFill
)
from hypercoil.data.wds import torch_wds


max_epoch = 1001
log_interval = 25
data_interval = 5
lr = 0.05
batch_size = 3
buffer_size = 10
device = 'cuda:0'
atlas_path = '/mnt/pulsar/Downloads/atlases/glasser.nii'
data_dir = '/mnt/andromeda/Data/HCP_wds/spl-{000..007}_shd-{000000..000004}.tar'
modal_cmap = pkgrf(
    'hypercoil',
    'viz/resources/cmap_modal.nii'
)
network_cmap = pkgrf(
    'hypercoil',
    'viz/resources/cmap_network.nii'
)


epochs = Epochs(max_epoch)


wndw = partial(
    window_map,
    keys=('images', 'confounds', 'tmask'),
    window_length=500
)
normalise = Normalise()
maps = {
    'images' : identity,
    'confounds' : identity,
    'tmask' : identity,
    't_r' : identity,
    'task' : identity,
}
ds = torch_wds(
    data_dir,
    keys=maps,
    batch_size=batch_size,
    shuffle=buffer_size,
    map=wndw
)
polybasis = torch.stack([
    torch.arange(500) ** i
    for i in range(3)
])


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
    device=device
)
atlas = AtlasLinear(dirichlet, mask_input=False,
                    domain=dirichlet.init['_all'].domain,
                    dtype=torch.float, device=device)

lh_coor = atlas.coors[dirichlet.compartments['cortex_L'][atlas.mask]].t()
rh_coor = atlas.coors[dirichlet.compartments['cortex_R'][atlas.mask]].t()

lh_mask = dirichlet.compartments['cortex_L'][atlas.mask].unsqueeze(0)
rh_mask = dirichlet.compartments['cortex_R'][atlas.mask].unsqueeze(0)

dispersion_nu = MultiplierSigmoidSchedule(
    epochs=epochs, base=10,
    transitions={(80, 120): 2, (200, 300): 0.5, (400, 500): 2, (600, 700): 0.5}
)
determinant_nu = MultiplierSigmoidSchedule(
    epochs=epochs, base=.005,
    transitions={(80, 120): .001, (200, 400): .0001}
)
equilibrium_nu = MultiplierSigmoidSchedule(
    epochs=epochs, base=1e7,
    transitions={(200, 300): 25000, (500, 700): 1e7}
)
entropy_nu = MultiplierSigmoidSchedule(
    epochs=epochs, base=0.1,
    transitions={(100, 200): 0.0001, (500, 800): 0.1}
)
compactness_nu = MultiplierSigmoidSchedule(
    epochs=epochs, base=2,
    transitions={(80, 120): 5, (400, 800): 20}
)

opt = torch.optim.Adam(params=atlas.parameters(), lr=lr)
loss = LossScheme((
    LossScheme((
        Entropy(nu=entropy_nu, name='LeftHemisphereEntropy'),
        Equilibrium(nu=equilibrium_nu, name='LeftHemisphereEquilibrium'),
        Compactness(nu=compactness_nu, radius=100, coor=lh_coor,
                    name='LeftHemisphereCompactness')
    ), apply=lambda arg: arg.lh),
    LossScheme((
        Entropy(nu=entropy_nu, name='RightHemisphereEntropy'),
        Equilibrium(nu=equilibrium_nu, name='RightHemisphereEquilibrium'),
        Compactness(nu=compactness_nu, radius=100, coor=rh_coor,
                    name='RightHemisphereCompactness')
    ), apply=lambda arg: arg.rh),
    LossApply(
        HemisphericTether(nu=0.2, name='InterHemisphericTether'),
        apply=lambda arg: UnpackingLossArgument(
            lh=arg.lh,
            rh=arg.rh,
            lh_coor=arg.lh_coor,
            rh_coor=arg.rh_coor)
    ),
    LossApply(
        VectorDispersion(nu=dispersion_nu, metric=spherical_geodesic,
                         name='LeftHemisphereDispersion'),
        apply=lambda arg: cmass_coor(arg.lh, arg.lh_coor, 100).t()),
    LossApply(
        VectorDispersion(nu=dispersion_nu, metric=spherical_geodesic,
                         name='RightHemisphereDispersion'),
        apply=lambda arg: cmass_coor(arg.rh, arg.rh_coor, 100).t()),
    LossApply(
        LogDetCorr(nu=determinant_nu, psi=0.01, xi=0.01),
        apply=lambda arg: arg.ts)
))
terminal_L = ReactiveTerminal(
    loss=SecondMoment(nu=5, standardise=False,
                      skip_normalise=True,
                      name='LeftHemisphereSecondMoment'),
    slice_target='data',
    slice_axis=-1,
    max_slice=10,
    pretransforms={'weight': atlas.domain.image}
)
terminal_R = ReactiveTerminal(
    loss=SecondMoment(nu=5, standardise=False,
                      skip_normalise=True,
                      name='RightHemisphereSecondMoment'),
    slice_target='data',
    slice_axis=-1,
    max_slice=10,
    pretransforms={'weight': atlas.domain.image}
)
loss_tape = LossArchive(epochs=epochs)
loss.register_sentry(loss_tape)


for epoch in epochs:
    if epoch % data_interval == 0:
        for sample in ds:
            X = sample[0][:, :atlas.mask.sum(), :].float()
            gs = X.mean(-2)
            regs = torch.cat((
                polybasis.tile(X.shape[0], 1, 1),
                gs.unsqueeze(-2)
            ), -2).to('cuda')
            X = X.to('cuda')
            data = residualise(X, regs, driver='gels')
            data = normalise(data)
            break

    if epoch == 500:
        opt.param_groups[0]['lr'] /= 2.5

    # sometimes, torch is really stupid
    shape_L = (data.shape[0], lh_mask.sum(), -1)
    shape_R = (data.shape[0], rh_mask.sum(), -1)
    mask_L = lh_mask.tile(data.shape[0], 1)
    mask_R = rh_mask.tile(data.shape[0], 1)
    data_L = data[mask_L].view(*shape_L)
    data_R = data[mask_R].view(*shape_R)
    ts = atlas(data)
    arg = LossArgument(
        ts=ts,
        lh=atlas.weight['cortex_L'],
        rh=atlas.weight['cortex_R'],
        lh_coor=lh_coor, rh_coor=rh_coor
    )
    arg_L = {'data': data_L, 'weight': atlas.preweight['cortex_L']}
    arg_R = {'data': data_R, 'weight': atlas.preweight['cortex_R']}
    out = terminal_L(arg=arg_L)
    print(f'- {terminal_L.loss} : {out}')
    #if (epoch % log_interval == 0): print(f'{terminal_L.loss} : {out}')
    out = terminal_R(arg=arg_R)
    print(f'- {terminal_R.loss} : {out}')
    #if (epoch % log_interval == 0): print(f'{terminal_R.loss} : {out}')
    #out = loss(arg, verbose=(epoch % log_interval == 0))
    out = loss(arg, verbose=True)
    print(f'[ Epoch {epoch} | Total loss {out} ]\n')
    out.backward()
    opt.step()
    opt.zero_grad()

    if epoch % log_interval == 0:
        views = ('medial', 'lateral')
        results = '/tmp/parc-spatiotemporal'
        plotter = fsLRAtlasParcels(atlas)
        plotter(
            cmap=modal_cmap,
            views=views,
            save=f'{results}/epoch-{epoch:08}'
        )

#"""
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
#"""
#plotter = fsLRAtlasMaps(atlas)
#plotter(save=f'{results}/parcellation_maps')
