# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Filter scaling
~~~~~~~~~~~~~~
Scaling the frequency filter to real data.
"""
import torch
import pathlib
import numpy as np
import pandas as pd
import webdataset as wds
from functools import partial
from hypercoil.functional import (
    corr
)
from hypercoil.engine import (
    Epochs,
    ModelArgument,
    UnpackingModelArgument,
    Origin,
    Conveyance,
    Conflux,
    DataPool,
    Terminal
)
from hypercoil.loss import (
    LossApply,
    LossScheme,
    MultivariateKurtosis,
    QCFC,
    SymmetricBimodalNorm,
    SmoothnessPenalty,
    NormedLoss,
    Entropy
)
from hypercoil.nn import (
    FrequencyDomainFilter,
    HybridInterpolate,
    UnaryCovarianceUW
)
from hypercoil.data.functional import window_map
from hypercoil.functional.domain import (
    AmplitudeMultiLogit,
    AmplitudeAtanh
)
from hypercoil.init.freqfilter import FreqFilterSpec
from hypercoil.functional.noise import UnstructuredDropoutSource


# Note / advisory: If somebody happens to encounter this, they should be
# advised *not* to use conveyances in the way they are used here. This approach
# has negative utility, as it delays memory release. Substantial caution must
# be taken, as there is no engine yet for optimising resource allocation. The
# use here of this experimental feature is strictly errrr experimental.


qcfc_tol = 0
window_size = 135
batch_size = 50
max_epoch = 1000
time_dim = window_size
freq_dim = time_dim // 2 + 1
data_dir = [str(path) for path in pathlib.Path(
    '/mnt/andromeda/Data/DiFuMo-fMRIPrep/upstream/ds002790/').glob(
    '*.tar')]
confounds_dir = '/mnt/blazar/Data/DiFuMo-fMRIPrep/confounds/'

loss_nu = 0.001
smoothness_nu = 0.4
symbimodal_nu = 0.05
l2_nu = 0.015
entropy_nu = 0.1
lossfn = 'qcfc'
objective = 'max'
n_bands = 2
n_networks = 50


if ((objective == 'min' and lossfn == 'mvk')
    or (objective == 'max' and lossfn == 'qcfc')):
    loss_nu *= -1

if lossfn == 'mvk':
    loss_base = [LossApply(
        MultivariateKurtosis(nu=loss_nu, l2=0.01),
        apply=lambda arg: arg.bold_filtered
    )]
    entropy_nu = 0
elif lossfn == 'qcfc':
    loss_base = [LossApply(
        QCFC(nu=loss_nu),
        apply=lambda arg: UnpackingModelArgument(FC=arg.corr_mat, QC=arg.fd)
    )]
    entropy_nu = 0
elif lossfn == 'partition':
    loss_base = [LossApply(
        SymmetricBimodalNorm(nu=loss_nu, modes=(-1, 1)),
        apply=lambda arg: arg.corr_mat
    )]

loss = [None for _ in range(n_networks)]
for i in range(n_networks):
    loss[i] = LossScheme([
        *loss_base,
        LossScheme([
            SmoothnessPenalty(nu=smoothness_nu),
            SymmetricBimodalNorm(nu=symbimodal_nu),
            NormedLoss(nu=l2_nu),
            Entropy(nu=entropy_nu)
        ], apply=lambda arg: arg.weight)
    ])


wndw = partial(
    window_map,
    keys=('bold.pyd', 'confounds', 'tmask'),
    window_length=window_size
)

def transpose(x):
    ret = {}
    for k, v in x.items():
        if isinstance(v, np.ndarray):
            try:
                ret[k] = v.swapaxes(-2, -1)
            except np.AxisError:
                ret[k] = v
        else:
            ret[k] = v
    return ret

def get_confounds(key, filter=None):
    key = '_'.join(key.split('_')[1:])
    path = pathlib.Path(f'{confounds_dir}/').glob(
        f'{key}*desc-confounds_regressors.tsv')
    path = str(list(path)[0])
    data = pd.read_csv(path, sep='\t')
    if filter is not None:
        data = pd.DataFrame(data[filter])
    return {'confounds': data.values.astype(float),
            'conf_names': tuple(data.columns)}

def append_qc(dict):
    confs = get_confounds(
        dict['__key__'],
        ['framewise_displacement', 'std_dvars']
    )
    mask = confs['confounds'][:, 0] <= 0.25
    dict.update(confs)
    dict.update({'tmask' : np.reshape(mask, (-1, 1))})
    return dict

dpl = wds.DataPipeline(
    wds.PytorchShardList(data_dir, verbose=True),
    wds.tarfile_to_samples(),
    wds.decode(lambda x, y: wds.autodecode.basichandlers('pyd', y)),
    wds.map(append_qc),
    wds.map(transpose),
    wds.map(wndw),
    #wds.map(to_float)
)
origin = Origin(
    dpl,
    lines='main',
    transmit_filters={'main': ('bold', 'tmask', 'fd')},
    num_workers=0,
    batch_size=batch_size,
    efflux=(lambda arg: ModelArgument(
        bold=arg['bold.pyd'],
        confounds=arg['confounds'],
        confounds_names=arg['conf_names'],
        t_r=arg['t_r.pyd'],
        tmask=arg['tmask'],
        __key__=arg['__key__'],
        fd=torch.mean(arg['confounds'][:, 0], -1, keepdim=True
            ).t().to(dtype=torch.float)
    ))
)


filter_spec = lambda: FreqFilterSpec(
    Wn=None, ftype='randn', btype=None, clamps=[{0: 0}],
    ampl_scale=0.01, phase_scale=0.01)
survival_prob=0.5
drop = UnstructuredDropoutSource(
    distr=torch.distributions.Bernoulli(survival_prob),
    training=True
)

fftfilter = [
    Conveyance(
        lines=[('main', 'loss'), ('main', 'main')],
        influx=(lambda arg: UnpackingModelArgument(input=arg.bold)),
        efflux=(lambda output, arg: ModelArgument.swap(
            arg, ('bold', 'bold_filtered'), output.squeeze(1))),
        transmit_filters={'loss': ('bold_filtered', 'fd')},
    ) for _ in range(n_networks)]
cov = [
    Conveyance(
        lines=[('main', 'loss'), ('main', 'main')],
        influx=(lambda arg: UnpackingModelArgument(
            input=arg.bold_filtered, mask=arg.tmask)),
        efflux=(lambda output, arg: ModelArgument.swap(
            arg, ('bold_filtered', 'corr_mat'), output)),
        transmit_filters={'loss': ('corr_mat',)},
    ) for _ in range(n_networks)]
interpol = Conveyance(
    model=HybridInterpolate(),
    lines='main',
    influx=(lambda arg: UnpackingModelArgument(
        input=arg.bold.unsqueeze(-3),
        mask=arg.tmask
    )),
    efflux=(lambda output, arg: ModelArgument.replaced(
        arg, {'bold' : output.squeeze(1)}))
)
terminal = [None for _ in range(n_networks)]
origin.connect_downstream(interpol)

for i in range(n_networks):
    if lossfn == 'partition':
        fftfilter[i].model = FrequencyDomainFilter(
            dim=freq_dim,
            filter_specs=[filter_spec() for _ in range(n_bands + 1)],
            domain=AmplitudeMultiLogit(axis=0))
    else:
        fftfilter[i].model = FrequencyDomainFilter(
            dim=freq_dim,
            filter_specs=[filter_spec()],
            domain=AmplitudeAtanh())

    cov[i].model = UnaryCovarianceUW(
        dim=time_dim,
        estimator=corr,
        dropout=drop)

    terminal[i] = Terminal(
        loss=loss[i],
        lines='loss',
        argbase=ModelArgument(
            weight=fftfilter[i].model.weight
        ),
        args=('weight', 'bold_filtered', 'corr_mat')
    )

    fftfilter[i].connect_downstream(terminal[i])
    cov[i].connect_downstream(terminal[i])
    interpol.connect_downstream(fftfilter[i])
    fftfilter[i].connect_downstream(cov[i])


"""
epochs = Epochs(max_epoch)
for epoch in epochs:
    print(f'[ Epoch {epoch} ]')
    for _ in range(20):
        origin(line='source')
        for i in range(n_networks):
            pass
        pool.reset()
"""
