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
    ModelArgument,
    UnpackingModelArgument,
    Origin,
    Conveyance,
    Conflux,
    DataPool
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


qcfc_tol = 0
window_size = 135
batch_size = 50
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
lossfn = 'mvk'
objective = 'max'
n_bands = 2
n_networks = 50


if ((objective == 'min' and lossfn == 'mvk')
    or (objective == 'max' and lossfn == 'qcfc')):
    loss_nu *= -1

if lossfn == 'mvk':
    loss = [LossApply(
        MultivariateKurtosis(nu=loss_nu, l2=0.01),
        apply=lambda arg: arg.ts_filtered
    )]
    entropy_nu = 0
elif lossfn == 'qcfc':
    loss = [LossApply(
        QCFC(nu=loss_nu),
        apply=lambda arg: arg.corr_mat
    )]
    entropy_nu = 0
elif lossfn == 'partition':
    loss = [LossApply(
        SymmetricBimodalNorm(nu=loss_nu, modes=(-1, 1)),
        apply=lambda arg: arg.corr_mat
    )]
loss = LossScheme([
    *loss,
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
    return {'confounds': data.values, 'conf_names': tuple(data.columns)}

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
    #transmit_filters={'main': None},
    num_workers=8,
    batch_size=batch_size,
    efflux=(lambda arg: ModelArgument(
        images=arg['bold.pyd'],
        confounds=arg['confounds'],
        confounds_names=arg['conf_names'],
        t_r=arg['t_r.pyd'],
        tmask=arg['tmask'],
        __key__=arg['__key__']
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

image_influx = (lambda arg: UnpackingModelArgument(input=arg.images))
image_efflux = (lambda output, arg: ModelArgument.replaced(
    arg, {'images': output}))

fftfilter = [
    Conveyance(
        lines='main',
        influx=image_influx,
        efflux=image_efflux
    ) for _ in range(n_networks)]
cov = [
    Conveyance(
        lines='main',
        influx=image_influx,
        efflux=image_efflux
    ) for _ in range(n_networks)]
interpol = Conveyance(
    model=HybridInterpolate(),
    lines='main',
    influx=(lambda arg: UnpackingModelArgument(
        input=arg.images.unsqueeze(-3),
        mask=arg.tmask
    )),
    efflux=(lambda output, arg: ModelArgument.replaced(
        arg, {'images' : output.squeeze(1)}))
)
pool = DataPool(lines='main')
origin.connect_downstream(interpol)

for i in range(2) : #n_networks):
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
    #loss_pool[i] = Conflux(lines='loss')
    interpol.connect_downstream(fftfilter[i])
    #fftfilter[i].connect_downstream(cov[i])
    #cov[i].connect_downstream(loss_pool)

    fftfilter[i].connect_downstream(pool)
