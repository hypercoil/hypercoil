# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Filter scaling
~~~~~~~~~~~~~~
Scaling the frequency filter to real data.
"""
import torch
import webdataset as wds
from hypercoil.functional import (
    corr
)
from hypercoil.engine import (
    ModelArgument,
    UnpackingModelArgument,
    Origin,
    Conveyance,
    Conflux
)
from hypercoil.loss import (
    LossApply,
    MultivariateKurtosis,
    QCFC,
    SymmetricBimodalNorm
)
from hypercoil.nn import (
    FrequencyDomainFilter,
    HybridInterpolate,
    UnaryCovarianceUW
)
from hypercoil.functional.domain import (
    AmplitudeMultiLogit,
    AmplitudeAtanh
)
from hypercoil.init.freqfilter import FreqFilterSpec
from hypercoil.functional.noise import UnstructuredDropoutSource


qcfc_tol = 0
window_size = 800
time_dim = window_size
freq_dim = time_dim // 2 + 1
data_dir = None

lossfn = 'mvk'
objective = 'max'
n_bands = 2
n_networks = 20


if (objective == 'min' and lossfn == 'mvk')
    or (objective == 'max' and lossfn == 'qcfc'):
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
    keys=('images',),
    window_length=window_size
)

def get_confounds(key):
    pass

def ffd(confs):
    pass

def append_qc(dict):
    return dict

dpl = wds.DataPipeline(
    wds.PytorchShardList(data_dir, verbose=True),
    wds.tarfile_to_samples(),
    wds.decode(lambda x, y: wds.autodecode.torch_loads(y)),
    wds.map(append_qc),
    wds.map(wndw),
    wds.map(to_float)
)
origin = Origin(
    dpl,
    lines='main',
    transmit_filters={'main': None},
    num_workers=0,
    batch_size=50
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
image_efflux = (lambda out, arg: ModelArgument.replaced(
    arg, {'images': out[0]}))

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
    influx=image_influx,
    efflux=image_efflux
)
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
    loss_pool[i] = Conflux(lines='loss')
    interpol.connect_downstream(fftfilter[i])
    fftfilter[i].connect_downstream(cov[i])
    cov[i].connect_downstream(loss_pool)
