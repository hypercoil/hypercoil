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
from torch.nn import ModuleList
from hypercoil.functional import (
    corr, sym2vec, complex_decompose
)
from hypercoil.engine import (
    Epochs,
    ModelArgument,
    UnpackingModelArgument,
    Replaced,
    Swap,
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
    Entropy,
    VectorDispersion
)
from hypercoil.nn import (
    FrequencyDomainFilter,
    HybridInterpolate,
    UnaryCovarianceUW
)
from hypercoil.data.functional import window_map
from hypercoil.init.domain import (
    AmplitudeMultiLogit,
    AmplitudeAtanh
)
from hypercoil.init.freqfilter import FreqFilterSpec
from hypercoil.functional.noise import UnstructuredDropoutSource
from hypercoil.viz.filter import StreamPlot


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

lr = 0.02
symb_nu = 0.001
disp_nu = 1e-4
mvk_nu  = 0.01
qcfc_nu = 10
smoothness_nu = 5
symbimodal_nu = 1
l2_nu = 0.015
entropy_nu = 0.1
lossfn = 'partition'
objective = 'min'
n_bands = 3
n_networks = 8

l2_nu = torch.logspace(0.5, 1, n_networks).tolist()


def amplitude(x):
    a, _ = complex_decompose(x)
    return a


if ((objective == 'min' and lossfn == 'mvk')
    or (objective == 'max' and lossfn == 'qcfc')):
    qcfc_nu *= -1
    mvk_nu *= -1

if lossfn == 'mvk':
    loss_base = [LossApply(
        MultivariateKurtosis(nu=mvk_nu, l2=0.01),
        apply=lambda arg: arg.bold_filtered
    )]
    entropy_nu = 0
elif lossfn == 'qcfc':
    loss_base = [LossApply(
        QCFC(nu=qcfc_nu),
        apply=lambda arg: UnpackingModelArgument(FC=arg.corr_mat, QC=arg.fd)
    )]
    entropy_nu = 0
elif lossfn == 'partition':
    loss_base = [LossScheme([
        VectorDispersion(nu=disp_nu),
        SymmetricBimodalNorm(nu=symb_nu, modes=(-1, 1))
    ], apply=lambda arg: arg.corr_mat),
    LossApply(
        QCFC(nu=qcfc_nu),
        apply=lambda arg: UnpackingModelArgument(FC=arg.corr_mat, QC=arg.fd)
    )]

loss = [None for _ in range(n_networks)]
for i in range(n_networks):
    loss[i] = LossScheme([
        *loss_base,
        LossScheme([
            SmoothnessPenalty(nu=smoothness_nu),
            SymmetricBimodalNorm(nu=symbimodal_nu),
            NormedLoss(nu=l2_nu[i]),
            Entropy(nu=entropy_nu)
        ], apply=lambda arg: amplitude(arg.weight))
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
        fd=torch.nanmean(arg['confounds'][:, 0], -1, keepdim=True
            ).t().to(dtype=torch.float)
    ))
)


filter_spec = lambda: FreqFilterSpec(
    Wn=None, ftype='randn', btype=None, # clamps=[{0: 0}],
    ampl_scale=0.01, phase_scale=0.01)
survival_prob=0.5
drop = UnstructuredDropoutSource(
    distr=torch.distributions.Bernoulli(survival_prob),
    training=True
)

class FilterStreams(torch.nn.Module):
    def __init__(self, n_networks, origin, loss, lossfn='partition'):
        super().__init__()
        self.origin = origin
        self.n_networks = n_networks
        self.interpol = Conveyance(
            model=HybridInterpolate(),
            lines='main',
            influx=(lambda arg: UnpackingModelArgument(
                input=arg.bold.unsqueeze(-3),
                mask=arg.tmask
            )),
            efflux=(lambda output, arg: ModelArgument.replaced(
                arg, {'bold' : output.squeeze(1)}))
        )
        if lossfn == 'partition':
            fft_efflux = (lambda output, arg: ModelArgument.swap(
                arg, ('bold', 'bold_filtered'), output[:, :(n_bands)]))
        else:
            fft_efflux = (lambda output, arg: ModelArgument.swap(
                arg, ('bold', 'bold_filtered'), output))
        self.fftfilter = ModuleList([
            Conveyance(
                lines=[('main', 'loss'), ('main', 'main')],
                influx=(lambda arg: UnpackingModelArgument(input=arg.bold)),
                efflux=fft_efflux,
                transmit_filters={'loss': ('bold_filtered', 'fd')},
            ) for _ in range(n_networks)
        ])
        self.cov = ModuleList([
            Conveyance(
                lines=[('main', 'loss'), ('main', 'main')],
                influx=(lambda arg: UnpackingModelArgument(
                    input=arg.bold_filtered, mask=arg.tmask.unsqueeze(1))),
                efflux=Swap(('bold_filtered', 'corr_mat'),
                            val_map=lambda arg, output: output),
                transmit_filters={'loss': ('corr_mat',)},
            ) for _ in range(n_networks)
        ])
        self.terminal = ModuleList([None for _ in range(n_networks)])
        def terminal_argfn(filter, n_bands=None):
            if n_bands is not None:
                return filter.weight[:(n_bands)]
            return filter.weight

        self.origin.connect_downstream(self.interpol)

        for i in range(n_networks):
            if lossfn == 'partition':
                self.fftfilter[i].model = FrequencyDomainFilter(
                    dim=freq_dim,
                    filter_specs=[filter_spec() for _ in range(n_bands + 1)],
                    domain=AmplitudeMultiLogit(axis=0))
                terminal_arg = partial(
                    terminal_argfn, n_bands=n_bands,
                    filter=self.fftfilter[i].model)
            else:
                self.fftfilter[i].model = FrequencyDomainFilter(
                    dim=freq_dim,
                    filter_specs=[filter_spec()],
                    domain=AmplitudeAtanh())
                terminal_arg = partial(terminal_argfn,
                    filter=self.fftfilter[i].model)

            self.cov[i].model = UnaryCovarianceUW(
                dim=time_dim,
                estimator=corr,
                dropout=drop
            )

            self.terminal[i] = Terminal(
                loss=loss[i],
                lines='loss',
                arg_factory={'weight': terminal_arg},
                args=('weight', 'bold_filtered', 'corr_mat'),
                influx=Replaced(
                    'corr_mat',
                    replace_map=(lambda arg: sym2vec(arg.corr_mat)))
            )

            self.fftfilter[i].connect_downstream(self.terminal[i])
            self.cov[i].connect_downstream(self.terminal[i])
            self.interpol.connect_downstream(self.fftfilter[i])
            self.fftfilter[i].connect_downstream(self.cov[i])

    def forward(self):
        self.origin(line='source')

stream = FilterStreams(
    n_networks=n_networks,
    origin=origin,
    loss=loss,
    lossfn=lossfn
)


opt = torch.optim.Adam(lr=lr, params=stream.fftfilter.parameters())

epochs = Epochs(max_epoch)
for epoch in epochs:
    print(f'\n\n[ Epoch {epoch} ]')
    for _ in range(1): #20):
        stream()
        opt.step()
        opt.zero_grad()
    plotter = StreamPlot(stream=stream, objective=lossfn)
    plotter(f'/tmp/desc-{lossfn}-epoch{epoch:04}_filter.png')
