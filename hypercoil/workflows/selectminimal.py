import torch
import pathlib
import pandas as pd
import webdataset as wds
from functools import partial
from hypercoil.data.transforms import Normalise
from hypercoil.data.functional import window_map
from hypercoil.data.collate import gen_collate, extend_and_bind
from hypercoil.engine import (
    Epochs,
    LossArchive,
    ModelArgument,
    UnpackingModelArgument
)
from hypercoil.functional import conditionalcorr, sym2vec
from hypercoil.functional.domainbase import Identity
from hypercoil.functional.utils import conform_mask
from hypercoil.init.freqfilter import FreqFilterSpec
from hypercoil.loss import (
    ReducingLoss, QCFC, SoftmaxEntropy, LossScheme, LossApply
)
from hypercoil.loss.batchcorr import qcfc_loss
from hypercoil.nn.interpolate import HybridInterpolate
from hypercoil.nn.freqfilter import FrequencyDomainFilter
from hypercoil.nn.select import (
    BOLDPredict, QCPredict, ResponseFunctionLinearSelector
)
from hypercoil.nn.window import WindowAmplifier


model_dim = 1
batch_size = 100
batch_size_val = 100
max_epoch = 1000
window_size = 200
n_response_functions = 5
n_ts = 400
lr = 0.005
wd = 1e-3
momentum=0.9
train_dir = '/mnt/andromeda/Data/HCP_parcels_wds/spl-{000..005}_shd-{000000..000004}.tar'
val_dir = '/mnt/andromeda/Data/HCP_parcels_wds/spl-{006..007}_shd-{000000..000004}.tar'
confounds_path = '/tmp/confounds/sub-{}_ses-{}_run-{}.pt'
confounds_path_orig = '/mnt/andromeda/Data/HCP_S1200/data/{}/MNINonLinear/Results/rfMRI_REST{}_{}/Confound_Regressors.tsv'
dtype = torch.float
save_dev = 'cpu'
device = 'cuda:0'
cols = ['framewise_displacement', 'std_dvars']


wndw = partial(
    window_map,
    keys=('images', 'confounds'),
    window_length=window_size
)


collate = partial(gen_collate, concat=extend_and_bind)
dpl = wds.DataPipeline(
    wds.ResampledShards(train_dir),
    wds.shuffle(100),
    wds.tarfile_to_samples(1),
    wds.decode(lambda x, y: wds.torch_loads(y)),
    wds.shuffle(100),
)
dl = torch.utils.data.DataLoader(
    dpl, num_workers=2, batch_size=batch_size, collate_fn=collate)

dpl_val = wds.DataPipeline(
    wds.ResampledShards(val_dir),
    wds.shuffle(100),
    wds.tarfile_to_samples(1),
    wds.decode(lambda x, y: wds.torch_loads(y)),
    wds.shuffle(batch_size_val),
)
dl_val = torch.utils.data.DataLoader(
    dpl_val, num_workers=2, batch_size=batch_size_val, collate_fn=collate)

def get_confs(keys):
    basenames = [d.split('/')[-1] for d in keys]
    subs = [b.split('_')[0].split('-')[-1] for b in basenames]
    sess = [b.split('_')[2].split('-')[-1] for b in basenames]
    runs = [b.split('_')[3].split('-')[-1] for b in basenames]
    confs = [(confounds_path.format(sub, ses, run), (sub, ses, run))
             for sub, ses, run in zip(subs, sess, runs)]
    confs_dne = [(confounds_path_orig.format(*attr), attr)
                 for path, attr in confs
                 if not pathlib.Path(path).is_file()]
    confs_dne_out = [confounds_path.format(*attr) for (_, attr) in confs_dne]
    confs_dne = [pd.read_csv(path, sep='\t') for path, attr in confs_dne]
    for c_df, path in zip(confs_dne, confs_dne_out):
        torch.save(
            torch.tensor(c_df[cols].values.T, dtype=dtype, device=save_dev),
            path)
    confs = extend_and_bind([torch.load(c[0]) for c in confs])
    return confs


basis_functions = [
    #lambda x : torch.ones_like(x),
    lambda x: x,
    #lambda x: torch.log(torch.abs(x) + 1e-8),
    #torch.exp
]


"""
qc_model = QCPredict(
    n_ts=n_ts,
    basis_functions=basis_functions,
    n_qc=len(cols)
)
ts_model = BOLDPredict(
    n_ts=n_ts,
    basis_functions=basis_functions,
    n_qc=len(cols)
)
ts_model_deep = BOLDPredict(
    n_ts=n_ts,
    basis_functions=basis_functions,
    n_intermediate_layers=5,
    n_qc=len(cols)
)
"""
interpol = HybridInterpolate()

fft_init = FreqFilterSpec(
    Wn=(0.01, 0.1),
    ftype='ideal',
    btype='bandpass',
    fs=0.72
)
bpfft = FrequencyDomainFilter(
    filter_specs=[fft_init],
    time_dim=window_size,
    domain=Identity(),
    device=device
)
bpfft.preweight.requires_grad = False

selector = ResponseFunctionLinearSelector(
    model_dim=model_dim,
    n_columns=57,
    leak=0.01,
    n_response_functions=n_response_functions,
    softmax=False,
    device=device
)


qcfc = QCFC(nu=1, name='QC-FC')
entropy = SoftmaxEntropy(nu=10, name='Entropy')

loss_train = LossScheme([
    #LossApply(entropy, apply=(lambda arg: arg.weight)),
    LossApply(
        qcfc,
        apply=(lambda arg: UnpackingModelArgument(FC=arg.fc, QC=arg.qc)))
])


qcfc_val = LossScheme([LossApply(
    QCFC(nu=1, name='QC-FC_Val'),
    apply=(lambda arg: UnpackingModelArgument(FC=arg.fc, QC=arg.qc))
)])


epochs = Epochs(max_epoch)
loss_tape = LossArchive(epochs=epochs)
loss_train.register_sentry(loss_tape)
qcfc_val.register_sentry(loss_tape)


#opt = torch.optim.Adam(lr=lr, params=selector.parameters())

opt = torch.optim.SGD(
    lr=lr, momentum=momentum, weight_decay=wd,
    params=selector.parameters())

n = Normalise()


wndw = WindowAmplifier(window_size=window_size, augmentation_factor=5)


def model(bold, confs, mask):
    assert mask.dtype == torch.bool
    ms = (mask.sum(-1).squeeze())
    #if torch.any(ms == 0):
    #    print(ms)
    bold = interpol(bold.unsqueeze(1), mask).squeeze()
    bold = bpfft(bold).squeeze()
    confs = interpol(confs.unsqueeze(1), mask).squeeze()
    confs = bpfft(confs).squeeze()
    confmodel = selector(confs)
    return bold, confmodel, confs


def ingest_data(dl, wndw):
    data = next(dl)
    qc = get_confs(data['__key__'])[:, 0]
    bold = n(data['images'])
    confounds = n(data['confounds'])
    nanmask = (torch.isnan(bold).sum(-2) == 0)
    bold[torch.isnan(bold)] = 0.001 * torch.randn_like(bold[torch.isnan(bold)])
    confounds[torch.isnan(confounds)] = 0.001 * torch.randn_like(
        confounds[torch.isnan(confounds)])
    qc[torch.isnan(qc)] = 0.001 * torch.randn_like(qc[torch.isnan(qc)])

    [bold, confounds, qc, nanmask] = wndw((bold, confounds, qc, nanmask))
    return (
        bold.to(device=device),
        confounds.to(device=device),
        qc.to(device=device),
        nanmask.to(device=device)
    )


def model_forward(model, bold, confounds, mask, qc, lossfn):
    bold, confmodel, confs = model(bold, confounds, mask)
    cor = conditionalcorr(bold, confmodel)
    fc = sym2vec(cor)
    arg = ModelArgument(
        fc=fc,
        qc=qc.nanmean(1).unsqueeze(0),
        weight=selector.weight['lin']
    )
    loss = lossfn(arg, verbose=True)
    return loss, confmodel, bold, confs


val_iter = iter(dl_val)
for epoch in epochs:
    print(f'[ Epoch {epoch} ]')
    if epoch > 2:
        print(loss_tape.archive['QC-FC'][-3:],
              loss_tape.archive['QC-FC_Val'][-3:])
    dl_iter = iter(dl)
    selector.train()
    #qcfc.tol = 0.1 * torch.rand(1).item()
    for _ in range(10):
        bold, confounds, qc, nanmask = ingest_data(dl_iter, wndw)
        loss, _, _, _ = model_forward(
            model, bold, confounds,
            nanmask.unsqueeze(1),
            qc, loss_train)

        if torch.any(torch.isnan(loss)):
            assert 0

        loss.backward()
        opt.step()
        opt.zero_grad()


    refqcfc = 0
    with torch.no_grad():
        selector.eval()
        for _ in range(10):
            bold, confounds, qc, nanmask = ingest_data(val_iter, wndw)
            loss, confmodel, bold, confs = model_forward(
                model, bold, confounds,
                nanmask.unsqueeze(1),
                qc, qcfc_val)
            reffc = sym2vec(conditionalcorr(bold, confs[:, 8].unsqueeze(1)))
            refqcfc += qcfc_loss(QC=qc.nanmean(1).unsqueeze(0), FC=reffc).mean().item()
        print(f'  (Reference : {refqcfc / 10})')
