#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Denoising benchmarks
~~~~~~~~~~~~~~~~~~~~
Simple denoising benchmark workflow. Full of hacks and hard codes, and not in
the least bit ready for general use.
"""
import click
import json
import torch
import pathlib
import webdataset as wds
from functools import partial
from hypercoil.data.collate import (
    gen_collate,
    extend_and_bind
)
from hypercoil.data.transforms import Normalise
from hypercoil.eval.denoise import DenoisingEval
from hypercoil.formula.fc import FCShorthand
from hypercoil.init import FreqFilterSpec
from hypercoil.functional import conditionalcorr
from hypercoil.functional.domainbase import Identity
from hypercoil.nn import (
    FrequencyDomainFilter,
    HybridInterpolate,
    WindowAmplifier,
    ResponseFunctionLinearSelector
)


confounds_path = '/tmp/confounds/sub-{}_ses-{}_run-{}.pt'
confounds_path_orig = '/mnt/andromeda/Data/HCP_S1200/data/{}/MNINonLinear/Results/rfMRI_REST{}_{}/Confound_Regressors.tsv'


#TODO: implement a way of getting these from expansions in the formula module
# because this is terrible and hideous
PARAM1 = [
    'global_signal'
]
PARAM9 = [
    'trans_x',
    'trans_y',
    'trans_z',
    'rot_x',
    'rot_y',
    'rot_z',
    'white_matter',
    'csf',
    'global_signal',
]
PARAM36 = [
    'trans_x',
    'trans_y',
    'trans_z',
    'rot_x',
    'rot_y',
    'rot_z',
    'white_matter',
    'csf',
    'global_signal',
    'trans_x_derivative1',
    'trans_y_derivative1',
    'trans_z_derivative1',
    'rot_x_derivative1',
    'rot_y_derivative1',
    'rot_z_derivative1',
    'white_matter_derivative1',
    'csf_derivative1',
    'global_signal_derivative1',
    'trans_x_power2',
    'trans_y_power2',
    'trans_z_power2',
    'rot_x_power2',
    'rot_y_power2',
    'rot_z_power2',
    'white_matter_power2',
    'csf_power2',
    'global_signal_power2',
    'trans_x_derivative1_power2',
    'trans_y_derivative1_power2',
    'trans_z_derivative1_power2',
    'rot_x_derivative1_power2',
    'rot_y_derivative1_power2',
    'rot_z_derivative1_power2',
    'white_matter_derivative1_power2',
    'csf_derivative1_power2',
    'global_signal_derivative1_power2',
]
ACOMPCOR5 = [
    'a_comp_cor_wm_00',
    'a_comp_cor_wm_01',
    'a_comp_cor_wm_02',
    'a_comp_cor_wm_03',
    'a_comp_cor_wm_04',
    'a_comp_cor_csf_00',
    'a_comp_cor_csf_01',
    'a_comp_cor_csf_02',
    'a_comp_cor_csf_03',
    'a_comp_cor_csf_04',
]
ACOMPCOR5GS = [
    'a_comp_cor_wm_00',
    'a_comp_cor_wm_01',
    'a_comp_cor_wm_02',
    'a_comp_cor_wm_03',
    'a_comp_cor_wm_04',
    'a_comp_cor_csf_00',
    'a_comp_cor_csf_01',
    'a_comp_cor_csf_02',
    'a_comp_cor_csf_03',
    'a_comp_cor_csf_04',
    'global_signal',
]
ACOMPCOR10 = [
    'a_comp_cor_wm_00',
    'a_comp_cor_wm_01',
    'a_comp_cor_wm_02',
    'a_comp_cor_wm_03',
    'a_comp_cor_wm_04',
    'a_comp_cor_wm_05',
    'a_comp_cor_wm_06',
    'a_comp_cor_wm_07',
    'a_comp_cor_wm_08',
    'a_comp_cor_wm_09',
    'a_comp_cor_csf_00',
    'a_comp_cor_csf_01',
    'a_comp_cor_csf_02',
    'a_comp_cor_csf_03',
    'a_comp_cor_csf_04',
    'a_comp_cor_csf_05',
    'a_comp_cor_csf_06',
    'a_comp_cor_csf_07',
    'a_comp_cor_csf_08',
    'a_comp_cor_csf_09',
]
ACOMPCOR10GS = [
    'a_comp_cor_wm_00',
    'a_comp_cor_wm_01',
    'a_comp_cor_wm_02',
    'a_comp_cor_wm_03',
    'a_comp_cor_wm_04',
    'a_comp_cor_wm_05',
    'a_comp_cor_wm_06',
    'a_comp_cor_wm_07',
    'a_comp_cor_wm_08',
    'a_comp_cor_wm_09',
    'a_comp_cor_csf_00',
    'a_comp_cor_csf_01',
    'a_comp_cor_csf_02',
    'a_comp_cor_csf_03',
    'a_comp_cor_csf_04',
    'a_comp_cor_csf_05',
    'a_comp_cor_csf_06',
    'a_comp_cor_csf_07',
    'a_comp_cor_csf_08',
    'a_comp_cor_csf_09',
    'global_signal',
]
BADMODEL = [
    'rot_y_power2'
]
BASELINE_MODELS = {
    'PARAM1' : PARAM1,
    'PARAM9' : PARAM9,
    'PARAM36' : PARAM36,
    'ACOMPCOR5' : ACOMPCOR5,
    'ACOMPCOR5GS' : ACOMPCOR5GS,
    'ACOMPCOR10' : ACOMPCOR10,
    'ACOMPCOR10GS' : ACOMPCOR10GS,
    'BADMODEL' : BADMODEL
}


def _read_reg_names(txt):
    with open(txt) as file:
        t = file.readlines()
    return [l.rstrip() for l in t]


def _select_model(spec, confounds, names, device, gs_augment=True):
    if spec not in BASELINE_MODELS:
        ##TODO: we'll want this to be more flexible instead of hard coding
        # the current rudimentary model.
        state_dict = torch.load(spec)
        model_dim, _ = state_dict['weight.lin'].shape
        n_response_functions, _, _, response_function_len = state_dict['weight.rf'].shape
        ##TODO: hard codes here are gonna be a major problem someday if we
        # don't change this
        selector = ResponseFunctionLinearSelector(
            model_dim=model_dim,
            n_columns=len(names),
            leak=0.01,
            n_response_functions=n_response_functions,
            softmax=False,
            device=device
        )
        selector.load_state_dict(state_dict)
        if gs_augment:
            gs_idx = [i for i, n in enumerate(names) if n == 'global_signal']
            gsselect = torch.zeros(len(names), device=device)
            gsselect[gs_idx] = 1
            #print(gsselect)
            return torch.cat(
                ((gsselect @ confounds).unsqueeze(1), selector(confounds)),
            1)
        return selector(confounds)
    else:
        regs = BASELINE_MODELS[spec]
        idx = [i for i, name in enumerate(names) if name in regs]
        idx = torch.tensor(idx, dtype=torch.long, device=device)
        selector = torch.eye(len(names), device=device)[idx]
        return selector @ confounds


def get_confs(keys):
    ##TODO: this only exists here because of bad decisions regarding data
    # management that are difficult to fix under the current time frame
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


def _fwd_init(fs, window_size, device):
    interpol = HybridInterpolate()
    fft_init = FreqFilterSpec(
        Wn=(0.01, 0.1),
        ftype='ideal',
        btype='bandpass',
        fs=fs
    )
    bpfft = FrequencyDomainFilter(
        filter_specs=[fft_init],
        time_dim=window_size,
        domain=Identity(),
        device=device
    )
    return interpol, bpfft


def _get_connectomes(bold, confounds, mask, spec, names, fs,
                     window_size, gs_augment, device):
    with torch.no_grad():
        interpol, bpfft = _fwd_init(fs, window_size, device)
        bold = interpol(bold.unsqueeze(1), mask).squeeze()
        bold = bpfft(bold).squeeze()
        confounds = interpol(confounds.unsqueeze(1), mask).squeeze()
        confounds = bpfft(confounds).squeeze()
        confmodel = _select_model(spec, confounds, names, device, gs_augment)
        print(confmodel.shape, confmodel[0])
        """
        from hypercoil.functional import sym2vec
        zz0, zz1, zz2 = (
            conditionalcorr(bold, confmodel),
            conditionalcorr(bold, confmodel[:, 0]),
            conditionalcorr(bold, confmodel[:, 1])
        )
        print(zz0.mean(0), zz1.mean(0), zz2.mean(0))
        print(zz0.mean(0) - zz1.mean(0))
        print(sym2vec(zz0.mean(0)).max(), sym2vec(zz1.mean(0)).max())
        print(sym2vec(conditionalcorr(bold, confmodel)).mean(0).max())
        assert 0
        """
        return conditionalcorr(bold, confmodel), confmodel, confounds


def ingest_data(data, wndw, device):
    ##TODO: so much redundancy between this and `selectminimal`, and so much
    # of it made necessary by early bad decisions regarding data management
    #data = next(dl)
    qc = get_confs(data['__key__'])[:, 0]
    n = Normalise()
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


def prepare_data(data, model_spec, names, window_size, device, fs, gs_augment):
    collate = partial(gen_collate, concat=extend_and_bind)
    dpl = wds.DataPipeline(
        wds.SimpleShardList(data),
        wds.tarfile_to_samples(),
        wds.decode(lambda x, y: wds.torch_loads(y)),
    )
    ##TODO: the hard codes . . .
    dl = torch.utils.data.DataLoader(
        dpl, num_workers=1, batch_size=100, collate_fn=collate)
    window = WindowAmplifier(window_size=window_size, augmentation_factor=1)

    qc, fc, confmodel, confounds = [], [], [], []
    for data in dl:
        bold, confs, qc_i, mask = ingest_data(
            data=data,
            wndw=window,
            device=device
        )

        fc_i, confmodel_i, confounds_i = _get_connectomes(
            bold=bold,
            confounds=confs,
            mask=mask,
            spec=model_spec,
            names=names,
            fs=fs,
            window_size=window_size,
            gs_augment=gs_augment,
            device=device
        )
        qc += [qc_i]
        fc += [fc_i]
        confmodel += [confmodel_i]
        confounds += [confounds_i]
    qc = torch.cat(qc, 0)
    fc = torch.cat(fc, 0)
    confmodel = torch.cat(confmodel, 0)
    confounds = torch.cat(confounds, 0)
    #assert 0
    return qc, fc, confmodel, confounds


def run_benchmark(data, out, model_spec, names, window_size, device,
                  fs, evaluate_qcfc, evaluate_varexp, plot_result,
                  atlas=None, significance=None, gs_augment=False):
    eval = DenoisingEval(
        confound_names=names,
        evaluate_qcfc=evaluate_qcfc,
        evaluate_varexp=evaluate_varexp,
        plot_result=plot_result,
        atlas=atlas,
        significance=significance
    )
    qc, fc, confmodel, confounds = prepare_data(
        data=data,
        model_spec=model_spec,
        names=names,
        window_size=window_size,
        device=device,
        fs=fs,
        gs_augment=gs_augment
    )
    eval.evaluate(
        connectomes=fc,
        model=confmodel,
        confounds=confounds,
        qc=qc.nanmean(1).unsqueeze(0),
        save=out
    )
    return eval


@click.command()
@click.option('-o', '--out', required=True, type=str)
@click.option('-n', '--namefile', required=True, type=str)
@click.option('-d', '--data', required=True, type=str)
@click.option('-m', '--model-spec', required=True, type=str)
@click.option('-a', '--atlas', default=None, type=str)
@click.option('-s', '--significance', default=None, type=float)
@click.option('-c', '--device', default='cuda:0', type=str)
@click.option('-w', '--window_size', default=None, type=int)
@click.option('--fs', required=True, type=float)
@click.option('--augment-gsr', default=False, type=bool, is_flag=True)
@click.option('--skip-qcfc', default=False, type=bool, is_flag=True)
@click.option('--skip-varexp', default=False, type=bool, is_flag=True)
@click.option('--skip-plot', default=False, type=bool, is_flag=True)
def main(out, namefile, data, model_spec, atlas, significance, device,
         window_size, fs, skip_qcfc, skip_varexp, skip_plot, augment_gsr):
    if not skip_plot:
        if not atlas or not significance:
            raise ValueError(
                'QC-FC plot requires specifying an atlas and a significance '
                'threshold. Either specify both or call with `--skip-plot` '
                'to skip the QC-FC plot.')

    names = _read_reg_names(namefile)

    eval = run_benchmark(
        data=data,
        out=out,
        model_spec=model_spec,
        names=names,
        window_size=window_size,
        device=device,
        fs=fs,
        evaluate_qcfc=(not skip_qcfc),
        evaluate_varexp=(not skip_varexp),
        plot_result=(not skip_plot),
        atlas=atlas,
        significance=significance,
        gs_augment=augment_gsr
    )

    with open(f'{out}.json', 'w') as fp:
        json.dump(
            eval.results, fp,
            sort_keys=True,
            indent=4,
            separators=(',', ': ')
        )


if __name__ == '__main__':
    main()
