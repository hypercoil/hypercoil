#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas benchmarks
~~~~~~~~~~~~~~~~
Benchmark parcellations.
"""
import click
import torch
import pathlib
import numpy as np
import nibabel as nb
import webdataset as wds
import templateflow.api as tflow
from collections import OrderedDict
from hypercoil.eval.atlas import AtlasEval
from hypercoil.functional import (
    corr,
    residualise
)
from hypercoil.data.functional import identity
from hypercoil.data.transforms import (
    Normalise
)


def run_benchmark(
    atlas_list,
    data_dir,
    batch_size=1,
    buffer_size=1,
    atlas_dim=59412,
    verbose=False,
    device='cuda:0',
    lstsq_driver='gels',
    compute_full_matrix=False
):

    normalise = Normalise()
    maps = {
        'images' : identity,
        'confounds' : identity,
        'tmask' : identity,
        't_r' : identity,
        'task' : identity,
    }
    ds = wds.DataPipeline(
        wds.ResampledShards(data_dir),
        wds.tarfile_to_samples(),
        wds.decode(lambda x, y: wds.autodecode.torch_loads(y))
    )
    polybasis = torch.stack([
        torch.arange(1200, device=device) ** i
        for i in range(3)
    ])

    mask = torch.zeros(91282, dtype=torch.bool, device=device)
    mask[:atlas_dim] = 1
    eval = AtlasEval(
        mask=mask,
        lstsq_driver=lstsq_driver
    )

    print('[ Loading parcellations ]')
    parcellations = list(atlas_list)
    parcellations.sort()
    for i, atlas in enumerate(parcellations):
        name = '.'.join(atlas.split('.')[:-1])
        name = name.split('/')[-1]
        eval.add_voxel_assignment(
            name=atlas,
            asgt=torch.tensor(
                nb.load(atlas).get_fdata(),
                dtype=torch.long,
                device=device
            )
        )
        break

    for s, sample in enumerate(ds):
        print(f'[ Preparing next sample {s} ]')
        key = sample['__key__'].split('/')[-1]
        X = sample['images'].squeeze().to(device=device, dtype=torch.float)
        if X.shape[-1] != 1200:
            print(f'[ Incorrect time dim: {X.shape[-1]}. Skipping ]')
            continue
        gs = X.mean(-2, keepdim=True)
        regs = torch.cat((polybasis, gs), -2)
        data = residualise(X, regs, driver=lstsq_driver)
        data = normalise(data)

        eval.evaluate(
            data, id=key,
            verbose=verbose,
            compute_full_matrix=compute_full_matrix
        )

        break
    return eval


@click.command()
@click.option('-a', '--atlas', 'atlas_list', multiple=True, required=True)
@click.option('-d', '--data', 'data_dir', required=True, type=str)
@click.option('-b', '--batch-size', default=1, type=int)
@click.option('-s', '--buffer-size', default=1, type=int)
@click.option('-l', '--atlas-dim', default=59412, type=int)
@click.option('-v', '--verbose', default=False, type=bool, is_flag=True)
@click.option('-c', '--device', default='cuda:0', type=str)
@click.option('--compute-full-matrix', default=False, type=bool)
@click.option('--lstsq-driver', default='gels',
              type=click.Choice(['gels', 'gelsy', 'gelss', 'gelsd']))
def main(atlas_list, data_dir, batch_size, buffer_size,
         atlas_dim, verbose, device, compute_full_matrix, lstsq_driver):
    eval = run_benchmark(
        atlas_list=atlas_list,
        data_dir=data_dir,
        batch_size=batch_size,
        buffer_size=buffer_size,
        atlas_dim=atlas_dim,
        verbose=verbose,
        device=device,
        lstsq_driver=lstsq_driver,
        compute_full_matrix=compute_full_matrix
    )
    print(eval.results)


if __name__ == '__main__':
    main()
