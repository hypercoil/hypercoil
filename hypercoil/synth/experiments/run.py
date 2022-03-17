#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Experiment deployment
~~~~~~~~~~~~~~~~~~~~~
Framework for deploying synthetic data experiments.
"""
import os
import re
import json
import glob
import click
import hypercoil
from importlib import import_module


def run_experiment(layer, expt, index=None):
    experiments = (
        f'{os.path.dirname(hypercoil.__file__)}/synth/experiments/config'
    )
    experiment = f'{experiments}/layer-{layer}_expt-{expt}.json'

    with open(experiment) as json_file:
        experiment = json.load(json_file)

    function = (f'hypercoil.synth.experiments.'
                f'{experiment["layer"]}.{experiment["type"]}')
    mname, fname = function.rsplit('.', 1)
    module = import_module(mname)
    experiment_function = getattr(module, fname)

    if index is not None:
        index = f' {index}'
    else:
        index = ''

    results = (
        f'{os.path.dirname(hypercoil.__file__)}/results'
    )
    os.makedirs(f'{results}/layer-{layer}_expt-{expt}', exist_ok=True)
    print('\n---------------------------------------'
          '---------------------------------------\n'
          f'Experiment{index}: {experiment["name"]}\n'
          '---------------------------------------'
          '---------------------------------------')
    experiment_function(
        **experiment['parameters'],
        save=f'{results}/layer-{layer}_expt-{expt}/layer-{layer}_expt-{expt}'
    )


def run_layer_experiments(layer):
    exptdir = (
        f'{os.path.dirname(hypercoil.__file__)}/synth/experiments/config'
    )
    experiments = glob.glob(f'{exptdir}/layer-{layer}*.json')
    for i, experiment in enumerate(experiments):
        search = re.search(
            'layer-(?P<layer>[^_.]*)_expt-(?P<expt>[^_.]*)',
            experiment
        )
        run_experiment(
            layer=search.group('layer'),
            expt=search.group('expt'),
            index=(i + 1)
        )


@click.command()
@click.argument('layer')
@click.argument('experiment')
def main(layer, experiment):
    run_experiment(layer=layer, expt=experiment)


if __name__ == '__main__':
    main()
