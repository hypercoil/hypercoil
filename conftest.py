# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
PyTest Configuration
~~~~~~~~~~~~~~~~~~~~
Settings configuring pytest for the differentiable programming library.
"""
import torch
import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--sims', action='store_true', default=False, help='Run simulations'
    )
    parser.addoption(
        '--ci', action='store_true', default=False,
        help='Skip tests unsupported on CI server'
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers',
        'sim: mark test as learning simulation'
    )
    config.addinivalue_line(
        'markers',
        'cuda: mark test as CUDA-only'
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption('--sims'):
        # --sims given in cli: do not skip simulations
        pass
    else:
        skip_sims = pytest.mark.skip(
            reason='--sims option must be set to run simulations')
        for item in items:
            if 'sim' in item.keywords:
                item.add_marker(skip_sims)

    if config.getoption('--ci'):
        # --ci given in cli: skip unsupported tests
        skip_ci_unsupported = pytest.mark.skip(
            reason='--ci option indicates not to run tests unsupported '
                   'in CI environment')
        for item in items:
            if 'ci_unsupported' in item.keywords:
                item.add_marker(skip_ci_unsupported)
    else:
        pass

    if torch.cuda.is_available():
        pass
    else:
        skip_cuda = pytest.mark.skip(
            reason='PyTorch is not installed with CUDA integration')
        for item in items:
            if 'cuda' in item.keywords:
                item.add_marker(skip_cuda)
