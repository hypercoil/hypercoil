# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Minimal datasets
~~~~~~~~~~~~~~~~
Minimal neuroimaging datasets for testing or rapid prototyping.
"""
import os
import shutil
import requests, zipfile
from pkg_resources import resource_filename as pkgrf


def minimal_msc_download(delete_if_exists=False):
    output = pkgrf(
        'hypercoil',
        'neuro/datasets'
    )
    if delete_if_exists:
        shutil.rmtree(f'{output}/MSCMinimal')
    #TODO: unsafe. No guarantee that the right files are inside the directory.
    if os.path.isdir(f'{output}/MSCMinimal'):
        return f'{output}/MSCMinimal'

    os.makedirs(output, exist_ok=True)

    #TODO: not a long-term solution. Ideally we'd do this the lazy way, but
    # there just isn't time now.
    url = (
        'https://github.com/rciric/xwave/archive/'
        'cf862c1dc3ead168db68abe4c870a509285fc729.zip'
    )
    r = requests.get(url, allow_redirects=True)
    zip_target = f'{output}/MSCMinimal.zip'
    with open(zip_target, 'wb') as f:
        f.write(r.content)
    with zipfile.ZipFile(zip_target, 'r') as zip_ref:
        zip_ref.extractall(f'{output}')
    shutil.move(f'{output}/xwave-cf862c1dc3ead168db68abe4c870a509285fc729',
                f'{output}/MSCMinimal')
    os.remove(zip_target)
    return f'{output}/MSCMinimal'
