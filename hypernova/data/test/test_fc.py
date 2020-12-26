# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Transforms
~~~~~~~~~~
Transforms applied to columns of a pandas DataFrame.
"""
import hypernova
import pandas as pd
from pathlib import Path
from pkg_resources import resource_filename as pkgrf


confpath = pkgrf('hypernova', 'testdata/desc-confounds_timeseries.tsv')
metapath = pkgrf('hypernova', 'testdata/desc-confounds_timeseries.json')
metadata = hypernova.data.load_metadata(metapath)
df = pd.read_csv(confpath, sep='\t')


def test_36ev_expr():
    model_formula = 'dd1((rps + wm + csf + gsr)^^2)'
    shfc = hypernova.data.fc.FCShorthand()
    df = pd.read_csv(confpath, sep='\t')
    spec_sh = shfc(model_formula, df.columns, metadata)
    assert(spec_sh == (
        'dd1((trans_x + trans_y + trans_z + rot_x + rot_y + rot_z + '
        'white_matter + csf + global_signal)^^2)'))
    expr = hypernova.data.Expression(
        spec_sh,
        transforms=[hypernova.data.fc.PowerTransform(),
                    hypernova.data.fc.DerivativeTransform()]
    )
    assert(expr.children[0].children[0].n_children == 9)


def test_36ev_spec():
    model_formula = 'dd1((rps + wm + csf + gsr)^^2)'
    fcms = hypernova.data.FCConfoundModelSpec(model_formula, '36ev')
    out = fcms(df, metadata)
    out[['trans_y', 'trans_y_power2', 'trans_y_derivative1']]
    assert(out.shape[1] == 36)
