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
shfc = hypernova.data.fc.FCShorthand()
df = pd.read_csv(confpath, sep='\t')


def test_36ev_expr():
    model_formula = 'dd1((rps + wm + csf + gsr)^^2)'
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
    assert('trans_y_power2_derivative1' in out.columns)
    assert(out.shape[1] == 36)


def test_acc_expr():
    model_formula = 'd1(rps) + acc<v=29.9, mask=CSF+WM>'
    spec_sh = shfc(model_formula, df.columns, metadata)
    assert(spec_sh == (
        'd1(trans_x + trans_y + trans_z + rot_x + rot_y + rot_z) + ' +
        ' + '.join([f'a_comp_cor_{i:02}' for i in range(4)]) + ' + ' +
        ' + '.join([f'a_comp_cor_{i:02}' for i in range(9, 22)])))
    expr = hypernova.data.Expression(
        spec_sh,
        transforms=[hypernova.data.fc.PowerTransform(),
                    hypernova.data.fc.DerivativeTransform()]
    )
    assert(expr.n_children == 18)
    assert(expr.children[0].children[0].n_children == 6)


def test_acc_spec():
    model_formula = 'd1(rps) + acc<v=29.9, mask=CSF+WM>'
    fcms = hypernova.data.FCConfoundModelSpec(model_formula, '36ev')
    out = fcms(df, metadata)
    assert('trans_y' not in out.columns)
    assert('a_comp_cor_02' in out.columns)
    assert(out.shape[1] == 23)


def test_aroma_expr():
    model_formula = 'wm + csf + aroma'
    spec_sh = shfc(model_formula, df.columns, metadata)
    assert(len(spec_sh.split(' + ')) == 41)
    expr = hypernova.data.Expression(
        spec_sh,
        transforms=[hypernova.data.fc.PowerTransform(),
                    hypernova.data.fc.DerivativeTransform()]
    )
    assert(expr.n_children == 41)


def test_aroma_spec():
    model_formula = 'wm + csf + aroma'
    fcms = hypernova.data.FCConfoundModelSpec(model_formula, '36ev')
    out = fcms(df, metadata)
    assert('aroma_motion_57' in out.columns)
    assert('aroma_motion_38' not in out.columns)
    assert(out.shape[1] == 41)
