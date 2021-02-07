# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Functional connectivity
~~~~~~~~~~~~~~~~~~~~~~~
Tests for functional connectivity-specific data loading.
"""
import pytest
import hypernova
import pandas as pd
from pathlib import Path
from pkg_resources import resource_filename as pkgrf


class TestModelSpec:
    filenames = {
        'confdata': 'desc-confounds_timeseries.tsv',
        'confmeta': 'desc-confounds_timeseries.json'
    }

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.confpath = self.path_from_examples('confdata')
        self.metapath = self.path_from_examples('confmeta')
        self.metadata = hypernova.data.functional.read_json(self.metapath)
        self.shfc = hypernova.formula.fc.FCShorthand()
        self.df = pd.read_csv(self.confpath, sep='\t')

    def path_from_examples(self, key):
        return pkgrf('hypernova', 'examples/{}'.format(self.filenames[key]))

    def expr_base(self, model_formula, expanded_spec, n_children):
        spec_sh = self.shfc(model_formula, self.df.columns, self.metadata)
        if expanded_spec:
            assert(spec_sh == expanded_spec)
        expr = hypernova.formula.Expression(
            spec_sh,
            transforms=hypernova.formula.fc.fc_transforms()
        )
        assert(expr.n_children == n_children)

    def spec_base(self, model_formula, name, shape):
        fcms = hypernova.formula.FCConfoundModelSpec(model_formula, name)
        out = fcms(self.df, self.metadata)
        assert(out.shape[1] == shape)
        return out

    def test_36ev_expr(self):
        self.expr_base(
            model_formula='dd1((rps + wm + csf + gsr)^^2)',
            expanded_spec=(
                'dd1((trans_x + trans_y + trans_z + rot_x + rot_y + rot_z + '
                'white_matter + csf + global_signal)^^2)'),
            n_children=1
        )

    def test_36ev_spec(self):
        out = self.spec_base(
            model_formula='dd1((rps + wm + csf + gsr)^^2)',
            name='36ev',
            shape=36)
        out[['trans_y', 'trans_y_power2', 'trans_y_derivative1']]
        assert('trans_y_power2_derivative1' in out.columns)

    def test_acc_expr(self):
        self.expr_base(
            model_formula='d1(rps) + acc<v=29.9, mask=CSF+WM>',
            expanded_spec=(
                'd1(trans_x + trans_y + trans_z + rot_x + rot_y + rot_z) + ' +
                ' + '.join([f'a_comp_cor_{i:02}' for i in range(4)]) + ' + ' +
                ' + '.join([f'a_comp_cor_{i:02}' for i in range(9, 22)])),
            n_children=18
        )

    def test_acc_spec(self):
        out = self.spec_base(
            model_formula='d1(rps) + acc<v=29.9, mask=CSF+WM>',
            name='acc',
            shape=23)
        assert('trans_y' not in out.columns)
        assert('a_comp_cor_02' in out.columns)

    def test_accn_expr(self):
        self.expr_base(
            model_formula='d0-1(rps) + acc<n=5, mask=CSF+WM>',
            expanded_spec=(
                'd0-1(trans_x + trans_y + trans_z + rot_x + rot_y + rot_z) + '
                + ' + '.join([f'a_comp_cor_{i:02}' for i in range(5)]) + ' + '
                + ' + '.join([f'a_comp_cor_{i:02}' for i in range(9, 14)])),
            n_children=11
        )

    def test_accn_spec(self):
        out = self.spec_base(
            model_formula='d0-1(rps) + acc<n=5, mask=CSF+WM>',
            name='acc',
            shape=22)
        assert('a_comp_cor_30' not in out.columns)
        assert('a_comp_cor_02' in out.columns)

    def test_aroma_expr(self):
        self.expr_base(
            model_formula='wm + csf + aroma',
            expanded_spec=None,
            n_children=41
        )

    def test_aroma_spec(self):
        out = self.spec_base(
            model_formula='wm + csf + aroma',
            name='aroma',
            shape=41)
        assert('aroma_motion_57' in out.columns)
        assert('aroma_motion_38' not in out.columns)

    def test_parens_expr(self):
        self.expr_base(
            model_formula='(rps)^^2 + dd1(rps)',
            expanded_spec=(
                '(trans_x + trans_y + trans_z + rot_x + rot_y + rot_z)^^2 + '
                'dd1(trans_x + trans_y + trans_z + rot_x + rot_y + rot_z)'
            ),
            n_children=2
        )

    def test_parens_spec(self):
        out = self.spec_base(
            model_formula='(rps)^^2 + dd1(rps)',
            name='parens',
            shape=18
        )

    def test_tmask_expr(self):
        self.expr_base(
            model_formula='and(1_[fd<0.5] + 1_[dv<1.5])',
            expanded_spec=None,
            n_children=1
        )

    def test_tmask_spec(self):
        out = self.spec_base(
            model_formula='and(1_[fd<0.5] + 1_[dv<1.5])',
            name='tmask',
            shape=1
        )
        assert out.values.sum() == (len(out) - 3)
        out = self.spec_base(
            model_formula='not(or(1_[fd>0.5] + 1_[dv>1.5]))',
            name='tmask',
            shape=1
        )
        # Note the inconsistency because nan is always false at comparison
        assert out.values.sum() == (len(out) - 2)
