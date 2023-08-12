# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for dataframe formula grammar
"""
import json
import pandas as pd
from pkg_resources import resource_filename as pkgrf
from hypercoil.formula.dfops import (
    ConfoundFormulaGrammar
)


def path_from_examples(fname):
    return pkgrf('hypercoil_examples', f'data/confounds/{fname}')


#TODO: remove this after we upstream the ``data`` submodule.
def read_json(path):
    """
    Load JSON-formatted metadata into a python dictionary.

    Parameters
    ----------
    path : str
        Path to the JSON-formatted metadata file.

    Returns
    -------
    metadata : dict
        Python dictionary containing all metadata in the JSON file.
    """
    with open(path) as file:
        metadata = json.load(file)
    return metadata


class TestModelSpec:
    filenames = {
        'confdata': 'desc-confounds_timeseries.tsv',
        'confmeta': 'desc-confounds_timeseries.json'
    }
    confdata = path_from_examples(filenames['confdata'])
    confmeta = path_from_examples(filenames['confmeta'])
    df = pd.read_csv(confdata, sep='\t')
    metadata = read_json(confmeta)

    def spec_base(
        self, model_formula, shape
    ):
        f = ConfoundFormulaGrammar().compile(model_formula)
        out, meta = f(self.df, self.metadata)
        assert(out.shape[1] == shape)
        return out, meta

    def test_36ev_spec(self):
        out, _ = self.spec_base(
            model_formula='dd1((rps + wm + csf + gsr)^^2)',
            shape=36)
        out[['trans_y', 'trans_y_power2', 'trans_y_derivative1']]
        assert('trans_y_power2_derivative1' in out.columns)

    def test_parens_spec(self):
        self.spec_base(
            model_formula='(rps)^^2 + dd1(rps)',
            shape=18
        )

    def test_tmask_spec(self):
        out, _ = self.spec_base(
            model_formula='[AND](1_[<0.5](fd) + 1_[<1.5](dv))',
            shape=1
        )
        assert out.values.sum() == (len(out) - 3)
        out, _ = self.spec_base(
            model_formula='[NOT]([OR](1_[>0.5](fd) + 1_[>1.5](dv)))',
            shape=1
        )
        # Note the inconsistency because nan is always false at comparison
        assert out.values.sum() == (len(out) - 2)

    def test_spikes_spec(self):
        out, _ = self.spec_base(
            model_formula='[SCATTER]([OR](1_[>0.5](fd) + 1_[>1.5](dv)))',
            shape=2
        )
        assert (out.values.sum(0) == 1).all()

    def test_accn_spec(self):
        out, _ = self.spec_base(
            model_formula='d0-1(rps) + n_{{5; acc; Mask=CSF,WM}}',
            shape=22)
        assert('a_comp_cor_30' not in out.columns)
        assert('a_comp_cor_02' in out.columns)

    def test_acc_spec(self):
        out, _ = self.spec_base(
            model_formula='d1(rps) + v_{{29.9; acc; Mask=CSF,WM}}',
            shape=23)
        assert('trans_y' not in out.columns)
        assert('a_comp_cor_02' in out.columns)

    def test_aroma_spec(self):
        out, _ = self.spec_base(
            model_formula='wm + csf + {{aroma; MotionNoise=True}}',
            shape=41)
        assert('aroma_motion_57' in out.columns)
        assert('aroma_motion_38' not in out.columns)
