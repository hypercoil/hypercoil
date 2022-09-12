# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for image maths operations.
"""
import jax.numpy as jnp
import nibabel as nb
from hypercoil.formula.imops import (
    ImageMathsGrammar,
)


class TestImageMaths:

    def test_thresh(self):
        img = nb.load(
            '/Users/rastkociric/Downloads/MBP_BKP/'
            'Downloads/hypercoil/hypercoil/examples/ds-synth/'
            'func/sub-0_run-0_task-rest_desc-preproc_bold.nii.gz')
        grammar = ImageMathsGrammar()
        op_str = 'IMG -thr (IMG -bin[0.5])'
        f = grammar.compile(op_str)
        img_out, _ = f(img)(img)
        img_dataobj = img.get_fdata()
        img_thr = (img_dataobj > 0.5)
        img_ref = jnp.where(img_dataobj > img_thr, img_dataobj, 0)
        assert (img_out == img_ref).all()
