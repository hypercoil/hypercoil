# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for image maths operations.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pkg_resources import resource_filename as pkgrf
from hypercoil.formula.imops import (
    ImageMathsGrammar,
    LeafInterpreter,
    scalar_leaf_ingress,
    select_args,
)


class TensorLeafInterpreter(LeafInterpreter):
    def __call__(self, leaf):
        def img_and_meta(*args, side='left'):
            arg = select_args(args, side=side, num=1)[0]
            if leaf[:3] == 'ARG':
                return arg, {}
            else:
                return scalar_leaf_ingress(leaf)
        return img_and_meta


class TestImageMaths:

    def test_thresh(self):
        key = jax.random.PRNGKey(0)
        arg = jax.random.uniform(key, (10, 10, 10))
        grammar = ImageMathsGrammar(default_interpreter=TensorLeafInterpreter())
        op_str = 'ARGa -thr (ARGb -bin[0.5])'
        f = grammar.compile(op_str)
        img_out, _ = f(arg, arg)
        img_dataobj = arg
        img_thr = (img_dataobj > 0.5)
        img_ref = jnp.where(img_dataobj > img_thr, img_dataobj, 0)
        assert (img_out == img_ref).all()

    def test_morphological_atoms(self):
        grammar = ImageMathsGrammar(default_interpreter=TensorLeafInterpreter())

        data = np.zeros((10, 10))
        data[2:8, 2:8] = 1
        data[2, 3] = 0
        data[4, 6] = 0

        op_str = 'ARG -dil[1]'
        f = grammar.compile(op_str)
        dil, _ = f(data)

        op_str = 'ARG -ero[1]'
        f = grammar.compile(op_str)
        ero, _ = f(data)

        op_str = 'ARG -opening[1]'
        f = grammar.compile(op_str)
        opened, _ = f(data)

        op_str = 'ARG -closing[1]'
        f = grammar.compile(op_str)
        closed, _ = f(data)

        op_str = 'ARG -fillholes[1|5]'
        f = grammar.compile(op_str)
        filled, _ = f(data)

        fig, ax = plt.subplots(2, 3, figsize=(18, 12))
        ax[0, 0].imshow(data, cmap='gray')
        ax[0, 1].imshow(dil, cmap='gray')
        ax[0, 2].imshow(ero, cmap='gray')
        ax[1, 0].imshow(opened, cmap='gray')
        ax[1, 1].imshow(closed, cmap='gray')
        ax[1, 2].imshow(filled, cmap='gray')

        results = pkgrf(
            'hypercoil',
            'results/'
        )
        fig.savefig(f'{results}/imops_morphology.png', bbox_inches='tight')

    def test_logical_atoms(self):
        grammar = ImageMathsGrammar(default_interpreter=TensorLeafInterpreter())

        data0 = np.zeros((10, 10))
        data0[:, :7] = 1
        data0[:, :1] = 0
        data1 = np.zeros((10, 10))
        data1[:, 3:] = 1
        data1[:, 9:] = 0

        op_str = 'ARG -neg'
        f = grammar.compile(op_str)
        negation0, _ = f(data0)
        negation1, _ = f(data1)

        op_str = 'ARGa -or ARGb'
        f = grammar.compile(op_str)
        union, _ = f(data0, data1)

        op_str = 'ARGa -and ARGb'
        f = grammar.compile(op_str)
        intersection, _ = f(data0, data1)

        fig, ax = plt.subplots(2, 3, figsize=(18, 12))
        ax[0, 0].imshow(data0, cmap='gray')
        ax[0, 1].imshow(data1, cmap='gray')
        ax[0, 2].imshow(negation0, cmap='gray')
        ax[1, 0].imshow(negation1, cmap='gray')
        ax[1, 1].imshow(union, cmap='gray')
        ax[1, 2].imshow(intersection, cmap='gray')

        results = pkgrf(
            'hypercoil',
            'results/'
        )
        fig.savefig(f'{results}/imops_logic.png', bbox_inches='tight')

    def test_compositions(self):
        grammar = ImageMathsGrammar(default_interpreter=TensorLeafInterpreter())

        key = jax.random.PRNGKey(0)
        key0, key1, key2 = jax.random.split(key, 3)
        data0 = jax.random.uniform(key0, (10, 10))
        data1 = jax.random.uniform(key1, (10, 10))
        data2 = jax.random.uniform(key2, (10, 10))

        op_str = (
            r'ARGa -sub{1} -thr[0.1]{0.5} -mul ARGb '
            r'-add{1} -uthr[-1] (ARGc -div{2,...})'
        )
        f = grammar.compile(op_str)
        img_out, _ = f(data0, data1, data2)
        img_ref = data0 - 1
        img_ref = jnp.where(img_ref > 0.5, img_ref, 0.1)
        img_ref = img_ref * data1 + 1
        img_ref = jnp.where(img_ref < 2 / data2, img_ref, -1)
        assert (img_out == img_ref).all()
