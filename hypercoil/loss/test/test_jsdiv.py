# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for JS divergence
"""
import pytest
import torch
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import softmax
from hypercoil.loss.jsdiv import (
    js_divergence, js_divergence_logit, JSDivergence, SoftmaxJSDivergence
)


class TestJSDivergence:

    def test_js_div(self):
        jsdiv = JSDivergence()
        p = torch.rand(30, 10)
        p /= p.sum(-1, keepdim=True)
        q = torch.rand(30, 10)
        q /= q.sum(-1, keepdim=True)
        out = js_divergence(p, q, axis=-1).squeeze().sqrt()
        ref = jensenshannon(p.numpy(), q.numpy(), axis=-1)
        assert np.allclose(out, ref)

        out = jsdiv(p, q)
        assert np.allclose(out, (ref ** 2).mean())

    def test_js_div_logit(self):
        torch.random.manual_seed(0)
        jsdiv = SoftmaxJSDivergence()
        p = torch.rand(30, 10)
        q = torch.rand(30, 10)
        p_norm = softmax(p, axis=-1)
        q_norm = softmax(q, axis=-1)
        out = js_divergence_logit(p, q, axis=-1).squeeze().sqrt()
        ref = jensenshannon(p_norm, q_norm, axis=-1)
        assert np.allclose(out, ref)

        out = jsdiv(p, q)
        assert np.allclose(out, (ref ** 2).mean())
