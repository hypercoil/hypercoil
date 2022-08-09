# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for polynomial convolution
"""
import pytest
import numpy as np
from hypercoil.functional.tsconv import (
    tsconv2d, polyconv2d, basisconv2d, polychan, basischan
)


def known_filter():
    weight = np.array([
        [0, 0, 1, 0, 0],
        [0, 0, 0.3, 0, 0],
        [0, 0, -0.1, 0, 0]
    ])
    return weight.reshape(1, weight.shape[0], 1, weight.shape[1])


class TestPolynomial:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = np.random.rand(7, 100)
        self.approx = np.allclose

    def test_tsconv(self):
        def zeropad_and_shift(x, n):
            if n == 0:
                return x
            out = np.zeros_like(x)
            if n > 0:
                out[..., n:] = x[..., :-n]
            else:
                out[..., :n] = x[..., -n:]
            return out
        X = np.random.rand(7, 100)
        weight = np.array([0., 0., 2., 1., 0.])
        out = tsconv2d(X, weight)
        assert np.allclose(2 * X + zeropad_and_shift(X, -1), out)

        weight = np.array([2., 1., 0.])
        out2 = tsconv2d(X, weight, padding='final')
        assert np.allclose(out2, out)

        weight = np.array([0., 1., 2.])
        out3 = np.flip(
            tsconv2d(np.flip(X, -1), weight, padding='initial'),
            -1)
        assert np.allclose(out3, out)

        # test different input and weight shapes
        shapes_to_test = [
            [(2, 4, 10), (2, 1, 3), (1, 1, 4, 10)],
            [(2, 4, 10), (4, 2, 1, 3), (1, 4, 4, 10)],
            [(8, 2, 4, 10), (4, 2, 1, 3), (8, 4, 4, 10)]
        ]
        for i, w, o in shapes_to_test:
            assert tsconv2d(
                np.random.rand(*i), np.random.rand(*w)
            ).shape == o

    def test_polychan(self):
        X = np.random.rand(10)
        out = polychan(X, degree=2, include_const=False)
        assert out.shape == (1, 2, 1, 10)
        assert self.approx(out[0, 0, 0, :], X)
        assert self.approx(out[0, 1, 0, :], X ** 2)
        out = polychan(X, degree=2, include_const=True)
        assert out.shape == (1, 3, 1, 10)
        assert self.approx(out[0, 0, 0, :], 1)
        assert self.approx(out[0, 1, 0, :], X)
        assert self.approx(out[0, 2, 0, :], X ** 2)

        basis = [
            (lambda x: x ** 1),
            (lambda x: x ** 2),
            (lambda x: x ** 3),
        ]
        out2 = basischan(X, basis_functions=basis, include_const=True)
        assert self.approx(out, out2[:, :-1])

    def test_polyconv2d(self):
        out = polyconv2d(self.X, known_filter())
        ref = self.X + 0.3 * self.X ** 2 - 0.1 * self.X ** 3
        assert self.approx(out, ref)

    def test_basisconv2d(self):
        basis = [
            (lambda x: x ** 1),
            (lambda x: x ** 2),
            (lambda x: x ** 3),
        ]
        out = basisconv2d(
            self.X,
            basis_functions=basis,
            weight=known_filter()
        )
        ref = self.X + 0.3 * self.X ** 2 - 0.1 * self.X ** 3
        assert self.approx(out, ref)
