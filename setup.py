#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Differentiable programming for functional brain mapping

This package intends to implement atoms and compositions to support writing
differentiable programs for neuroimaging analysis. With numerous analytic
options available, it is often unclear how to select the best option for
satisfying a particular objective. Provided the objective can be formulated or
approximated as a differentiable function, this package can help optimise over
the space of possible parametrisations.

Essentially, this system implements common analytic steps as neural network
layers that can be initialised to reasonable defaults that yield performance
identical to a conventional pipeline. They can then be concatenated with neural
network models and trained via backpropagation to improve that performance.
"""
from setuptools import setup


NAME = 'hypercoil'


def main():
    setup(
        name=NAME
    )


if __name__ == '__main__':
    main()
