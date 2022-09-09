# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for artefact models.
"""
import jax
import jax.numpy as jnp
from hypercoil.engine.paramutil import _to_jax_array
from hypercoil.nn.confound import (
    LinearRFNN,
    QCPredict,
    LinearCombinationSelector,
    EliminationSelector,
)


#TODO: All of these tests are only testing that the forward pass doesn't
#      crash.
class TestArtefactModels:
    def test_rfnn(self):
        key = jax.random.PRNGKey(0)
        confounds = jnp.zeros((2, 1, 36, 100))

        model = LinearRFNN(
            model_dim=1,
            num_columns=36,
            key=key,
        )
        out = model(confounds)
        assert out.shape == (2, 1, 1, 100)

        confounds = jnp.zeros((36, 100))
        out = model(confounds)
        assert out.shape == (1, 1, 1, 100)

    def test_qcpredict(self):
        key = jax.random.PRNGKey(0)
        data = jnp.zeros((3, 1, 12, 100))

        model = QCPredict(
            num_columns=12,
            key=key,
        )
        out = model(data)
        assert out.shape == (3, 1, 1, 100)

    def test_lcselector(self):
        key = jax.random.PRNGKey(0)
        data = jnp.zeros((3, 1, 12, 100))

        model = LinearCombinationSelector(
            model_dim=3,
            num_columns=12,
            key=key,
        )
        out = model(data)
        assert out.shape == (3, 1, 3, 100)

    def test_elimination(self):
        key = jax.random.PRNGKey(0)
        data = jnp.zeros((3, 1, 12, 100))

        model = EliminationSelector(
            num_columns=12,
            key=key,
        )
        out = model(data)
        assert (_to_jax_array(model.weight) >= 0).all()
        assert (_to_jax_array(model.weight) <= 1).all()
        assert out.shape == data.shape
