# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for model argument
"""
from hypercoil.engine.argument import ModelArgument


class TestModelArgument:
    def test_argument(self):
        agmt0 = ModelArgument(
            name='test',
            default=0,
            dtype=int,
            doc='Test argument',
        )
        assert agmt0.name == 'test'
        assert agmt0['default'] == 0
        assert len(agmt0) == 4

        assert ModelArgument.add(agmt0, rabbit=None) == ModelArgument(
            name='test',
            default=0,
            dtype=int,
            doc='Test argument',
            rabbit=None,
        )

        assert ModelArgument.all_except(agmt0, 'name') == ModelArgument(
            default=0,
            dtype=int,
            doc='Test argument',
        )

        assert ModelArgument.replaced(agmt0, name='test2') == ModelArgument(
            name='test2',
            default=0,
            dtype=int,
            doc='Test argument',
        )

        assert ModelArgument.replaced(agmt0, name='test2', rabbit=None) == ModelArgument(
            name='test2',
            default=0,
            dtype=int,
            doc='Test argument',
        )

        assert ModelArgument.swap(agmt0, 'name', type='anonymous') == ModelArgument(
            type='anonymous',
            default=0,
            dtype=int,
            doc='Test argument',
        )

        agmt1 = ModelArgument(
            name='val',
            dtype=float,
        )
        assert agmt1.name == 'val'
        assert len(agmt1) == 2

        assert agmt0 + agmt1 == ModelArgument(
            name='val',
            default=0,
            dtype=float,
            doc='Test argument',
        )
