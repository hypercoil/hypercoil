# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for formula grammar
"""
from hypercoil.formula.grammar import (
    Grammar,
    GroupingPool, Grouping,
    TransformPool, ConcatenateNode, BackwardDifferenceNode, PowerNode
)


class MinimalGrammar(Grammar):
    groupings: GroupingPool = GroupingPool(
        Grouping(open='(', close=')'),
        Grouping(open='[', close=']'),
        Grouping(open='{', close='}'),
    )
    transforms: TransformPool = TransformPool(
        ConcatenateNode(),
        BackwardDifferenceNode(),
        PowerNode(),
    )


class TestGrammar:
    def test_grouping(self):
        test_str = 'The (dog) is not very [(dog)gy]'
        tree = MinimalGrammar().parse_groups(test_str)
        assert len(tree.children) == 2
        assert sorted([
            len(v.children) for v in tree.children.values()
        ]) == [0, 1]

        intermediate = [
            v for v in tree.children.values()
            if len(v.children) == 1
        ][0]
        final = {
            k: v for k, v in tree.children.items()
            if len(v.children) == 0
        }
        assert intermediate.children == final

    def test_transform_parse(self):
        masked = '▒'
        test_str = 'x^^2 + x^2 + x^3-5'
        test_str = MinimalGrammar().delete_whitespace(test_str)
        tree = MinimalGrammar().tokenise_transforms(test_str)

        assert tree.materialise(recursive=True) == test_str
        assert (
            tree.materialise(recursive=False) ==
            f'x{masked}{masked}x{masked}{masked}x{masked}'
        )
        assert tree.materialise(repr=True) == (
            'x⌈^^2⌋⌈+⌋x⌈^2⌋⌈+⌋x⌈^3-5⌋'
        )

        ledger = MinimalGrammar().make_transform_ledger(tree)
        MinimalGrammar().parse_transforms(tree, ledger)
        children = set(tree.children.values())
        out = set([
            tree.materialise_recursive(c.content, tree.children)
            for c in children
        ])
        ref = {'x^^2+x^2+x^3-5', 'x^^2+x^2', 'x^3-5', 'x^^2', 'x^2', 'x'}
        assert out == ref
