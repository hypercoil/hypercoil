# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for formula grammar
"""
import pandas as pd
from hypercoil.formula.grammar import (
    Grammar, UnparsedTreeError,
    GroupingPool, Grouping, TransformPool,
)
from hypercoil.formula.dfops import (
    ColumnSelectInterpreter,
    ConcatenateNode, PowerNode,
)


class MinimalGrammar(Grammar):
    groupings: GroupingPool = GroupingPool(
        Grouping(open='(', close=')'),
        Grouping(open='[', close=']'),
        Grouping(open='{', close='}'),
    )
    transforms: TransformPool = TransformPool(
        ConcatenateNode(),
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
        assert intermediate.children.keys() == final.keys()

    def test_shorthand(self):
        grammar = MinimalGrammar(shorthand={'vars': '(x + y + z)'})
        test_str = 'vars^^2 + vars + (vars^2 + vars)^3-5'
        test_str = grammar.expand_shorthand(test_str)
        test_str = grammar.delete_whitespace(test_str)
        assert test_str == '(x+y+z)^^2+(x+y+z)+((x+y+z)^2+(x+y+z))^3-5'

    def test_transform_parse(self):
        grammar = MinimalGrammar()
        masked = '▒'
        test_str = 'x^^2 + x^2 + x^3-5'
        test_str = grammar.delete_whitespace(test_str)
        tree = grammar.tokenise_transforms(test_str)

        assert tree.materialise(recursive=True) == test_str
        assert (
            tree.materialise(recursive=False) ==
            f'x{masked}{masked}x{masked}{masked}x{masked}'
        )
        assert tree.materialise(repr=True) == (
            'x⌈^^2⌋⌈+⌋x⌈^2⌋⌈+⌋x⌈^3-5⌋'
        )

        ledger = grammar.make_transform_ledger(tree)
        tree = grammar.parse_transforms(tree, ledger)
        children = set(tree.children.values())
        out = set([
            tree.materialise_recursive(c.content, tree.children)
            for c in children
        ])
        ref = {'x^^2+x^2', 'x^3-5', 'x^^2', 'x^2', 'x'}
        assert out == ref

        xdict = {
            tree.materialise_recursive(
                c.content, tree.children
            ): (c.transform_root.metadata['transform']
                if c.transform_root is not None else None)
            for c in tree.children.values()
        }
        assert isinstance(tree.transform_root.metadata['transform'],
                          ConcatenateNode)
        assert isinstance(xdict['x^^2+x^2'], ConcatenateNode)
        assert isinstance(xdict['x^3-5'], PowerNode)
        assert isinstance(xdict['x^^2'], PowerNode)
        assert isinstance(xdict['x^2'], PowerNode)
        assert xdict['x'] is None

    def test_parse(self):
        def recursive_children(tree):
            return [
                (c.materialise(recursive=True), c.transform_root.__repr__(), recursive_children(c))
                for c in tree.children.values()
            ]

        grammar = MinimalGrammar()
        test_str = 'x^^2 + (x^2 + x)^3-5'

        tree = grammar.parse_groups(test_str)
        try:
            grammar.verify_parse(tree)
            assert 0
        except UnparsedTreeError:
            pass

        tree = grammar.parse(test_str)
        out = [(
            tree.materialise(recursive=True),
            tree.transform_root.__repr__(),
            recursive_children(tree),
        )]
        ref = [
            ('x^^2+(x^2+x)^3-5', '⌈+⌋', [
                ('x^^2', '⌈^^2⌋', [
                    ('x', 'None', [])]),
                ('(x^2+x)^3-5', '⌈^3-5⌋', [
                    ('x^2+x', '⌈+⌋', [
                        ('x', 'None', []),
                        ('x^2', '⌈^2⌋', [
                            ('x', 'None', [])
                        ])
                    ])
                ])
            ])
        ]
        assert out == ref
        grammar.verify_parse(tree)

    def test_transform(self):
        grammar = MinimalGrammar()
        test_str = 'x^^2 + x + (x^2 + x)^3-5'
        tree = grammar.parse(test_str)
        tree = grammar.transform(tree)
        assert isinstance(tree.transform, ConcatenateNode)
        assert len(tree.children) == 3
        #TODO: tree.children[0] is not a PowerNode, but it should be if
        #      the grammar is to handle noncommutative operations properly
        child = tree.children[-1]
        assert isinstance(child.transform, PowerNode)
        assert child.parameters == {'num_leaves': 2, 'order': (3, 4, 5)}
        assert len(child.children) == 1
        assert isinstance(child.children[0].transform, ConcatenateNode)
        assert len(child.children[0].children) == 2

    def test_compile(self):
        data = {
            'x': [1, 2, 3, 4, 5],
        }
        data = pd.DataFrame(data)

        grammar = MinimalGrammar()
        test_str = 'x^^2 + x + (x^2 + x)^3-5'
        tree = grammar.parse(test_str)
        tree = grammar.transform(tree)
        f = tree.compile(interpreter=ColumnSelectInterpreter())
        out, _ = f(data)
        cols = set(out.columns)
        ref_cols = {
            'x', 'x_power2', 'x_power3', 'x_power4', 'x_power5',
            'x_power2_power3', 'x_power2_power4', 'x_power2_power5',
        }
        assert cols == ref_cols
        assert (out['x_power2_power5'] == (data['x']**2)**5).all()

    def test_trivial(self):
        grammar = MinimalGrammar()
        test_str = 'x'
        tree = grammar.parse(test_str)
        tree = grammar.transform(tree)
        f = tree.compile(interpreter=ColumnSelectInterpreter())
        inp = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        out, _ = f(inp)
        assert (out[['x']].values == inp.values).all()
