# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Grammar
~~~~~~~
Elementary formula grammar. For use with the confound modelling and parameter
specification subsystems.
"""
import re
import equinox as eqx
from abc import abstractclassmethod
from dataclasses import field
from hashlib import sha256
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple


class Literalisation(eqx.Module):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix']
    regex : str



class TransformNode(eqx.Module):
    min_arity : int
    max_arity : int
    priority : int
    canonical_literal: Literalisation
    literals: Sequence[Literalisation]
    associative: bool = False
    commutative: bool = False


import pandas as pd
class ConcatenateInfixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'infix'
    regex : str = r'\+'

class ConcatenatePrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'\+'

class ConcatenateNode(TransformNode):
    min_arity: int = 2
    max_arity: int = float('inf')
    priority: int = 3
    associative: bool = True
    commutative: bool = True
    canonical_literal: Literalisation = ConcatenatePrefixLiteralisation()
    literals: Sequence[Literalisation] = field(default_factory = lambda: [
        ConcatenatePrefixLiteralisation(),
        ConcatenateInfixLiteralisation(),
    ])

    def __call__(self, *pparams, **params):
        return pd.concat(pparams, axis=1)


class BackwardDifferencePrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'd[0-9]+'

class BackwardDifferenceInclPrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'dd[0-9]+'

class BackwardDifferenceRangePrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'd[0-9]+\-[0-9]+'

class BackwardDifferenceEnumPrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'd\{[0-9\,]+\}'

class BackwardDifferenceNode(TransformNode):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 2
    canonical_literal: Literalisation = BackwardDifferenceEnumPrefixLiteralisation()
    literals: Sequence[Literalisation] = field(default_factory = lambda: [
        BackwardDifferencePrefixLiteralisation(),
        BackwardDifferenceInclPrefixLiteralisation(),
        BackwardDifferenceRangePrefixLiteralisation(),
        BackwardDifferenceEnumPrefixLiteralisation(),
    ])

    def __call__(self, *pparams, **params):
        arg = pparams[0]
        acc = [arg]
        for o in range(max(params['order'])):
            arg = arg.diff()
            if o in params['order']:
                acc.append(arg)
        return pd.concat(acc, axis=1)


class PowerSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'suffix'
    regex : str = r'\^[0-9]+'

class PowerInclSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'suffix'
    regex : str = r'\^\^[0-9]+'

class PowerRangeSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'suffix'
    regex : str = r'\^[0-9]+\-[0-9]+'

class PowerEnumSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'suffix'
    regex : str = r'\^\{[0-9\,]+\}'

class PowerNode(TransformNode):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 1
    canonical_literal: Literalisation = PowerEnumSuffixLiteralisation()
    literals: Sequence[Literalisation] = field(default_factory = lambda: [
        PowerSuffixLiteralisation(),
        PowerInclSuffixLiteralisation(),
        PowerRangeSuffixLiteralisation(),
        PowerEnumSuffixLiteralisation(),
    ])

    def __call__(self, *pparams, **params):
        arg = pparams[0]
        acc = [arg ** o for o in params['order']]
        return pd.concat(acc, axis=1)


class IndexedNestedString(eqx.Module):
    content: Tuple[Any]
    index: Tuple[int]

    def __init__(self, content, index=None):
        if index is None:
            total_length = sum(len(c) for c in content)
            index = range(total_length)
        self.content = tuple(content)
        self.index = tuple(index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return self.content[self.index[start]:self.index[stop]:step]
        else:
            return self.content[self.index[key]]

    def __iter__(self):
        return iter(self.content)

    def idx(self, key):
        return self.content.__getitem__(key)

    def substitute(self, content, start, end=None):
        if end is None:
            end = start + len(content)
        new_content = (
            list(self.content[:self.index[start]]) +
            list(content) +
            list(self.content[self.index[end]:])
        )
        new_index = (
            list(self.index[:start]) +
            [self.index[start]] * (end - start) +
            list(range(
                self.index[start] + 1,
                self.index[start] + 1 + len(self.index) - end
            ))
        )
        return IndexedNestedString(new_content, new_index)


class AbstractChild:
    def __init__(self, hash, length=1):
        self.hash = hash
        self.length = length

    def __repr__(self):
        return '⟨⟨{}⟩⟩'.format(self.hash[:4])

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter([self])


class SyntacticTree:
    def __init__(
        self,
        content: str,
        circumfix: Tuple[str, str] = ('', ''),
        hashfn: Callable = sha256,
    ):
        self.hashfn = hashfn
        self.circumfix = circumfix
        if isinstance(content, SyntacticTree):
            # self.content = content.content
            # self.index = content.index
            self.content = IndexedNestedString(
                content=content.content,
                index=content.index,
            )
            self.children = content.children
        else:
            # self.content = tuple(content)
            # self.index = tuple(range(len(self.content) + 1))
            self.content = IndexedNestedString(
                content=tuple(content),
                index=tuple(range(len(content) + 1)),
            )
            self.children = {}

    def materialise(self, repr=False, recursive=False):
        if repr:
            reprstr = [
                c.__repr__()
                if isinstance(c, AbstractChild) else c
                for c in self.content
            ]
        elif recursive:
            reprstr = [
                self.children[c.hash].apply_circumfix(
                    self.children[c.hash].materialise(recursive=True))
                if isinstance(c, AbstractChild) else c
                for c in self.content
            ]
        else:
            reprstr = [
                '▒' if isinstance(c, AbstractChild) else c
                for c in self.content
            ]
        return ''.join(reprstr)

    def apply_circumfix(self, s):
        return self.circumfix[0] + s + self.circumfix[1]

    def _scan_content_and_embed_child(self, loc, child_id):
        # content = []
        # last = 0
        # index = list(self.index)
        content = self.content
        for start, end in loc:
            content = content.substitute(
                content=AbstractChild(child_id, length=end - start),
                start=start,
                end=end,
            )
        #     content += list(self.content[self.index[last]:self.index[start]])
        #     content += [AbstractChild(child_id, length=end - start)]
        #     last = end
        #     index[start:end] = [index[start]] * (end - start)
        #     index[end:] = range(
        #         index[start] + 1,
        #         index[start] + 1 + len(index) - end
        #     )
        # content += self.content[last:]
        # return content, index
        return content

    def _child_carry_nested_content(self, loc, circumfix, drop_circumfix):
        start, end = loc
        if drop_circumfix:
            start += 1
            end -= 1
        # child_index = tuple([
        #     i - self.index[start] for i in self.index[start:end]
        # ])
        # child_content = self.content[self.index[start]:self.index[end]]
        low = self.content.index[start]
        child_index = tuple([
            i - low for i in self.content.index[start:end]
        ])
        child_content = self.content[start:end]
        child_str = IndexedNestedString(
            content=child_content,
            index=child_index,
        )
        if len(child_content) == 1:
            if isinstance(child_content[0], AbstractChild):
                return # already nested
        return SyntacticTree.from_parsed(
            #content=child_content,
            #index=child_index,
            content=child_str,
            children=self.children,
            circumfix=circumfix,
            hashfn=self.hashfn,
        )

    def create_child(
        self,
        query,
        recursive=False,
        nest=True,
        drop_circumfix=False,
    ):
        child_text = query
        circumfix = ('', '')
        if drop_circumfix:
            circumfix = (query[0], query[-1])
            child_text = query[1:-1]
        child_id = self.hashfn(child_text.encode('utf-8')).hexdigest()

        contentstr = self.materialise(recursive=recursive)
        loc = [m.span() for m in re.finditer(re.escape(query), contentstr)]

        # content, index = self._scan_content_and_embed_child(loc, child_id)
        content = self._scan_content_and_embed_child(loc, child_id)

        if nest:
            child = self._child_carry_nested_content(
                loc[0], circumfix, drop_circumfix,
            )
        else:
            child = SyntacticTree(
                child_text,
                circumfix=circumfix,
                hashfn=self.hashfn
            )
    
        # self.content = tuple(content)
        # self.index = tuple(index)
        self.content = content
        self.children[child_id] = child

    @classmethod
    def from_parsed(
        cls,
        content,
        #index,
        children=None,
        circumfix=('', ''),
        hashfn=sha256
    ):
        if children is None:
            children = {}
        present_children = [c.hash for c in content
                            if isinstance(c, AbstractChild)]
        children = {c: v for c, v in children.items()
                    if c in present_children}
        tree = object.__new__(cls)
        tree.content = content
        #tree.index = index
        tree.children = children
        tree.circumfix = circumfix
        tree.hashfn = hashfn
        return tree

    def __repr__(self):
        return self.materialise(repr=True)


class Grouping(eqx.Module):
    open: str
    close: str


class GroupingPool(eqx.Module):
    groupings: Sequence[Grouping]
    open: Tuple[str]
    close: Tuple[str]
    open_to_close: Dict[str, str]
    close_to_open: Dict[str, str]

    def __init__(self, *groupings) -> None:
        self.groupings = groupings
        self.open, self.close = self.open_and_close()
        self.open_to_close = dict(zip(self.open, self.close))
        self.close_to_open = dict(zip(self.close, self.open))

    def open_and_close(self):
        open, close = zip(*[(g.open, g.close) for g in self.groupings])
        return tuple(open), tuple(close)

    def eval_stack(self, stack, char):
        if char in self.open:
            return 1, stack + [char]
        elif char in self.close:
            if not stack:
                raise ValueError('Unmatched closing bracket')
            if stack[-1] == self.close_to_open[char]:
                #stack.pop()
                return -1, stack[:-1]
        return 0, stack


class Grammar(eqx.Module):
    groupings: GroupingPool
    transforms: Sequence[TransformNode]

    @staticmethod
    def delete_whitespace(s: str) -> str:
        whitespace = r'[\s]+'
        whitespace = re.compile(whitespace)
        return re.sub(whitespace, '', s)

    def parse_groups(self, s: str) -> SyntacticTree:
        tree = SyntacticTree(s)
        stack = []
        pointer = 0
        for i, c in enumerate(s):
            val, stack = self.groupings.eval_stack(stack, c)
            if val == 1 and len(stack) == 1:
                pointer = i
            elif val == -1 and not stack:
                tree.create_child(
                    s[pointer:i + 1],
                    drop_circumfix=True,
                    recursive=True,
                )
        return tree


class TestGrammar(Grammar):
    groupings: GroupingPool = GroupingPool(
        Grouping(open='(', close=')'),
        Grouping(open='[', close=']'),
        Grouping(open='{', close='}'),
    )
    transforms: Sequence[TransformNode] = (
        ConcatenateNode(),
    )
