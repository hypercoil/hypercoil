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
from abc import abstractmethod
from collections import defaultdict
from dataclasses import field
from hashlib import sha256
from typing import (
    Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
)


class UnparsedTreeError(Exception):
    pass


class TransformArityError(Exception):
    pass


class Literalisation(eqx.Module):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix']
    regex : str

    #@abstractmethod
    def parse_params(self, params):
        pass

    def parameterise(self, string):
        params = re.search(self.regex, string).groupdict()
        return self.parse_params(params)



class TransformPrimitive(eqx.Module):
    min_arity : int
    max_arity : int
    priority : int
    canonical_literal: Literalisation
    literals: Sequence[Literalisation]
    associative: bool = False
    commutative: bool = False

    @property
    def _key(self):
        canonical_literal = self.literals[0]
        return (
            canonical_literal.affix,
            canonical_literal.regex,
        )

    def __hash__(self):
        return hash(self._key)


class LeafTransform(TransformPrimitive):
    min_arity: int = 0
    max_arity: int = 0
    priority: float = float('nan')
    canonical_literal: Optional[Literalisation] = None
    literals: Sequence[Literalisation] = ()


import pandas as pd
class ConcatenateInfixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'infix'
    regex : str = r'\+'

class ConcatenatePrefixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'prefix'
    regex : str = r'\+\{[^\{^\}]+\}'

class ConcatenateNode(TransformPrimitive):
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

class BackwardDifferenceNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 2
    canonical_literal: Literalisation = BackwardDifferenceEnumPrefixLiteralisation()
    literals: Sequence[Literalisation] = field(default_factory = lambda: [
        BackwardDifferenceEnumPrefixLiteralisation(),
        BackwardDifferenceInclPrefixLiteralisation(),
        BackwardDifferenceRangePrefixLiteralisation(),
        BackwardDifferencePrefixLiteralisation(),
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
    regex : str = r'\^(?P<order>[0-9]+)'

    def parse_params(self, params):
        params['order'] = (int(params['order']),)
        return params

class PowerInclSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'suffix'
    regex : str = r'\^\^(?P<order>[0-9]+)'

    def parse_params(self, params):
        params['order'] = tuple(range(1, int(params['order']) + 1))
        return params

class PowerRangeSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'suffix'
    regex : str = r'\^(?P<order>[0-9]+\-[0-9]+)'

    def parse_params(self, params):
        lim = [int(z) for z in params['order'].split('-')]
        params['order'] = tuple(range(lim[0], lim[1] + 1))
        return params

class PowerEnumSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'suffix'
    regex : str = r'\^\{(?P<order>[0-9\,]+)\}'

    def parse_params(self, params):
        params['order'] = tuple(int(z) for z in params['order'].split(','))
        return params

class PowerNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 1
    canonical_literal: Literalisation = PowerEnumSuffixLiteralisation()
    literals: Sequence[Literalisation] = field(default_factory = lambda: [
        PowerEnumSuffixLiteralisation(),
        PowerInclSuffixLiteralisation(),
        PowerRangeSuffixLiteralisation(),
        PowerSuffixLiteralisation(),
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
            index = range(total_length + 1)
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

    def substitute(
        self, content, start, end=None, loc_type='index'
    ):
        if end is None:
            end = start + len(content)
        if loc_type == 'index':
            start = self.index[start]
            end = self.index[end]
        if start == end:
            return self
        new_content = (
            list(self.content[:start]) +
            list(content) +
            list(self.content[end:])
        )
        new_index = []
        for i in self.index:
            if i < start:
                new_index.append(i)
            elif i < end:
                new_index.append(start)
            else:
                new_index.append(i - (end - start) + 1)
        return IndexedNestedString(new_content, new_index)


class ChildToken:
    def __init__(self, hash, length=1):
        self.hash = hash
        self.length = length

    def __hash__(self):
        return hash(self.hash)

    def __repr__(self):
        return '⟨⟨{}⟩⟩'.format(self.hash[:4])

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter([self])


class TransformToken:
    def __init__(self, string, metadata=None, length=1):
        self.string = string
        self.metadata = metadata
        self.length = length

    def __hash__(self):
        return hash(id(self))

    def __repr__(self):
        return '⌈{}⌋'.format(self.string)

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter([self])


class SyntacticTree:
    def __init__(
        self,
        content: str,
        circumfix: Optional[Tuple[str, str]] = None,
        hashfn: Callable = sha256,
    ):
        self.hashfn = hashfn
        if isinstance(content, SyntacticTree):
            self.content = content.content
            self.children = content.children
            self.transform_root = content.transform_root
            self.circumfix = circumfix or content.circumfix
        else:
            self.content = IndexedNestedString(
                content=tuple(content),
                index=tuple(range(len(content) + 1)),
            )
            self.children = {}
            self.circumfix = circumfix or ('', '')
        self.transform_root = None

    @staticmethod
    def materialise_recursive(content, children):
        content = [
            c.string if isinstance(c, TransformToken) else c
            for c in content
        ]
        content = [
            children[c.hash].apply_circumfix(
                children[c.hash].materialise(recursive=True))
            if isinstance(c, ChildToken) else c
            for c in content
        ]
        return ''.join(content)

    @staticmethod
    def materialise_masked(content):
        content = [
            '▒' if isinstance(c, ChildToken)
            or isinstance(c, TransformToken) else c
            for c in content
        ]
        return ''.join(content)

    @staticmethod
    def materialise_repr(content):
        content = [
            repr(c) if isinstance(c, ChildToken)
            or isinstance(c, TransformToken) else c
            for c in content
        ]
        return ''.join(content)

    def materialise(self, repr=False, recursive=False):
        if repr:
            return self.materialise_repr(self.content)
        elif recursive:
            return self.materialise_recursive(self.content, self.children)
        else:
            return self.materialise_masked(self.content)

    def apply_circumfix(self, s):
        return self.circumfix[0] + s + self.circumfix[1]

    def _scan_content_and_embed_child(self, loc, child_id):
        content = self.content
        for start, end in loc:
            content = content.substitute(
                content=ChildToken(child_id, length=end - start),
                start=start,
                end=end,
                loc_type='index',
            )
        return content

    def _scan_content_and_embed_token(self, loc, content, metadata=None):
        for start, end in loc:
            new_token = TransformToken(
                ''.join(content.idx(slice(start, end))),
                metadata=metadata,
                length=end - start
            )
            content = content.substitute(
                content=new_token,
                start=start,
                end=end,
                loc_type='content',
            )
        return content

    def _child_carry_nested_content(
        self,
        loc,
        circumfix=('', ''),
        drop_circumfix=False,
    ):
        start, end = loc
        if drop_circumfix:
            start += 1
            end -= 1
        low = self.content.index[start]
        child_index = tuple([
            i - low for i in self.content.index[start:(end + 1)]
        ])
        child_content = self.content[start:end]
        child_str = IndexedNestedString(
            content=child_content,
            index=child_index,
        )
        if len(child_content) == 1:
            if isinstance(child_content[0], ChildToken):
                return # already nested
        return SyntacticTree.from_parsed(
            content=child_str,
            children=self.children,
            circumfix=circumfix,
            hashfn=self.hashfn,
        )

    def create_child(
        self,
        query: str,
        *,
        recursive: bool = False,
        nest: bool = True,
        drop_circumfix: bool = False,
    ) -> None:
        child_text = query
        circumfix = ('', '')
        if drop_circumfix:
            circumfix = (query[0], query[-1])
            child_text = query[1:-1]
        child_id = self.hashfn(child_text.encode('utf-8')).hexdigest()

        contentstr = self.materialise(recursive=recursive)
        loc = [m.span() for m in re.finditer(re.escape(query), contentstr)]
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

        self.content = content
        self.children[child_id] = child
        return child_id

    def create_token(
        self,
        queries: Sequence[str],
        metadata: Sequence[Dict[str, Any]] = None,
        recursive: bool = False,
    ):
        content = self.content
        for i, query in enumerate(queries):
            contentstr = self.materialise(recursive=recursive)
            loc = [m.span() for m in re.finditer(query, contentstr)]
            meta = metadata[i] if metadata else None
            content = self._scan_content_and_embed_token(
                loc, self.content, metadata=meta)
            self.content = content

    @staticmethod
    def find_present_children(content):
        return tuple([c.hash for c in content if isinstance(c, ChildToken)])

    @staticmethod
    def prune_children(content, children):
        present = SyntacticTree.find_present_children(content)
        return {c: v for c, v in children.items()
                if c in present}

    def prune(self):
        self.children = SyntacticTree.prune_children(
            self.content, self.children)
        for c in self.children.values():
            c.prune()

    @classmethod
    def from_parsed(
        cls,
        content,
        children=None,
        circumfix=('', ''),
        hashfn=sha256
    ):
        if children is None:
            children = {}
        children = cls.prune_children(content, children)
        tree = object.__new__(cls)
        tree.content = content
        tree.children = children
        tree.circumfix = circumfix
        tree.hashfn = hashfn
        tree.transform_root = None
        return tree

    def __repr__(self):
        return self.materialise(repr=True)


class TransformTree(eqx.Module):
    transform: TransformPrimitive
    parameters: Dict[str, Any]
    children: List['TransformTree']


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
                return -1, stack[:-1]
        return 0, stack


class TransformPool(eqx.Module):
    transforms: Sequence[TransformPrimitive]
    transform_to_priority: Dict[TransformPrimitive, int]
    priority_to_transform: List[Sequence[TransformPrimitive]]

    def __init__(self, *transforms) -> None:
        self.transforms = transforms
        self.transform_to_priority = {
            t: t.priority for t in self.transforms
        }
        priority_to_transform = defaultdict(list)
        for t in self.transforms:
            priority_to_transform[t.priority].append(t)
        self.priority_to_transform = [
            priority_to_transform[p] for p in sorted(priority_to_transform)
        ]

    @staticmethod
    def specify_transform_arg_search(affix, pointer):
        pfx = (pointer, [], 1)
        sfx = (pointer, [], -1)
        if affix == 'prefix':
            search = (pfx,)
        elif affix == 'suffix':
            search = (sfx,)
        elif affix == 'infix':
            search = (sfx, pfx,)
        elif affix == 'circumfix':
            raise NotImplementedError(
                'Circumfix parse is not yet implemented')
        else:
            raise ValueError(
                'Invalid affix')
        return search

    def search_for_transform_args(
        self,
        tree: SyntacticTree,
        pointer: int,
        incr: str,
        stack: List[str],
    ) -> Tuple[List[str], int]:
        pointer += incr
        accounted = set()
        while True:
            if pointer < 0 or pointer >= len(tree.content.index) - 1:
                return stack
            incr, stack, accounted = self.eval_stack(
                stack, tree.content[pointer], incr, accounted=accounted)
            if incr == 0:
                return stack
            pointer += incr

    def eval_stack(self, stack, char, incr, accounted):
        if isinstance(char, TransformToken):
            return 0, stack, accounted
        elif isinstance(char, ChildToken):
            if char in accounted:
                return incr, stack, accounted
            else:
                accounted = accounted.union({char})
        if incr == 1:
            stack = stack + [char]
        elif incr == -1:
            stack = [char] + stack
        return incr, stack, accounted

    def transform_args(
        self,
        tree: SyntacticTree,
        pointer: int,
        affix: str,
    ):
        search = self.specify_transform_arg_search(
            affix, pointer)
        args = []

        for s in search:
            pointer, stack, incr = s
            stack = self.search_for_transform_args(
                tree, pointer, incr, stack)
            if len(stack) > 1 or not isinstance(stack[0], ChildToken):
                stack = tree.materialise_recursive(stack, tree.children)
                tree.create_child(stack, recursive=True)
            args += stack
        return args

    @staticmethod
    def transform_expr(tree, token, affix, args):
        if affix == 'prefix':
            expr = tree.materialise_recursive((token, *args), tree.children)
        elif affix == 'suffix':
            expr = tree.materialise_recursive((*args, token), tree.children)
        if affix == 'infix':
            expr = tree.materialise_recursive(
                (args[0], token, args[1]),
                tree.children
            )
        return expr


class Grammar(eqx.Module):
    groupings: GroupingPool
    transforms: TransformPool
    whitespace: bool = False

    @staticmethod
    def delete_whitespace(s: str) -> str:
        whitespace = r'[\s]+'
        whitespace = re.compile(whitespace)
        return re.sub(whitespace, '', s)

    def parse_groups(
        self,
        s: Union[str, SyntacticTree]
    ) -> SyntacticTree:
        tree = SyntacticTree(s)
        stack = []
        pointer = 0
        for i, c in enumerate(tree.content):
            val, stack = self.groupings.eval_stack(stack, c)
            if val == 1 and len(stack) == 1:
                pointer = i
            elif val == -1 and not stack:
                s = tree.materialise_recursive(
                    tree.content, tree.children)
                tree.create_child(
                    s[pointer:i + 1],
                    drop_circumfix=True,
                    recursive=True,
                )
        return tree

    def tokenise_transforms(
        self,
        s: Union[str, SyntacticTree]
    ) -> SyntacticTree:
        tree = SyntacticTree(s)
        for transform in self.transforms.transforms:
            tree.create_token(
                [l.regex for l in transform.literals],
                metadata=[{
                    'transform': transform,
                    'literal': l,
                } for l in transform.literals],
            )
        return tree

    def make_transform_ledger(
        self,
        tree: SyntacticTree
    ) -> Dict[str, TransformPrimitive]:
        ledger = defaultdict(list)
        accounted = set()
        for i, ix in enumerate(tree.content.index[:-1]): # null terminator
            token = tree.content.content[ix]
            if isinstance(token, TransformToken) and token not in accounted:
                accounted = accounted.union({token})
                ledger[token.metadata['transform']].append(i)
        return ledger

    def parse_transforms(
        self,
        tree: SyntacticTree,
        ledger: Dict[str, TransformPrimitive]
    ):
        for priority in self.transforms.priority_to_transform:
            for transform in priority:
                idx = ledger.get(transform, [])
                for i in idx:
                    token = tree.content[i]
                    affix = token.metadata['literal'].affix
                    args = self.transforms.transform_args(
                        tree=tree,
                        pointer=i,
                        affix=affix,
                    )
                    expr = self.transforms.transform_expr(
                        tree=tree, token=token, affix=affix, args=args)
                    if (tree.materialise_recursive(expr, tree.children)
                        != tree.materialise(recursive=True)):
                        ch = tree.create_child(expr, recursive=True)
                        tree.children[ch].transform_root = token
                    else:
                        tree.transform_root = token
        return tree

    @staticmethod
    def recur_depth_first(tree, f, skip_transform_roots=False):
        if (not skip_transform_roots) or (tree.transform_root is None):
            tree = f(tree)
        tree.children = tree.prune_children(tree.content, tree.children)
        for key, child in tree.children.items():
            tree.children[key] = Grammar.recur_depth_first(
                tree=child,
                f=f,
                skip_transform_roots=skip_transform_roots
            )
        return tree

    def parse_level(
        self,
        tree: SyntacticTree,
    ) -> SyntacticTree:
        tree = self.parse_groups(tree)
        tree = self.tokenise_transforms(tree)
        ledger = self.make_transform_ledger(tree)
        tree = self.parse_transforms(tree, ledger)
        return tree

    def parse(
        self,
        s: str,
    ):
        if not self.whitespace:
            s = self.delete_whitespace(s)
        tree = SyntacticTree(s)
        tree = Grammar.recur_depth_first(
            tree=tree,
            f=self.parse_level,
            skip_transform_roots=True
        )
        return tree

    def verify_level(
        self,
        tree: SyntacticTree,
    ):
        if len(tree.children) != 0:
            raise UnparsedTreeError(
                f'Unparsed non-transform node {tree} '
                f'(full version: {tree.materialise(recursive=True)}) '
                f'has children: {[v for v in tree.children.values()]}. '
                'All nodes must be either transforms or terminal (leaves).'
            )
        return tree

    def verify_parse(
        self,
        tree: SyntacticTree,
    ) -> None:
        Grammar.recur_depth_first(
            tree=tree,
            f=self.verify_level,
            skip_transform_roots=True
        )

    def transform_impl(
        self,
        tree: SyntacticTree,
    ):
        if tree.transform_root:
            transform = tree.transform_root.metadata['transform']
            if len(tree.children) > transform.max_arity:
                raise TransformArityError(
                    f'Transform {transform} has arity '
                    f'{len(tree.children)} but max arity is '
                    f'{transform.max_arity}.'
                )
            elif len(tree.children) < transform.min_arity:
                raise TransformArityError(
                    f'Transform {transform} has arity '
                    f'{len(tree.children)} but min arity is '
                    f'{transform.min_arity}.'
                )
        else:
            transform = LeafTransform()

        children = list(tree.children.values())
        if transform.associative:
            new_children = children
            running_arity = len(children)
            num_children = len(children)
            num_kept_children = 0
            while running_arity < transform.max_arity:
                if num_children == num_kept_children:
                    break
                children = new_children
                num_children = len(children)
                new_children = []
                num_kept_children = 0
                for child in children:
                    if child.transform_root:
                        child_transform = child.transform_root.metadata['transform']
                        if child_transform == transform:
                            child_children = list(child.children.values())
                            new_children += child_children
                            running_arity += len(child_children) - 1
                        else:
                            new_children.append(child)
                            num_kept_children += 1
                    else:
                        new_children.append(child)
                        num_kept_children += 1

        if tree.transform_root:
            parser = tree.transform_root.metadata['literal']
            string = tree.transform_root.string
            parameters = parser.parameterise(string)
        else:
            parameters = {}

        return TransformTree(
            transform=transform,
            children=tuple(self.transform_impl(c) for c in children),
            parameters=parameters,
        )

    def transform(
        self,
        tree: SyntacticTree,
    ) -> TransformTree:
        self.verify_parse(tree)
        return self.transform_impl(tree)
