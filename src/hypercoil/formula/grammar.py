# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Elementary formula grammar.

The grammar underlies confound modelling and parameter specification.
"""
from __future__ import annotations
import re
from abc import abstractmethod, abstractstaticmethod
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import equinox as eqx


class UnparsedTreeError(Exception):
    """Exception for a parse-tree node that is neither transform nor leaf."""
    pass


class TransformArityError(Exception):
    """Exception for a transform with too few or too many arguments."""
    pass


class Literalisation(eqx.Module):
    """
    Literal form of a transform in a formula.

    A subclass sets ``affix`` (``'prefix'``, ``'suffix'``, ``'infix'`` or
    ``'leaf'``) and ``regex``, whose named groups capture the parameters of the
    transform, and implements :meth:`parse_params`.
    """
    affix: Literal['prefix', 'suffix', 'infix', 'leaf']
    regex: str

    @abstractmethod
    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the groups captured by ``regex`` to transform parameters."""
        pass

    def parameterise(self, string: str) -> Dict[str, Any]:
        """Return the parameters of the literal ``string``."""
        params = re.search(self.regex, string).groupdict()
        return self.parse_params(params)


class TransformPrimitive(eqx.Module):
    """
    Transform of a formula grammar.

    A subclass implements ``ascend``, which takes the compiled function of each
    argument and the parameters of the literal, and returns the compiled
    function of the transform.

    Parameters
    ----------
    min_arity
        Minimum number of arguments.
    max_arity
        Maximum number of arguments.
    priority
        Parse priority; transforms of lower priority are parsed first.
    literals
        Literal forms of the transform. The first is canonical: it identifies
        the transform.
    associative
        The transform is associative.
    commutative
        The transform is commutative. A transform that is associative and
        commutative absorbs the arguments of nested instances of itself.
    """
    min_arity: int
    max_arity: int
    priority: int
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
    """Terminal of a parse tree, holding the leaf string ``leaf``."""
    leaf: Any = None
    min_arity: int = 0
    max_arity: int = 0
    priority: float = float('nan')
    literals: Sequence[Literalisation] = ()
    associative: bool = False
    commutative: bool = False


class LeafInterpreter(eqx.Module):
    """
    Interpreter that compiles each leaf of a formula into a function.

    Calling an interpreter on a leaf string returns a function; a subclass
    implements the call.
    """
    @abstractstaticmethod
    def __call__(leaf: Any) -> Callable:
        pass


class IndexedNestedString(eqx.Module):
    """
    Sequence of characters and tokens, indexed by source position.

    Each entry of ``index`` maps a character position of the source string to a
    position in ``content``, so that a token that replaces several characters
    keeps the positions of the characters around it.

    Parameters
    ----------
    content
        Characters and tokens.
    index
        Position in ``content`` of each source position, with a final
        terminator. Defaults to the identity over the total length.
    """
    content: Tuple[Any]
    index: Tuple[int]

    def __init__(
        self,
        content: Sequence[Any],
        index: Sequence[int] = None,
    ):
        if index is None:
            total_length = sum(len(c) for c in content)
            index = range(total_length + 1)
        self.content = tuple(content)
        self.index = tuple(index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, key: Union[int, slice]) -> Any:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return self.content[self.index[start] : self.index[stop] : step]
        else:
            return self.content[self.index[key]]

    def __iter__(self):
        return iter(self.content)

    def idx(self, key: Union[int, slice]) -> Any:
        """Return the content at content position ``key``."""
        return self.content.__getitem__(key)

    def substitute(
        self,
        content: Sequence[Any],
        start: int,
        end: Optional[int] = None,
        loc_type: Literal['index', 'content'] = 'index',
    ) -> 'IndexedNestedString':
        """
        Replace a span of the content.

        Parameters
        ----------
        content
            Replacement content.
        start
            Start of the span.
        end
            End of the span. Defaults to ``start + len(content)``.
        loc_type
            ``'index'`` reads ``start`` and ``end`` as source positions;
            ``'content'`` as content positions.

        Returns
        -------
        IndexedNestedString
            The string with the span replaced, or the string itself if the span
            is
            empty.
        """
        if end is None:
            end = start + len(content)
        if loc_type == 'index':
            start = self.index[start]
            end = self.index[end]
        if start == end:
            return self
        new_content = (
            list(self.content[:start])
            + list(content)
            + list(self.content[end:])
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
    """
    Placeholder for a child subtree in the content of a syntactic tree.

    Parameters
    ----------
    hash
        Hash of the child, which keys the child in the parent tree.
    length
        Number of source characters that the token replaces.
    """
    def __init__(
        self,
        hash: str,
        length: int = 1,
    ):
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
    """
    Token of a transform literal in the content of a syntactic tree.

    Parameters
    ----------
    string
        Literal string.
    metadata
        The transform and its literal form, keyed ``'transform'`` and
        ``'literal'``.
    length
        Number of source characters that the token replaces.
    """
    def __init__(
        self,
        string: str,
        metadata: Optional[Dict[str, Any]] = None,
        length: int = 1,
    ):
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
    """
    Parse tree of a formula string.

    Parameters
    ----------
    content
        Formula string, or a tree whose content and children are reused.
    circumfix
        Grouping delimiters around the content, such as ``('(', ')')``.
    hashfn
        Hash function that keys the children.
    """
    def __init__(
        self,
        content: Union[str, 'SyntacticTree'],
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
    def materialise_recursive(
        content: Sequence[Any],
        children: Dict[str, 'SyntacticTree'],
    ) -> str:
        """Return the string of ``content`` with each child expanded."""
        content = [
            c.string if isinstance(c, TransformToken) else c for c in content
        ]
        content = [
            children[c.hash].apply_circumfix(
                children[c.hash].materialise(recursive=True)
            )
            if isinstance(c, ChildToken)
            else c
            for c in content
        ]
        return ''.join(content)

    @staticmethod
    def materialise_masked(content: Sequence[Any]) -> str:
        """Return the string of ``content`` with each token masked."""
        content = [
            '▒'
            if isinstance(c, ChildToken) or isinstance(c, TransformToken)
            else c
            for c in content
        ]
        return ''.join(content)

    @staticmethod
    def materialise_repr(content: Sequence[Any]) -> str:
        """Return the string of ``content`` with each token by its repr."""
        content = [
            repr(c)
            if isinstance(c, ChildToken) or isinstance(c, TransformToken)
            else c
            for c in content
        ]
        return ''.join(content)

    def materialise(
        self,
        repr: bool = False,
        recursive: bool = False,
    ) -> str:
        """Return the string of the tree: masked, by repr, or expanded."""
        if repr:
            return self.materialise_repr(self.content)
        elif recursive:
            return self.materialise_recursive(self.content, self.children)
        else:
            return self.materialise_masked(self.content)

    def apply_circumfix(self, s: str) -> str:
        """Return ``s`` enclosed in the circumfix of the tree."""
        return self.circumfix[0] + s + self.circumfix[1]

    def _scan_content_and_embed_child(
        self,
        loc: Tuple[int, int],
        child_id: str,
    ) -> IndexedNestedString:
        content = self.content
        for start, end in loc:
            content = content.substitute(
                content=ChildToken(child_id, length=end - start),
                start=start,
                end=end,
                loc_type='index',
            )
        return content

    def _scan_content_and_embed_token(
        self,
        loc: Tuple[int, int],
        content: IndexedNestedString,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IndexedNestedString:
        for start, end in loc:
            new_token = TransformToken(
                ''.join(content.idx(slice(start, end))),
                metadata=metadata,
                length=end - start,
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
        loc: Tuple[int, int],
        circumfix: Tuple[str, str] = ('', ''),
        drop_circumfix: bool = False,
    ) -> 'SyntacticTree':
        start, end = loc
        if drop_circumfix:
            start += 1
            end -= 1
        low = self.content.index[start]
        child_index = tuple(
            [i - low for i in self.content.index[start : (end + 1)]]
        )
        child_content = self.content[start:end]
        child_str = IndexedNestedString(
            content=child_content,
            index=child_index,
        )
        if len(child_content) == 1:
            if isinstance(child_content[0], ChildToken):
                return  # already nested
        return SyntacticTree.from_parsed(
            content=child_str,
            children=deepcopy(self.children),
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
    ) -> str:
        """
        Move each occurrence of a substring into a child subtree.

        Parameters
        ----------
        query
            Substring to move.
        recursive
            Match ``query`` against the fully expanded string.
        nest
            Carry the tokens of the matched span into the child; otherwise the
            child is parsed from the text of ``query``.
        drop_circumfix
            Strip the first and last characters of ``query`` into the circumfix
            of the child.

        Returns
        -------
        str
            Hash of the child.
        """
        child_text = query
        circumfix = ('', '')
        if drop_circumfix:
            circumfix = (query[0], query[-1])
            child_text = query[1:-1]
        child_id = self.hashfn(child_text.encode('utf-8')).hexdigest()
        if child_id in self.children:
            return child_id

        contentstr = self.materialise(recursive=recursive)
        loc = [m.span() for m in re.finditer(re.escape(query), contentstr)]
        content = self._scan_content_and_embed_child(loc, child_id)

        if nest:
            child = self._child_carry_nested_content(
                loc[0],
                circumfix,
                drop_circumfix,
            )
        else:
            child = SyntacticTree(
                child_text,
                circumfix=circumfix,
                hashfn=self.hashfn,
            )

        self.content = content
        self.children[child_id] = child
        return child_id

    def create_token(
        self,
        queries: Sequence[str],
        metadata: Sequence[Dict[str, Any]] = None,
        recursive: bool = False,
    ) -> None:
        """
        Replace the first match of each pattern by a transform token.

        Parameters
        ----------
        queries
            Regular expressions to match.
        metadata
            Metadata of the token for each pattern.
        recursive
            Match against the fully expanded string.
        """
        content = self.content
        for i, query in enumerate(queries):
            contentstr = self.materialise(recursive=recursive)
            result = re.search(query, contentstr)
            if result is None:
                continue
            loc = [result.span()]
            meta = metadata[i] if metadata else None
            content = self._scan_content_and_embed_token(
                loc, self.content, metadata=meta
            )
            self.content = content

    @staticmethod
    def find_present_children(content: IndexedNestedString) -> Tuple[str]:
        """Return the hashes of the children referenced in ``content``."""
        return tuple([c.hash for c in content if isinstance(c, ChildToken)])

    @staticmethod
    def prune_children(
        content: IndexedNestedString,
        children: Dict[str, 'SyntacticTree'],
    ) -> Dict[str, 'SyntacticTree']:
        """Return the entries of ``children`` referenced in ``content``."""
        present = SyntacticTree.find_present_children(content)
        return {c: v for c, v in children.items() if c in present}

    def prune(self):
        """Remove unreferenced children, recursively."""
        self.children = SyntacticTree.prune_children(
            self.content, self.children
        )
        for c in self.children.values():
            c.prune()

    @classmethod
    def from_parsed(
        cls,
        content: IndexedNestedString,
        children: Optional[Dict[str, 'SyntacticTree']] = None,
        circumfix: Tuple[str, str] = ('', ''),
        hashfn: Callable = sha256,
    ) -> 'SyntacticTree':
        """
        Create a tree from parsed content without parsing it.

        Parameters
        ----------
        content
            Parsed content.
        children
            Children referenced in ``content``; others are dropped.
        circumfix
            Grouping delimiters around the content.
        hashfn
            Hash function that keys the children.

        Returns
        -------
        SyntacticTree
            The tree.
        """
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

    def __repr__(self) -> str:
        return self.materialise(repr=True)


class TransformTree(eqx.Module):
    """
    Tree of transforms produced from a parse tree.

    Parameters
    ----------
    transform
        Transform at the root.
    parameters
        Parameters of the root transform.
    children
        Subtrees of the arguments.
    """
    transform: TransformPrimitive
    parameters: Dict[str, Any]
    children: List['TransformTree']

    def descend(
        self,
        interpreter: Optional[LeafInterpreter] = None,
    ) -> Callable:
        """Compile the tree, with ``interpreter`` for the leaves."""
        if isinstance(self.transform, LeafTransform):
            return interpreter(self.transform.leaf)
        functions = [c.descend(interpreter=interpreter) for c in self.children]
        return self.transform.ascend(*functions, **self.parameters)

    def compile(
        self,
        interpreter: Optional[LeafInterpreter] = None,
        root_transform: Optional[TransformPrimitive] = None,
    ) -> Callable:
        """Compile the tree, then apply ``root_transform`` if it is given."""
        if root_transform is None:
            return self.descend(interpreter=interpreter)
        return root_transform(self.descend(interpreter=interpreter))


class Grouping(eqx.Module):
    """Pair of grouping delimiters, ``open`` and ``close``."""
    open: str
    close: str


class GroupingPool(eqx.Module):
    """Set of grouping delimiters of a grammar."""
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
        """Return the opening and the closing delimiters."""
        open, close = zip(*[(g.open, g.close) for g in self.groupings])
        return tuple(open), tuple(close)

    def eval_stack(self, stack: List[str], char: str) -> Tuple[int, Sequence]:
        """
        Update the grouping stack with one character.

        Parameters
        ----------
        stack
            Open delimiters, innermost last.
        char
            Character.

        Returns
        -------
        tuple
            1 if ``char`` opens a group, -1 if it closes the innermost group,
            and
            0 otherwise; then the updated stack.

        Raises
        ------
        ValueError
            If ``char`` closes a group and ``stack`` is empty.
        """
        if char in self.open:
            return 1, stack + [char]
        elif char in self.close:
            if not stack:
                raise ValueError('Unmatched closing bracket')
            if stack[-1] == self.close_to_open[char]:
                return -1, stack[:-1]
        return 0, stack


class TransformPool(eqx.Module):
    """Set of the transforms of a grammar, grouped by priority."""
    transforms: Sequence[TransformPrimitive]
    transform_to_priority: Dict[TransformPrimitive, int]
    priority_to_transform: List[Sequence[TransformPrimitive]]

    def __init__(self, *transforms) -> None:
        self.transforms = transforms
        self.transform_to_priority = {t: t.priority for t in self.transforms}
        priority_to_transform = defaultdict(list)
        for t in self.transforms:
            priority_to_transform[t.priority].append(t)
        self.priority_to_transform = [
            priority_to_transform[p] for p in sorted(priority_to_transform)
        ]

    @staticmethod
    def specify_transform_arg_search(
        affix: Literal['prefix', 'suffix', 'infix'],
        pointer: int,
    ) -> Tuple[Tuple[int, Sequence, int], ...]:
        """Return the start, stack and direction of each argument search."""
        pfx = (pointer, [], 1)
        sfx = (pointer, [], -1)
        if affix == 'prefix':
            search = (pfx,)
        elif affix == 'suffix':
            search = (sfx,)
        elif affix == 'infix':
            search = (
                sfx,
                pfx,
            )
        elif affix == 'leaf':
            search = ()
        else:
            raise ValueError('Invalid affix')
        return search

    def search_for_transform_args(
        self,
        tree: 'SyntacticTree',
        pointer: int,
        incr: str,
        stack: List[str],
    ) -> Tuple[List[str], int]:
        """
        Collect the content of one argument of a transform.

        Parameters
        ----------
        tree
            Syntactic tree.
        pointer
            Content position of the transform token.
        incr
            Direction of the search: 1 forward, -1 backward.
        stack
            Content collected so far.

        Returns
        -------
        list
            Content of the argument; the search stops at another transform
            token
            or at the end of the content.
        """
        accounted = set(tree.content[pointer])
        pointer += incr
        while True:
            if pointer < 0 or pointer >= len(tree.content.index) - 1:
                return stack
            incr, stack, accounted = self.eval_stack(
                stack, tree.content[pointer], incr, accounted=accounted
            )
            if incr == 0:
                return stack
            pointer += incr

    def eval_stack(
        self,
        stack: List[str],
        char: Union[str, 'ChildToken', 'TransformToken'],
        incr: int,
        accounted: Set,
    ) -> Tuple[int, List[str], Set]:
        """
        Add one item of content to an argument stack.

        Parameters
        ----------
        stack
            Content collected so far.
        char
            Character or token.
        incr
            Direction of the search: 1 forward, -1 backward.
        accounted
            Tokens already in the argument.

        Returns
        -------
        tuple
            Direction of the next step (0 to stop), the stack and the updated
            ``accounted``.
        """
        if isinstance(char, TransformToken):
            if char not in accounted:
                return 0, stack, accounted
            else:
                return incr, stack, accounted
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
    ) -> List:
        """
        Collect the arguments of a transform token, as child subtrees.

        Parameters
        ----------
        tree
            Syntactic tree. Arguments of more than one item become children.
        pointer
            Content position of the transform token.
        affix
            Affix of the literal.

        Returns
        -------
        list
            Content of each argument.
        """
        search = self.specify_transform_arg_search(affix, pointer)
        args = []

        for s in search:
            pointer, stack, incr = s
            stack = self.search_for_transform_args(tree, pointer, incr, stack)
            if len(stack) == 0:
                pass
            elif len(stack) > 1 or not isinstance(stack[0], ChildToken):
                stack = tree.materialise_recursive(stack, tree.children)
                tree.create_child(stack, recursive=True)
            args += [stack]
        return args

    @staticmethod
    def transform_expr(
        tree: 'SyntacticTree',
        token: 'TransformToken',
        affix: Literal['prefix', 'suffix', 'infix'],
        args: Sequence,
    ) -> str:
        """Return the string of a transform token with its arguments."""
        if affix == 'prefix':
            expr = tree.materialise_recursive((token, *args[0]), tree.children)
        elif affix == 'suffix':
            expr = tree.materialise_recursive((*args[0], token), tree.children)
        elif affix == 'infix':
            expr = tree.materialise_recursive(
                (*args[0], token, *args[1]),
                tree.children,
            )
        elif affix == 'leaf':
            expr = tree.materialise_recursive((token,), tree.children)
        return expr


class Grammar(eqx.Module):
    """
    Formula grammar: groupings, transforms and leaf interpretation.

    :meth:`compile` parses a formula, builds its transform tree and compiles it
    into a function.

    Parameters
    ----------
    groupings
        Grouping delimiters.
    transforms
        Transforms.
    whitespace
        Whitespace is significant: runs of whitespace are reduced to one space.
        Otherwise whitespace is deleted.
    shorthand
        Regular-expression substitutions applied to a formula before parsing.
    default_interpreter
        Leaf interpreter of :meth:`compile`.
    default_root_transform
        Transform applied to the compiled function by :meth:`compile`.
    """
    groupings: GroupingPool
    transforms: TransformPool
    whitespace: bool = False
    shorthand: Optional[Dict[str, str]] = None
    default_interpreter: Optional[LeafInterpreter] = None
    default_root_transform: Optional[TransformPrimitive] = None

    @staticmethod
    def delete_whitespace(s: str) -> str:
        """Return ``s`` without whitespace."""
        whitespace = r'[\s]+'
        whitespace = re.compile(whitespace)
        return re.sub(whitespace, '', s)

    @staticmethod
    def sanitise_whitespace(s: str) -> str:
        """Return ``s`` stripped, with whitespace runs as single spaces."""
        whitespace = r'[\s]+'
        whitespace = re.compile(whitespace)
        return re.sub(whitespace, ' ', s.strip())

    def expand_shorthand(self, s: str) -> str:
        """Return ``s`` with the shorthand substitutions applied."""
        if not self.shorthand:
            return s
        for k, v in self.shorthand.items():
            s = re.sub(k, v, s)
        return s

    def parse_groups(
        self,
        s: Union[str, SyntacticTree],
    ) -> SyntacticTree:
        """Move each outermost group of ``s`` into a child subtree."""
        tree = SyntacticTree(s)
        stack = []
        pointer = 0
        for i, c in enumerate(tree.content):
            val, stack = self.groupings.eval_stack(stack, c)
            if val == 1 and len(stack) == 1:
                pointer = i
            elif val == -1 and not stack:
                s = tree.materialise_recursive(tree.content, tree.children)
                tree.create_child(
                    s[pointer : i + 1],
                    drop_circumfix=True,
                    recursive=True,
                )
        return tree

    def tokenise_transforms(
        self,
        s: Union[str, SyntacticTree],
    ) -> SyntacticTree:
        """Replace each transform literal of ``s`` by a transform token."""
        tree = SyntacticTree(s)
        for transform in self.transforms.transforms:
            content = tree.content.content
            while True:
                tree.create_token(
                    [literal.regex for literal in transform.literals],
                    metadata=[
                        {
                            'transform': transform,
                            'literal': literal,
                        }
                        for literal in transform.literals
                    ],
                )
                if content == tree.content.content:
                    break
                else:
                    content = tree.content.content
        return tree

    def make_transform_ledger(
        self,
        tree: SyntacticTree,
    ) -> Dict[str, TransformPrimitive]:
        """Return the content positions of the tokens of each transform."""
        ledger = defaultdict(list)
        accounted = set()
        for i, ix in enumerate(tree.content.index[:-1]):  # null terminator
            token = tree.content.content[ix]
            if isinstance(token, TransformToken) and token not in accounted:
                accounted = accounted.union({token})
                ledger[token.metadata['transform']].append(i)
        return ledger

    def parse_transforms(
        self,
        tree: SyntacticTree,
        ledger: Dict[str, TransformPrimitive],
        parse_order: Literal['left', 'right'] = 'right',
    ) -> SyntacticTree:
        """
        Move the arguments of each transform of a tree into child subtrees.

        Parameters
        ----------
        tree
            Tokenised tree.
        ledger
            Content positions of the tokens of each transform, from
            :meth:`make_transform_ledger`.
        parse_order
            ``'right'`` processes the tokens of equal priority in order of
            position; ``'left'`` in reverse order.

        Returns
        -------
        SyntacticTree
            The tree, with the transform at the root of each subtree recorded.
        """
        for priority in self.transforms.priority_to_transform:
            idx = []
            for transform in priority:
                idx += ledger.get(transform, [])
            idx = sorted(idx, reverse=(parse_order == 'left'))
            for i in idx:
                token = tree.content[i]
                affix = token.metadata['literal'].affix
                args = self.transforms.transform_args(
                    tree=tree,
                    pointer=i,
                    affix=affix,
                )
                expr = self.transforms.transform_expr(
                    tree=tree, token=token, affix=affix, args=args
                )
                if tree.materialise_recursive(
                    expr, tree.children
                ) != tree.materialise(recursive=True):
                    ch = tree.create_child(expr, recursive=True)
                    tree.children[ch].transform_root = token
                else:
                    tree.transform_root = token
        return tree

    @staticmethod
    def recur_depth_first(
        tree: 'SyntacticTree',
        f: Callable,
        skip_transform_roots: bool = False,
    ) -> 'SyntacticTree':
        """
        Apply a function to a tree and its descendants, depth first.

        Parameters
        ----------
        tree
            Tree.
        f
            Function of a tree that returns a tree.
        skip_transform_roots
            Skip the trees whose transform root is set.

        Returns
        -------
        SyntacticTree
            The transformed tree, with unreferenced children pruned.
        """
        if (not skip_transform_roots) or (tree.transform_root is None):
            tree = f(tree)
        tree.children = tree.prune_children(tree.content, tree.children)
        for key, child in tree.children.items():
            tree.children[key] = Grammar.recur_depth_first(
                tree=child,
                f=f,
                skip_transform_roots=skip_transform_roots,
            )
        return tree

    def parse_level(
        self,
        tree: SyntacticTree,
    ) -> SyntacticTree:
        """Parse the groups and transforms of one level of ``tree``."""
        tree = self.parse_groups(tree)
        tree = self.tokenise_transforms(tree)
        ledger = self.make_transform_ledger(tree)
        tree = self.parse_transforms(tree, ledger)
        return tree

    def parse(
        self,
        s: str,
    ) -> SyntacticTree:
        """Parse the formula ``s`` into a syntactic tree."""
        s = self.expand_shorthand(s)
        if not self.whitespace:
            s = self.delete_whitespace(s)
        else:
            s = self.sanitise_whitespace(s)
        tree = SyntacticTree(s)
        tree = Grammar.recur_depth_first(
            tree=tree,
            f=self.parse_level,
            skip_transform_roots=True,
        )
        return tree

    def verify_level(
        self,
        tree: SyntacticTree,
    ) -> SyntacticTree:
        """
        Check that each node without a transform root is a leaf.

        Raises
        ------
        UnparsedTreeError
            If the node has children.
        """
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
        """
        Check that every node of a parse tree is a transform or a leaf.

        Raises
        ------
        UnparsedTreeError
            If a node is neither.
        """
        Grammar.recur_depth_first(
            tree=tree,
            f=self.verify_level,
            skip_transform_roots=True,
        )

    def transform_impl(
        self,
        tree: SyntacticTree,
    ) -> TransformTree:
        """
        Build the transform tree of a parse tree.

        Raises
        ------
        TransformArityError
            If a transform has fewer than ``min_arity`` or more than
            ``max_arity``
            arguments.
        """
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
            transform = LeafTransform(leaf=tree.materialise())

        children = [
            tree.children[token.hash]
            for token in tree.content
            if isinstance(token, ChildToken)
        ]
        # TODO: check that children are in the right order for noncommutative
        #       transforms when there are nested groups
        if transform.associative and transform.commutative:
            new_children = children
            running_arity = len(children)
            num_children = len(children)
            num_kept_children = 0
            # TODO: Careful! We can exceed the max arity here! We should check
            #       that the arity is correct with a lookahead before adding
            #       children. We haven't run into this since in practice all
            #       of our transforms have either max_arity=1, max_arity=2, or
            #       max_arity=inf.
            while running_arity < transform.max_arity:
                if num_children == num_kept_children:
                    break
                children = new_children
                num_children = len(children)
                new_children = []
                num_kept_children = 0
                for child in children:
                    if child.transform_root:
                        # fmt: off
                        child_transform = (
                            child.transform_root.metadata["transform"])
                        # fmt: on
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

    @staticmethod
    def annotate_leaf_count(tree):
        """Add ``num_leaves`` to the parameters of each node of ``tree``."""
        def leaf_tree(tree):
            if len(tree.children) == 0:
                return (1, None)
            else:
                out = tuple(leaf_tree(c) for c in tree.children)
                acc, _ = zip(*out)
                return sum(acc), out

        def tree_with_leaves(tree, leaves):
            if len(tree.children) == 0:
                return TransformTree(
                    transform=tree.transform,
                    children=tree.children,
                    parameters={**tree.parameters, 'num_leaves': 1},
                )
            else:
                return TransformTree(
                    transform=tree.transform,
                    children=tuple(
                        tree_with_leaves(ch, lf)
                        for ch, lf in zip(tree.children, leaves[1])
                    ),
                    parameters={**tree.parameters, 'num_leaves': leaves[0]},
                )

        leaves = leaf_tree(tree)
        return tree_with_leaves(tree, leaves)

    def transform(
        self,
        tree: SyntacticTree,
    ) -> TransformTree:
        """
        Build the transform tree of a parse tree, with leaf counts.

        Raises
        ------
        UnparsedTreeError
            If a node of ``tree`` is neither a transform nor a leaf.
        TransformArityError
            If a transform has too few or too many arguments.
        """
        self.verify_parse(tree)
        tree = self.transform_impl(tree)
        return Grammar.annotate_leaf_count(tree)

    def compile(
        self,
        s: str,
        interpreter: Optional[LeafInterpreter] = None,
        root_transform: Optional[TransformPrimitive] = None,
    ) -> Callable:
        """
        Compile a formula into a function.

        Parameters
        ----------
        s
            Formula.
        interpreter
            Leaf interpreter. Defaults to ``default_interpreter``.
        root_transform
            Transform applied to the compiled function. Defaults to
            ``default_root_transform``.

        Returns
        -------
        Callable
            Compiled function.

        Raises
        ------
        UnparsedTreeError
            If the formula does not parse into transforms and leaves.
        TransformArityError
            If a transform has too few or too many arguments.
        """
        if interpreter is None:
            interpreter = self.default_interpreter
        if root_transform is None:
            root_transform = self.default_root_transform
        tree = self.parse(s)
        transform = self.transform(tree)
        return transform.compile(
            interpreter=interpreter,
            root_transform=root_transform,
        )
