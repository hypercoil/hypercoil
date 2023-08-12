# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Grammar
~~~~~~~
Elementary formula grammar. For use with the confound modelling and parameter
specification subsystems.
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
    pass


class TransformArityError(Exception):
    pass


class Literalisation(eqx.Module):
    affix: Literal["prefix", "suffix", "infix", "leaf"]
    regex: str

    @abstractmethod
    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def parameterise(self, string: str) -> Dict[str, Any]:
        params = re.search(self.regex, string).groupdict()
        return self.parse_params(params)


class TransformPrimitive(eqx.Module):
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
    leaf: Any = None
    min_arity: int = 0
    max_arity: int = 0
    priority: float = float("nan")
    literals: Sequence[Literalisation] = ()
    associative: bool = False
    commutative: bool = False


class LeafInterpreter(eqx.Module):
    @abstractstaticmethod
    def __call__(leaf: Any) -> Callable:
        pass


class IndexedNestedString(eqx.Module):
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
        return self.content.__getitem__(key)

    def substitute(
        self,
        content: Sequence[Any],
        start: int,
        end: Optional[int] = None,
        loc_type: Literal["index", "content"] = "index",
    ) -> "IndexedNestedString":
        if end is None:
            end = start + len(content)
        if loc_type == "index":
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
        return "⟨⟨{}⟩⟩".format(self.hash[:4])

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter([self])


class TransformToken:
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
        return "⌈{}⌋".format(self.string)

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter([self])


class SyntacticTree:
    def __init__(
        self,
        content: Union[str, "SyntacticTree"],
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
            self.circumfix = circumfix or ("", "")
        self.transform_root = None

    @staticmethod
    def materialise_recursive(
        content: Sequence[Any],
        children: Dict[str, "SyntacticTree"],
    ) -> str:
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
        return "".join(content)

    @staticmethod
    def materialise_masked(content: Sequence[Any]) -> str:
        content = [
            "▒"
            if isinstance(c, ChildToken) or isinstance(c, TransformToken)
            else c
            for c in content
        ]
        return "".join(content)

    @staticmethod
    def materialise_repr(content: Sequence[Any]) -> str:
        content = [
            repr(c)
            if isinstance(c, ChildToken) or isinstance(c, TransformToken)
            else c
            for c in content
        ]
        return "".join(content)

    def materialise(
        self,
        repr: bool = False,
        recursive: bool = False,
    ) -> str:
        if repr:
            return self.materialise_repr(self.content)
        elif recursive:
            return self.materialise_recursive(self.content, self.children)
        else:
            return self.materialise_masked(self.content)

    def apply_circumfix(self, s: str) -> str:
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
                loc_type="index",
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
                "".join(content.idx(slice(start, end))),
                metadata=metadata,
                length=end - start,
            )
            content = content.substitute(
                content=new_token,
                start=start,
                end=end,
                loc_type="content",
            )
        return content

    def _child_carry_nested_content(
        self,
        loc: Tuple[int, int],
        circumfix: Tuple[str, str] = ("", ""),
        drop_circumfix: bool = False,
    ) -> "SyntacticTree":
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
        child_text = query
        circumfix = ("", "")
        if drop_circumfix:
            circumfix = (query[0], query[-1])
            child_text = query[1:-1]
        child_id = self.hashfn(child_text.encode("utf-8")).hexdigest()
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
        return tuple([c.hash for c in content if isinstance(c, ChildToken)])

    @staticmethod
    def prune_children(
        content: IndexedNestedString,
        children: Dict[str, "SyntacticTree"],
    ) -> Dict[str, "SyntacticTree"]:
        present = SyntacticTree.find_present_children(content)
        return {c: v for c, v in children.items() if c in present}

    def prune(self):
        self.children = SyntacticTree.prune_children(
            self.content, self.children
        )
        for c in self.children.values():
            c.prune()

    @classmethod
    def from_parsed(
        cls,
        content: IndexedNestedString,
        children: Optional[Dict[str, "SyntacticTree"]] = None,
        circumfix: Tuple[str, str] = ("", ""),
        hashfn: Callable = sha256,
    ) -> "SyntacticTree":
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
    transform: TransformPrimitive
    parameters: Dict[str, Any]
    children: List["TransformTree"]

    def descend(
        self,
        interpreter: Optional[LeafInterpreter] = None,
    ) -> Callable:
        if isinstance(self.transform, LeafTransform):
            return interpreter(self.transform.leaf)
        functions = [c.descend(interpreter=interpreter) for c in self.children]
        return self.transform.ascend(*functions, **self.parameters)

    def compile(
        self,
        interpreter: Optional[LeafInterpreter] = None,
        root_transform: Optional[TransformPrimitive] = None,
    ) -> Callable:
        if root_transform is None:
            return self.descend(interpreter=interpreter)
        return root_transform(self.descend(interpreter=interpreter))


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

    def eval_stack(self, stack: List[str], char: str) -> Tuple[int, Sequence]:
        if char in self.open:
            return 1, stack + [char]
        elif char in self.close:
            if not stack:
                raise ValueError("Unmatched closing bracket")
            if stack[-1] == self.close_to_open[char]:
                return -1, stack[:-1]
        return 0, stack


class TransformPool(eqx.Module):
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
        affix: Literal["prefix", "suffix", "infix"],
        pointer: int,
    ) -> Tuple[Tuple[int, Sequence, int], ...]:
        pfx = (pointer, [], 1)
        sfx = (pointer, [], -1)
        if affix == "prefix":
            search = (pfx,)
        elif affix == "suffix":
            search = (sfx,)
        elif affix == "infix":
            search = (
                sfx,
                pfx,
            )
        elif affix == "leaf":
            search = ()
        else:
            raise ValueError("Invalid affix")
        return search

    def search_for_transform_args(
        self,
        tree: "SyntacticTree",
        pointer: int,
        incr: str,
        stack: List[str],
    ) -> Tuple[List[str], int]:
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
        char: Union[str, "ChildToken", "TransformToken"],
        incr: int,
        accounted: Set,
    ) -> Tuple[int, List[str], Set]:
        if isinstance(char, TransformToken):
            if not char in accounted:
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
        tree: "SyntacticTree",
        token: "TransformToken",
        affix: Literal["prefix", "suffix", "infix"],
        args: Sequence,
    ) -> str:
        if affix == "prefix":
            expr = tree.materialise_recursive((token, *args[0]), tree.children)
        elif affix == "suffix":
            expr = tree.materialise_recursive((*args[0], token), tree.children)
        elif affix == "infix":
            expr = tree.materialise_recursive(
                (*args[0], token, *args[1]),
                tree.children,
            )
        elif affix == "leaf":
            expr = tree.materialise_recursive((token,), tree.children)
        return expr


class Grammar(eqx.Module):
    groupings: GroupingPool
    transforms: TransformPool
    whitespace: bool = False
    shorthand: Optional[Dict[str, str]] = None
    default_interpreter: Optional[LeafInterpreter] = None
    default_root_transform: Optional[TransformPrimitive] = None

    @staticmethod
    def delete_whitespace(s: str) -> str:
        whitespace = r"[\s]+"
        whitespace = re.compile(whitespace)
        return re.sub(whitespace, "", s)

    @staticmethod
    def sanitise_whitespace(s: str) -> str:
        whitespace = r"[\s]+"
        whitespace = re.compile(whitespace)
        return re.sub(whitespace, " ", s.strip())

    def expand_shorthand(self, s: str) -> str:
        if not self.shorthand:
            return s
        for k, v in self.shorthand.items():
            s = re.sub(k, v, s)
        return s

    def parse_groups(
        self,
        s: Union[str, SyntacticTree],
    ) -> SyntacticTree:
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
        tree = SyntacticTree(s)
        for transform in self.transforms.transforms:
            content = tree.content.content
            while True:
                tree.create_token(
                    [l.regex for l in transform.literals],
                    metadata=[
                        {
                            "transform": transform,
                            "literal": l,
                        }
                        for l in transform.literals
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
        ledger = defaultdict(list)
        accounted = set()
        for i, ix in enumerate(tree.content.index[:-1]):  # null terminator
            token = tree.content.content[ix]
            if isinstance(token, TransformToken) and token not in accounted:
                accounted = accounted.union({token})
                ledger[token.metadata["transform"]].append(i)
        return ledger

    def parse_transforms(
        self,
        tree: SyntacticTree,
        ledger: Dict[str, TransformPrimitive],
        parse_order: Literal["left", "right"] = "right",
    ) -> SyntacticTree:
        for priority in self.transforms.priority_to_transform:
            idx = []
            for transform in priority:
                idx += ledger.get(transform, [])
            idx = sorted(idx, reverse=(parse_order == "left"))
            for i in idx:
                token = tree.content[i]
                affix = token.metadata["literal"].affix
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
        tree: "SyntacticTree",
        f: Callable,
        skip_transform_roots: bool = False,
    ) -> "SyntacticTree":
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
        tree = self.parse_groups(tree)
        tree = self.tokenise_transforms(tree)
        ledger = self.make_transform_ledger(tree)
        tree = self.parse_transforms(tree, ledger)
        return tree

    def parse(
        self,
        s: str,
    ) -> SyntacticTree:
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
        if len(tree.children) != 0:
            raise UnparsedTreeError(
                f"Unparsed non-transform node {tree} "
                f"(full version: {tree.materialise(recursive=True)}) "
                f"has children: {[v for v in tree.children.values()]}. "
                "All nodes must be either transforms or terminal (leaves)."
            )
        return tree

    def verify_parse(
        self,
        tree: SyntacticTree,
    ) -> None:
        Grammar.recur_depth_first(
            tree=tree,
            f=self.verify_level,
            skip_transform_roots=True,
        )

    def transform_impl(
        self,
        tree: SyntacticTree,
    ) -> TransformTree:
        if tree.transform_root:
            transform = tree.transform_root.metadata["transform"]
            if len(tree.children) > transform.max_arity:
                raise TransformArityError(
                    f"Transform {transform} has arity "
                    f"{len(tree.children)} but max arity is "
                    f"{transform.max_arity}."
                )
            elif len(tree.children) < transform.min_arity:
                raise TransformArityError(
                    f"Transform {transform} has arity "
                    f"{len(tree.children)} but min arity is "
                    f"{transform.min_arity}."
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
            parser = tree.transform_root.metadata["literal"]
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
                    parameters={**tree.parameters, "num_leaves": 1},
                )
            else:
                return TransformTree(
                    transform=tree.transform,
                    children=tuple(
                        tree_with_leaves(c, l)
                        for c, l in zip(tree.children, leaves[1])
                    ),
                    parameters={**tree.parameters, "num_leaves": leaves[0]},
                )

        leaves = leaf_tree(tree)
        return tree_with_leaves(tree, leaves)

    def transform(
        self,
        tree: SyntacticTree,
    ) -> TransformTree:
        self.verify_parse(tree)
        tree = self.transform_impl(tree)
        return Grammar.annotate_leaf_count(tree)

    def compile(
        self,
        s: str,
        interpreter: Optional[LeafInterpreter] = None,
        root_transform: Optional[TransformPrimitive] = None,
    ) -> Callable:
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
