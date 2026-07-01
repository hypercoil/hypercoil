#!/usr/bin/env python3
"""Assert a docstring migration changed *only docstrings* — no code/behaviour.

Parses a source file at a git ref (default HEAD) and in the working tree, removes
every docstring node from both, and compares the resulting ASTs. If they are
identical the edit touched only docstrings (PASS); any other difference — a
changed signature, default, expression, import, ``__all__`` — fails the file.
Also confirms the working-tree file still compiles.

Docstring semantics deliberately ignore line numbers, so shifting code down by
editing a docstring is invisible; adding or removing a docstring is normalised
away by dropping the leading string-literal statement.

Exit 0 = only docstrings changed (and compiles); 1 = FAIL; 2 = usage/IO error.

Usage:
  check_docstring_migration.py <file.py> [--base HEAD]
"""
import argparse
import ast
import pathlib
import subprocess
import sys


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
    return tree


def code_ast(src):
    return ast.dump(strip_docstrings(ast.parse(src)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--base", default="HEAD", help="git ref to compare against")
    a = p.parse_args()

    path = pathlib.Path(a.file).resolve()
    if not path.is_file():
        print(f"FAIL: not a file: {path}", file=sys.stderr)
        return 2
    new_src = path.read_text()
    try:
        compile(new_src, str(path), "exec")
    except SyntaxError as e:
        print(f"FAIL: {path.name} does not compile: {e}")
        return 1

    root = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True, text=True)
    if root.returncode != 0:
        print(f"FAIL: not in a git repo: {path}", file=sys.stderr)
        return 2
    repo = pathlib.Path(root.stdout.strip())
    rel = path.relative_to(repo)
    show = subprocess.run(["git", "-C", str(repo), "show", f"{a.base}:{rel}"],
                          capture_output=True, text=True)
    if show.returncode != 0:
        print(f"FAIL: cannot read {a.base}:{rel} — {show.stderr.strip()}")
        return 1

    try:
        if code_ast(show.stdout) == code_ast(new_src):
            print(f"PASS: {rel} — only docstrings changed")
            return 0
    except SyntaxError as e:
        print(f"FAIL: base {rel} does not parse: {e}")
        return 1
    print(f"FAIL: {rel} — code (non-docstring) AST changed; migration must edit "
          "docstrings only")
    return 1


if __name__ == "__main__":
    sys.exit(main())
