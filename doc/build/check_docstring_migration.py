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


def _is_str_expr(stmt):
    """A bare string-literal statement — a docstring (leading) or a PEP 224
    attribute docstring. Always a no-op, so removing it can't change behaviour."""
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            node.body = [s for s in node.body if not _is_str_expr(s)] or [ast.Pass()]
    return tree


def code_ast(src):
    return ast.dump(strip_docstrings(ast.parse(src)))


# Control chars that signal LaTeX-in-a-non-raw-docstring corruption: \r->CR,
# \t->TAB, \b->BS, \f->FF, \v->VT, \a->BEL are *valid* escapes (so they never
# raise SyntaxWarning) yet turn e.g. `\rho`/`\top`/`\beta` into garbage. Newline
# (\n) is legitimate and excluded.
_CORRUPTING = {"\x07": r"\a", "\x08": r"\b", "\x09": r"\t", "\x0b": r"\v",
               "\x0c": r"\f", "\x0d": r"\r"}


def corrupted_docstrings(src):
    hits = []
    for node in ast.walk(ast.parse(src)):
        if _is_str_expr(node):   # docstrings AND PEP 224 attribute docstrings
            bad = {_CORRUPTING[c] for c in node.value.value if c in _CORRUPTING}
            if bad:
                hits.append(f"line {node.lineno} (likely un-raw LaTeX: "
                            f"{', '.join(sorted(bad))})")
    return hits


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
        code_changed = code_ast(show.stdout) != code_ast(new_src)
    except SyntaxError as e:
        print(f"FAIL: base {rel} does not parse: {e}")
        return 1
    if code_changed:
        print(f"FAIL: {rel} — code (non-docstring) AST changed; migration must "
              "edit docstrings only")
        return 1
    corrupt = corrupted_docstrings(new_src)
    if corrupt:
        print(f"FAIL: {rel} — control chars in docstrings (un-raw LaTeX escapes): "
              + "; ".join(corrupt))
        return 1
    print(f"PASS: {rel} — only docstrings changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
