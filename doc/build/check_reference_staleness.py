#!/usr/bin/env python3
"""Flag API-reference stubs whose underlying signatures have drifted (PLAN §3.2).

Recomputes each generated stub's signature hash from the current source (using
the SAME logic as gen_reference_stubs, so the two can never disagree) and compares
it to the `signature-hash` recorded in the stub's front matter. A mismatch means
the code changed since the stub was written/last blessed — surfaced for human
attention. This does NOT auto-edit prose (SPEC §7.7: no freshness guarantee, only
a drift signal).

  * default: report drift; exit non-zero if any (CI gate).
  * --update: re-bless — rewrite each stub's stored hash to current (run after a
    human updates the prose to match a new signature). Touches only the front
    matter line.

Pure stdlib.

Usage:
  check_reference_staleness.py nitrix [--reference PATH] [--src PATH] [--update]
"""
import argparse
import ast
import pathlib
import re
import sys

import gen_reference_stubs as g   # sibling module: shared hashing logic

CODE = pathlib.Path(__file__).resolve().parent.parent.parent.parent


def front_matter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip().strip('"')
    return fm


def current_hash(modname, library, src_root):
    parts = modname.split(".")[1:]
    modfile = src_root.joinpath(*parts).with_suffix(".py")
    if not modfile.exists():
        return None
    tree = ast.parse(modfile.read_text())
    names = g.literal_all(tree)
    if not names:
        return None
    _, sig_hash = g.page_for_module(modname, names, g.collect(tree))
    return sig_hash


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("library")
    p.add_argument("--reference", default=None)
    p.add_argument("--src", default=None)
    p.add_argument("--update", action="store_true", help="re-bless stored hashes")
    a = p.parse_args()

    ref = pathlib.Path(a.reference) if a.reference else CODE / a.library / "docs" / "reference"
    src_root = pathlib.Path(a.src) if a.src else CODE / a.library / "src" / a.library

    ok = drift = missing = blessed = 0
    for qmd in sorted(ref.glob("*.qmd")):
        text = qmd.read_text()
        fm = front_matter(text)
        if fm.get("generated") != "true" or "signature-hash" not in fm:
            continue
        modname = fm.get("title", "")
        stored = fm["signature-hash"]
        cur = current_hash(modname, a.library, src_root)
        if cur is None:
            print(f"  ? {qmd.name}: source module '{modname}' gone or no __all__ — REVIEW")
            missing += 1
        elif cur == stored:
            ok += 1
        elif a.update:
            qmd.write_text(text.replace(f"signature-hash: {stored}",
                                        f"signature-hash: {cur}", 1))
            print(f"  ~ {qmd.name}: re-blessed {stored} -> {cur}")
            blessed += 1
        else:
            print(f"  ! {qmd.name}: signature DRIFT ({modname}: stored {stored} != current {cur})")
            drift += 1

    print(f"reference staleness: {ok} current, {drift} drifted, "
          f"{missing} missing-source, {blessed} re-blessed")
    if drift and not a.update:
        sys.exit(1)


if __name__ == "__main__":
    main()
